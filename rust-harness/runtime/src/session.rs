/*
 * session — conversation handle owning the KV cache.
 *
 * The harness appends transcript tokens; the session prefills only tokens
 * beyond the cached prefix. Divergence from the cached prefix truncates the
 * cache to the common prefix and re-prefills from there. Forking a session
 * is the spawn primitive for subagents: the child inherits the cached prefix
 * without re-prefilling it.
 */

use anyhow::Result;

use super::grammar::{Grammar, Vocab};
use super::model::{Logits, Model, Token};
use super::sampler::Sampler;

pub struct Session {
    model: Box<dyn Model>,
    // Token history mirroring the KV cache, for common-prefix computation.
    tokens: Vec<Token>,
}

/// Outcome of one generation: sampled tokens (EOS excluded) and whether EOS
/// ended it (as opposed to the max-token bound).
#[derive(Debug, PartialEq)]
pub struct Generation {
    pub tokens: Vec<Token>,
    pub stopped_at_eos: bool,
}

impl Session {
    pub fn new(model: Box<dyn Model>) -> Self {
        assert_eq!(
            model.cached_len(),
            0,
            "session requires a fresh model cache"
        );
        Self {
            model,
            tokens: Vec::new(),
        }
    }

    pub fn cached_len(&self) -> usize {
        self.tokens.len()
    }

    /// Sync the cache to `transcript`: truncate to the common prefix, prefill
    /// the rest. Returns the number of tokens actually prefilled — the delta.
    pub fn sync(&mut self, transcript: &[Token]) -> Result<usize> {
        let common = self
            .tokens
            .iter()
            .zip(transcript)
            .take_while(|(a, b)| a == b)
            .count();
        if common < self.tokens.len() {
            self.model.truncate(common)?;
            self.tokens.truncate(common);
        }
        let delta = &transcript[common..];
        if !delta.is_empty() {
            self.model.prefill(delta)?;
            self.tokens.extend_from_slice(delta);
        }
        Ok(delta.len())
    }

    /// Start a generation after syncing to `transcript`. The returned handle
    /// advances one token per `step` call — the token-boundary yield point
    /// the subagent interleaver schedules around.
    pub fn begin_generation<'a>(
        &'a mut self,
        transcript: &[Token],
        vocab: &'a Vocab,
        grammar: Option<&'a Grammar>,
    ) -> Result<ActiveGeneration<'a>> {
        anyhow::ensure!(!transcript.is_empty(), "generate on empty transcript");
        self.sync(transcript)?;
        let logits = self.last_logits()?;
        let state = grammar.map(|g| g.start());
        Ok(ActiveGeneration {
            session: self,
            vocab,
            grammar,
            state,
            logits,
            done: false,
        })
    }

    /// Generate after syncing to `transcript`. Each sampled token is fed back
    /// through a single-token prefill; `on_token` observes tokens as they are
    /// produced (streaming seam). Grammar masking applies when provided.
    pub fn generate(
        &mut self,
        transcript: &[Token],
        vocab: &Vocab,
        grammar: Option<&Grammar>,
        sampler: &mut Sampler,
        max_tokens: usize,
        mut on_token: impl FnMut(Token),
    ) -> Result<Generation> {
        let mut active = self.begin_generation(transcript, vocab, grammar)?;
        let mut out = Vec::new();
        while out.len() < max_tokens {
            match active.step(sampler)? {
                Some(tok) => {
                    on_token(tok);
                    out.push(tok);
                }
                None => {
                    return Ok(Generation {
                        tokens: out,
                        stopped_at_eos: true,
                    })
                }
            }
        }
        Ok(Generation {
            tokens: out,
            stopped_at_eos: false,
        })
    }

    /// Logits at the current cache tail, obtained by replaying the final
    /// cached token (truncate one, prefill one). Cost is one token.
    fn last_logits(&mut self) -> Result<Logits> {
        let last = *self.tokens.last().expect("non-empty cache");
        self.model.truncate(self.tokens.len() - 1)?;
        let logits = self.model.prefill(&[last])?;
        Ok(logits)
    }

    /// Fork at the current boundary; the child pays no prefill for the prefix.
    pub fn fork(&self) -> Result<Session> {
        Ok(Session {
            model: self.model.fork()?,
            tokens: self.tokens.clone(),
        })
    }
}

/// An in-progress generation. Each `step` samples one token and prefills it;
/// EOS ends the generation with `None`. Dropping the handle simply stops at
/// the last completed token boundary — the cache stays valid.
pub struct ActiveGeneration<'a> {
    session: &'a mut Session,
    vocab: &'a Vocab,
    grammar: Option<&'a Grammar>,
    state: Option<crate::grammar::GrammarState>,
    logits: Logits,
    done: bool,
}

impl ActiveGeneration<'_> {
    pub fn step(&mut self, sampler: &mut Sampler) -> Result<Option<Token>> {
        anyhow::ensure!(!self.done, "step after EOS");
        let mask = self.state.as_ref().map(|st| {
            self.grammar
                .expect("state implies grammar")
                .mask(st, self.vocab)
        });
        let tok = sampler.sample(&self.logits, mask.as_deref())?;
        if tok == self.vocab.eos() {
            self.done = true;
            return Ok(None);
        }
        if let (Some(g), Some(st)) = (self.grammar, self.state.as_mut()) {
            for c in self.vocab.text(tok).chars() {
                *st = g
                    .advance(st, c)
                    .ok_or_else(|| anyhow::anyhow!("sampled grammar-illegal token"))?;
            }
        }
        self.logits = self.session.model.prefill(&[tok])?;
        self.session.tokens.push(tok);
        Ok(Some(tok))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mock::MockModel;
    use crate::sampler::SamplingParams;

    fn vocab() -> Vocab {
        let mut tokens: Vec<String> = (b'a'..=b'z').map(|b| (b as char).to_string()).collect();
        tokens.push("</s>".to_string());
        let eos = (tokens.len() - 1) as u32;
        Vocab::new(tokens, eos)
    }

    /// Script that deterministically emits `output` (token ids) after a
    /// prompt of length `start`, then EOS. Logits at position p select the
    /// token appearing at position p + 1.
    fn script_emitting(vocab_len: usize, start: usize, output: &[u32], eos: u32) -> Vec<Vec<f32>> {
        let mut script = vec![vec![0.0; vocab_len]; start + output.len()];
        for (i, &tok) in output.iter().enumerate() {
            script[start - 1 + i][tok as usize] = 1.0;
        }
        script[start - 1 + output.len()][eos as usize] = 1.0;
        script
    }

    fn greedy() -> Sampler {
        Sampler::new(SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: 0,
        })
    }

    #[test]
    fn sync_prefills_only_the_delta() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])));
        assert_eq!(s.sync(&[1, 2, 3]).unwrap(), 3);
        assert_eq!(s.sync(&[1, 2, 3, 4, 5]).unwrap(), 2);
        assert_eq!(s.cached_len(), 5);
    }

    #[test]
    fn sync_truncates_to_common_prefix_on_divergence() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])));
        s.sync(&[1, 2, 3, 4]).unwrap();
        // Diverges after position 1: re-prefill is the suffix from there.
        assert_eq!(s.sync(&[1, 2, 9, 9, 9]).unwrap(), 3);
        assert_eq!(s.cached_len(), 5);
    }

    #[test]
    fn generate_emits_scripted_tokens_until_eos() {
        let v = vocab();
        let prompt = [0u32, 1, 2];
        let output = [7u32, 8, 9];
        let script = script_emitting(v.len(), prompt.len(), &output, v.eos());
        let mut s = Session::new(Box::new(MockModel::new(v.len(), script)));
        let mut streamed = Vec::new();
        let gen = s
            .generate(&prompt, &v, None, &mut greedy(), 32, |t| streamed.push(t))
            .unwrap();
        assert_eq!(gen.tokens, output);
        assert!(gen.stopped_at_eos);
        assert_eq!(streamed, output);
        assert_eq!(s.cached_len(), prompt.len() + output.len());
    }

    #[test]
    fn generate_respects_max_tokens() {
        let v = vocab();
        // Script always points at token 3; no EOS in range.
        let mut always = vec![vec![0.0; v.len()]; 64];
        for l in always.iter_mut() {
            l[3] = 1.0;
        }
        let mut s = Session::new(Box::new(MockModel::new(v.len(), always)));
        let gen = s
            .generate(&[0, 1], &v, None, &mut greedy(), 4, |_| {})
            .unwrap();
        assert_eq!(gen.tokens, vec![3, 3, 3, 3]);
        assert!(!gen.stopped_at_eos);
    }

    #[test]
    fn second_turn_prefills_only_the_appended_delta() {
        let v = vocab();
        let prompt = [0u32, 1, 2];
        let output = [7u32];
        let script = script_emitting(v.len(), prompt.len(), &output, v.eos());
        let mut s = Session::new(Box::new(MockModel::new(v.len(), script)));
        s.generate(&prompt, &v, None, &mut greedy(), 8, |_| {})
            .unwrap();
        let len_after_turn1 = s.cached_len();
        // Turn 2 transcript: everything so far plus a two-token user delta.
        let mut transcript: Vec<u32> = prompt.to_vec();
        transcript.extend([7u32]); // model output now part of transcript
        transcript.extend([4u32, 5]);
        assert_eq!(s.sync(&transcript).unwrap(), 2);
        assert_eq!(s.cached_len(), len_after_turn1 + 2);
    }
}
