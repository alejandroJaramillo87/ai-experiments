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

const SNAPSHOT_MAGIC: &[u8; 4] = b"HKV1";
const SNAPSHOT_VERSION: u32 = 1;
const SNAPSHOT_DTYPE_F32: u8 = 0;

/// FNV-1a 64-bit hash, used to fingerprint model weights for snapshot
/// compatibility checks. Corruption and mismatch detection only — not a
/// cryptographic guarantee.
pub fn fingerprint(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub struct Session {
    model: Box<dyn Model>,
    // Hard cap on cached tokens: transcript sync and generation both error
    // instead of growing the KV cache past it.
    max_context: usize,
    // Token history mirroring the KV cache, for common-prefix computation.
    tokens: Vec<Token>,
    // Logits at the cache tail from the most recent prefill, when still
    // valid. Reusing them saves a redundant replay forward pass per
    // generation; a truncate-only sync invalidates them.
    tail_logits: Option<Logits>,
}

/// Outcome of one generation: sampled tokens (EOS excluded) and whether EOS
/// ended it (as opposed to the max-token bound).
#[derive(Debug, PartialEq)]
pub struct Generation {
    pub tokens: Vec<Token>,
    pub stopped_at_eos: bool,
}

impl Session {
    pub fn new(model: Box<dyn Model>, max_context: usize) -> Result<Session> {
        anyhow::ensure!(
            model.cached_len() == 0,
            "session requires a fresh model cache, found {} cached tokens",
            model.cached_len()
        );
        anyhow::ensure!(max_context > 0, "context budget must be positive");
        Ok(Self {
            model,
            max_context,
            tokens: Vec::new(),
            tail_logits: None,
        })
    }

    pub fn cached_len(&self) -> usize {
        self.tokens.len()
    }

    pub fn max_context(&self) -> usize {
        self.max_context
    }

    /// Sync the cache to `transcript`: truncate to the common prefix, prefill
    /// the rest. Returns the number of tokens actually prefilled — the delta.
    ///
    /// `Model::prefill` is required to be atomic — all of its tokens land in
    /// the KV cache, or none do when it errors — and `Session` is the only
    /// place that requirement is load-bearing: `tokens` is a mirror of the
    /// cache, and a backend that commits part of a batch before failing puts
    /// the two out of step. Sync therefore reconciles against the model
    /// before prefilling instead of trusting the mirror, so a failed sync
    /// stays recoverable by retrying it.
    pub fn sync(&mut self, transcript: &[Token]) -> Result<usize> {
        let span = tracing::info_span!(
            "session.sync",
            cached_len = self.tokens.len(),
            delta_len = tracing::field::Empty
        );
        let _enter = span.enter();
        anyhow::ensure!(
            transcript.len() <= self.max_context,
            "transcript of {} tokens exceeds the context budget of {}",
            transcript.len(),
            self.max_context
        );
        let common = self
            .tokens
            .iter()
            .zip(transcript)
            .take_while(|(a, b)| a == b)
            .count();
        // Only positions the model and the token history agree on survive:
        // the model runs ahead when a prefill committed part of its batch
        // before failing, and behind when a tail replay dropped a position
        // and then failed. Both lengths already equal `common` on the normal
        // path, where the two truncates below are no-ops.
        let keep = common.min(self.model.cached_len());
        if self.model.cached_len() > keep {
            self.model.truncate(keep)?;
        }
        if keep < self.tokens.len() {
            self.tokens.truncate(keep);
            self.tail_logits = None;
        }
        let delta = &transcript[keep..];
        span.record("delta_len", delta.len());
        if !delta.is_empty() {
            self.tail_logits = Some(self.model.prefill(delta)?);
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
        self.ensure_tail_logits()?;
        let state = grammar.map(|g| g.start());
        Ok(ActiveGeneration {
            session: self,
            vocab,
            grammar,
            state,
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

    /// Make tail logits available, replaying the final cached token
    /// (truncate one, prefill one) only when the cached copy was invalidated.
    /// A replay that fails after its truncate leaves the model one position
    /// behind `tokens`; the next `sync` reconciles that back.
    fn ensure_tail_logits(&mut self) -> Result<()> {
        if self.tail_logits.is_none() {
            let last = *self.tokens.last().expect("non-empty cache");
            self.model.truncate(self.tokens.len() - 1)?;
            self.tail_logits = Some(self.model.prefill(&[last])?);
        }
        Ok(())
    }

    /// Serialize the session state — token history plus the model's KV
    /// cache — so a restarted process can resume without re-prefilling.
    /// `model_hash` fingerprints the weights the cache was computed from;
    /// `restore` refuses a mismatch.
    ///
    /// Format (all little-endian): magic "HKV1", version u32, dtype u8
    /// (0 = f32; 1 reserved for f16 payloads), model hash u64, token count
    /// u64, tokens u32 each, then the backend-defined cache payload.
    pub fn save(&self, w: &mut dyn std::io::Write, model_hash: u64) -> Result<()> {
        w.write_all(SNAPSHOT_MAGIC)?;
        w.write_all(&SNAPSHOT_VERSION.to_le_bytes())?;
        w.write_all(&[SNAPSHOT_DTYPE_F32])?;
        w.write_all(&model_hash.to_le_bytes())?;
        w.write_all(&(self.tokens.len() as u64).to_le_bytes())?;
        for &t in &self.tokens {
            w.write_all(&t.to_le_bytes())?;
        }
        self.model.save_state(w)
    }

    /// Restore a snapshot written by `save` into a fresh session. Any header
    /// mismatch — magic, version, dtype, model hash — is a hard error: a
    /// cache from different weights must never be trusted silently. Whether
    /// the restored tokens still match the transcript is not checked here;
    /// the next `sync` prefix-matches and falls back to re-prefilling from
    /// the divergence point, so a stale-but-well-formed snapshot degrades
    /// safely.
    pub fn restore(&mut self, r: &mut dyn std::io::Read, model_hash: u64) -> Result<()> {
        anyhow::ensure!(self.tokens.is_empty(), "restore into a non-fresh session");
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        anyhow::ensure!(&magic == SNAPSHOT_MAGIC, "not a session snapshot file");
        let mut u32buf = [0u8; 4];
        r.read_exact(&mut u32buf)?;
        let version = u32::from_le_bytes(u32buf);
        anyhow::ensure!(
            version == SNAPSHOT_VERSION,
            "snapshot version {version} unsupported (expected {SNAPSHOT_VERSION})"
        );
        let mut dtype = [0u8; 1];
        r.read_exact(&mut dtype)?;
        anyhow::ensure!(
            dtype[0] == SNAPSHOT_DTYPE_F32,
            "snapshot dtype {} unsupported",
            dtype[0]
        );
        let mut u64buf = [0u8; 8];
        r.read_exact(&mut u64buf)?;
        let saved_hash = u64::from_le_bytes(u64buf);
        anyhow::ensure!(
            saved_hash == model_hash,
            "snapshot was written for different model weights \
             (hash {saved_hash:016x}, expected {model_hash:016x})"
        );
        r.read_exact(&mut u64buf)?;
        let n_tokens = u64::from_le_bytes(u64buf) as usize;
        anyhow::ensure!(
            n_tokens <= self.max_context,
            "snapshot of {n_tokens} tokens exceeds the context budget of {}",
            self.max_context
        );
        let mut tokens = Vec::with_capacity(n_tokens);
        for _ in 0..n_tokens {
            r.read_exact(&mut u32buf)?;
            tokens.push(u32::from_le_bytes(u32buf));
        }
        self.model.load_state(r)?;
        anyhow::ensure!(
            self.model.cached_len() == tokens.len(),
            "restored cache length {} does not match {} saved tokens",
            self.model.cached_len(),
            tokens.len()
        );
        self.tokens = tokens;
        self.tail_logits = None; // ensure_tail_logits replays one token
        Ok(())
    }

    /// Fork at the current boundary; the child pays no prefill for the prefix.
    pub fn fork(&self) -> Result<Session> {
        Ok(Session {
            model: self.model.fork()?,
            max_context: self.max_context,
            tokens: self.tokens.clone(),
            tail_logits: self.tail_logits.clone(),
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
    done: bool,
}

impl ActiveGeneration<'_> {
    pub fn step(&mut self, sampler: &mut Sampler) -> Result<Option<Token>> {
        anyhow::ensure!(!self.done, "step after EOS");
        let span = tracing::info_span!(
            "session.step",
            position = self.session.tokens.len(),
            token = tracing::field::Empty
        );
        let _enter = span.enter();
        let mask = self.state.as_ref().map(|st| {
            self.grammar
                .expect("state implies grammar")
                .mask(st, self.vocab)
        });
        let logits = self
            .session
            .tail_logits
            .as_ref()
            .expect("begin_generation established tail logits");
        let tok = sampler.sample(logits, mask.as_deref())?;
        span.record("token", tok);
        if tok == self.vocab.eos() {
            self.done = true;
            return Ok(None);
        }
        // Deliberate ordering: the budget is checked after sampling because
        // EOS is legal at exactly the budget; the failing step therefore
        // consumes one sampler draw.
        anyhow::ensure!(
            self.session.tokens.len() < self.session.max_context,
            "context budget of {} reached during generation",
            self.session.max_context
        );
        // Compute the next grammar state without mutating, and commit it only
        // after the prefill succeeds: a transient prefill error must not
        // leave the state one token ahead of the cache for a retried step.
        let next_state = match (self.grammar, self.state.as_ref()) {
            (Some(g), Some(st)) => Some(
                // Defense in depth: the mask already excludes illegal tokens.
                g.advance_str(st, self.vocab.text(tok))
                    .ok_or_else(|| anyhow::anyhow!("sampled grammar-illegal token"))?,
            ),
            _ => None,
        };
        self.session.tail_logits = Some(self.session.model.prefill(&[tok])?);
        self.session.tokens.push(tok);
        if let Some(st) = next_state {
            self.state = Some(st);
        }
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
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        assert_eq!(s.sync(&[1, 2, 3]).unwrap(), 3);
        assert_eq!(s.sync(&[1, 2, 3, 4, 5]).unwrap(), 2);
        assert_eq!(s.cached_len(), 5);
    }

    #[test]
    fn sync_truncates_to_common_prefix_on_divergence() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
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
        let mut s = Session::new(Box::new(MockModel::new(v.len(), script)), 4096).unwrap();
        let mut streamed = Vec::new();
        let gen = s
            .generate(&prompt, &v, None, &mut greedy(), 32, |t| streamed.push(t))
            .unwrap();
        assert_eq!(gen.tokens, output);
        assert!(gen.stopped_at_eos);
        assert_eq!(streamed, output);
        assert_eq!(s.cached_len(), prompt.len() + output.len());
    }

    /// Delegating wrapper that fails exactly one prefill call (the `fail_at`-th),
    /// then recovers: the transient-error case a retried `step` must survive.
    /// The failing call first commits `commit_on_failure` tokens of its batch
    /// to the inner cache, modelling a backend whose prefill is not atomic.
    struct FlakyModel {
        inner: MockModel,
        calls: usize,
        fail_at: usize,
        commit_on_failure: usize,
    }

    impl crate::model::Model for FlakyModel {
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
        fn cached_len(&self) -> usize {
            self.inner.cached_len()
        }
        fn prefill(&mut self, tokens: &[Token]) -> Result<crate::model::Logits> {
            self.calls += 1;
            if self.calls == self.fail_at {
                if self.commit_on_failure > 0 {
                    self.inner.prefill(&tokens[..self.commit_on_failure])?;
                }
                anyhow::bail!("transient prefill failure");
            }
            self.inner.prefill(tokens)
        }
        fn truncate(&mut self, len: usize) -> Result<()> {
            self.inner.truncate(len)
        }
        fn fork(&self) -> Result<Box<dyn crate::model::Model>> {
            unimplemented!("not forked in this test")
        }
    }

    #[test]
    fn retried_step_after_transient_prefill_error_stays_grammar_consistent() {
        // Regression: the grammar state used to advance before the token's
        // prefill, so a failed prefill left the state one token ahead and a
        // retried step skipped a character of the skeleton.
        let skeleton = r#"{"tool": "ab", "args": {}}"#;
        let mut tokens: Vec<String> = skeleton
            .chars()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|c| c.to_string())
            .collect();
        tokens.push("</s>".to_string());
        let eos = (tokens.len() - 1) as u32;
        let v = Vocab::new(tokens.clone(), eos);
        let grammar = Grammar::compile(&crate::grammar::ToolSchema {
            name: "ab".into(),
            params: vec![],
        });
        // Uniform logits: the grammar mask forces the single legal token.
        let script = vec![vec![1.0; v.len()]; 128];
        let model = FlakyModel {
            inner: MockModel::new(v.len(), script),
            calls: 0,
            fail_at: 6, // prompt prefill is call 1; fail mid-generation
            commit_on_failure: 0,
        };
        let mut s = Session::new(Box::new(model), 4096).unwrap();
        let mut active = s.begin_generation(&[0], &v, Some(&grammar)).unwrap();
        let mut sampler = greedy();
        let mut out = String::new();
        loop {
            match active.step(&mut sampler) {
                Ok(Some(tok)) => out.push_str(&tokens[tok as usize]),
                Ok(None) => break,
                Err(e) => assert!(e.to_string().contains("transient prefill failure")),
            }
        }
        assert_eq!(out, skeleton);
    }

    #[test]
    fn sync_after_a_prefill_committed_tokens_before_failing_matches_a_clean_run() {
        // Regression: a prefill that committed part of its batch before
        // erroring left the model cache ahead of the session's history, and
        // the retried sync prefilled the whole prompt on top of the orphaned
        // positions — every later logit came from the wrong position instead
        // of the retry either working or failing.
        let v = vocab();
        let prompt = [0u32, 1, 2, 3, 4];
        let script = script_emitting(v.len(), prompt.len(), &[7, 8], v.eos());

        let mut clean =
            Session::new(Box::new(MockModel::new(v.len(), script.clone())), 4096).unwrap();
        let expected = clean
            .generate(&prompt, &v, None, &mut greedy(), 8, |_| {})
            .unwrap();

        let model = FlakyModel {
            inner: MockModel::new(v.len(), script),
            calls: 0,
            fail_at: 1,           // the prompt prefill
            commit_on_failure: 3, // three of the five tokens land anyway
        };
        let mut s = Session::new(Box::new(model), 4096).unwrap();
        let err = s.sync(&prompt).unwrap_err();
        assert!(err.to_string().contains("prefill failure"), "{err}");
        assert_eq!(s.cached_len(), 0);

        let gen = s
            .generate(&prompt, &v, None, &mut greedy(), 8, |_| {})
            .unwrap();
        assert_eq!(gen, expected);
        assert_eq!(s.tail_logits, clean.tail_logits);
        assert_eq!(s.cached_len(), clean.cached_len());
    }

    #[test]
    fn sync_re_prefills_the_position_a_failed_tail_replay_dropped_from_the_model() {
        // The mirror case: a replay inside ensure_tail_logits truncates one
        // position and then fails to prefill it, leaving the model behind the
        // session's history. Without reconciling, the next sync writes its
        // delta over the dropped position and generation silently continues
        // from logits one position early.
        let v = vocab();
        let transcript = [0u32, 1, 2, 4, 5];
        let script = script_emitting(v.len(), transcript.len(), &[7, 8], v.eos());

        let mut clean =
            Session::new(Box::new(MockModel::new(v.len(), script.clone())), 4096).unwrap();
        let expected = clean
            .generate(&transcript, &v, None, &mut greedy(), 8, |_| {})
            .unwrap();

        let model = FlakyModel {
            inner: MockModel::new(v.len(), script),
            calls: 0,
            fail_at: 2, // the tail replay, not the prompt prefill
            commit_on_failure: 0,
        };
        let mut s = Session::new(Box::new(model), 4096).unwrap();
        s.sync(&[0, 1, 2, 3]).unwrap();
        // A truncate-only sync drops the cached tail logits, so the next
        // generation has to replay the final token — the failing prefill.
        s.sync(&[0, 1, 2]).unwrap();
        // ActiveGeneration is not Debug; discard the Ok arm to unwrap_err.
        let err = s
            .begin_generation(&[0, 1, 2], &v, None)
            .map(|_| ())
            .unwrap_err();
        assert!(err.to_string().contains("prefill failure"), "{err}");

        let gen = s
            .generate(&transcript, &v, None, &mut greedy(), 8, |_| {})
            .unwrap();
        assert_eq!(gen, expected);
        assert_eq!(s.tail_logits, clean.tail_logits);
        assert_eq!(s.cached_len(), clean.cached_len());
    }

    #[test]
    fn generate_respects_max_tokens() {
        let v = vocab();
        // Script always points at token 3; no EOS in range.
        let mut always = vec![vec![0.0; v.len()]; 64];
        for l in always.iter_mut() {
            l[3] = 1.0;
        }
        let mut s = Session::new(Box::new(MockModel::new(v.len(), always)), 4096).unwrap();
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
        let mut s = Session::new(Box::new(MockModel::new(v.len(), script)), 4096).unwrap();
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

    #[test]
    fn begin_generation_reuses_sync_logits_without_a_replay_prefill() {
        let v = vocab();
        let prompt = [0u32, 1, 2];
        let script = script_emitting(v.len(), prompt.len(), &[7], v.eos());
        let mock = MockModel::new(v.len(), script);
        let recorder = mock.recorder();
        let mut s = Session::new(Box::new(mock), 4096).unwrap();
        s.begin_generation(&prompt, &v, None).unwrap();
        // Only the 3-token prompt prefill; no truncate-and-replay of the tail.
        assert_eq!(recorder.total_prefilled(), 3);
    }

    #[test]
    fn sync_to_empty_transcript_truncates_to_zero() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        s.sync(&[1, 2, 3]).unwrap();
        assert_eq!(s.sync(&[]).unwrap(), 0);
        assert_eq!(s.cached_len(), 0);
    }

    #[test]
    fn step_after_eos_is_an_error() {
        let v = vocab();
        let prompt = [0u32, 1];
        let script = script_emitting(v.len(), prompt.len(), &[], v.eos());
        let mut s = Session::new(Box::new(MockModel::new(v.len(), script)), 4096).unwrap();
        let mut active = s.begin_generation(&prompt, &v, None).unwrap();
        let mut sampler = greedy();
        assert_eq!(active.step(&mut sampler).unwrap(), None);
        assert!(active.step(&mut sampler).is_err());
    }

    #[test]
    fn max_tokens_zero_generates_nothing() {
        let v = vocab();
        let prompt = [0u32, 1];
        let script = script_emitting(v.len(), prompt.len(), &[5], v.eos());
        let mut s = Session::new(Box::new(MockModel::new(v.len(), script)), 4096).unwrap();
        let gen = s
            .generate(&prompt, &v, None, &mut greedy(), 0, |_| {})
            .unwrap();
        assert_eq!(gen.tokens, Vec::<u32>::new());
        assert!(!gen.stopped_at_eos);
    }

    #[test]
    fn sync_beyond_the_context_budget_is_an_error() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4).unwrap();
        assert_eq!(s.sync(&[1, 2, 3, 4]).unwrap(), 4);
        let err = s.sync(&[1, 2, 3, 4, 5]).unwrap_err();
        assert!(err.to_string().contains("context budget"), "{err}");
        // Failed sync left the cache untouched.
        assert_eq!(s.cached_len(), 4);
    }

    #[test]
    fn generation_stops_with_an_error_at_the_context_budget() {
        let v = vocab();
        // Script always points at token 3; no EOS in range.
        let mut always = vec![vec![0.0; v.len()]; 16];
        for l in always.iter_mut() {
            l[3] = 1.0;
        }
        let mut s = Session::new(Box::new(MockModel::new(v.len(), always)), 4).unwrap();
        let err = s
            .generate(&[0, 1], &v, None, &mut greedy(), 32, |_| {})
            .unwrap_err();
        assert!(err.to_string().contains("during generation"), "{err}");
    }

    #[test]
    fn non_fresh_model_and_zero_budget_are_construction_errors() {
        let mut used = MockModel::new(4, vec![]);
        used.prefill(&[1]).unwrap();
        assert!(Session::new(Box::new(used), 16).is_err());
        assert!(Session::new(Box::new(MockModel::new(4, vec![])), 0).is_err());
    }

    #[test]
    fn snapshot_round_trip_restores_tokens_and_skips_the_prefill() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        s.sync(&[1, 2, 3, 4]).unwrap();
        let mut buf = Vec::new();
        s.save(&mut buf, 0xabcd).unwrap();

        let mock = MockModel::new(27, vec![]);
        let recorder = mock.recorder();
        let mut restored = Session::new(Box::new(mock), 4096).unwrap();
        restored.restore(&mut buf.as_slice(), 0xabcd).unwrap();
        assert_eq!(restored.cached_len(), 4);
        // Syncing to the same transcript prefills nothing.
        assert_eq!(restored.sync(&[1, 2, 3, 4]).unwrap(), 0);
        assert_eq!(recorder.total_prefilled(), 0);
        // A longer transcript prefills only the appended delta.
        assert_eq!(restored.sync(&[1, 2, 3, 4, 5, 6]).unwrap(), 2);
    }

    #[test]
    fn restore_refuses_a_different_model_hash() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        s.sync(&[1, 2]).unwrap();
        let mut buf = Vec::new();
        s.save(&mut buf, 1).unwrap();
        let mut fresh = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        let err = fresh.restore(&mut buf.as_slice(), 2).unwrap_err();
        assert!(err.to_string().contains("different model weights"), "{err}");
        assert_eq!(fresh.cached_len(), 0);
    }

    #[test]
    fn restore_refuses_bad_magic_version_and_oversized_snapshots() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        s.sync(&[1, 2]).unwrap();
        let mut buf = Vec::new();
        s.save(&mut buf, 7).unwrap();

        let mut fresh = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        let err = fresh.restore(&mut b"nope".as_slice(), 7).unwrap_err();
        assert!(err.to_string().contains("not a session snapshot"), "{err}");

        let mut wrong_version = buf.clone();
        wrong_version[4] = 99;
        let err = fresh.restore(&mut wrong_version.as_slice(), 7).unwrap_err();
        assert!(err.to_string().contains("version"), "{err}");

        // The two saved tokens exceed a budget of one.
        let mut tiny = Session::new(Box::new(MockModel::new(27, vec![])), 1).unwrap();
        let err = tiny.restore(&mut buf.as_slice(), 7).unwrap_err();
        assert!(err.to_string().contains("context budget"), "{err}");
    }

    #[test]
    fn restore_into_a_used_session_is_an_error() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        s.sync(&[1, 2]).unwrap();
        let mut buf = Vec::new();
        s.save(&mut buf, 7).unwrap();
        let err = s.restore(&mut buf.as_slice(), 7).unwrap_err();
        assert!(err.to_string().contains("non-fresh"), "{err}");
    }

    #[test]
    fn restored_session_falls_back_on_a_diverged_transcript() {
        let mut s = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        s.sync(&[1, 2, 3, 4]).unwrap();
        let mut buf = Vec::new();
        s.save(&mut buf, 7).unwrap();

        let mut restored = Session::new(Box::new(MockModel::new(27, vec![])), 4096).unwrap();
        restored.restore(&mut buf.as_slice(), 7).unwrap();
        // The transcript moved on differently: prefix matching truncates to
        // the common prefix and prefills only from the divergence point.
        assert_eq!(restored.sync(&[1, 2, 9, 9]).unwrap(), 2);
        assert_eq!(restored.cached_len(), 4);
    }

    #[test]
    fn generation_under_a_grammar_yields_schema_valid_json() {
        use crate::grammar::{Grammar, ParamType, ToolSchema};
        // Character-level vocab covering the skeleton plus value chars.
        let mut tokens: Vec<String> = "{}\"argstol: ,ping123"
            .chars()
            .map(|c| c.to_string())
            .collect();
        tokens.push("</s>".to_string());
        let eos = (tokens.len() - 1) as u32;
        let v = Vocab::new(tokens, eos);
        let grammar = Grammar::compile(&ToolSchema {
            name: "ping".into(),
            params: vec![("n".into(), ParamType::Int)],
        });
        // Flat positive logits: the mask alone steers the walk.
        let script = vec![vec![1.0; v.len()]; 512];
        let mut s = Session::new(Box::new(MockModel::new(v.len(), script)), 4096).unwrap();
        let mut sampler = Sampler::new(SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            seed: 5,
        });
        let gen = s
            .generate(&[0, 1], &v, Some(&grammar), &mut sampler, 256, |_| {})
            .unwrap();
        assert!(gen.stopped_at_eos, "grammar walk must reach completion");
        let text: String = gen.tokens.iter().map(|&t| v.text(t)).collect();
        let parsed: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(parsed["tool"], "ping");
        assert!(parsed["args"]["n"].is_number());
    }
}
