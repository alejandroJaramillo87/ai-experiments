/*
 * generator — the text↔token seam between the agent loop and the runtime.
 *
 * `SessionGenerator` implements the loop's text-level `Generator` trait over
 * a runtime `Session` plus `Tokenizer`: encode the rendered prompt, generate
 * with the persistent KV cache (delta prefill comes free from Session::sync),
 * decode the sampled tokens back to text.
 */

use anyhow::Result;

use runtime::grammar::Vocab;
use runtime::sampler::{Sampler, SamplingParams};
use runtime::session::Session;
use runtime::tokenizer::Tokenizer;

use crate::agent::Generator;

pub struct SessionGenerator {
    session: Session,
    tokenizer: Tokenizer,
    // Grammar-mask view over the same vocabulary table as the tokenizer.
    vocab: Vocab,
    sampler: Sampler,
    max_tokens: usize,
}

impl SessionGenerator {
    pub fn new(
        session: Session,
        tokenizer: Tokenizer,
        params: SamplingParams,
        max_tokens: usize,
    ) -> Result<SessionGenerator> {
        let texts = (0..tokenizer.vocab_len())
            .map(|i| Ok(tokenizer.piece(i as u32)?.to_string()))
            .collect::<Result<Vec<_>>>()?;
        let vocab = Vocab::new(texts, tokenizer.eos());
        Ok(SessionGenerator {
            session,
            tokenizer,
            vocab,
            sampler: Sampler::new(params),
            max_tokens,
        })
    }

    pub fn session(&self) -> &Session {
        &self.session
    }
}

impl Generator for SessionGenerator {
    fn complete(&mut self, prompt: &str) -> Result<String> {
        let mut ids = vec![self.tokenizer.bos()];
        ids.extend(self.tokenizer.encode(prompt)?);
        let generation = self.session.generate(
            &ids,
            &self.vocab,
            None,
            &mut self.sampler,
            self.max_tokens,
            |_| {},
        )?;
        self.tokenizer.decode(&generation.tokens)
    }

    fn count_tokens(&mut self, text: &str) -> Result<usize> {
        // +1 for the BOS `complete` prepends to every prompt.
        Ok(self.tokenizer.encode(text)?.len() + 1)
    }

    fn context_budget(&self) -> Option<usize> {
        Some(self.session.max_context())
    }
}
