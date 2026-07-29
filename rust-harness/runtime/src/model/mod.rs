/*
 * model — the seam between the harness stack and the forward pass.
 *
 * Everything above this trait (sessions, sampler, grammar, scheduler, KV
 * bookkeeping) is written against it and tested with the scripted `MockModel`.
 * The real Gemma 4 E4B backend implements the same trait behind the `cuda`
 * feature.
 */

pub mod mock;

use anyhow::Result;

pub type Token = u32;

/// Logits for the final position of a prefill call. The mock returns a dense
/// vector; the real backend copies a trimmed slice off the GPU.
#[derive(Debug, Clone, PartialEq)]
pub struct Logits(pub Vec<f32>);

/// A causal LM with an owned, persistent KV cache.
///
/// Prefill and decode are the same operation at different lengths: `prefill`
/// appends tokens to the cached sequence and returns the logits at the final
/// position. A decode step is `prefill` of the single sampled token. This
/// makes delta prefill the only code path rather than a special case.
pub trait Model: Send {
    fn vocab_size(&self) -> usize;

    /// Number of tokens currently held in the KV cache.
    fn cached_len(&self) -> usize;

    /// Append `tokens` after the cached prefix and return logits for the
    /// final appended position. Cost is proportional to `tokens.len()`, never
    /// to `cached_len()`.
    fn prefill(&mut self, tokens: &[Token]) -> Result<Logits>;

    /// Drop cached state beyond `len` tokens. Used when a transcript diverges
    /// from the cached prefix: truncate to the common prefix, re-prefill the
    /// rest.
    fn truncate(&mut self, len: usize) -> Result<()>;

    /// Fork at the current cache boundary. The child starts with the parent's
    /// full cached prefix and pays no prefill for it; subsequent state is
    /// independent. Basis of subagent spawn.
    fn fork(&self) -> Result<Box<dyn Model>>;
}
