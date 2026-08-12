/*
 * model — the seam between the harness stack and the forward pass.
 *
 * Everything above this trait (sessions, sampler, grammar, scheduler, KV
 * bookkeeping) is written against it and tested with the scripted `MockModel`.
 * The real Gemma 4 E4B backend implements the same trait behind the `cuda`
 * feature.
 */

pub mod cpu;
pub mod mock;
pub mod probe;
pub mod record;

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
    ///
    /// On error the cache is left at *some* prefix of what it held, possibly
    /// empty, and never in the middle of a batch. The strong form — the cache
    /// is unchanged — holds wherever the backend can still rewind: a
    /// sliding-window cache that evicted during the failed call cannot, since
    /// the rows are gone, and clears instead.
    ///
    /// What must not happen is a *partial* commit. `Session` mirrors
    /// `cached_len()` in its own token list, and a backend that kept half a
    /// failed batch desynchronizes that mirror with no error to see, so every
    /// later position is computed against context the transcript does not
    /// contain. A shorter prefix is recoverable and `Session::sync` recovers
    /// it by reconciling against `cached_len()`; a wrong one is not.
    fn prefill(&mut self, tokens: &[Token]) -> Result<Logits>;

    /// Append `tokens` and return logits at every appended position, one row
    /// per token. Perplexity evaluation needs all positions, not just the
    /// tail. The default steps one token at a time — correct for any impl at
    /// O(n) forward passes; real backends override with one batched pass.
    fn prefill_all(&mut self, tokens: &[Token]) -> Result<Vec<Logits>> {
        tokens
            .iter()
            .map(|&t| self.prefill(&[t]))
            .collect::<Result<Vec<_>>>()
    }

    /// Append `tokens` and return logits from appended position `first`
    /// onward, one row per token from there.
    ///
    /// Perplexity scores only the second half of each window, and the tied LM
    /// head is the largest tensor in the checkpoint, so the rows a caller will
    /// not read are worth not computing. The default computes them and drops
    /// them: correct for any impl, and all the mock needs. A backend with a
    /// head worth skipping overrides *this* and defines `prefill_all` as
    /// `prefill_from(tokens, 0)`, rather than the other way round.
    fn prefill_from(&mut self, tokens: &[Token], first: usize) -> Result<Vec<Logits>> {
        anyhow::ensure!(
            first <= tokens.len(),
            "prefill from position {first} of only {} tokens",
            tokens.len()
        );
        let mut rows = self.prefill_all(tokens)?;
        Ok(rows.split_off(first))
    }

    /// The earliest position this cache can still be truncated to.
    ///
    /// Zero for a backend that retains everything, which is the default and
    /// what the mock and the recorder report. A backend with sliding-window
    /// layers evicts rather than masks, so once a window has filled the
    /// positions before this one are gone and no rewind reaches them.
    ///
    /// A query rather than a discovery: the alternative to asking is finding
    /// out mid-experiment, from a `truncate` that fails after the interesting
    /// state has already been built. `truncate(0)` is always available
    /// regardless — that is a reset, not a rewind.
    fn rewindable_from(&self) -> usize {
        0
    }

    /// Drop cached state beyond `len` tokens. Used when a transcript diverges
    /// from the cached prefix: truncate to the common prefix, re-prefill the
    /// rest.
    ///
    /// Errors below `rewindable_from`. Callers that may need to go further
    /// back — conversation compaction is the one that always does — ask first
    /// and fall back to `truncate(0)` plus a re-prefill.
    fn truncate(&mut self, len: usize) -> Result<()>;

    /// Fork at the current cache boundary. The child starts with the parent's
    /// full cached prefix and pays no prefill for it; subsequent state is
    /// independent. Basis of subagent spawn.
    fn fork(&self) -> Result<Box<dyn Model>>;

    /// Serialize the KV cache so a later process can resume without
    /// re-prefilling. The format is backend-defined; `load_state` of the
    /// same backend over the same weights must restore it exactly. Backends
    /// without persistent state (the mock) keep the default refusal.
    fn save_state(&self, _w: &mut dyn std::io::Write) -> Result<()> {
        anyhow::bail!("state serialization unsupported by this backend")
    }

    /// Restore a cache written by `save_state`. Only valid on a fresh model
    /// (empty cache) loaded from the same weights.
    fn load_state(&mut self, _r: &mut dyn std::io::Read) -> Result<()> {
        anyhow::bail!("state deserialization unsupported by this backend")
    }
}
