/*
 * subagents — fork-and-interleave scheduling.
 *
 * Subagents are cooperative: each starts as a fork of the parent session
 * (paying prefill only for its own task prompt) and generation interleaves
 * at token boundaries under a single GPU lock — exactly one generation step
 * in flight at any instant. Tool execution happens outside the lock, so one
 * agent's tool time overlaps another agent's model time. Batched decode is
 * out of scope by design.
 */

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use runtime::grammar::{Grammar, Vocab};
use runtime::model::Token;
use runtime::sampler::Sampler;
use runtime::session::{Generation, Session};
use tokio::sync::Mutex;

/// Serializes generation steps: the single-generation-in-flight invariant.
pub type GpuLock = Arc<Mutex<()>>;

pub fn gpu_lock() -> GpuLock {
    Arc::new(Mutex::new(()))
}

/// Spawn a subagent: fork the parent at its cache boundary and sync to the
/// child transcript (parent prefix + task prompt). Returns the child session
/// and the number of tokens prefilled, which is the task-prompt delta only.
pub fn spawn(parent: &Session, child_transcript: &[Token]) -> Result<(Session, usize)> {
    let mut child = parent.fork()?;
    let delta = child.sync(child_transcript)?;
    Ok((child, delta))
}

/// Generate while yielding the GPU at every token boundary. The lock is held
/// for exactly one step at a time; `step_time` models per-token compute so
/// schedule overlap is observable under tokio's paused test clock.
#[allow(clippy::too_many_arguments)]
pub async fn generate_interleaved(
    gpu: &GpuLock,
    session: &mut Session,
    transcript: &[Token],
    vocab: &Vocab,
    grammar: Option<&Grammar>,
    sampler: &mut Sampler,
    max_tokens: usize,
    step_time: Duration,
    mut on_token: impl FnMut(Token),
) -> Result<Generation> {
    let mut active = {
        // Prefill is GPU work: sync and first-logits happen under the lock.
        let _hold = gpu.lock().await;
        session.begin_generation(transcript, vocab, grammar)?
    };
    let mut out = Vec::new();
    while out.len() < max_tokens {
        let step = {
            let _hold = gpu.lock().await;
            tokio::time::sleep(step_time).await;
            active.step(sampler)?
        };
        match step {
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
        // Token-boundary yield point: let another subagent take the GPU.
        tokio::task::yield_now().await;
    }
    Ok(Generation {
        tokens: out,
        stopped_at_eos: false,
    })
}
