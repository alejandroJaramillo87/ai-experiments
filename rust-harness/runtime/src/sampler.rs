/*
 * sampler — temperature/top-p sampling with optional grammar masking.
 *
 * Deterministic given a seed: randomness comes from a 64-bit LCG rather than
 * an OS entropy source, so Tier 2 tests replay exactly. The mask is applied
 * before temperature and top-p, making grammar-illegal tokens unrepresentable
 * regardless of their probability mass.
 */

use anyhow::{ensure, Result};

use super::model::{Logits, Token};

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            seed: 0,
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    state: u64,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        // Avoid the LCG's zero fixed point without changing seeded behavior.
        Self {
            params,
            state: params.seed.wrapping_add(0x9e3779b97f4a7c15),
        }
    }

    fn next_unit(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 40) as f32) / ((1u64 << 24) as f32)
    }

    /// Sample one token. `mask`, when present, restricts the support; an
    /// all-false mask is an error rather than a silent fallback.
    pub fn sample(&mut self, logits: &Logits, mask: Option<&[bool]>) -> Result<Token> {
        let mut candidates: Vec<(Token, f32)> = logits
            .0
            .iter()
            .enumerate()
            .filter(|(i, _)| mask.is_none_or(|m| m[*i]))
            .map(|(i, &l)| (i as Token, l))
            .collect();
        ensure!(
            !candidates.is_empty(),
            "empty sampling support (mask excludes all tokens)"
        );

        if self.params.temperature <= 0.0 {
            let best = candidates
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .expect("non-empty candidates");
            return Ok(best.0);
        }

        // Softmax at temperature over the masked support.
        let max = candidates
            .iter()
            .map(|c| c.1)
            .fold(f32::NEG_INFINITY, f32::max);
        for c in candidates.iter_mut() {
            c.1 = ((c.1 - max) / self.params.temperature).exp();
        }
        let total: f32 = candidates.iter().map(|c| c.1).sum();
        for c in candidates.iter_mut() {
            c.1 /= total;
        }

        // Top-p: keep the smallest probability-sorted prefix covering top_p.
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut cum = 0.0;
        let mut cut = candidates.len();
        for (i, c) in candidates.iter().enumerate() {
            cum += c.1;
            if cum >= self.params.top_p {
                cut = i + 1;
                break;
            }
        }
        candidates.truncate(cut);

        let kept: f32 = candidates.iter().map(|c| c.1).sum();
        let mut draw = self.next_unit() * kept;
        for c in &candidates {
            draw -= c.1;
            if draw <= 0.0 {
                return Ok(c.0);
            }
        }
        Ok(candidates.last().expect("non-empty after truncate").0)
    }
}
