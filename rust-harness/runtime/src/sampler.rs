/*
 * sampler — temperature/top-p sampling with optional grammar masking.
 *
 * Deterministic given a seed: randomness comes from a 64-bit LCG rather than
 * an OS entropy source, so Tier 2 tests replay exactly. The mask is applied
 * before temperature and top-p, making grammar-illegal tokens unrepresentable
 * regardless of their probability mass. Non-finite logits (NaN, +/-inf) are
 * excluded from the support: a backend bug producing NaN surfaces as an
 * empty-support error instead of silently skewed sampling.
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
    /// Panics on out-of-range parameters: they are configuration bugs, not
    /// runtime conditions.
    pub fn new(params: SamplingParams) -> Self {
        assert!(
            params.temperature >= 0.0 && params.temperature.is_finite(),
            "temperature must be finite and >= 0, got {}",
            params.temperature
        );
        assert!(
            params.top_p > 0.0 && params.top_p <= 1.0,
            "top_p must be in (0, 1], got {}",
            params.top_p
        );
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
    /// all-false mask (or all-non-finite logits) is an error rather than a
    /// silent fallback.
    pub fn sample(&mut self, logits: &Logits, mask: Option<&[bool]>) -> Result<Token> {
        if let Some(m) = mask {
            ensure!(
                m.len() == logits.0.len(),
                "mask length {} does not match logits length {}",
                m.len(),
                logits.0.len()
            );
        }
        let supported = |i: usize, l: f32| l.is_finite() && mask.is_none_or(|m| m[i]);

        if self.params.temperature <= 0.0 {
            // Argmax without the candidate allocation. `>=` keeps the last
            // index on ties, matching the previous max_by behavior that the
            // Tier 2 scripts rely on.
            let mut best: Option<(Token, f32)> = None;
            for (i, &l) in logits.0.iter().enumerate() {
                if supported(i, l) && best.is_none_or(|(_, b)| l >= b) {
                    best = Some((i as Token, l));
                }
            }
            let (token, _) =
                best.ok_or_else(|| anyhow::anyhow!("empty sampling support (argmax)"))?;
            return Ok(token);
        }

        let mut candidates: Vec<(Token, f32)> = logits
            .0
            .iter()
            .enumerate()
            .filter(|(i, l)| supported(*i, **l))
            .map(|(i, &l)| (i as Token, l))
            .collect();
        ensure!(
            !candidates.is_empty(),
            "empty sampling support (mask excludes all tokens)"
        );

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

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(v: &[f32]) -> Logits {
        Logits(v.to_vec())
    }

    fn sampler(temperature: f32, top_p: f32, seed: u64) -> Sampler {
        Sampler::new(SamplingParams {
            temperature,
            top_p,
            seed,
        })
    }

    #[test]
    fn same_seed_replays_the_same_sequence() {
        let l = logits(&[0.1, 0.5, 0.2, 0.9]);
        let mut a = sampler(0.8, 0.95, 42);
        let mut b = sampler(0.8, 0.95, 42);
        for _ in 0..20 {
            assert_eq!(a.sample(&l, None).unwrap(), b.sample(&l, None).unwrap());
        }
    }

    #[test]
    fn temperature_zero_is_argmax_and_ties_pick_the_last_index() {
        let mut s = sampler(0.0, 0.95, 0);
        assert_eq!(s.sample(&logits(&[0.1, 2.0, 0.3]), None).unwrap(), 1);
        // Flat logits: last index wins, preserved for script determinism.
        assert_eq!(s.sample(&logits(&[1.0, 1.0, 1.0]), None).unwrap(), 2);
    }

    #[test]
    fn mask_restricts_the_support() {
        let mut s = sampler(0.0, 0.95, 0);
        let l = logits(&[0.0, 9.0, 1.0]);
        let mask = [true, false, true];
        assert_eq!(s.sample(&l, Some(&mask)).unwrap(), 2);
    }

    #[test]
    fn all_false_mask_is_an_error() {
        let mut s = sampler(0.8, 0.95, 0);
        let mask = [false, false];
        assert!(s.sample(&logits(&[0.0, 1.0]), Some(&mask)).is_err());
        let mut greedy = sampler(0.0, 0.95, 0);
        assert!(greedy.sample(&logits(&[0.0, 1.0]), Some(&mask)).is_err());
    }

    #[test]
    fn mask_length_mismatch_is_an_error_not_a_panic() {
        let mut s = sampler(0.8, 0.95, 0);
        let mask = [true];
        let err = s.sample(&logits(&[0.0, 1.0]), Some(&mask)).unwrap_err();
        assert!(err.to_string().contains("mask length"));
    }

    #[test]
    fn non_finite_logits_are_excluded_from_the_support() {
        // NaN and -inf tokens never win, in either sampling path.
        let l = logits(&[f32::NAN, 5.0, f32::NEG_INFINITY]);
        let mut greedy = sampler(0.0, 0.95, 0);
        assert_eq!(greedy.sample(&l, None).unwrap(), 1);
        let mut stochastic = sampler(0.8, 0.95, 7);
        for _ in 0..50 {
            assert_eq!(stochastic.sample(&l, None).unwrap(), 1);
        }
    }

    #[test]
    fn all_non_finite_logits_are_an_error() {
        let l = logits(&[f32::NAN, f32::INFINITY]);
        assert!(sampler(0.8, 0.95, 0).sample(&l, None).is_err());
        assert!(sampler(0.0, 0.95, 0).sample(&l, None).is_err());
    }

    #[test]
    fn top_p_one_keeps_the_full_support() {
        // With top_p = 1 every token remains reachable across draws.
        let l = logits(&[1.0, 1.0, 1.0]);
        let mut s = sampler(1.0, 1.0, 3);
        let mut seen = [false; 3];
        for _ in 0..200 {
            seen[s.sample(&l, None).unwrap() as usize] = true;
        }
        assert_eq!(seen, [true, true, true]);
    }

    #[test]
    fn tiny_top_p_degenerates_to_the_most_probable_token() {
        let l = logits(&[0.0, 3.0, 1.0]);
        let mut s = sampler(0.5, 0.001, 11);
        for _ in 0..50 {
            assert_eq!(s.sample(&l, None).unwrap(), 1);
        }
    }

    #[test]
    #[should_panic(expected = "temperature")]
    fn negative_temperature_is_rejected_at_construction() {
        sampler(-1.0, 0.95, 0);
    }

    #[test]
    #[should_panic(expected = "top_p")]
    fn out_of_range_top_p_is_rejected_at_construction() {
        sampler(0.8, 1.5, 0);
    }
}
