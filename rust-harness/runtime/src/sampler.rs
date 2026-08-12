/*
 * sampler — temperature/top-k/top-p sampling with optional grammar masking.
 *
 * Deterministic given a seed: randomness comes from a 64-bit LCG rather than
 * an OS entropy source, so Tier 2 tests replay exactly. The mask is applied
 * before temperature and the truncations, making grammar-illegal tokens
 * unrepresentable regardless of their probability mass. Non-finite logits
 * (NaN, +/-inf) are excluded from the support: a backend bug producing NaN
 * surfaces as an empty-support error instead of silently skewed sampling.
 */

use anyhow::{ensure, Result};

use super::model::{Logits, Token};

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    /// Rank cutoff, or `None` for no rank cutoff at all.
    ///
    /// `Option` rather than the conventional sentinel zero, so "disabled" has
    /// exactly one spelling. A `top_k` of zero would otherwise have to mean
    /// either "keep nothing", which is not a distribution, or "keep
    /// everything", which reads as the opposite of what it says.
    pub top_k: Option<usize>,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 0.95,
            top_k: None,
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
        assert!(
            params.top_k != Some(0),
            "top_k must be >= 1; omit it (None) to sample without a rank cutoff"
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
        let span = tracing::info_span!(
            "sampler.sample",
            temperature = self.params.temperature,
            candidates = tracing::field::Empty
        );
        let _enter = span.enter();
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
            // Argmax without the candidate allocation. Strictly `>`, so a tie
            // keeps the *first* supported index, which is what `torch.argmax`
            // returns and therefore what greedy decoding has to do to be
            // comparable against the reference implementation at all.
            //
            // This read `>=` until the correctness ladder needed greedy
            // parity. Leaving the sampler on the other convention while the
            // ladder used torch's would have made the ladder pass while the
            // shipping decoder did something else -- the precise shape of
            // gate that proves nothing. Exact ties are rare in f32 over a
            // 262144-wide vocabulary but not absent: grammar masking and
            // logit-softcap saturation both produce them, and both are here.
            //
            // `metrics::argmax_first` is the same rule without a mask, and
            // the two are pinned to agree by a test below rather than shared
            // as one function: threading an optional mask through a metrics
            // helper would complicate it for a single caller.
            let mut best: Option<(Token, f32)> = None;
            let mut support = 0usize;
            for (i, &l) in logits.0.iter().enumerate() {
                if supported(i, l) {
                    support += 1;
                    if best.is_none_or(|(_, b)| l > b) {
                        best = Some((i as Token, l));
                    }
                }
            }
            span.record("candidates", support);
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

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Top-k before top-p, which is the conventional order and also the
        // cheap one: both truncations want the same descending sort, and
        // taking the rank cutoff first shortens the prefix scan below. The
        // order is observable -- top-k then top-p keeps at most k tokens,
        // while the reverse could keep more -- so it is stated rather than
        // left to fall out of the code. A k at or above the support truncates
        // nothing.
        if let Some(k) = self.params.top_k {
            candidates.truncate(k);
        }

        // Top-p: keep the smallest probability-sorted prefix covering top_p.
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
        span.record("candidates", candidates.len());

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
            top_k: None,
            seed,
        })
    }

    /// The same, with a rank cutoff. Separate from `sampler` so the existing
    /// tests keep reading as assertions about temperature and top-p alone.
    fn sampler_k(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> Sampler {
        Sampler::new(SamplingParams {
            temperature,
            top_p,
            top_k: Some(top_k),
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
    fn temperature_zero_is_argmax_and_ties_pick_the_first_index_as_torch_does() {
        let mut s = sampler(0.0, 0.95, 0);
        assert_eq!(s.sample(&logits(&[0.1, 2.0, 0.3]), None).unwrap(), 1);
        // Flat logits: the first index wins. This test asserted the last
        // index until greedy parity with transformers became a gate; the
        // choice is arbitrary in isolation and is not arbitrary once a
        // reference implementation exists to disagree with.
        assert_eq!(s.sample(&logits(&[1.0, 1.0, 1.0]), None).unwrap(), 0);
        assert_eq!(s.sample(&logits(&[0.0, 2.0, 2.0, 1.0]), None).unwrap(), 1);
    }

    #[test]
    fn greedy_sampling_and_the_metrics_argmax_resolve_ties_the_same_way() {
        // The two implementations exist separately because the sampler
        // carries a mask and the metric does not. That is a reason for two
        // functions, not for two conventions, so the agreement is asserted
        // rather than assumed -- a later edit to either one fails here.
        let mut s = sampler(0.0, 0.95, 0);
        for row in [
            vec![1.0f32, 1.0, 1.0],
            vec![0.0, 2.0, 2.0, 1.0],
            vec![-3.0, -1.0, -5.0],
            vec![5.0, 4.0, 3.0],
        ] {
            let got = s.sample(&logits(&row), None).unwrap();
            let want = crate::metrics::argmax_first(&row) as Token;
            assert_eq!(got, want, "row {row:?}");
        }
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
    fn top_k_of_one_degenerates_to_the_most_probable_token() {
        // The rank sibling of tiny_top_p_degenerates_to_the_most_probable_token:
        // with top_p wide open, the cutoff is entirely top-k's doing.
        let l = logits(&[0.0, 3.0, 1.0]);
        let mut s = sampler_k(1.0, 1.0, 1, 11);
        for _ in 0..50 {
            assert_eq!(s.sample(&l, None).unwrap(), 1);
        }
    }

    #[test]
    fn top_k_excludes_a_token_top_p_alone_would_have_kept() {
        // Three equally probable tokens: top_p = 1.0 keeps all three, and a
        // seeded run reaches every one of them. Ranking to two must make the
        // third unreachable, which is the property top-k exists for -- and
        // which asserting only the kept tokens' presence would not show.
        let l = logits(&[1.0, 1.0, 1.0]);
        let mut open = sampler(1.0, 1.0, 3);
        let mut seen_open = [false; 3];
        for _ in 0..200 {
            seen_open[open.sample(&l, None).unwrap() as usize] = true;
        }
        assert_eq!(seen_open, [true, true, true]);

        let mut ranked = sampler_k(1.0, 1.0, 2, 3);
        let mut seen_ranked = [false; 3];
        for _ in 0..200 {
            seen_ranked[ranked.sample(&l, None).unwrap() as usize] = true;
        }
        assert!(
            !seen_ranked[2],
            "top_k = 2 sampled the third-ranked token: {seen_ranked:?}"
        );
    }

    #[test]
    fn a_top_k_at_or_above_the_support_changes_nothing() {
        // Not merely "still valid": bit-identical to the same seed without a
        // cutoff, because a no-op truncation must not perturb the draw stream.
        let l = logits(&[0.1, 0.5, 0.2, 0.9]);
        let mut without = sampler(0.8, 0.95, 42);
        let mut at = sampler_k(0.8, 0.95, 4, 42);
        let mut above = sampler_k(0.8, 0.95, 999, 42);
        for _ in 0..20 {
            let want = without.sample(&l, None).unwrap();
            assert_eq!(at.sample(&l, None).unwrap(), want);
            assert_eq!(above.sample(&l, None).unwrap(), want);
        }
    }

    #[test]
    fn top_k_ranks_the_masked_support_not_the_whole_vocabulary() {
        // The mask is applied first, so the two highest logits being masked
        // out must not consume the k. Were the rank taken over the raw
        // vocabulary, top_k = 1 here would keep a masked token and the sample
        // would have no legal support at all.
        let l = logits(&[9.0, 8.0, 1.0, 0.5]);
        let mask = [false, false, true, true];
        let mut s = sampler_k(1.0, 1.0, 1, 5);
        for _ in 0..50 {
            assert_eq!(s.sample(&l, Some(&mask)).unwrap(), 2);
        }
    }

    #[test]
    fn top_k_does_not_reach_the_greedy_path() {
        // Temperature zero is an argmax; a rank cutoff is meaningless there
        // and must not become a special case that quietly changes the answer.
        let l = logits(&[0.1, 2.0, 0.3]);
        assert_eq!(sampler_k(0.0, 0.95, 1, 0).sample(&l, None).unwrap(), 1);
        assert_eq!(sampler_k(0.0, 0.95, 3, 0).sample(&l, None).unwrap(), 1);
    }

    #[test]
    #[should_panic(expected = "top_k")]
    fn a_top_k_of_zero_is_rejected_at_construction() {
        sampler_k(0.8, 0.95, 0, 0);
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
