/*
 * eval — output-correctness evaluation: perplexity over a fixed corpus.
 *
 * Perplexity is the regression scalar for engine changes: a wrong RoPE base,
 * transposed weight, or bad quant kernel moves it sharply while spot checks
 * stay plausible. Evaluation is chunked llama.cpp-style: the corpus splits
 * into fixed windows, each window is scored with fresh context, and only the
 * second half of each window is scored so every scored token has at least
 * half a window of context.
 */

use anyhow::{bail, ensure, Result};

use super::model::{Logits, Model, Token};

/// Accumulated negative log-likelihood over scored tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PplResult {
    pub nll_sum: f64,
    pub tokens: usize,
}

impl PplResult {
    pub fn mean_nll(&self) -> f64 {
        self.nll_sum / self.tokens as f64
    }

    pub fn ppl(&self) -> f64 {
        self.mean_nll().exp()
    }
}

/// Negative log-probability of `target` under a softmax of `logits`.
/// f64 throughout: f32 accumulation drifts measurably over 100k+ tokens.
fn nll(logits: &Logits, target: Token) -> Result<f64> {
    let target = target as usize;
    ensure!(
        target < logits.0.len(),
        "target token {target} outside logits width {}",
        logits.0.len()
    );
    // `f64::max` returns the non-NaN operand, so a NaN never reaches `max`
    // and the finiteness check below cannot see it: the NaN would flow into
    // the sum and perplexity would report a number-shaped non-number. A -inf
    // entry stays legal — it is log-zero probability mass, not corruption.
    if let Some(i) = logits.0.iter().position(|l| l.is_nan()) {
        bail!("non-finite logits in evaluation: NaN at index {i}");
    }
    let max = logits
        .0
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, l| a.max(l as f64));
    ensure!(max.is_finite(), "non-finite logits in evaluation");
    let log_sum: f64 = logits
        .0
        .iter()
        .map(|&l| ((l as f64) - max).exp())
        .sum::<f64>()
        .ln()
        + max;
    Ok(log_sum - logits.0[target] as f64)
}

/// One scored token: where it sits in the corpus, and what it cost.
///
/// The corpus position travels with the number because a per-token comparison
/// against a reference is only actionable if a disagreement can be pointed at
/// a token. An index into the scored sequence is not that: warmup positions
/// are skipped and partial windows dropped, so the two numberings differ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredToken {
    /// Index in `corpus` of the token whose probability was scored.
    pub position: usize,
    pub target: Token,
    pub nll: f64,
}

/// Per-token negative log-likelihood over `corpus`, in scoring order.
///
/// The windowing is `perplexity`'s, and `perplexity` is defined as the sum of
/// this: one loop, so the two can never disagree about which tokens were
/// scored. Summation is left to the caller in the same order the tokens were
/// produced, which keeps the total bit-identical to the accumulating loop this
/// replaced.
pub fn per_token_nll(
    model: &mut dyn Model,
    corpus: &[Token],
    chunk: usize,
) -> Result<Vec<ScoredToken>> {
    ensure!(chunk >= 2, "chunk must be at least 2 tokens");
    ensure!(
        corpus.len() >= chunk,
        "corpus of {} tokens is shorter than one {chunk}-token chunk",
        corpus.len()
    );
    let warmup = chunk / 2;
    let mut scored = Vec::new();
    for (w, window) in corpus.chunks_exact(chunk).enumerate() {
        model.truncate(0)?;
        let rows = model.prefill_all(window)?;
        ensure!(
            rows.len() == window.len(),
            "prefill_all returned short rows"
        );
        // Row p holds logits after token p, predicting token p + 1.
        for p in warmup..chunk - 1 {
            scored.push(ScoredToken {
                position: w * chunk + p + 1,
                target: window[p + 1],
                nll: nll(&rows[p], window[p + 1])?,
            });
        }
    }
    ensure!(!scored.is_empty(), "no tokens scored");
    Ok(scored)
}

/// Perplexity of `corpus` under `model`, evaluated in windows of `chunk`
/// tokens. Each full window is prefilled from an empty cache; positions in
/// the second half score the next token. The trailing partial window is
/// dropped, matching the llama.cpp convention.
pub fn perplexity(model: &mut dyn Model, corpus: &[Token], chunk: usize) -> Result<PplResult> {
    let scored = per_token_nll(model, corpus, chunk)?;
    Ok(PplResult {
        nll_sum: scored.iter().map(|s| s.nll).sum(),
        tokens: scored.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::mock::MockModel;

    const TOL: f64 = 1e-9;

    #[test]
    fn uniform_logits_give_perplexity_equal_to_vocab_size() {
        // Empty script: every position yields all-zero (uniform) logits, so
        // each scored token costs exactly ln(V) nats.
        for chunk in [4, 8, 16] {
            let mut m = MockModel::new(32, vec![]);
            let corpus: Vec<Token> = (0..32).map(|i| i % 32).collect();
            let r = perplexity(&mut m, &corpus, chunk).unwrap();
            assert!((r.ppl() - 32.0).abs() < TOL, "chunk {chunk}: {}", r.ppl());
        }
    }

    #[test]
    fn hand_calculated_two_token_case() {
        // V = 2, every row is [ln 3, 0]: softmax gives p(token 0) = 3/4.
        // Corpus of all zeros scores each position at -ln(3/4), ppl = 4/3.
        let row = vec![3.0f32.ln(), 0.0];
        let mut m = MockModel::new(2, vec![row; 16]);
        let corpus = vec![0u32; 16];
        let r = perplexity(&mut m, &corpus, 8).unwrap();
        // 8-token chunks score positions 4..7: 3 tokens per chunk, 2 chunks.
        assert_eq!(r.tokens, 6);
        assert!((r.ppl() - 4.0 / 3.0).abs() < 1e-6, "{}", r.ppl());
    }

    #[test]
    fn confident_model_scores_below_uniform() {
        // Script peaks strongly on the actual next corpus token.
        let corpus: Vec<Token> = (0..64).map(|i| (i * 7) % 16).collect();
        let script: Vec<Vec<f32>> = (0..63)
            .map(|p| {
                let mut l = vec![0.0; 16];
                l[corpus[p + 1] as usize] = 10.0;
                l
            })
            .collect();
        let mut m = MockModel::new(16, script);
        let r = perplexity(&mut m, &corpus, 16).unwrap();
        assert!(r.ppl() > 1.0 && r.ppl() < 2.0, "{}", r.ppl());
    }

    #[test]
    fn windows_are_scored_with_fresh_context() {
        // Position-keyed mock: after the truncate(0) between windows, the
        // second window must replay script positions from 0, so both windows
        // of an identical-logits script contribute identical NLL.
        let row = vec![2.0, 0.0, 0.0, 0.0];
        let mut m = MockModel::new(4, vec![row; 8]);
        let corpus = vec![0u32; 16];
        let one = perplexity(
            &mut MockModel::new(4, vec![vec![2.0, 0.0, 0.0, 0.0]; 8]),
            &corpus[..8],
            8,
        )
        .unwrap();
        let two = perplexity(&mut m, &corpus, 8).unwrap();
        assert_eq!(two.tokens, 2 * one.tokens);
        assert!((two.nll_sum - 2.0 * one.nll_sum).abs() < TOL);
    }

    #[test]
    fn trailing_partial_window_is_dropped() {
        let mut m = MockModel::new(8, vec![]);
        let corpus = vec![0u32; 20];
        let r = perplexity(&mut m, &corpus, 8).unwrap();
        // Two full windows only; positions 4..7 of each: 3 + 3 scored.
        assert_eq!(r.tokens, 6);
    }

    #[test]
    fn a_nan_logit_is_an_error_rather_than_a_nan_perplexity() {
        // `f64::max` returns the non-NaN operand, so the max-subtraction
        // guard never sees a NaN and it flows into the sum: without an
        // explicit check the caller gets Ok(NaN) and reports a number.
        let mut row = vec![0.0f32; 4];
        row[2] = f32::NAN;
        let mut m = MockModel::new(4, vec![row; 8]);
        let err = perplexity(&mut m, &[0u32; 8], 8).expect_err("expected error");
        assert!(err.to_string().contains("NaN at index 2"), "{err}");
    }

    #[test]
    fn a_negative_infinity_logit_stays_legal_as_log_zero_mass() {
        // -inf is a token the model assigns zero probability, not corruption.
        // Three of four tokens stay equiprobable, so perplexity is exactly 3.
        let row = vec![0.0, f32::NEG_INFINITY, 0.0, 0.0];
        let mut m = MockModel::new(4, vec![row; 8]);
        let r = perplexity(&mut m, &[0u32; 8], 8).unwrap();
        assert!((r.ppl() - 3.0).abs() < 1e-6, "{}", r.ppl());
    }

    #[test]
    fn scored_positions_are_corpus_indices_of_the_predicted_token() {
        // The per-token view exists so a disagreement can be pointed at a
        // token, which requires the corpus index rather than an index into
        // the scored sequence: warmup skips four positions per window and the
        // two numberings part company immediately.
        let mut m = MockModel::new(4, vec![]);
        let corpus: Vec<Token> = (0..16).map(|i| i % 4).collect();
        let scored = per_token_nll(&mut m, &corpus, 8).unwrap();
        let positions: Vec<usize> = scored.iter().map(|s| s.position).collect();
        assert_eq!(positions, vec![5, 6, 7, 13, 14, 15], "{positions:?}");
        for s in &scored {
            assert_eq!(s.target, corpus[s.position], "position {}", s.position);
        }
    }

    #[test]
    fn perplexity_is_exactly_the_sum_of_the_per_token_view() {
        // Not approximately: `perplexity` is implemented as this sum, and the
        // test pins that so a later fast path cannot quietly reorder the
        // accumulation.
        let corpus: Vec<Token> = (0..32).map(|i| (i * 5) % 16).collect();
        let script: Vec<Vec<f32>> = (0..32)
            .map(|p| {
                let mut l = vec![0.0; 16];
                l[p % 16] = 3.0;
                l
            })
            .collect();
        let scored = per_token_nll(&mut MockModel::new(16, script.clone()), &corpus, 16).unwrap();
        let r = perplexity(&mut MockModel::new(16, script), &corpus, 16).unwrap();
        assert_eq!(r.tokens, scored.len());
        let sum: f64 = scored.iter().map(|s| s.nll).sum();
        assert_eq!(r.nll_sum, sum, "{} vs {sum}", r.nll_sum);
    }

    #[test]
    fn degenerate_inputs_are_errors() {
        let mut m = MockModel::new(8, vec![]);
        assert!(perplexity(&mut m, &[0; 4], 8).is_err()); // corpus < chunk
        assert!(perplexity(&mut m, &[0; 8], 1).is_err()); // chunk < 2
                                                          // Target token wider than the logits row.
        let mut narrow = MockModel::new(2, vec![]);
        assert!(perplexity(&mut narrow, &[0, 5, 0, 5], 4).is_err());
    }
}
