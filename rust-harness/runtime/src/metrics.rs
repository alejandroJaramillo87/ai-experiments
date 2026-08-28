/*
 * metrics — comparison arithmetic for checking our numbers against a reference.
 *
 * Two families. `vec_diff` compares an activation row against the same row
 * from transformers: cosine for direction, relative L2 for magnitude, max
 * absolute for the outlier. `kl_divergence` and friends compare a
 * distribution against a reference distribution, which is a different
 * question and needs different arithmetic.
 *
 * Both live here rather than in their callers because two of them exist. The
 * layer comparison started inside `bin/layer-diff.rs`, and the ladder needs
 * the identical function -- a second copy would be a second set of thresholds
 * and eventually a second answer.
 *
 * f64 throughout. The quantities are small differences of large numbers
 * summed over thousands of terms, and f32 accumulation loses the difference
 * before the caller sees it.
 */

use anyhow::{bail, Result};

/// Direction, magnitude, and worst-element agreement of one row against a
/// reference row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VecDiff {
    pub cosine: f64,
    pub max_abs: f64,
    pub rel_l2: f64,
}

/// Compare `got` against `want`, elementwise.
///
/// Cosine and relative L2 are both reported because they fail independently.
/// A residual stream that is correct in shape and wrong in size passes cosine
/// perfectly -- the exact signature of a missing `layer_scalar`, a tensor the
/// released checkpoints carry at values as low as 0.049. Direction and
/// magnitude are separate defects and one number cannot carry both.
///
/// Rows of unequal length are a caller error and are not checked here; every
/// caller in the tree compares widths first so it can name both.
pub fn vec_diff(got: &[f32], want: &[f32]) -> VecDiff {
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    let mut max_abs = 0.0f64;
    let mut diff_sq = 0.0f64;
    for (a, b) in got.iter().zip(want) {
        let (a, b) = (*a as f64, *b as f64);
        dot += a * b;
        ng += a * a;
        nw += b * b;
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        diff_sq += (a - b) * (a - b);
    }
    // A zero row on either side has no direction, so cosine is undefined
    // rather than 1.0. Reported as 0.0, which fails any floor and prints,
    // instead of silently passing as a perfect match of nothing.
    let cosine = match ng > 0.0 && nw > 0.0 {
        true => dot / (ng.sqrt() * nw.sqrt()),
        false => 0.0,
    };
    let rel_l2 = match nw > 0.0 {
        true => diff_sq.sqrt() / nw.sqrt(),
        false => f64::INFINITY,
    };
    VecDiff {
        cosine,
        max_abs,
        rel_l2,
    }
}

/// Log-softmax of raw logits, max-subtracted, in f64.
///
/// Takes logits rather than probabilities everywhere in this module, so no
/// caller hand-rolls a softmax and gets the max-subtraction wrong.
fn log_softmax(logits: &[f32]) -> Result<Vec<f64>> {
    // `f64::max` returns the non-NaN operand, so a NaN never reaches the
    // running maximum and the finiteness check below cannot see it: it would
    // flow into the sum and every metric downstream would report a
    // number-shaped non-number. The same guard `eval::nll` carries, for the
    // same reason.
    if let Some(i) = logits.iter().position(|l| l.is_nan()) {
        bail!("non-finite logits in comparison: NaN at index {i}");
    }
    let max = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, l| a.max(l as f64));
    if !max.is_finite() {
        bail!("non-finite logits in comparison: no finite entry");
    }
    let log_sum: f64 = logits
        .iter()
        .map(|&l| ((l as f64) - max).exp())
        .sum::<f64>()
        .ln()
        + max;
    Ok(logits.iter().map(|&l| (l as f64) - log_sum).collect())
}

/// Kullback-Leibler divergence in nats, `KL(reference || ours)`.
///
/// The direction is deliberate and stated because an unstated one is a future
/// argument. With the reference as `p`, the sum is weighted by where the
/// reference puts its mass, so the number answers "how much probability does
/// the reference assign that we do not" -- the failure that matters. The
/// other direction would let us miss a mode the reference is confident about
/// as long as we were diffuse.
///
/// Preferred over a tolerance on raw logits: KL is dimensionless and
/// scale-free, which a 262144-wide softcapped vocabulary needs, and it is
/// sensitive to the shape of the whole distribution rather than to its
/// largest entry. A monotone error such as a missing logit softcap cannot
/// move an argmax at all and moves this immediately.
pub fn kl_divergence(reference: &[f32], ours: &[f32]) -> Result<f64> {
    if reference.len() != ours.len() {
        bail!(
            "kl_divergence: reference is {} wide, ours is {} wide",
            reference.len(),
            ours.len()
        );
    }
    let (lp, lq) = (log_softmax(reference)?, log_softmax(ours)?);
    let mut kl = 0.0f64;
    for (lp, lq) in lp.iter().zip(&lq) {
        // exp(-inf) is 0 and 0 * anything is the correct contribution, but
        // -inf - -inf is NaN, so a token both sides call impossible is
        // skipped rather than summed.
        let p = lp.exp();
        if p > 0.0 {
            kl += p * (lp - lq);
        }
    }
    Ok(kl)
}

/// Fraction of the reference's top `k` tokens that appear in ours.
///
/// Reported alongside argmax agreement because argmax is a step function: it
/// is unchanged by a reordered tail that will later surface as a sampling
/// difference. Set membership rather than rank equality, so a swap within the
/// top k does not read as a miss -- rank order inside the set is what
/// temperature is for.
pub fn topk_overlap(reference: &[f32], ours: &[f32], k: usize) -> f64 {
    let top = |l: &[f32]| -> Vec<usize> {
        let mut idx: Vec<usize> = (0..l.len()).collect();
        // Descending by value, ties by index, so the set is deterministic for
        // a distribution with equal entries.
        idx.sort_by(|&a, &b| {
            l[b].partial_cmp(&l[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(k);
        idx
    };
    let want = top(reference);
    if want.is_empty() {
        return 1.0;
    }
    let got: std::collections::HashSet<usize> = top(ours).into_iter().collect();
    let hits = want.iter().filter(|i| got.contains(i)).count();
    hits as f64 / want.len() as f64
}

/// Index of the largest logit, resolving ties to the **first** index.
///
/// The tie rule is the contract, not an implementation detail: `torch.argmax`
/// returns the first maximal index, and greedy decoding is only meaningful as
/// a comparison against the reference implementation. A tie broken the other
/// way produces a different token, a different continuation, and a test
/// failure with no defect behind it.
///
/// Exact ties are rare in f32 over a wide vocabulary but not absent: grammar
/// masking and logit softcap saturation both produce them, and both are in
/// this tree.
pub fn argmax_first(logits: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &l) in logits.iter().enumerate() {
        // Strictly greater, so an equal value never displaces the earlier
        // index. This is the whole tie rule.
        if l > logits[best] {
            best = i;
        }
    }
    best
}

/// Every measure of one position's distribution against the reference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitDiff {
    pub kl: f64,
    pub top_k: f64,
    pub argmax_same: bool,
    pub max_abs: f64,
}

/// Compare one position's logits against the reference's, on all four axes.
pub fn logit_diff(reference: &[f32], ours: &[f32], k: usize) -> Result<LogitDiff> {
    Ok(LogitDiff {
        kl: kl_divergence(reference, ours)?,
        top_k: topk_overlap(reference, ours, k),
        argmax_same: argmax_first(reference) == argmax_first(ours),
        max_abs: vec_diff(ours, reference).max_abs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_rows_compare_as_a_perfect_match() {
        let x = [1.0f32, -2.0, 3.5, 0.25];
        let d = vec_diff(&x, &x);
        assert!((d.cosine - 1.0).abs() < 1e-12, "{}", d.cosine);
        assert_eq!(d.max_abs, 0.0);
        assert_eq!(d.rel_l2, 0.0);
    }

    #[test]
    fn a_uniformly_scaled_row_keeps_its_direction_and_fails_on_magnitude() {
        // The shape a missing output scale produces: same direction, wrong
        // size. Cosine alone would pass it, which is why rel_l2 is reported
        // beside it rather than cosine on its own.
        let x = [1.0f32, -2.0, 3.5, 0.25];
        let scaled: Vec<f32> = x.iter().map(|v| v * 0.5).collect();
        let d = vec_diff(&scaled, &x);
        assert!((d.cosine - 1.0).abs() < 1e-12, "{}", d.cosine);
        assert!((d.rel_l2 - 0.5).abs() < 1e-12, "{}", d.rel_l2);
    }

    #[test]
    fn a_zero_row_is_reported_as_no_match_rather_than_a_perfect_one() {
        // 0/0 would be NaN, and NaN < floor is false -- so an unhooked layer
        // capturing zeros would read as passing. Reported as 0.0 instead,
        // which fails any floor.
        let d = vec_diff(&[0.0; 4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(d.cosine, 0.0);
        assert!(d.rel_l2.is_finite(), "{}", d.rel_l2);
    }

    #[test]
    fn a_distribution_has_zero_divergence_from_itself() {
        // The defining property, and the one a sign error breaks first.
        let x = [0.5f32, -1.25, 3.0, 0.0, 2.5];
        let kl = kl_divergence(&x, &x).unwrap();
        assert!(kl.abs() < 1e-12, "{kl}");
    }

    #[test]
    fn divergence_is_never_negative_and_grows_with_disagreement() {
        let reference = [3.0f32, 0.0, 0.0, 0.0];
        let near = [2.5f32, 0.0, 0.0, 0.0];
        let far = [0.0f32, 3.0, 0.0, 0.0];
        let (a, b) = (
            kl_divergence(&reference, &near).unwrap(),
            kl_divergence(&reference, &far).unwrap(),
        );
        assert!(a >= 0.0 && b >= 0.0, "{a}, {b}");
        assert!(a < b, "near {a} should diverge less than far {b}");
    }

    #[test]
    fn divergence_is_asymmetric_so_the_argument_order_is_load_bearing() {
        // If these were equal the documented direction would be decorative,
        // and a caller could swap the arguments without the tests noticing.
        // The asymmetry is why the docstring names which side is `p`.
        let p = [4.0f32, 0.0, 0.0];
        let q = [0.0f32, 1.0, 1.0];
        let forward = kl_divergence(&p, &q).unwrap();
        let backward = kl_divergence(&q, &p).unwrap();
        assert!(
            (forward - backward).abs() > 1e-3,
            "forward {forward}, backward {backward}"
        );
    }

    #[test]
    fn a_hand_computed_two_token_divergence_matches() {
        // p = softmax([ln 3, 0]) = [3/4, 1/4]; q = softmax([0, 0]) = [1/2, 1/2].
        // KL(p||q) = 3/4 ln(3/2) + 1/4 ln(1/2).
        let p = [3.0f32.ln(), 0.0];
        let q = [0.0f32, 0.0];
        let want = 0.75 * (1.5f64).ln() + 0.25 * (0.5f64).ln();
        let got = kl_divergence(&p, &q).unwrap();
        // 1e-7 rather than f64 epsilon: the input is f32, so `3.0f32.ln()` is
        // not exactly ln 3 and p is not exactly 3/4. The residual is the
        // input's rounding, not the summation's -- tightening this would be
        // asserting a precision the argument cannot carry.
        assert!((got - want).abs() < 1e-7, "got {got}, want {want}");
    }

    #[test]
    fn a_nan_logit_is_an_error_rather_than_a_nan_divergence() {
        // `f64::max` returns the non-NaN operand, so the max-subtraction
        // guard never sees a NaN and it flows into the sum. Without the
        // explicit scan the caller gets Ok(NaN), and `NaN > threshold` is
        // false -- the metric would pass by being broken.
        let mut row = vec![0.0f32; 4];
        row[2] = f32::NAN;
        let err = kl_divergence(&row, &[0.0; 4]).expect_err("expected error");
        assert!(err.to_string().contains("NaN at index 2"), "{err}");
    }

    #[test]
    fn a_token_both_sides_call_impossible_contributes_nothing() {
        // -inf - -inf is NaN. Skipping zero-probability entries is what keeps
        // a shared hard zero -- what grammar masking produces on both sides --
        // from poisoning the sum.
        let p = [f32::NEG_INFINITY, 0.0, 1.0];
        let q = [f32::NEG_INFINITY, 0.0, 1.0];
        let kl = kl_divergence(&p, &q).unwrap();
        assert!(kl.abs() < 1e-12, "{kl}");
    }

    #[test]
    fn overlap_ignores_reordering_inside_the_top_k_but_not_outside_it() {
        // Set membership, not rank equality: a swap within the top k is what
        // temperature is for, while a token entering or leaving the set is a
        // real change to what can be sampled.
        let reference = [5.0f32, 4.0, 3.0, 0.0];
        let reordered = [4.0f32, 5.0, 3.0, 0.0];
        assert_eq!(topk_overlap(&reference, &reordered, 3), 1.0);

        let displaced = [5.0f32, 4.0, 0.0, 3.0];
        assert!(
            (topk_overlap(&reference, &displaced, 3) - 2.0 / 3.0).abs() < 1e-12,
            "{}",
            topk_overlap(&reference, &displaced, 3)
        );
    }

    #[test]
    fn overlap_of_a_k_wider_than_the_vocabulary_is_the_whole_vocabulary() {
        let x = [1.0f32, 2.0];
        assert_eq!(topk_overlap(&x, &x, 50), 1.0);
    }

    #[test]
    fn argmax_resolves_a_tie_to_the_first_index_as_torch_does() {
        // The contract this module exists to pin. Resolving the other way
        // makes greedy decoding disagree with the reference implementation on
        // a tie, which reads as a defect in the forward pass and is not one.
        assert_eq!(argmax_first(&[1.0, 1.0, 1.0]), 0);
        assert_eq!(argmax_first(&[0.0, 2.0, 2.0, 1.0]), 1);
        assert_eq!(argmax_first(&[-3.0, -1.0, -5.0]), 1);
    }
}
