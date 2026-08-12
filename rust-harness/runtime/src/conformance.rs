/*
 * conformance — invariants every backend owes, stated once.
 *
 * These are obligations, not tests. `model::cpu` satisfies them today and the
 * CUDA backend will have to satisfy them to be accepted; the checks live here
 * rather than inside `cpu.rs`'s `mod tests` because a backend in another crate
 * cannot call into that. The precedent is `ladder.rs`, which holds the verdict
 * logic so that `bin/ladder.rs` can stay a driver.
 *
 * They return `Result` rather than panicking, so a caller that is not
 * `cargo test` -- an acceptance binary, a bring-up harness for a new backend --
 * can report the failure in its own format.
 *
 * What these claim
 * ----------------
 *
 * That how a request is *internally divided* is not observable in its output.
 * A backend may prefill a thousand tokens in one pass, in groups of 64, or one
 * at a time; it may compute logits for every position or only the ones asked
 * for. None of those choices may move a single bit of any row it returns.
 *
 * That property is what makes a batched fast path safe to add, and it is the
 * one an accelerator is most likely to break, because the natural GPU
 * implementations of prefill and decode are different kernels that need not
 * agree in the last bits.
 *
 * What these do not claim
 * -----------------------
 *
 * Nothing about whether the logits are *right*. Every check here compares the
 * backend against itself, so a backend that is uniformly wrong passes all of
 * them. Correctness against an independent implementation is the ladder's job
 * (`ladder.rs`, and `runtime/tests/ladder_tiny.rs` for the offline tier);
 * conformance is what remains to be shown once the ladder is green.
 *
 * Exact equality throughout, never a tolerance. Dividing a request differently
 * reorders no summation -- see `.styles/RUST_STYLE.md` on determinism -- so the
 * only admissible result is the same bits, and a tolerance here would hide
 * precisely the drift these exist to catch.
 */

use anyhow::{ensure, Result};

use crate::model::{Logits, Model, Token};

/// Builds a fresh model. One per composition being compared.
///
/// Boxed rather than generic so a backend crate can hand over closures over
/// its own type without this module naming it, and so a slice may hold
/// several differently-configured builders of the same backend.
pub type Builder<'a> = &'a dyn Fn() -> Result<Box<dyn Model>>;

/// The first index at which two rows differ, with both values.
///
/// Rows are vocabulary-wide -- 262144 at E4B -- so a failure names the one
/// element that moved rather than printing either row.
fn first_difference(a: &Logits, b: &Logits) -> Option<(usize, f32, f32)> {
    if a.0.len() != b.0.len() {
        return Some((0, a.0.len() as f32, b.0.len() as f32));
    }
    a.0.iter()
        .zip(b.0.iter())
        .position(|(x, y)| x.to_bits() != y.to_bits())
        .map(|i| (i, a.0[i], b.0[i]))
}

/// Every builder must return identical logits at every position.
///
/// `builders[0]` is the reference; each later one is compared against it. The
/// labels appear in the failure, so name them for the composition they set up
/// ("stepped", "batch 64") rather than by index.
///
/// Callers supply builders rather than one model reconfigured in place because
/// a composition is chosen before the first token is prefilled, and because
/// the knob that selects it is backend-specific: `CpuModel` has `set_batch`,
/// an accelerator will have something else, and this module should not know
/// which.
pub fn check_batch_composition_invariance(
    builders: &[(&str, Builder)],
    tokens: &[Token],
) -> Result<()> {
    ensure!(
        builders.len() >= 2,
        "batch-composition invariance needs at least two compositions to compare, given {}",
        builders.len()
    );

    let (reference_label, build_reference) = &builders[0];
    let mut reference_model = build_reference()?;
    let reference = reference_model.prefill_all(tokens)?;
    ensure!(
        reference.len() == tokens.len(),
        "composition {reference_label}: prefill_all returned {} rows for {} tokens",
        reference.len(),
        tokens.len()
    );

    for (label, build) in &builders[1..] {
        let mut model = build()?;
        let rows = model.prefill_all(tokens)?;
        ensure!(
            rows.len() == reference.len(),
            "composition {label}: {} rows against {} from {reference_label}, over {} tokens",
            rows.len(),
            reference.len(),
            tokens.len()
        );
        for (position, (row, want)) in rows.iter().zip(reference.iter()).enumerate() {
            if let Some((i, got, expected)) = first_difference(row, want) {
                anyhow::bail!(
                    "composition {label} differs from {reference_label} at position \
                     {position} of {}, logit {i}: {got:e} against {expected:e}",
                    tokens.len()
                );
            }
        }
        ensure!(
            model.cached_len() == tokens.len(),
            "composition {label}: cached {} of {} tokens; skipping logits must not \
             skip the forward pass",
            model.cached_len(),
            tokens.len()
        );
    }
    Ok(())
}

/// A row must not depend on which *other* rows were asked for.
///
/// `prefill_from(tokens, first)` must be exactly the tail of
/// `prefill_all(tokens)` at every `first`, bit for bit. `first` selects which
/// rows are computed and may change nothing about how any of them is folded.
///
/// The failure mode this rules out is a backend that batches the head across
/// the rows of one request and lets the batch's composition -- its width, or
/// which rows are live in it -- reach the arithmetic. mistral.rs answers the
/// same risk by isolating logit-returning requests from each other; we permit
/// them to share a pass and owe this check in exchange.
///
/// `firsts` is supplied rather than swept because each offset costs a fresh
/// model and a full prefill: exhaustive at nine tokens is cheap and is what
/// `cpu.rs` does, exhaustive at several hundred is seconds of gate time for
/// offsets that differ in nothing. Choose the ones that straddle a group
/// boundary, and the two ends.
pub fn check_logit_rows_are_batch_isolated(
    build: Builder,
    tokens: &[Token],
    firsts: &[usize],
) -> Result<()> {
    let mut model = build()?;
    let reference = model.prefill_all(tokens)?;

    for &first in firsts {
        ensure!(
            first <= tokens.len(),
            "offset {first} is past the {} tokens supplied",
            tokens.len()
        );
        let mut model = build()?;
        let rows = model.prefill_from(tokens, first)?;
        ensure!(
            rows.len() == tokens.len() - first,
            "prefill_from(.., {first}) returned {} rows, wanted {}",
            rows.len(),
            tokens.len() - first
        );
        for (i, row) in rows.iter().enumerate() {
            if let Some((j, got, expected)) = first_difference(row, &reference[first + i]) {
                anyhow::bail!(
                    "prefill_from(.., {first}) row {i} (position {}) differs from \
                     prefill_all, logit {j}: {got:e} against {expected:e}",
                    first + i
                );
            }
        }
        ensure!(
            model.cached_len() == tokens.len(),
            "prefill_from(.., {first}) cached {} of {} tokens; the rows a caller \
             did not ask for must still be prefilled",
            model.cached_len(),
            tokens.len()
        );
    }
    Ok(())
}

/// What the committed long-context golden is a golden *of*.
///
/// The binary that writes `fixtures/long-context-logits.json` and the test
/// that reads it must agree on the model, the tokens and the sampled
/// positions, or the comparison is between two different things and its
/// failure says nothing. They agree by both reading this, rather than by
/// carrying two copies that drift.
pub mod golden {
    use crate::model::Token;
    use crate::synth::TinySpec;

    /// Interleaved sliding and global layers over a two-layer shared-KV tail,
    /// window 3 -- the configuration whose storage cap is the next package.
    pub fn spec() -> TinySpec {
        TinySpec::e4b_shaped()
    }

    /// Long enough for three full groups at `DEFAULT_BATCH` and a short
    /// trailing one, so the recorded rows depend on the window and the group
    /// boundary interacting rather than on either alone.
    pub const TOKENS: usize = 200;

    /// Both ends, both sides of the first two group boundaries, and one
    /// position interior to a group.
    ///
    /// Not every row: 200 rows of a 288-wide vocabulary is a 700 KB fixture to
    /// detect what six rows detect, since a change past the window moves every
    /// row after it rather than one.
    pub const POSITIONS: [usize; 6] = [0, 1, 63, 64, 127, 199];

    /// `n` deterministic in-vocabulary tokens, and the only definition of the
    /// sequence -- `runtime/tests/long_context.rs` calls this rather than
    /// repeating the stride, since a golden compared against a different
    /// prompt fails for a reason that has nothing to do with the engine.
    ///
    /// Strided rather than sequential so neighbouring positions do not carry
    /// neighbouring embeddings, where a mask error can average away instead of
    /// showing up. The offset clears the control specials at the base of the
    /// synthetic vocabulary.
    pub fn tokens(n: usize) -> Vec<Token> {
        (0..n).map(|i| ((i * 37 + 11) % 250 + 5) as Token).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model whose rows are a function of position, and optionally wrong.
    ///
    /// Deliberately not `MockModel`: these tests are about whether the checks
    /// above *fail* when they should, so the fixture has to be able to
    /// misbehave in exactly one way at a time.
    struct Fake {
        cached: usize,
        /// Added to the row at this position, if any. The injected divergence.
        skew: Option<usize>,
        /// Stop advancing the cache, modelling a backend that skips the
        /// forward pass for rows nobody asked for.
        freeze_cache: bool,
        /// Perturb the first returned row whenever `prefill_from` is asked
        /// for a proper suffix.
        ///
        /// The divergence has to be conditioned on `first` specifically. A
        /// fixture that is merely *wrong* passes the isolation check, because
        /// that check compares the backend against its own `prefill_all` and
        /// a uniform error lands on both sides -- which is the same reason
        /// the module banner says these checks cannot detect a backend that
        /// is uniformly wrong.
        corrupt_partial: bool,
    }

    impl Fake {
        fn row(&self, position: usize) -> Logits {
            let bump = f32::from(self.skew == Some(position));
            Logits(vec![position as f32, 1.0 + bump])
        }
    }

    impl Model for Fake {
        fn vocab_size(&self) -> usize {
            2
        }
        fn cached_len(&self) -> usize {
            self.cached
        }
        fn prefill(&mut self, tokens: &[Token]) -> Result<Logits> {
            let last = self.cached + tokens.len() - 1;
            if !self.freeze_cache {
                self.cached += tokens.len();
            }
            Ok(self.row(last))
        }
        fn prefill_from(&mut self, tokens: &[Token], first: usize) -> Result<Vec<Logits>> {
            let corrupt = self.corrupt_partial && first > 0;
            let mut rows = self.prefill_all(tokens)?;
            let mut rows = rows.split_off(first);
            if corrupt {
                if let Some(row) = rows.first_mut() {
                    row.0[1] += 1.0;
                }
            }
            Ok(rows)
        }
        fn truncate(&mut self, len: usize) -> Result<()> {
            self.cached = len;
            Ok(())
        }
        fn fork(&self) -> Result<Box<dyn Model>> {
            anyhow::bail!("the conformance fixture does not fork")
        }
    }

    /// A well-behaved fixture, or one broken in exactly one way.
    #[derive(Default)]
    struct Break {
        skew: Option<usize>,
        freeze_cache: bool,
        corrupt_partial: bool,
    }

    fn build(b: Break) -> Box<dyn Model> {
        Box::new(Fake {
            cached: 0,
            skew: b.skew,
            freeze_cache: b.freeze_cache,
            corrupt_partial: b.corrupt_partial,
        })
    }

    fn sound() -> Box<dyn Model> {
        build(Break::default())
    }

    const TOKENS: [Token; 6] = [1, 2, 3, 4, 5, 6];

    #[test]
    fn agreeing_compositions_pass() {
        check_batch_composition_invariance(
            &[("reference", &|| Ok(sound())), ("twin", &|| Ok(sound()))],
            &TOKENS,
        )
        .unwrap();
    }

    /// The check must be able to fail, and must say where.
    ///
    /// Without this the whole module could be vacuous -- a conformance harness
    /// that passes everything is the exact failure this package exists to
    /// prevent, and it looks identical to a working one from the gate's exit
    /// code.
    #[test]
    fn the_invariant_check_fails_when_a_builder_disagrees() {
        let err = check_batch_composition_invariance(
            &[
                ("reference", &|| Ok(sound())),
                ("skewed", &|| {
                    Ok(build(Break {
                        skew: Some(3),
                        ..Break::default()
                    }))
                }),
            ],
            &TOKENS,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("skewed"), "{err}");
        assert!(err.contains("position 3"), "{err}");
        assert!(err.contains("logit 1"), "{err}");
    }

    #[test]
    fn one_composition_is_not_a_comparison() {
        let err = check_batch_composition_invariance(&[("lonely", &|| Ok(sound()))], &TOKENS)
            .unwrap_err()
            .to_string();
        assert!(err.contains("at least two"), "{err}");
    }

    #[test]
    fn a_backend_that_stops_advancing_its_cache_is_caught() {
        let err = check_batch_composition_invariance(
            &[
                ("reference", &|| Ok(sound())),
                ("frozen", &|| {
                    Ok(build(Break {
                        freeze_cache: true,
                        ..Break::default()
                    }))
                }),
            ],
            &TOKENS,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("frozen"), "{err}");
    }

    #[test]
    fn a_row_that_changes_with_the_rows_beside_it_is_caught() {
        let firsts: Vec<usize> = (0..=TOKENS.len()).collect();
        check_logit_rows_are_batch_isolated(&|| Ok(sound()), &TOKENS, &firsts).unwrap();
        let err = check_logit_rows_are_batch_isolated(
            &|| {
                Ok(build(Break {
                    corrupt_partial: true,
                    ..Break::default()
                }))
            },
            &TOKENS,
            &firsts,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("prefill_from"), "{err}");
        assert!(err.contains("logit 1"), "{err}");
    }
}
