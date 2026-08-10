/*
 * ladder — the five-rung correctness gate against a reference implementation.
 *
 * Every rung compares us against transformers. None compares us against our
 * own earlier output, which is the property the whole module exists for: the
 * numeric gate was green for nine work packages while the forward pass
 * produced noise, because it compared cpu.rs against a numpy oracle the same
 * author transcribed from the same documents, on random weights. Two
 * implementations of one belief agree perfectly.
 *
 *   L0  token ids for the rendered prompt
 *   L1  per-layer residual stream and the stages before it
 *   L2  the final-position distribution
 *   L3  perplexity over a token window
 *   L4  greedy continuations
 *
 * A library rather than only a binary, so the machinery -- ordering,
 * short-circuiting, thresholds, rendering -- is testable in the sandbox with
 * no model at all. A rung that names the wrong divergent position is itself a
 * defect, and that is exactly the kind a real-weight-only test cannot catch
 * because nobody can tell the two failures apart from the output.
 */

use std::fmt::Write as _;

use anyhow::{ensure, Result};

use crate::model::{Model, Token};
use crate::sampler::{Sampler, SamplingParams};

/// One rung of the ladder, in the order they are climbed.
///
/// The order is the whole design. Rung N is not evidence about anything if
/// rung N-1 failed: wrong token ids make every residual wrong, a wrong
/// residual stream makes every logit wrong, and wrong logits make perplexity
/// and greedy decoding meaningless. Reporting all five regardless would
/// produce four numbers that look like findings and are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rung {
    L0Tokenizer,
    L1Layers,
    L2Logits,
    L3Perplexity,
    L4Greedy,
}

impl Rung {
    /// Every rung, in climb order.
    pub const ALL: [Rung; 5] = [
        Rung::L0Tokenizer,
        Rung::L1Layers,
        Rung::L2Logits,
        Rung::L3Perplexity,
        Rung::L4Greedy,
    ];

    /// The short label used in the table's first column.
    pub fn label(self) -> &'static str {
        match self {
            Rung::L0Tokenizer => "L0",
            Rung::L1Layers => "L1",
            Rung::L2Logits => "L2",
            Rung::L3Perplexity => "L3",
            Rung::L4Greedy => "L4",
        }
    }

    /// What the rung compares, for the table's second column.
    pub fn subject(self) -> &'static str {
        match self {
            Rung::L0Tokenizer => "token ids",
            Rung::L1Layers => "residual stream",
            Rung::L2Logits => "final distribution",
            Rung::L3Perplexity => "perplexity",
            Rung::L4Greedy => "greedy continuation",
        }
    }

    /// Whether the rung's cost scales with the run size.
    ///
    /// L0 through L2 are one forward pass over a fixed prompt; only
    /// perplexity and greedy decoding grow with the corpus. The table says so
    /// per row, so nobody reads a quick run's L2 as weaker than a deep run's.
    pub fn scales_with_size(self) -> bool {
        matches!(self, Rung::L3Perplexity | Rung::L4Greedy)
    }
}

/// How a rung ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail,
    /// The rung's reference is absent, so it was not run. A skipped rung does
    /// not stop the climb -- it is missing evidence, not contrary evidence.
    Skipped,
    /// An earlier rung failed, so this one was never attempted.
    ///
    /// Distinct from `Skipped` on purpose, and both are printed rather than
    /// omitted: a missing row reads as "the tool crashed", `skipped` reads as
    /// "we had nothing to compare against", and `not reached` reads as "we
    /// know why, look up".
    NotReached,
}

impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Status::Pass => "PASS",
            Status::Fail => "FAIL",
            Status::Skipped => "skipped",
            Status::NotReached => "not reached",
        }
    }
}

/// Which side of the threshold a measure has to land on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    /// Divergences: smaller is better.
    AtMost,
    /// Agreements: larger is better.
    AtLeast,
}

impl Cmp {
    fn holds(self, value: f64, threshold: f64) -> bool {
        // A NaN fails either comparison rather than passing one. Written as
        // an explicit guard because `!(nan > t)` is true and the negation of
        // a float comparison is the classic way a broken metric reports
        // success.
        if value.is_nan() {
            return false;
        }
        match self {
            Cmp::AtMost => value <= threshold,
            Cmp::AtLeast => value >= threshold,
        }
    }

    fn symbol(self) -> &'static str {
        match self {
            Cmp::AtMost => "<=",
            Cmp::AtLeast => ">=",
        }
    }
}

/// One number a rung measured, and the threshold it had to meet.
#[derive(Debug, Clone)]
pub struct Measure {
    pub name: &'static str,
    pub value: f64,
    pub threshold: f64,
    pub cmp: Cmp,
}

impl Measure {
    pub fn new(name: &'static str, value: f64, threshold: f64, cmp: Cmp) -> Self {
        Self {
            name,
            value,
            threshold,
            cmp,
        }
    }

    /// Whether this measure met its threshold.
    ///
    /// Derived rather than stored, so a caller cannot construct a `Measure`
    /// whose recorded verdict disagrees with its own numbers.
    pub fn ok(&self) -> bool {
        self.cmp.holds(self.value, self.threshold)
    }
}

/// What one rung found.
#[derive(Debug, Clone)]
pub struct RungResult {
    pub rung: Rung,
    pub status: Status,
    pub measures: Vec<Measure>,
    /// One line of context: which layer diverged, which prompt, why skipped.
    /// Empty when the rung has nothing to add beyond its measures.
    pub note: String,
}

impl RungResult {
    /// A rung that ran, with its status derived from its measures.
    ///
    /// The status cannot be passed in: a rung reporting `Pass` beside a
    /// measure that missed its threshold is a contradiction the type should
    /// not be able to express.
    pub fn measured(rung: Rung, measures: Vec<Measure>, note: impl Into<String>) -> Self {
        let status = match measures.iter().all(Measure::ok) {
            true => Status::Pass,
            false => Status::Fail,
        };
        Self {
            rung,
            status,
            measures,
            note: note.into(),
        }
    }

    /// A rung whose reference is absent.
    pub fn skipped(rung: Rung, why: impl Into<String>) -> Self {
        Self {
            rung,
            status: Status::Skipped,
            measures: Vec::new(),
            note: why.into(),
        }
    }

    fn not_reached(rung: Rung, blamed: Rung) -> Self {
        Self {
            rung,
            status: Status::NotReached,
            measures: Vec::new(),
            note: format!("{} failed", blamed.label()),
        }
    }
}

/// How much work a run does. Only L3 and L4 read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Size {
    Quick,
    Standard,
    Deep,
}

impl Size {
    pub const ALL: [Size; 3] = [Size::Quick, Size::Standard, Size::Deep];

    pub fn label(self) -> &'static str {
        match self {
            Size::Quick => "quick",
            Size::Standard => "standard",
            Size::Deep => "deep",
        }
    }

    pub fn parse(s: &str) -> Option<Size> {
        Size::ALL.into_iter().find(|z| z.label() == s)
    }

    /// Tokens of the corpus this size scores.
    ///
    /// PROVISIONAL. These are placeholders until the CPU backend's real
    /// tokens/sec on E4B is measured; the counts then land here with the
    /// measurement quoted beside them. A size that takes an unknown amount of
    /// time is a size nobody will run.
    pub fn tokens(self) -> usize {
        match self {
            Size::Quick => 512,
            Size::Standard => 2048,
            Size::Deep => 8192,
        }
    }

    /// Prompts the greedy rung continues, from the front of the prompt list.
    pub fn prompts(self) -> usize {
        match self {
            Size::Quick => 2,
            Size::Standard => 8,
            Size::Deep => 32,
        }
    }
}

/// Window length `eval::perplexity` scores in, and the divisor every size's
/// token count must be a multiple of.
///
/// Fixed across sizes on purpose. Perplexity depends on how much context each
/// scored token had, so the same corpus scored in different windows gives
/// different numbers -- varying the chunk with the size would make the sizes
/// incomparable in a way that is invisible in the output.
pub const CHUNK: usize = 256;

/// The prefix of `corpus` this size scores.
///
/// Sizes are nested prefixes of one corpus rather than separate corpora, so a
/// deep run reproduces a quick run's number on its first `Quick::tokens()`
/// tokens exactly. Perplexity over 512 tokens and over 8192 tokens are not
/// otherwise comparable quantities, and someone will eventually diff them.
///
/// Returns `None` when the corpus is too short for the size, rather than
/// silently scoring fewer tokens than the header will claim.
pub fn corpus_for(corpus: &[Token], size: Size) -> Option<&[Token]> {
    corpus.get(..size.tokens())
}

/// A complete run: what was compared, against what, and what every rung said.
#[derive(Debug, Clone)]
pub struct Ladder {
    pub size: Size,
    pub model: String,
    pub reference: String,
    pub results: Vec<RungResult>,
}

/// Climb the rungs in order, stopping at the first failure.
///
/// Short-circuiting is structural rather than a `break` in the caller: every
/// rung after the failure is recorded as `NotReached` and still printed, so
/// the table always has five rows and the reader can see both what failed and
/// what consequently went unmeasured.
///
/// `run` returns `None` for a rung whose reference is absent; that becomes
/// `Skipped` and the climb continues.
pub fn climb(
    size: Size,
    model: impl Into<String>,
    reference: impl Into<String>,
    mut run: impl FnMut(Rung) -> Option<RungResult>,
) -> Ladder {
    let mut results = Vec::with_capacity(Rung::ALL.len());
    let mut failed: Option<Rung> = None;
    for rung in Rung::ALL {
        match failed {
            Some(blamed) => results.push(RungResult::not_reached(rung, blamed)),
            None => {
                let result = run(rung)
                    .unwrap_or_else(|| RungResult::skipped(rung, "no reference for this rung"));
                if result.status == Status::Fail {
                    failed = Some(rung);
                }
                results.push(result);
            }
        }
    }
    Ladder {
        size,
        model: model.into(),
        reference: reference.into(),
        results,
    }
}

impl Ladder {
    /// The first rung that failed, if any.
    pub fn first_failure(&self) -> Option<&RungResult> {
        self.results.iter().find(|r| r.status == Status::Fail)
    }

    /// Whether the run is a pass. A skipped rung does not fail the ladder,
    /// but it is visible in the table and in the summary count.
    pub fn ok(&self) -> bool {
        self.first_failure().is_none()
    }

    /// Rungs that had no reference to compare against.
    pub fn skipped(&self) -> usize {
        self.results
            .iter()
            .filter(|r| r.status == Status::Skipped)
            .count()
    }
}

/// The results table.
///
/// The header carries the size and its token count because a perplexity or
/// agreement number means nothing without them, and a number copied out of a
/// terminal loses its context immediately otherwise.
pub fn render(l: &Ladder) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "size={} tokens={} prompts={}",
        l.size.label(),
        l.size.tokens(),
        l.size.prompts()
    );
    let _ = writeln!(s, "model={}", l.model);
    let _ = writeln!(s, "reference={}", l.reference);
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "rung  {:<22}  {:<12}  {:<14}  verdict",
        "check", "value", "threshold"
    );

    for r in &l.results {
        let _ = writeln!(
            s,
            "{:<4}  {:<22}  {:<12}  {:<14}  {}",
            r.rung.label(),
            r.rung.subject(),
            "",
            "",
            r.status.label()
        );
        for m in &r.measures {
            let mark = match m.ok() {
                true => "ok",
                false => "FAIL",
            };
            let _ = writeln!(
                s,
                "      {:<22}  {:<12.6}  {} {:<11.6}  {}",
                m.name,
                m.value,
                m.cmp.symbol(),
                m.threshold,
                mark
            );
        }
        if !r.note.is_empty() {
            let _ = writeln!(s, "      {}", r.note);
        }
    }

    let _ = writeln!(s);
    match l.first_failure() {
        None => {
            let _ = writeln!(
                s,
                "[OK] Ladder: every rung within tolerance ({} skipped)",
                l.skipped()
            );
        }
        Some(r) => {
            let _ = writeln!(
                s,
                "[ERROR] Ladder: {} failed on {}",
                r.rung.label(),
                r.rung.subject()
            );
            if !r.note.is_empty() {
                let _ = writeln!(s, "  {}", r.note);
            }
            let _ = writeln!(
                s,
                "  Rungs below it were not reached: each one reads the output of the \
                 one above, so they would measure a run already known to be wrong."
            );
        }
    }
    s
}

/// The tolerances every rung is judged against.
///
/// PROVISIONAL, all of them. Not one has been set by a measurement of this
/// engine against transformers, because that measurement is what the ladder
/// exists to make possible. They are collected here rather than spread
/// through the rung bodies so the provisional ones can be counted, and so a
/// change to one is a diff a reviewer sees.
///
/// **A threshold is never tuned to make the current run pass.** A value that
/// lands between "clearly fine" and the threshold is a finding, and moving
/// the threshold to swallow it converts a finding into a green gate, which is
/// the failure the whole package is a response to.
pub mod thresholds {
    /// Relative perplexity delta against the reference. PROVISIONAL.
    pub const PPL_DELTA: f64 = 0.01;
    /// Worst single-token NLL delta, in nats. PROVISIONAL.
    ///
    /// Separate from the mean because perplexity is an average and averages
    /// hide one badly wrong token in a thousand ordinary ones -- which is
    /// exactly what a rare-path bug (a sliding-window boundary, a shared-KV
    /// layer) looks like.
    pub const PPL_MAX_TOKEN_DELTA: f64 = 0.05;
    /// Fraction of greedy positions that must match the reference token.
    /// PROVISIONAL.
    pub const GREEDY_AGREEMENT: f64 = 0.98;
    /// Mean common prefix, as a fraction of the continuation length.
    /// PROVISIONAL.
    ///
    /// A fraction rather than a token count so the threshold survives a
    /// change to the continuation length. Position-wise agreement and prefix
    /// length measure different failures: one late wrong token barely moves
    /// agreement but halves the prefix, and greedy decoding is
    /// autoregressive, so everything after a divergence is a different run
    /// rather than more evidence.
    pub const GREEDY_PREFIX: f64 = 0.75;
}

/// Reference perplexity for one corpus, as the reference implementation
/// scored it.
///
/// Carries `chunk` because a perplexity computed in different windows is a
/// different quantity, and an artifact that does not say which windows it
/// used cannot be compared to anything.
#[derive(Debug, Clone)]
pub struct PplReference {
    pub chunk: usize,
    /// Per-token NLL in scoring order, paired with the corpus index of the
    /// scored token.
    pub scored: Vec<(usize, f64)>,
}

impl PplReference {
    fn nll_sum(&self) -> f64 {
        self.scored.iter().map(|(_, n)| n).sum()
    }

    fn ppl(&self) -> f64 {
        (self.nll_sum() / self.scored.len() as f64).exp()
    }
}

/// L3: perplexity over a token window, against the reference's own NLLs.
///
/// Errors rather than fails when the artifact and the run disagree about
/// *which* tokens were scored: that is a broken or mismatched reference, not
/// evidence about the engine, and reporting it as a red rung would send the
/// reader looking for a numerics bug that is not there.
pub fn rung_perplexity(
    model: &mut dyn Model,
    corpus: &[Token],
    reference: &PplReference,
) -> Result<RungResult> {
    let scored = crate::eval::per_token_nll(model, corpus, reference.chunk)?;
    ensure!(
        scored.len() == reference.scored.len(),
        "reference scored {} tokens, this run scored {} -- the corpus or the \
         {}-token window does not match the artifact",
        reference.scored.len(),
        scored.len(),
        reference.chunk
    );

    let mut worst = (0usize, 0.0f64);
    for (ours, &(position, want)) in scored.iter().zip(&reference.scored) {
        ensure!(
            ours.position == position,
            "reference scores corpus position {position} where this run scores {}",
            ours.position
        );
        let delta = (ours.nll - want).abs();
        if delta > worst.1 {
            worst = (position, delta);
        }
    }

    let ours_ppl = (scored.iter().map(|s| s.nll).sum::<f64>() / scored.len() as f64).exp();
    let want_ppl = reference.ppl();
    // Relative, because perplexity's scale depends on the corpus: half a nat
    // is nothing at ppl 200 and enormous at ppl 1.2.
    let delta = ((ours_ppl - want_ppl) / want_ppl).abs();

    Ok(RungResult::measured(
        Rung::L3Perplexity,
        vec![
            Measure::new("ppl delta", delta, thresholds::PPL_DELTA, Cmp::AtMost),
            Measure::new(
                "worst token nll",
                worst.1,
                thresholds::PPL_MAX_TOKEN_DELTA,
                Cmp::AtMost,
            ),
        ],
        format!(
            "ppl {ours_ppl:.4} against {want_ppl:.4} over {} tokens; worst token at corpus \
             position {}",
            scored.len(),
            worst.0
        ),
    ))
}

/// One greedy continuation the reference implementation produced.
#[derive(Debug, Clone)]
pub struct GreedyReference {
    pub prompt: Vec<Token>,
    /// What `generate(do_sample=False)` emitted, in order.
    pub continuation: Vec<Token>,
}

/// Whether the largest finite logit occurs more than once.
///
/// Counted rather than assumed absent. Exact ties in f32 over a 262144-wide
/// vocabulary are rare but not absent -- logit softcapping saturates and
/// grammar masking flattens -- and a tie is the one position where our
/// tie-break convention, not our arithmetic, decides the token. A future
/// disagreement is attributable only if the count was reported at the time.
fn has_tie(logits: &[f32]) -> bool {
    let best = logits
        .iter()
        .copied()
        .filter(|l| l.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    logits.iter().filter(|&&l| l == best).count() > 1
}

/// L4: greedy continuations, decoded through the shipping sampler.
///
/// Through `Sampler` at temperature zero rather than a local argmax: a
/// ladder-only argmax would let this rung go green while the decoder every
/// caller actually uses did something else, which is the shape of gate this
/// package exists to remove.
pub fn rung_greedy(model: &mut dyn Model, references: &[GreedyReference]) -> Result<RungResult> {
    ensure!(!references.is_empty(), "no greedy references to compare");
    let mut sampler = Sampler::new(SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    });

    let mut matched = 0usize;
    let mut total = 0usize;
    let mut prefix_fraction = 0.0f64;
    let mut ties = 0usize;
    // The earliest divergence anywhere, as (prompt index, offset). The report
    // names one position rather than all of them because after a greedy
    // decode diverges its remaining tokens are a different continuation, not
    // more findings.
    let mut first: Option<(usize, usize)> = None;

    for (p, r) in references.iter().enumerate() {
        ensure!(!r.prompt.is_empty(), "greedy reference {p} has no prompt");
        ensure!(
            !r.continuation.is_empty(),
            "greedy reference {p} has no continuation"
        );
        model.truncate(0)?;
        let mut logits = model.prefill(&r.prompt)?;
        let mut prefix = r.continuation.len();
        for (i, &want) in r.continuation.iter().enumerate() {
            if has_tie(&logits.0) {
                ties += 1;
            }
            let got = sampler.sample(&logits, None)?;
            total += 1;
            if got == want {
                matched += 1;
            } else {
                if prefix == r.continuation.len() {
                    prefix = i;
                }
                if first.is_none() {
                    first = Some((p, i));
                }
            }
            // Fed our own token, not the reference's. Feeding the reference's
            // would measure teacher-forced agreement, which is L2 again with
            // extra steps; the question here is whether an unattended decode
            // stays on the same trajectory.
            logits = model.prefill(&[got])?;
        }
        prefix_fraction += prefix as f64 / r.continuation.len() as f64;
    }

    let agreement = matched as f64 / total as f64;
    let mean_prefix = prefix_fraction / references.len() as f64;
    let note = match first {
        None => format!("{} prompts, {ties} ties", references.len()),
        Some((p, i)) => {
            format!("first divergence in prompt {p} at continuation offset {i}; {ties} ties")
        }
    };

    Ok(RungResult::measured(
        Rung::L4Greedy,
        vec![
            Measure::new(
                "token agreement",
                agreement,
                thresholds::GREEDY_AGREEMENT,
                Cmp::AtLeast,
            ),
            Measure::new(
                "mean common prefix",
                mean_prefix,
                thresholds::GREEDY_PREFIX,
                Cmp::AtLeast,
            ),
        ],
        note,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(rung: Rung) -> RungResult {
        RungResult::measured(rung, vec![Measure::new("m", 0.0, 1.0, Cmp::AtMost)], "")
    }

    fn fail(rung: Rung) -> RungResult {
        RungResult::measured(rung, vec![Measure::new("m", 9.0, 1.0, Cmp::AtMost)], "")
    }

    #[test]
    fn a_failing_rung_leaves_every_later_rung_not_reached() {
        let l = climb(Size::Quick, "m", "r", |rung| {
            Some(match rung {
                Rung::L2Logits => fail(rung),
                _ => pass(rung),
            })
        });
        let statuses: Vec<Status> = l.results.iter().map(|r| r.status).collect();
        assert_eq!(
            statuses,
            vec![
                Status::Pass,
                Status::Pass,
                Status::Fail,
                Status::NotReached,
                Status::NotReached
            ],
            "{statuses:?}"
        );
        assert_eq!(l.first_failure().unwrap().rung, Rung::L2Logits);
        assert!(!l.ok());
    }

    #[test]
    fn a_short_circuited_table_still_prints_every_rung() {
        // A missing row reads as a crashed tool. The distinction between
        // "not reached" and "absent" is the reason Status has four variants
        // and not two.
        let l = climb(Size::Quick, "m", "r", |rung| {
            Some(match rung {
                Rung::L1Layers => fail(rung),
                _ => pass(rung),
            })
        });
        let out = render(&l);
        for rung in Rung::ALL {
            assert!(
                out.contains(rung.label()),
                "{} missing:\n{out}",
                rung.label()
            );
        }
        assert!(out.contains("not reached"), "{out}");
        assert!(out.contains("[ERROR] Ladder: L1 failed"), "{out}");
    }

    #[test]
    fn a_rung_with_no_reference_is_skipped_without_stopping_the_climb() {
        // Missing evidence is not contrary evidence: a reference we never
        // generated must not read as a defect, and must not hide the rungs
        // after it either.
        let l = climb(Size::Quick, "m", "r", |rung| match rung {
            Rung::L0Tokenizer => None,
            _ => Some(pass(rung)),
        });
        assert_eq!(l.results[0].status, Status::Skipped);
        assert!(l.results[1..].iter().all(|r| r.status == Status::Pass));
        assert!(l.ok());
        assert_eq!(l.skipped(), 1);
        assert!(render(&l).contains("(1 skipped)"), "{}", render(&l));
    }

    #[test]
    fn the_table_header_carries_the_size_and_its_token_count() {
        // A perplexity number without its corpus length is not a number
        // anybody can act on, and it leaves the terminal without one unless
        // the table puts them together.
        let l = climb(Size::Standard, "/models/e4b", "/ref/std", |r| Some(pass(r)));
        let out = render(&l);
        assert!(out.contains("size=standard"), "{out}");
        assert!(
            out.contains(&format!("tokens={}", Size::Standard.tokens())),
            "{out}"
        );
        assert!(out.contains("/models/e4b"), "{out}");
        assert!(out.contains("/ref/std"), "{out}");
    }

    #[test]
    fn a_rungs_status_is_derived_from_its_measures_and_cannot_disagree_with_them() {
        let r = RungResult::measured(
            Rung::L2Logits,
            vec![
                Measure::new("kl", 1e-6, 1e-3, Cmp::AtMost),
                Measure::new("top_k", 0.5, 0.98, Cmp::AtLeast),
            ],
            "",
        );
        assert_eq!(
            r.status,
            Status::Fail,
            "one missed threshold fails the rung"
        );
    }

    #[test]
    fn a_nan_measure_fails_rather_than_passing_a_negated_comparison() {
        // `!(nan <= t)` is true, so a metric that returned NaN would pass any
        // threshold written as a negation. Both directions are checked here
        // because only one of them is the obvious one.
        assert!(!Measure::new("kl", f64::NAN, 1e-3, Cmp::AtMost).ok());
        assert!(!Measure::new("agree", f64::NAN, 0.98, Cmp::AtLeast).ok());
    }

    #[test]
    fn sizes_are_nested_prefixes_of_one_corpus() {
        // The property that makes two runs comparable. Separate corpora per
        // size would produce numbers that look diffable and are not.
        let corpus: Vec<Token> = (0..Size::Deep.tokens() as u32).collect();
        let quick = corpus_for(&corpus, Size::Quick).unwrap();
        let deep = corpus_for(&corpus, Size::Deep).unwrap();
        assert_eq!(quick, &deep[..quick.len()]);
    }

    #[test]
    fn every_size_is_larger_than_the_last_and_a_whole_number_of_windows() {
        // Both are invariants the perplexity comparison depends on: unequal
        // ordering would make "deep" a misnomer, and a partial trailing
        // window is dropped by eval::perplexity, so a non-multiple would
        // score fewer tokens than the header claims.
        for pair in Size::ALL.windows(2) {
            assert!(
                pair[0].tokens() < pair[1].tokens(),
                "{} is not smaller than {}",
                pair[0].label(),
                pair[1].label()
            );
            assert!(pair[0].prompts() < pair[1].prompts());
        }
        for size in Size::ALL {
            assert_eq!(
                size.tokens() % CHUNK,
                0,
                "{} is {} tokens, not a multiple of the {CHUNK}-token window",
                size.label(),
                size.tokens()
            );
        }
    }

    #[test]
    fn a_corpus_too_short_for_the_size_is_none_rather_than_a_short_scoring() {
        let corpus: Vec<Token> = (0..8).collect();
        assert!(corpus_for(&corpus, Size::Quick).is_none());
    }

    #[test]
    fn only_perplexity_and_greedy_decoding_scale_with_the_size() {
        let scaling: Vec<Rung> = Rung::ALL
            .into_iter()
            .filter(|r| r.scales_with_size())
            .collect();
        assert_eq!(scaling, vec![Rung::L3Perplexity, Rung::L4Greedy]);
    }

    #[test]
    fn every_size_label_round_trips_through_parse() {
        for size in Size::ALL {
            assert_eq!(Size::parse(size.label()), Some(size));
        }
        assert_eq!(Size::parse("thorough"), None);
    }

    // The rung bodies below are tested against a scripted mock and a
    // reference derived from that same mock. That is circular as a
    // correctness check and is not meant to be one: what is under test is the
    // rung's own reporting -- that an exact match passes, that a single
    // perturbed token fails, and that the note names the position that moved.
    // A rung which localizes a divergence to the wrong token is a defect no
    // real-weight run could distinguish from the divergence itself.

    use crate::model::mock::MockModel;

    /// A mock whose logits at position `p` peak on `peak(p)`, so a scripted
    /// continuation is predictable by hand.
    fn peaked(vocab: usize, len: usize, peak: impl Fn(usize) -> usize) -> MockModel {
        let script = (0..len)
            .map(|p| {
                let mut l = vec![0.0; vocab];
                l[peak(p) % vocab] = 8.0;
                l
            })
            .collect();
        MockModel::new(vocab, script)
    }

    fn ppl_reference(model: &mut dyn Model, corpus: &[Token], chunk: usize) -> PplReference {
        let scored = crate::eval::per_token_nll(model, corpus, chunk).unwrap();
        PplReference {
            chunk,
            scored: scored.iter().map(|s| (s.position, s.nll)).collect(),
        }
    }

    #[test]
    fn perplexity_matching_the_reference_exactly_passes() {
        let corpus: Vec<Token> = (0..16).map(|i| (i * 3) % 8).collect();
        let reference = ppl_reference(&mut peaked(8, 16, |p| p + 1), &corpus, 8);
        let r = rung_perplexity(&mut peaked(8, 16, |p| p + 1), &corpus, &reference).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?}");
        for m in &r.measures {
            assert_eq!(m.value, 0.0, "{}", m.name);
        }
    }

    #[test]
    fn one_perturbed_token_fails_perplexity_and_the_note_names_its_position() {
        let corpus: Vec<Token> = (0..16).map(|i| (i * 3) % 8).collect();
        let mut reference = ppl_reference(&mut peaked(8, 16, |p| p + 1), &corpus, 8);
        // Position 6 of the corpus, the second scored token of the first
        // window. Perturbing the reference rather than the model keeps the
        // model the same scripted object in both tests, so the only
        // difference between passing and failing is the one number.
        let victim = reference.scored[1];
        reference.scored[1] = (victim.0, victim.1 + 1.5);
        let r = rung_perplexity(&mut peaked(8, 16, |p| p + 1), &corpus, &reference).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(
            r.note.contains(&format!("position {}", victim.0)),
            "note does not name the perturbed position {}: {}",
            victim.0,
            r.note
        );
        let worst = r
            .measures
            .iter()
            .find(|m| m.name == "worst token nll")
            .unwrap();
        assert!((worst.value - 1.5).abs() < 1e-9, "{}", worst.value);
    }

    #[test]
    fn a_reference_scoring_different_positions_is_an_error_not_a_failed_rung() {
        // A mismatched artifact is not evidence about the engine. Reporting
        // it as a red L3 sends the reader hunting a numerics bug that is not
        // there, and the ladder short-circuits every rung below it.
        let corpus: Vec<Token> = (0..16).map(|i| (i * 3) % 8).collect();
        let mut reference = ppl_reference(&mut peaked(8, 16, |p| p + 1), &corpus, 8);
        reference.scored[2].0 += 1;
        let err = rung_perplexity(&mut peaked(8, 16, |p| p + 1), &corpus, &reference)
            .expect_err("expected an error");
        assert!(err.to_string().contains("corpus position"), "{err}");

        reference.scored.pop();
        let err = rung_perplexity(&mut peaked(8, 16, |p| p + 1), &corpus, &reference)
            .expect_err("expected an error");
        assert!(
            err.to_string().contains("does not match the artifact"),
            "{err}"
        );
    }

    #[test]
    fn a_greedy_continuation_reproducing_the_reference_passes() {
        // The mock peaks on p % 4 at position p, and the prompt is 2 tokens,
        // so the continuation after prefill is positions 1, 2, 3, 4 -> tokens
        // 1, 2, 3, 0.
        let refs = vec![GreedyReference {
            prompt: vec![0, 0],
            continuation: vec![1, 2, 3, 0],
        }];
        let r = rung_greedy(&mut peaked(4, 16, |p| p), &refs).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?} note={}", r.note);
        assert!(r.note.contains("0 ties"), "{}", r.note);
    }

    #[test]
    fn one_wrong_greedy_token_fails_and_the_note_names_the_offset() {
        let refs = vec![GreedyReference {
            prompt: vec![0, 0],
            continuation: vec![1, 2, 3, 0],
        }];
        let mut wrong = refs.clone();
        wrong[0].continuation[2] = 1;
        let r = rung_greedy(&mut peaked(4, 16, |p| p), &wrong).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(
            r.note.contains("prompt 0 at continuation offset 2"),
            "{}",
            r.note
        );
        // Agreement stays high on a single wrong token while the prefix
        // collapses to half. That the two measures move differently is the
        // reason both are reported.
        let agree = r
            .measures
            .iter()
            .find(|m| m.name == "token agreement")
            .unwrap();
        let prefix = r
            .measures
            .iter()
            .find(|m| m.name == "mean common prefix")
            .unwrap();
        assert!((agree.value - 0.75).abs() < 1e-9, "{}", agree.value);
        assert!((prefix.value - 0.5).abs() < 1e-9, "{}", prefix.value);
    }

    #[test]
    fn greedy_reports_the_ties_it_had_to_break() {
        // A mock past the end of its script returns all-zero logits: every
        // token ties, and the first-index rule decides all four. The count is
        // what makes a later disagreement with torch attributable rather than
        // mysterious.
        let refs = vec![GreedyReference {
            prompt: vec![0, 0],
            continuation: vec![0, 0, 0, 0],
        }];
        let r = rung_greedy(&mut MockModel::new(4, vec![]), &refs).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?}");
        assert!(r.note.contains("4 ties"), "{}", r.note);
    }

    #[test]
    fn greedy_feeds_back_its_own_token_rather_than_the_references() {
        // Teacher forcing would make this rung L2 with extra steps: every
        // position would be scored from the reference's own trajectory and a
        // model that diverges once but recovers would look identical to one
        // that never diverged. The mock is position-keyed, so the check is
        // that exactly one token is prefilled per step and the cache advances
        // with the decode.
        let refs = vec![GreedyReference {
            prompt: vec![0, 0, 0],
            continuation: vec![3, 0, 1],
        }];
        let mut m = peaked(4, 16, |p| p);
        rung_greedy(&mut m, &refs).unwrap();
        let steps: Vec<usize> = m.records().iter().map(|r| r.tokens.len()).collect();
        assert_eq!(steps, vec![3, 1, 1, 1], "{steps:?}");
        assert_eq!(m.cached_len(), 6);
    }

    #[test]
    fn greedy_with_no_references_is_an_error_rather_than_a_vacuous_pass() {
        // Zero of zero positions agreeing is 100% agreement, and an empty
        // reference set would otherwise report a green rung having compared
        // nothing at all.
        let err = rung_greedy(&mut peaked(4, 8, |p| p), &[]).expect_err("expected an error");
        assert!(err.to_string().contains("no greedy references"), "{err}");
    }
}
