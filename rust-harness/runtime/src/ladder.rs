/*
 * ladder — the seven-rung correctness gate against a reference implementation.
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
 *   L5  logits past the sliding window
 *   L6  attention rows past the sliding window
 *
 * A library rather than only a binary, so the machinery -- ordering,
 * short-circuiting, thresholds, rendering -- is testable in the sandbox with
 * no model at all. A rung that names the wrong divergent position is itself a
 * defect, and that is exactly the kind a real-weight-only test cannot catch
 * because nobody can tell the two failures apart from the output.
 */

use std::fmt::Write as _;
use std::time::{Duration, Instant};

use anyhow::{bail, ensure, Context, Result};

use crate::kv::LayerKind;
use crate::model::cpu::{AttentionKey, AttentionRow};
use crate::model::{Model, Token};
use crate::sampler::{Sampler, SamplingParams};

/// One rung of the ladder, in the order they are climbed.
///
/// The order is the whole design. Rung N is not evidence about anything if
/// rung N-1 failed: wrong token ids make every residual wrong, a wrong
/// residual stream makes every logit wrong, and wrong logits make perplexity
/// and greedy decoding meaningless. Reporting all seven regardless would
/// produce six numbers that look like findings and are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rung {
    L0Tokenizer,
    L1Layers,
    L2Logits,
    L3Perplexity,
    L4Greedy,
    L5LongContext,
    L6Attention,
}

impl Rung {
    /// Every rung, in climb order.
    pub const ALL: [Rung; 7] = [
        Rung::L0Tokenizer,
        Rung::L1Layers,
        Rung::L2Logits,
        Rung::L3Perplexity,
        Rung::L4Greedy,
        Rung::L5LongContext,
        Rung::L6Attention,
    ];

    /// Whether this rung reads the sized corpus.
    ///
    /// Only perplexity and greedy decoding scale with `Size`; L0 through L2
    /// score one recorded prompt and report `tier n/a`. The header uses this
    /// to decide whether a token count describes anything that ran.
    pub fn uses_corpus(self) -> bool {
        matches!(self, Rung::L3Perplexity | Rung::L4Greedy)
    }

    /// The short label used in the table's first column.
    pub fn label(self) -> &'static str {
        match self {
            Rung::L0Tokenizer => "L0",
            Rung::L1Layers => "L1",
            Rung::L2Logits => "L2",
            Rung::L3Perplexity => "L3",
            Rung::L4Greedy => "L4",
            Rung::L5LongContext => "L5",
            Rung::L6Attention => "L6",
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
            Rung::L5LongContext => "logits past the window",
            Rung::L6Attention => "attention past the window",
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
    /// Wall clock for this rung's body, when it ran.
    ///
    /// Stamped by `climb` rather than by a rung body, so no rung can report a
    /// duration that excludes part of its own work. `None` for a rung that was
    /// never reached -- an absent time and a zero time are different claims.
    ///
    /// This is the only field in a result that is not reproducible, which is
    /// why `render` keeps it out of the table entirely. See the timing block.
    pub elapsed: Option<Duration>,
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
            elapsed: None,
        }
    }

    /// A rung whose reference is absent.
    pub fn skipped(rung: Rung, why: impl Into<String>) -> Self {
        Self {
            rung,
            status: Status::Skipped,
            measures: Vec::new(),
            note: why.into(),
            elapsed: None,
        }
    }

    fn not_reached(rung: Rung, blamed: Rung) -> Self {
        Self {
            rung,
            status: Status::NotReached,
            measures: Vec::new(),
            note: format!("{} failed", blamed.label()),
            elapsed: None,
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

/// How many tokens L4 continues each prompt for.
///
/// Fixed across sizes for the reason `CHUNK` is: `Size` varies how many
/// prompts are decoded, not how far each one runs, so a quick run's rows are
/// the same measurement as a deep run's rather than a shorter one. Long enough
/// that a divergence has somewhere to show -- greedy decoding is
/// autoregressive, so a wrong token at offset 3 sends the rest of the
/// continuation somewhere else entirely, and a length of 8 would report that
/// as a near miss.
///
/// The artifact records it as well, so a stored reference states its own
/// length rather than inheriting whatever this constant happens to be when it
/// is read back.
pub const GREEDY_TOKENS: usize = 64;

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
    /// Length of the prompt L0-L2 scored, when the caller knows it.
    ///
    /// Those rungs score one recorded prompt rather than the sized corpus, and
    /// its length appeared nowhere in the table -- the first real run reported
    /// agreement over five tokens under a header saying 512. `climb` cannot
    /// know it (the rung bodies read the artifact), so the runner fills it in.
    pub prompt_tokens: Option<usize>,
    /// Width of the CPU backend's row split, when the runner records it.
    ///
    /// Not a caveat on the numbers: the split is bit-identical at every
    /// width, so two tables taken at one thread and at sixteen must agree
    /// digit for digit. Printed because that is exactly the claim, and a
    /// table that does not say which run it came from cannot support it.
    pub threads: Option<usize>,

    /// Which dot product produced these numbers, when the runner said.
    ///
    /// Unlike `threads` this **is** a caveat on the numbers, and that is why
    /// it is printed rather than assumed: the block kinds are bit-identical
    /// to the default, but `lanes16` reorders the reduction, so a table taken
    /// under it is not comparable digit for digit with one taken without it.
    /// A run that changed the arithmetic and did not say so would be read as
    /// evidence about the engine rather than about the experiment.
    pub dot: Option<String>,

    /// How many positions a prefill put through the forward pass at once,
    /// when the runner said.
    ///
    /// Like `threads` and unlike `dot`, not a caveat on the numbers: every
    /// batch width folds each dot in the same order, so two tables taken
    /// stepped and batched must agree digit for digit. Printed for the same
    /// reason `threads` is — that agreement is the claim, and a table that
    /// does not say which run it came from cannot support it. It is also the
    /// only thing in the header that explains a large wall-clock difference
    /// between two otherwise identical runs.
    pub batch: Option<String>,
}

/// Climb the rungs in order, stopping at the first failure.
///
/// Short-circuiting is structural rather than a `break` in the caller: every
/// rung after the failure is recorded as `NotReached` and still printed, so
/// the table always has seven rows and the reader can see both what failed and
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
                let began = Instant::now();
                let mut result = run(rung)
                    .unwrap_or_else(|| RungResult::skipped(rung, "no reference for this rung"));
                // Stamped around the closure call, which includes a skipped
                // rung's attempt to find its reference. A skip is usually fast,
                // but "fast" is a measurement and this is what takes it.
                result.elapsed = Some(began.elapsed());
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
        prompt_tokens: None,
        threads: None,
        dot: None,
        batch: None,
    }
}

impl Ladder {
    /// Whether any rung that reads the sized corpus actually ran.
    ///
    /// A rung that failed still read it, so this is deliberately not "passed":
    /// a failure is exactly when the corpus length matters most.
    pub fn measured_the_corpus(&self) -> bool {
        self.results
            .iter()
            .any(|r| r.rung.uses_corpus() && matches!(r.status, Status::Pass | Status::Fail))
    }

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
///
/// The table only. The verdict is `summary`, because the two belong on
/// different streams: this is the ladder's answer and goes to stdout, and a
/// status line mixed into it would corrupt a captured comparison.
pub fn render(l: &Ladder) -> String {
    let mut s = String::new();
    // The counts are printed only when a rung that reads the corpus actually
    // read it. The size is what was asked for and always prints; a token count
    // beside four skipped rungs describes work that did not happen, which is
    // the opposite of what this line is for.
    if l.measured_the_corpus() {
        let _ = writeln!(
            s,
            "size={} tokens={} prompts={}",
            l.size.label(),
            l.size.tokens(),
            l.size.prompts()
        );
    } else {
        let _ = writeln!(s, "size={} (no rung read the corpus)", l.size.label());
    }
    if let Some(n) = l.prompt_tokens {
        let _ = writeln!(s, "prompt={n} tokens");
    }
    if let Some(n) = l.threads {
        let _ = writeln!(s, "threads={n}");
    }
    if let Some(d) = &l.dot {
        let _ = writeln!(s, "dot={d}");
    }
    if let Some(b) = &l.batch {
        let _ = writeln!(s, "batch={b}");
    }
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

    // Timing is a separate block rather than a column in the table above, and
    // the separation is the point. Two runs that differ only in `--batch` or
    // `--threads` must agree digit for digit, so the instruction for reading
    // an A/B is "the table is identical". A duration column would make that
    // "the table is identical except one column", and a qualified comparison
    // is one someone eventually reads as a pass.
    //
    // The total is the sum of the rung bodies, not the process. It excludes
    // model load, which for E4B is the larger part of a short run.
    let timed: Vec<_> = l
        .results
        .iter()
        .filter_map(|r| r.elapsed.map(|d| (r.rung, d)))
        .collect();
    if !timed.is_empty() {
        let total: Duration = timed.iter().map(|(_, d)| *d).sum();
        let _ = writeln!(s);
        let _ = writeln!(s, "timing (not compared; expected to differ between runs)");
        for (rung, d) in &timed {
            let _ = writeln!(s, "  {:<4}  {:>8.2}s", rung.label(), d.as_secs_f64());
        }
        let _ = writeln!(
            s,
            "  {:<4}  {:>8.2}s  (rung bodies only, excludes model load)",
            "sum",
            total.as_secs_f64()
        );
    }

    s
}

/// One line of the verdict: a tag, or `None` for an indented continuation of
/// the line above it.
///
/// `None` is not an oversight. The verdict's shape is one tagged statement
/// followed by indented detail -- the same shape `rust-harness/Makefile:70-86`
/// uses -- and tagging a continuation would produce `  [ERROR] Rungs below it
/// were not reached`, which reads as a second, separate failure. The indent
/// travels in the string because that is where it has always been.
pub type SummaryLine = (Option<crate::status::Tag>, String);

/// The ladder's verdict, as tagged lines rather than rendered text.
///
/// Split out of `render` because the two halves belong on different streams:
/// the table is the answer and goes to stdout, the verdict is status and goes
/// to stderr through `crate::status`. Returning tags rather than a string is
/// what keeps the verdict testable -- a test can assert no `Tag::Ok` is
/// present, which grepping rendered text for `"[OK]"` only appeared to do,
/// since a table that stopped carrying tags would satisfy it vacuously.
pub fn summary(l: &Ladder) -> Vec<SummaryLine> {
    use crate::status::Tag;
    match l.first_failure() {
        // A run that measured nothing is not a pass. The status of each rung
        // is honest either way, but the summary line is what a reader takes
        // away, and "every rung within tolerance" over six skips is a green
        // gate that compared nothing.
        None if l.skipped() == Rung::ALL.len() => vec![(
            Some(Tag::Warn),
            "Ladder: no rung had a reference; nothing was measured".to_string(),
        )],
        None => vec![(
            Some(Tag::Ok),
            format!(
                "Ladder: every rung within tolerance ({} skipped)",
                l.skipped()
            ),
        )],
        Some(r) => {
            let mut lines = vec![(
                Some(Tag::Error),
                format!("Ladder: {} failed on {}", r.rung.label(), r.rung.subject()),
            )];
            if !r.note.is_empty() {
                lines.push((None, format!("  {}", r.note)));
            }
            lines.push((
                None,
                "  Rungs below it were not reached: each one reads the output of the \
                 one above, so they would measure a run already known to be wrong."
                    .to_string(),
            ));
            lines
        }
    }
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
    /// L6: how far a captured attention row may sum away from 1.0.
    ///
    /// Not tiered: a post-softmax row sums to one by construction on both
    /// oracles, and the only thing this absorbs is the f32 accumulation of
    /// up to a few thousand terms.
    pub const ATTENTION_MASS: f64 = 1e-5;
    /// L6: how many of the reference's heaviest keys ours must also rank.
    ///
    /// Ten rather than L2's fifty because an attention row is peaked on a
    /// handful of keys and flat elsewhere; fifty would count agreement on
    /// the flat part, which is agreement about nothing.
    pub const ATTENTION_TOP_K: usize = 10;
}

/// The tolerances L1 and L2 are judged against, which differ by oracle.
///
/// One struct rather than two sets of loose constants, so a caller cannot mix
/// the tiny fixture's cosine floor with the real checkpoint's KL ceiling and
/// get a threshold nobody chose.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    /// L1: cosine floor. Direction.
    pub cosine: f64,
    /// L1: relative-L2 ceiling. Magnitude.
    ///
    /// Separate from `cosine` because the two failures are independent: a
    /// residual stream correct in shape and wrong in size passes any cosine
    /// floor perfectly, which is the exact signature of a missing
    /// `layer_scalar` -- a tensor released checkpoints carry at values as low
    /// as 0.049.
    pub rel_l2: f64,
    /// L2: mean KL in nats across the scored positions.
    pub mean_kl: f64,
    /// L2: worst single-position KL in nats.
    ///
    /// The mean alone would let one badly wrong position hide among many
    /// right ones, and one position is all a rare-path bug touches.
    pub worst_kl: f64,
    /// L2: how deep the overlap check looks.
    pub top_k: usize,
    /// L2: worst top-k overlap across the scored positions.
    pub overlap: f64,
    /// L2: fraction of positions whose argmax matches.
    ///
    /// Reported but weak on its own, and never the only L2 measure: a
    /// monotone error such as a missing logit softcap cannot move an argmax
    /// at all, so a rung resting on it would be blind to exactly the defect
    /// class this checkpoint has already produced once.
    pub argmax: f64,
    /// L6: worst single-key difference between an attention row of ours and
    /// the reference's, over every key the row may attend.
    ///
    /// Absolute rather than KL because the rows are probabilities already
    /// and most of a row is near zero: a KL over an attention row is
    /// dominated by the handful of heavy keys and would let a wrong weight
    /// on a light one hide, which is what a leaked key looks like.
    pub attention_max_abs: f64,
    /// L6: worst top-k key overlap across the compared rows.
    pub attention_overlap: f64,
}

/// Tier T: the tiny transformers-authored fixture, f32 against f32 on a few
/// dozen dimensions. Tight on purpose -- there is nothing here for a
/// tolerance to absorb except accumulation order, so if it needs more than
/// this to pass, something is wrong and loosening hides it.
pub const TIER_T: Tolerance = Tolerance {
    cosine: 0.99999,
    rel_l2: 1e-5,
    mean_kl: 1e-6,
    worst_kl: 1e-5,
    top_k: 50,
    overlap: 1.0,
    argmax: 1.0,
    attention_max_abs: 1e-5,
    attention_overlap: 1.0,
};

/// Tier E: the real checkpoint. Looser, because thousands of dimensions and
/// dozens of layers accumulate reordering differences that Tier T's handful
/// cannot. PROVISIONAL apart from the two L1 numbers, which `layer-diff`
/// has already argued and used against real weights.
pub const TIER_E: Tolerance = Tolerance {
    cosine: 0.9999,
    rel_l2: 1e-3,
    mean_kl: 1e-3,
    worst_kl: 5e-3,
    top_k: 50,
    overlap: 0.98,
    argmax: 0.98,
    // PROVISIONAL, both: no attention row of ours has been compared against
    // transformers on real weights yet. bf16 weights widened to f32 on both
    // sides, then a 2560-wide dot and a softmax over up to 2052 keys.
    attention_max_abs: 1e-3,
    attention_overlap: 0.9,
};

/// One tokenizer case the reference encoded.
///
/// Carries the text and not only the ids, because a disagreement is read by
/// eye and "case 3 differs" is not enough to act on. `allowed` is part of the
/// case rather than a property of the run: the same string encodes to
/// different ids depending on which special literals the caller vouched for,
/// and an artifact that did not record the allowlist would be comparing two
/// different questions.
#[derive(Debug, Clone)]
pub struct TokenizerCase {
    pub name: String,
    pub text: String,
    /// The special literals this case permits, for `encode_prompt`'s
    /// allowlist. Empty means `encode`: nothing is resolved, and a `<bos>` in
    /// the text is bytes.
    pub allowed: Vec<String>,
    pub ids: Vec<Token>,
}

/// The head of a token sequence, for a divergence note.
///
/// Truncated because a chat-rendered prompt runs to hundreds of ids and the
/// useful part of a tokenizer disagreement is always near its start -- BPE is
/// left-to-right, so the first differing id is the defect and everything after
/// it is that defect's consequence.
fn head(ids: &[Token]) -> String {
    const SHOWN: usize = 12;
    let shown: Vec<String> = ids.iter().take(SHOWN).map(|t| t.to_string()).collect();
    match ids.len() > SHOWN {
        true => format!("[{}, ... {} total]", shown.join(", "), ids.len()),
        false => format!("[{}]", shown.join(", ")),
    }
}

/// L0: token ids for the rendered prompt.
///
/// Exact equality, with no tolerance and deliberately no `Tolerance` field to
/// hold one. Every rung above this one assumes both sides fed the model the
/// same tokens, so a near miss here is not a small error -- it is the
/// withdrawal of that assumption, and L1's agreement on a different token
/// sequence would be luck rather than evidence.
///
/// Expressed as a `Measure` at threshold 1.0 like every other rung, so
/// `render` needs no special case for it.
pub fn rung_tokenizer(
    tok: &crate::tokenizer::Tokenizer,
    cases: &[TokenizerCase],
) -> Result<RungResult> {
    ensure!(
        !cases.is_empty(),
        "no tokenizer cases to compare -- an empty parity artifact would \
         otherwise report perfect agreement over nothing"
    );

    let mut matched = 0usize;
    // The first disagreement only. After it, the note would list consequences
    // of one wrong merge rather than additional findings.
    let mut first: Option<String> = None;

    for case in cases {
        let allowed: Vec<&str> = case.allowed.iter().map(String::as_str).collect();
        let ours = tok
            .encode_prompt(&case.text, &allowed)
            .with_context(|| format!("encode tokenizer case {}", case.name))?;
        if ours == case.ids {
            matched += 1;
        } else if first.is_none() {
            let at = ours
                .iter()
                .zip(&case.ids)
                .position(|(a, b)| a != b)
                .unwrap_or_else(|| ours.len().min(case.ids.len()));
            first = Some(format!(
                "case {} diverges at index {at}: ours {} against {}",
                case.name,
                head(&ours),
                head(&case.ids)
            ));
        }
    }

    let note = match &first {
        None => format!("{} cases, every id identical", cases.len()),
        Some(what) => what.clone(),
    };
    Ok(RungResult::measured(
        Rung::L0Tokenizer,
        vec![Measure::new(
            "case agreement",
            matched as f64 / cases.len() as f64,
            1.0,
            Cmp::AtLeast,
        )],
        note,
    ))
}

/// What a divergence at a named pre-stack stage points at.
///
/// Here rather than in the binary that prints it because the rung and the
/// tool have to name the same cause: two wordings for one failure is how a
/// reader ends up believing they are two failures.
pub fn stage_cause(stage: &str) -> &'static str {
    match stage {
        "embed" => {
            "the token IDs, the embedding table, or the sqrt(embed_dim) scale -- \
             not any decoder layer"
        }
        "ple_identity" => {
            "the packed per-layer table: its row stride, or the sqrt(ple_dim) embed scale"
        }
        "ple_context" => {
            "the per-layer projection: its weight orientation, the 1/sqrt(embed_dim) \
             scale, or the norm over each row"
        }
        // Both halves clean and the sum wrong leaves only the sum.
        _ => "the composition of the two halves, which is only the 1/sqrt(2) scale",
    }
}

/// Activations at one position: the named pre-stack stages, then one row per
/// decoder layer.
///
/// Both the reference and our own capture use this type, so `rung_layers`
/// compares two values of one shape and needs no model. That is what makes
/// the rung's own reporting testable: a rung that localizes a divergence to
/// the wrong row is a defect, and it is one no real-weight run can tell apart
/// from the divergence itself.
#[derive(Debug, Clone, Default)]
pub struct Activations {
    /// Named stages in computation order. `PRELUDE_STAGES` order matters: the
    /// stages are a chain, so the first divergence is the term rather than
    /// any sum it feeds.
    pub stages: Vec<(String, Vec<f32>)>,
    pub layers: Vec<Vec<f32>>,
}

/// Our side of L1: prefill `tokens` and collect the rows a capture recorded.
///
/// Specific to `CpuModel` because the tap is -- the `Model` trait has no
/// notion of an intermediate row -- and shared by the `ladder` binary and the
/// Tier T test rather than written once in each. Two copies of "drive the tap
/// and collect its rows" is two places for the stage order to drift from
/// `PRELUDE_STAGES`, and the whole rung rests on both sides meaning the same
/// thing by a stage name.
///
/// Only the stages `want` recorded are collected. Asking the tap for a stage
/// the reference never captured would raise an error inside the rung, where it
/// would read as a divergence rather than as a capture that did not run.
pub fn capture_ours(
    model: &mut crate::model::cpu::CpuModel,
    tokens: &[Token],
    want: &Activations,
) -> Result<Activations> {
    use crate::model::cpu::LastRowTap;

    let n_layers = model.config().n_layers;
    let tap = LastRowTap::new((0..n_layers).collect()).capturing_prelude();
    model.set_tap(Box::new(tap.clone()));
    model.truncate(0)?;
    model
        .prefill(tokens)
        .context("prefill the reference tokens")?;

    let mut stages = Vec::new();
    for (name, _) in &want.stages {
        let row = tap
            .stage(name)
            .with_context(|| format!("{name}: the tap captured no row"))?;
        stages.push((name.clone(), row));
    }
    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        layers.push(
            tap.row(i)
                .with_context(|| format!("layer {i}: the tap captured no row"))?,
        );
    }
    Ok(Activations { stages, layers })
}

/// L1: the residual stream and the stages feeding it.
///
/// First divergence, not worst. Once row k is wrong every later row is wrong
/// downstream of it, so a table read by magnitude points at the end of the
/// stack rather than at the cause. Stages are checked before layers and their
/// verdict displaces the layer verdict entirely: with the input to layer 0
/// already wrong, the layer rows measure a stack fed bad data and are not
/// evidence about the layers.
pub fn rung_layers(got: &Activations, want: &Activations, tol: Tolerance) -> Result<RungResult> {
    ensure!(
        got.layers.len() == want.layers.len(),
        "reference captured {} layers, this run captured {} -- the two are not the \
         same model",
        want.layers.len(),
        got.layers.len()
    );
    ensure!(!want.layers.is_empty(), "reference captured no layers");

    let mut worst_cosine = f64::INFINITY;
    let mut worst_rel_l2: f64 = 0.0;
    let mut first_bad: Option<String> = None;
    let mut in_prelude = true;

    // Stages the reference carried, then the layers. A stage absent from our
    // capture is an error rather than a skip: the reference measured it, so
    // not measuring it is missing evidence dressed as agreement.
    let stage_rows = want.stages.iter().map(|(name, row)| {
        let ours = got
            .stages
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| r.as_slice());
        (name.clone(), ours, row.as_slice())
    });
    let layer_rows = (0..want.layers.len()).map(|i| {
        (
            format!("layer {i}"),
            Some(got.layers[i].as_slice()),
            want.layers[i].as_slice(),
        )
    });

    for (label, ours, theirs) in stage_rows.chain(layer_rows) {
        // The stage half of the chain is exhausted once a "layer " label
        // appears, and only a stage divergence gets the displacing verdict.
        if label.starts_with("layer ") {
            if in_prelude && first_bad.is_some() {
                break;
            }
            in_prelude = false;
        }
        let ours = ours.with_context(|| format!("{label}: this run captured no row"))?;
        ensure!(
            ours.len() == theirs.len(),
            "{label}: this run's row is {} wide, the reference's is {} wide",
            ours.len(),
            theirs.len()
        );
        let d = crate::metrics::vec_diff(ours, theirs);
        worst_cosine = worst_cosine.min(d.cosine);
        worst_rel_l2 = worst_rel_l2.max(d.rel_l2);
        if (d.cosine < tol.cosine || d.rel_l2 > tol.rel_l2) && first_bad.is_none() {
            first_bad = Some(label);
        }
    }

    let note = match &first_bad {
        None => format!("{} layers within tolerance", want.layers.len()),
        Some(label) if label.starts_with("layer ") => {
            format!("first divergence at {label}; every later layer is downstream of it")
        }
        Some(stage) => format!(
            "the stack's input diverges at '{stage}' -- look at {}",
            stage_cause(stage)
        ),
    };

    Ok(RungResult::measured(
        Rung::L1Layers,
        vec![
            Measure::new("worst cosine", worst_cosine, tol.cosine, Cmp::AtLeast),
            Measure::new("worst rel_l2", worst_rel_l2, tol.rel_l2, Cmp::AtMost),
        ],
        note,
    ))
}

/// Reference logits at a recorded set of prefill positions.
#[derive(Debug, Clone)]
pub struct LogitReference {
    /// Indices into the prompt whose rows were captured.
    ///
    /// Recorded in the artifact and re-asserted here rather than implied by
    /// row order, because scoring every position of a long prompt is what
    /// makes L2 unaffordable, so the set is capped -- and a capped set that
    /// does not say which positions it holds is a set nobody can reproduce.
    pub positions: Vec<usize>,
    pub rows: Vec<Vec<f32>>,
}

/// L2: the output distribution at the recorded positions.
pub fn rung_logits(
    model: &mut dyn Model,
    prompt: &[Token],
    reference: &LogitReference,
    tol: Tolerance,
) -> Result<RungResult> {
    ensure!(
        reference.positions.len() == reference.rows.len(),
        "reference names {} positions but carries {} rows",
        reference.positions.len(),
        reference.rows.len()
    );
    ensure!(!reference.rows.is_empty(), "reference carries no rows");
    if let Some(&p) = reference.positions.iter().find(|&&p| p >= prompt.len()) {
        bail!(
            "reference scores position {p} of a {}-token prompt",
            prompt.len()
        );
    }

    model.truncate(0)?;
    let ours = model.prefill_all(prompt)?;
    ensure!(
        ours.len() == prompt.len(),
        "prefill_all returned {} rows for {} tokens",
        ours.len(),
        prompt.len()
    );

    let mut kl_sum = 0.0;
    let mut worst = (0usize, 0.0f64);
    let mut worst_overlap = f64::INFINITY;
    let mut argmax_matches = 0usize;
    for (&position, want) in reference.positions.iter().zip(&reference.rows) {
        let d = crate::metrics::logit_diff(want, &ours[position].0, tol.top_k)
            .with_context(|| format!("position {position}"))?;
        kl_sum += d.kl;
        if d.kl > worst.1 {
            worst = (position, d.kl);
        }
        worst_overlap = worst_overlap.min(d.top_k);
        if d.argmax_same {
            argmax_matches += 1;
        }
    }
    let n = reference.rows.len();
    let top_k_name: &'static str = match tol.top_k {
        50 => "worst top-50 overlap",
        _ => "worst top-k overlap",
    };

    Ok(RungResult::measured(
        Rung::L2Logits,
        vec![
            Measure::new("mean kl", kl_sum / n as f64, tol.mean_kl, Cmp::AtMost),
            Measure::new("worst kl", worst.1, tol.worst_kl, Cmp::AtMost),
            Measure::new(top_k_name, worst_overlap, tol.overlap, Cmp::AtLeast),
            Measure::new(
                "argmax agreement",
                argmax_matches as f64 / n as f64,
                tol.argmax,
                Cmp::AtLeast,
            ),
        ],
        format!("{n} positions scored; worst at prompt position {}", worst.0),
    ))
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

/// How many positions a long-context reference scores.
///
/// Fixed rather than derived from the prompt length, so a reference row list
/// and our own schedule can be paired by index and a length disagreement is an
/// artifact error rather than a silent short comparison.
pub const LONG_CONTEXT_POSITIONS: usize = 6;

/// The shortest prompt `window_schedule` can place its anchors in.
///
/// Four windows plus the pair either side of the second crossing. A prompt
/// shorter than this cannot carry two crossings and a distinct interior
/// position, and a schedule that silently collapsed onto duplicate positions
/// would report agreement over fewer places than the header claims.
pub fn min_long_context_tokens(window: usize) -> usize {
    4 * window + 4
}

/// The positions a long-context reference scores, derived from the window.
///
/// **The window is the whole content of this rung**, and it enters here rather
/// than through a threshold: both sides derive their scored positions from the
/// checkpoint's `sliding_window`, so a run whose loader read a different
/// window compares its own rows against the reference's at different places
/// and diverges. Every other rung is blind to the window entirely -- L0 to L2
/// score one short prompt and L3's 256-token perplexity window is shorter than
/// E4B's 512-token attention window, so no rung before this one crossed a real
/// one at Tier E.
///
/// The anchors, in order:
///
///   window - 1   the last position whose whole prefix is still in the window
///   window       the first position whose oldest token has left it
///   window + 1   one past that, so an off-by-one lands between two scored rows
///   2*window + 1 a second crossing, since a convention can be right once by
///                luck and cannot be right twice
///   len / 2      an interior position, well past the window
///   len - 1      the end, whose row depends on the mask at every layer
///
/// Errors rather than truncating when the prompt is too short: a reference
/// that never outruns the window is not a long-context reference, and scoring
/// it would produce a green rung about the regime it exists to measure.
pub fn window_schedule(window: usize, len: usize) -> Result<Vec<usize>> {
    ensure!(window > 0, "a sliding window of 0 has no boundary to score");
    let need = min_long_context_tokens(window);
    ensure!(
        len >= need,
        "a {len}-token prompt cannot cross a {window}-token window twice -- \
         a long-context reference needs at least {need} tokens"
    );
    let positions = vec![
        window - 1,
        window,
        window + 1,
        2 * window + 1,
        len / 2,
        len - 1,
    ];
    // Strictly increasing by construction once `len >= 4*window + 4`. Asserted
    // rather than assumed, because the alternative to noticing here is a rung
    // that scores one position twice and calls it two agreements.
    ensure!(
        positions.windows(2).all(|p| p[0] < p[1]),
        "window schedule {positions:?} is not strictly increasing for window \
         {window} over {len} tokens"
    );
    Ok(positions)
}

/// Reference logits from a prefill that outruns the model's sliding window.
///
/// Carries `sliding_window` because the rung's verdict depends on it: the
/// artifact states the window transformers attended under, and the run states
/// the window our loader read from the same checkpoint. Those are two claims
/// about one config, and until L5 nothing compared them.
#[derive(Debug, Clone)]
pub struct LongContextReference {
    pub sliding_window: usize,
    /// The prompt, as ids. Fed to the engine rather than re-derived from text,
    /// for the reason every artifact here records ids: the tokenizer is itself
    /// under test at L0.
    pub token_ids: Vec<Token>,
    /// The positions the reference scored, as it recorded them.
    pub positions: Vec<usize>,
    pub rows: Vec<Vec<f32>>,
}

/// L5: the output distribution at positions either side of the window.
///
/// `window` is what *this run* believes the checkpoint's sliding window is,
/// read from the loaded config rather than from the artifact. Our own schedule
/// is derived from it and the reference's rows are paired against it by index,
/// so a one-position disagreement about the boundary compares two different
/// positions and shows up as divergence rather than as a green rung over a
/// mislabelled comparison.
///
/// Inside and past the window are reported separately, in L1's spirit of
/// naming the term rather than the sum: divergence on both sides is ordinary
/// numerics, and divergence only past the window is the eviction, the mask, or
/// the shared-KV tail -- the three things that engage nowhere else in the
/// ladder.
pub fn rung_long_context(
    model: &mut dyn Model,
    window: usize,
    reference: &LongContextReference,
    tol: Tolerance,
) -> Result<RungResult> {
    ensure!(
        reference.positions.len() == reference.rows.len(),
        "reference names {} positions but carries {} rows",
        reference.positions.len(),
        reference.rows.len()
    );
    ensure!(
        reference.rows.len() == LONG_CONTEXT_POSITIONS,
        "reference carries {} rows and the schedule scores {LONG_CONTEXT_POSITIONS} \
         positions -- regenerate it with the current generator",
        reference.rows.len()
    );

    let prompt = &reference.token_ids;
    // Our schedule, from our window. An error here is a prompt too short to
    // cross our window, which is a mismatched artifact rather than evidence
    // about the engine.
    let ours_schedule = window_schedule(window, prompt.len())
        .context("this run's window schedule over the reference prompt")?;
    if let Some(&p) = ours_schedule.iter().find(|&&p| p >= prompt.len()) {
        bail!(
            "schedule scores position {p} of a {}-token prompt",
            prompt.len()
        );
    }

    model.truncate(0)?;
    let ours = model.prefill_all(prompt)?;
    ensure!(
        ours.len() == prompt.len(),
        "prefill_all returned {} rows for {} tokens",
        ours.len(),
        prompt.len()
    );

    let mut worst_inside = 0.0f64;
    let mut worst_past = 0.0f64;
    let mut worst = (0usize, 0.0f64);
    let mut worst_overlap = f64::INFINITY;
    let mut argmax_matches = 0usize;
    for (&position, want) in ours_schedule.iter().zip(&reference.rows) {
        let d = crate::metrics::logit_diff(want, &ours[position].0, tol.top_k)
            .with_context(|| format!("position {position}"))?;
        // Split by *our* boundary, which is the one the row was computed
        // under. Splitting by the reference's would label our rows with a
        // regime they did not run in.
        match position < window {
            true => worst_inside = worst_inside.max(d.kl),
            false => worst_past = worst_past.max(d.kl),
        }
        if d.kl > worst.1 {
            worst = (position, d.kl);
        }
        worst_overlap = worst_overlap.min(d.top_k);
        if d.argmax_same {
            argmax_matches += 1;
        }
    }

    let n = reference.rows.len();
    let aligned = ours_schedule
        .iter()
        .zip(&reference.positions)
        .filter(|(a, b)| a == b)
        .count();
    let top_k_name: &'static str = match tol.top_k {
        50 => "worst top-50 overlap",
        _ => "worst top-k overlap",
    };

    let note = match window == reference.sliding_window {
        false => format!(
            "this run reads a {window}-token window and the reference recorded \
             {}: the two scored different positions ({ours_schedule:?} against {:?})",
            reference.sliding_window, reference.positions
        ),
        true => format!(
            "{n} positions across a {window}-token window over {} tokens; worst at \
             position {}",
            prompt.len(),
            worst.0
        ),
    };

    Ok(RungResult::measured(
        Rung::L5LongContext,
        vec![
            // First, because it is the one that explains the rest: rows
            // compared at positions the two sides chose differently are not
            // evidence about arithmetic.
            Measure::new(
                "scheduled positions agree",
                aligned as f64 / n as f64,
                1.0,
                Cmp::AtLeast,
            ),
            Measure::new(
                "worst kl inside window",
                worst_inside,
                tol.worst_kl,
                Cmp::AtMost,
            ),
            Measure::new(
                "worst kl past window",
                worst_past,
                tol.worst_kl,
                Cmp::AtMost,
            ),
            Measure::new(top_k_name, worst_overlap, tol.overlap, Cmp::AtLeast),
            Measure::new(
                "argmax agreement",
                argmax_matches as f64 / n as f64,
                tol.argmax,
                Cmp::AtLeast,
            ),
        ],
        note,
    ))
}

/// One reference attention row: the dense distribution over keys
/// `0..=position` for one query head of one layer, masked keys included as
/// the exact zeros the reference wrote.
#[derive(Debug, Clone)]
pub struct AttentionReferenceRow {
    pub layer: usize,
    pub head: usize,
    pub position: usize,
    pub weights: Vec<f32>,
}

/// Reference attention rows from a prefill that outruns the sliding window,
/// written by `fixtures/reference_attention.py`.
///
/// Carries the layers, the positions and the head count the generator chose,
/// not so the rung can read them, but so it can refuse them: the rung
/// recomputes all three from its own config read, and rows compared at
/// layers or positions the two sides chose differently are not evidence
/// about attention.
#[derive(Debug, Clone)]
pub struct AttentionReference {
    pub sliding_window: usize,
    pub layers: Vec<usize>,
    pub heads: usize,
    pub positions: Vec<usize>,
    pub token_ids: Vec<Token>,
    pub rows: Vec<AttentionReferenceRow>,
}

/// The layers L6 compares, from the schedule the checkpoint declares.
///
/// Mirrors `reference_attention.attention_layers`, and
/// `runtime/tests/ladder_reference_contract.rs` pins the two together. The
/// first two layers, where a wrong input shows before anything can average
/// it away; the sliding layer nearest the middle of the stack, ties to the
/// lower index; and the last global layer, which on the released E-series
/// is inside the shared-KV tail and so reads a donor layer's keys. Derived
/// from `layer_types` rather than written as an index list so the same rule
/// picks the interesting layers on the real checkpoint (0, 1, 21, 41) and on
/// the four-layer fixture (every layer), and so a stride constant never
/// enters -- the mistake `docs/models/gemma-4.md` warns against by name.
pub fn attention_layers(kinds: &[LayerKind]) -> Vec<usize> {
    let n = kinds.len();
    let mut chosen: Vec<usize> = (0..n.min(2)).collect();
    let middle = n / 2;
    let nearest_sliding = (0..n)
        .filter(|&i| matches!(kinds[i], LayerKind::SlidingWindow(_)))
        .min_by_key(|&i| (i.abs_diff(middle), i));
    chosen.extend(nearest_sliding);
    let last_global = (0..n)
        .rev()
        .find(|&i| matches!(kinds[i], LayerKind::Global));
    chosen.extend(last_global);
    chosen.sort_unstable();
    chosen.dedup();
    chosen
}

/// Our side of L6: prefill the reference prompt with an attention tap on the
/// layers and positions the artifact names, over every query head.
///
/// Specific to `CpuModel` for the reason `capture_ours` is: the tap is. Every
/// head rather than the reference's count, so a head-count disagreement
/// reaches the rung as a measure rather than as rows that were never asked
/// for. The tap is removed afterwards so a later prefill does not pay for a
/// capture nobody reads.
pub fn capture_attention(
    model: &mut crate::model::cpu::CpuModel,
    reference: &AttentionReference,
) -> Result<Vec<(AttentionKey, AttentionRow)>> {
    use crate::model::cpu::AttentionTap;

    let heads: Vec<usize> = (0..model.config().n_heads).collect();
    let tap =
        AttentionTap::at_positions(reference.layers.clone(), heads, reference.positions.clone());
    model.set_tap(Box::new(tap.clone()));
    model.truncate(0)?;
    model
        .prefill(&reference.token_ids)
        .context("prefill the reference tokens")?;
    model.take_tap();
    Ok(tap.rows())
}

/// L6: the post-softmax attention rows at positions either side of the
/// window, against the reference's, key for key.
///
/// The reference scores every key and masks the forbidden ones to exact
/// zero; the engine never scores a forbidden key and its row is padded with
/// zeros here. Comparing the dense rows is what makes the zeros a checked
/// fact about the mask on both sides: a key the engine wrongly allows
/// carries weight where the reference has zero, and one it wrongly forbids
/// has zero where the reference has weight. `weight outside window` is the
/// reference's side of that on its own, at a threshold of exactly zero,
/// because a small nonzero there means the reference masked with a finite
/// floor and the comparison is no longer against exact zeros at all.
///
/// `kinds`, `n_heads` and `window` are what *this run* read from the
/// checkpoint, so the layers, positions and head count are recomputed here
/// and the artifact's are refused rather than trusted, in L5's spirit.
pub fn rung_attention(
    captured: &[(AttentionKey, AttentionRow)],
    kinds: &[LayerKind],
    n_heads: usize,
    window: usize,
    reference: &AttentionReference,
    tol: Tolerance,
) -> Result<RungResult> {
    ensure!(
        !reference.rows.is_empty(),
        "reference carries no attention rows"
    );
    let len = reference.token_ids.len();
    let ours_layers = attention_layers(kinds);
    let ours_schedule = window_schedule(window, len)
        .context("this run's window schedule over the reference prompt")?;
    let agreements = [
        reference.layers == ours_layers,
        reference.positions == ours_schedule,
        reference.heads == n_heads,
    ];
    let agreed = agreements.iter().filter(|&&a| a).count() as f64 / agreements.len() as f64;

    let ours: std::collections::BTreeMap<AttentionKey, &AttentionRow> =
        captured.iter().map(|(k, r)| (*k, r)).collect();
    let mut outside = 0.0f64;
    let mut worst_abs = 0.0f64;
    let mut worst_mass = 0.0f64;
    let mut worst_overlap = f64::INFINITY;
    let mut worst_at = (0usize, 0usize, 0usize);
    for row in &reference.rows {
        let (l, h, p) = (row.layer, row.head, row.position);
        ensure!(
            row.weights.len() == p + 1,
            "reference row layer {l} head {h} position {p} holds {} weights",
            row.weights.len()
        );
        let (first, w) = ours.get(&(l, h, p)).with_context(|| {
            format!("layer {l} head {h} position {p}: this run captured no row")
        })?;
        ensure!(
            first + w.len() == p + 1,
            "layer {l} head {h} position {p}: {} weights over a span starting at {first}",
            w.len()
        );
        let mut dense = vec![0.0f32; p + 1];
        dense[*first..].copy_from_slice(w);

        let mut mass = 0.0f64;
        for &x in w.iter() {
            ensure!(
                x.is_finite(),
                "layer {l} head {h} position {p}: non-finite weight {x}"
            );
            mass += f64::from(x);
        }
        worst_mass = worst_mass.max((mass - 1.0).abs());
        for (k, (&a, &b)) in dense.iter().zip(&row.weights).enumerate() {
            ensure!(
                b.is_finite(),
                "layer {l} head {h} position {p}: non-finite reference weight {b} at key {k}"
            );
            if k < *first {
                outside = outside.max(f64::from(b).abs());
            }
            let d = (f64::from(a) - f64::from(b)).abs();
            if d > worst_abs {
                worst_abs = d;
                worst_at = (l, h, p);
            }
        }
        worst_overlap = worst_overlap.min(crate::metrics::topk_overlap(
            &row.weights,
            &dense,
            thresholds::ATTENTION_TOP_K,
        ));
    }

    let note = match agreements.iter().all(|&a| a) {
        false => format!(
            "this run chose layers {ours_layers:?} at {ours_schedule:?} over {n_heads} heads \
             and the reference recorded {:?} at {:?} over {}: the two compared different rows",
            reference.layers, reference.positions, reference.heads
        ),
        true => format!(
            "{} rows across {} layers x {} heads at {} positions; window {window}; worst at \
             layer {} head {} position {}",
            reference.rows.len(),
            reference.layers.len(),
            reference.heads,
            reference.positions.len(),
            worst_at.0,
            worst_at.1,
            worst_at.2
        ),
    };

    Ok(RungResult::measured(
        Rung::L6Attention,
        vec![
            // First, for L5's reason: rows compared at layers or positions the
            // two sides chose differently are not evidence about attention.
            Measure::new("captured layers agree", agreed, 1.0, Cmp::AtLeast),
            Measure::new("weight outside window", outside, 0.0, Cmp::AtMost),
            Measure::new(
                "worst row max abs diff",
                worst_abs,
                tol.attention_max_abs,
                Cmp::AtMost,
            ),
            Measure::new(
                "worst row mass",
                worst_mass,
                thresholds::ATTENTION_MASS,
                Cmp::AtMost,
            ),
            Measure::new(
                "worst top-10 overlap",
                worst_overlap,
                tol.attention_overlap,
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
                Status::NotReached,
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
        // The verdict left `render` for `summary`: the table is payload on
        // stdout, the verdict is status on stderr. Asserted against the tags
        // rather than rendered text, which is what makes the negative
        // assertion below mean anything.
        let v = summary(&l);
        assert_eq!(v[0].0, Some(crate::status::Tag::Error), "{v:?}");
        assert!(v[0].1.starts_with("Ladder: L1 failed on "), "{v:?}");
        assert!(
            v.iter().any(|(_, line)| line.contains("not reached")),
            "{v:?}"
        );
    }

    #[test]
    fn a_rung_that_ran_is_timed_and_one_that_was_never_reached_is_not() {
        // The distinction the Option carries: a rung that did no work has no
        // duration, and must not report 0.00s beside rungs that took minutes.
        // A zero would read as "instant", which is a measurement nobody took.
        let l = climb(Size::Quick, "m", "r", |rung| {
            Some(match rung {
                Rung::L1Layers => fail(rung),
                _ => pass(rung),
            })
        });
        for r in &l.results {
            match r.status {
                Status::NotReached => assert!(r.elapsed.is_none(), "{:?}", r.rung),
                _ => assert!(r.elapsed.is_some(), "{:?}", r.rung),
            }
        }
    }

    #[test]
    fn timings_stay_out_of_the_compared_table() {
        // The A/B this exists for -- stepped against batched -- is read by
        // checking the table is identical. That instruction only survives if
        // the durations, which are expected to differ, sit outside it. Two
        // runs differing only in wall clock must render the same table.
        let build = |d: Duration| {
            let mut l = climb(Size::Quick, "m", "r", |rung| Some(pass(rung)));
            for r in &mut l.results {
                r.elapsed = Some(d);
            }
            render(&l)
        };
        let fast = build(Duration::from_millis(1));
        let slow = build(Duration::from_secs(600));
        let head = |s: &str| s.split("timing (").next().unwrap().to_string();
        assert_eq!(head(&fast), head(&slow), "\n{fast}\n---\n{slow}");
        assert_ne!(fast, slow, "the timing block itself must show the change");
        assert!(fast.contains("excludes model load"), "{fast}");
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
        let v = summary(&l);
        assert_eq!(v[0].0, Some(crate::status::Tag::Ok), "{v:?}");
        assert!(v[0].1.contains("(1 skipped)"), "{v:?}");
    }

    #[test]
    fn a_ladder_that_measured_nothing_does_not_summarize_as_a_pass() {
        // Six skips and an [OK] line is a green gate that compared nothing,
        // and the summary line is what a reader takes away from the table.
        let l = climb(Size::Quick, "m", "r", |_| None);
        let v = summary(&l);
        assert!(
            v.iter()
                .any(|(_, line)| line.contains("nothing was measured")),
            "{v:?}"
        );
        // The assertion this replaces was `!render(&l).contains("[OK]")`,
        // which searched rendered text for a tag. Once the tags left `render`
        // it would have passed for the wrong reason and stopped guarding
        // anything. Against `summary` it states the property directly: no
        // part of a six-skip verdict may be a pass.
        assert!(
            !v.iter()
                .any(|(tag, _)| *tag == Some(crate::status::Tag::Ok)),
            "{v:?}"
        );
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
    fn the_header_omits_the_corpus_counts_when_no_rung_consumed_the_corpus() {
        // The first real E4B run printed "tokens=512 prompts=2" with L3 and L4
        // -- the only rungs those numbers describe -- skipped for want of a
        // reference. Both numbers were of work that did not happen, in the one
        // line whose job is to keep a copied number's context. A size is what
        // was asked for and still prints; a count is what was done.
        let l = climb(Size::Quick, "m", "r", |r| {
            matches!(r, Rung::L1Layers | Rung::L2Logits).then(|| pass(r))
        });
        let out = render(&l);
        assert!(out.contains("size=quick"), "{out}");
        assert!(!out.contains("tokens="), "{out}");
        assert!(!out.contains("prompts="), "{out}");
    }

    #[test]
    fn the_header_carries_the_corpus_counts_when_a_corpus_rung_failed() {
        // A failed rung consumed the corpus, so its counts are exactly what
        // the reader needs. Absent-when-skipped must not become
        // absent-unless-passing.
        let l = climb(Size::Quick, "m", "r", |r| {
            Some(if r == Rung::L3Perplexity {
                fail(r)
            } else {
                pass(r)
            })
        });
        let out = render(&l);
        assert!(out.contains("tokens=512"), "{out}");
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

    /// The committed tiny BPE vocabulary, as a tokenizer to encode against.
    ///
    /// A real `Tokenizer` rather than a stub: `rung_tokenizer`'s job is to
    /// drive `encode_prompt` with a per-case allowlist, and a stub that
    /// returned canned ids would exercise the comparison while skipping the
    /// call the rung exists to make.
    fn tiny_tokenizer() -> crate::tokenizer::Tokenizer {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../fixtures/tiny-bpe.tokenizer.json");
        crate::tokenizer::Tokenizer::from_file(&path).unwrap()
    }

    /// A case whose ids are whatever our tokenizer produces for it, so a test
    /// can then perturb exactly one of them.
    fn parity_case(tok: &crate::tokenizer::Tokenizer, name: &str, text: &str) -> TokenizerCase {
        TokenizerCase {
            name: name.to_string(),
            text: text.to_string(),
            allowed: Vec::new(),
            ids: tok.encode(text).unwrap(),
        }
    }

    #[test]
    fn tokenizer_cases_that_encode_identically_pass() {
        let tok = tiny_tokenizer();
        let cases = vec![
            parity_case(&tok, "plain", "hello world"),
            parity_case(&tok, "punctuation", "a, b; c."),
        ];
        let r = rung_tokenizer(&tok, &cases).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?}");
        assert_eq!(r.measures[0].value, 1.0, "{:?}", r.measures);
    }

    #[test]
    fn one_wrong_id_fails_the_tokenizer_rung_and_the_note_names_its_case() {
        let tok = tiny_tokenizer();
        let mut cases = vec![
            parity_case(&tok, "plain", "hello world"),
            parity_case(&tok, "perturbed", "a, b; c."),
        ];
        // The reference is perturbed rather than the tokenizer, so the only
        // difference between this test and the one above is the single id.
        let victim = &mut cases[1].ids;
        victim[0] = victim[0].wrapping_add(1);
        let r = rung_tokenizer(&tok, &cases).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(
            r.note.contains("perturbed") && r.note.contains("index 0"),
            "note does not localize the divergence: {}",
            r.note
        );
        assert_eq!(r.measures[0].value, 0.5, "{:?}", r.measures);
    }

    #[test]
    fn a_tokenizer_artifact_with_no_cases_is_an_error_rather_than_a_vacuous_pass() {
        let tok = tiny_tokenizer();
        // `matched / cases.len()` over an empty list is 0/0 = NaN, which
        // `Cmp::holds` fails -- so this would report FAIL rather than PASS
        // even without the guard. Still an error: a red rung would send the
        // reader looking for a tokenizer defect that is not there, when what
        // happened is that the generator wrote an empty artifact.
        let e = rung_tokenizer(&tok, &[]).unwrap_err();
        assert!(
            e.to_string().contains("no tokenizer cases"),
            "unexpected error: {e}"
        );
    }

    #[test]
    fn a_case_resolves_only_the_specials_it_allows() {
        let tok = tiny_tokenizer();
        // Taken from the vocabulary rather than written as a literal: the
        // tiny fixture spells its sequence opener `<s>` where a Gemma
        // vocabulary spells it `<bos>`, and a hardcoded literal would make
        // this test silently stop resolving anything on one of them.
        let bos = tok.piece(tok.bos()).unwrap().to_string();
        let text = format!("{bos}hi");
        let bare = tok.encode(&text).unwrap();
        let resolved = tok.encode_prompt(&text, &[&bos]).unwrap();
        assert_ne!(
            bare, resolved,
            "the fixture vocabulary does not distinguish an allowed special, \
             so this test cannot measure that the rung honours the allowlist"
        );

        // The same text under the two allowlists, each judged against the
        // encoding that allowlist produces. Both pass only if the rung passes
        // each case's own `allowed` through rather than a single run-wide one.
        let cases = vec![
            TokenizerCase {
                name: "literal".to_string(),
                text: text.clone(),
                allowed: Vec::new(),
                ids: bare,
            },
            TokenizerCase {
                name: "resolved".to_string(),
                text: text.clone(),
                allowed: vec![bos.clone()],
                ids: resolved,
            },
        ];
        let r = rung_tokenizer(&tok, &cases).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?}");
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

    /// A capture with `stages` named stages and `layers` layers, every row a
    /// distinct direction so a perturbation cannot coincidentally agree.
    fn activations(stages: &[&str], layers: usize) -> Activations {
        let row = |seed: usize| -> Vec<f32> {
            (0..8).map(|j| (seed + 1) as f32 * (j + 1) as f32).collect()
        };
        Activations {
            stages: stages
                .iter()
                .enumerate()
                .map(|(i, n)| ((*n).to_string(), row(i)))
                .collect(),
            layers: (0..layers).map(|i| row(100 + i)).collect(),
        }
    }

    #[test]
    fn an_exact_activation_capture_passes_the_tight_tier_t_tolerance() {
        let want = activations(&["embed", "ple_context", "ple"], 4);
        let r = rung_layers(&want.clone(), &want, TIER_T).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?}");
        assert!(r.note.contains("4 layers within tolerance"), "{}", r.note);
    }

    #[test]
    fn a_perturbed_layer_fails_and_the_note_names_the_first_one_not_the_worst() {
        // Layer 1 is nudged off direction and layer 3 much further. The
        // report has to name 1: once a layer is wrong every later layer is
        // wrong downstream of it, so a table read by magnitude points at the
        // end of the stack instead of the cause.
        let want = activations(&["embed"], 5);
        let mut got = want.clone();
        got.layers[1][0] += 0.5;
        got.layers[3][0] += 40.0;
        let r = rung_layers(&got, &want, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(r.note.contains("first divergence at layer 1"), "{}", r.note);
    }

    #[test]
    fn a_diverging_stage_displaces_the_layer_verdict_and_names_a_cause() {
        // With the input to layer 0 already wrong, the layer rows measure a
        // stack fed bad data. Reporting a layer here would send the reader
        // to a decoder layer that is doing exactly what it was told.
        let want = activations(&["embed", "ple_context", "ple"], 3);
        let mut got = want.clone();
        got.stages[1].1[2] += 3.0;
        got.layers[0][0] += 9.0;
        let r = rung_layers(&got, &want, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(r.note.contains("'ple_context'"), "{}", r.note);
        assert!(r.note.contains("per-layer projection"), "{}", r.note);
        assert!(
            !r.note.contains("layer 0"),
            "a stage divergence must not be reported as a layer: {}",
            r.note
        );
    }

    #[test]
    fn a_uniformly_scaled_layer_fails_on_magnitude_with_its_direction_intact() {
        // The signature of a missing layer_scalar, and the reason cosine is
        // not the only L1 measure: every cosine here is exactly 1.
        let want = activations(&[], 3);
        let mut got = want.clone();
        for v in got.layers[2].iter_mut() {
            *v *= 1.5;
        }
        let r = rung_layers(&got, &want, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        let cosine = r
            .measures
            .iter()
            .find(|m| m.name == "worst cosine")
            .unwrap();
        assert!(cosine.ok(), "direction is intact: {}", cosine.value);
        let rel = r
            .measures
            .iter()
            .find(|m| m.name == "worst rel_l2")
            .unwrap();
        assert!(!rel.ok(), "{}", rel.value);
    }

    #[test]
    fn a_capture_missing_a_stage_the_reference_measured_is_an_error() {
        // Not a skip. The reference measured it, so not measuring it is
        // missing evidence dressed up as agreement.
        let want = activations(&["embed", "ple"], 2);
        let mut got = want.clone();
        got.stages.remove(1);
        let err = rung_layers(&got, &want, TIER_T).expect_err("expected an error");
        assert!(err.to_string().contains("captured no row"), "{err}");
    }

    #[test]
    fn a_reference_with_a_different_layer_count_is_a_different_model() {
        let want = activations(&[], 4);
        let got = activations(&[], 5);
        let err = rung_layers(&got, &want, TIER_T).expect_err("expected an error");
        assert!(err.to_string().contains("not the same model"), "{err}");
    }

    #[test]
    fn logits_reproducing_the_reference_exactly_pass() {
        let script: Vec<Vec<f32>> = (0..8)
            .map(|p| (0..6).map(|j| ((p + j) % 5) as f32).collect())
            .collect();
        let positions = vec![1usize, 3, 5];
        let reference = LogitReference {
            rows: positions.iter().map(|&p| script[p].clone()).collect(),
            positions,
        };
        let mut m = MockModel::new(6, script);
        let r = rung_logits(&mut m, &[0; 8], &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?}");
        assert!(r.note.contains("3 positions scored"), "{}", r.note);
    }

    #[test]
    fn a_single_wrong_position_fails_l2_and_the_note_names_that_position() {
        let script: Vec<Vec<f32>> = (0..8)
            .map(|p| (0..6).map(|j| ((p + j) % 5) as f32).collect())
            .collect();
        let positions = vec![1usize, 3, 5];
        let mut reference = LogitReference {
            rows: positions.iter().map(|&p| script[p].clone()).collect(),
            positions,
        };
        reference.rows[1][4] += 6.0;
        let mut m = MockModel::new(6, script);
        let r = rung_logits(&mut m, &[0; 8], &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(r.note.contains("worst at prompt position 3"), "{}", r.note);
    }

    #[test]
    fn l2_scores_exactly_the_positions_the_artifact_records() {
        // The set is capped -- scoring every position of a long prompt is
        // what makes L2 unaffordable -- so which positions were scored is
        // part of the artifact, and a position outside the prompt is a
        // mismatched artifact rather than a divergence.
        let script: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0; 6]).collect();
        let reference = LogitReference {
            positions: vec![0, 9],
            rows: vec![vec![0.0; 6], vec![0.0; 6]],
        };
        let mut m = MockModel::new(6, script);
        let err = rung_logits(&mut m, &[0; 4], &reference, TIER_T).expect_err("expected an error");
        assert!(err.to_string().contains("position 9"), "{err}");

        let ragged = LogitReference {
            positions: vec![0, 1],
            rows: vec![vec![0.0; 6]],
        };
        let err = rung_logits(&mut m, &[0; 4], &ragged, TIER_T).expect_err("expected an error");
        assert!(err.to_string().contains("carries 1 rows"), "{err}");
    }

    #[test]
    fn an_argmax_preserving_distortion_still_fails_l2() {
        // A monotone error -- a missing logit softcap is the one this
        // checkpoint has already produced -- cannot move an argmax at all.
        // A rung resting on argmax agreement would be blind to it, so this
        // test exists to keep argmax from becoming the measure that matters.
        let script: Vec<Vec<f32>> = (0..4)
            .map(|_| vec![4.0f32, 1.0, 0.5, 0.25, 0.1, 0.0])
            .collect();
        let reference = LogitReference {
            positions: vec![2],
            // Same ordering throughout, so the argmax and the whole top-k
            // ranking agree; only the shape of the distribution differs.
            rows: vec![vec![8.0f32, 2.0, 1.0, 0.5, 0.2, 0.0]],
        };
        let mut m = MockModel::new(6, script);
        let r = rung_logits(&mut m, &[0; 4], &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        let argmax = r
            .measures
            .iter()
            .find(|m| m.name == "argmax agreement")
            .unwrap();
        assert!(argmax.ok(), "the argmax never moved: {}", argmax.value);
        let overlap = r
            .measures
            .iter()
            .find(|m| m.name == "worst top-50 overlap")
            .unwrap();
        assert!(overlap.ok(), "the ranking never moved: {}", overlap.value);
        let kl = r.measures.iter().find(|m| m.name == "mean kl").unwrap();
        assert!(
            !kl.ok(),
            "KL is the measure that has to catch this: {}",
            kl.value
        );
    }

    #[test]
    fn greedy_with_no_references_is_an_error_rather_than_a_vacuous_pass() {
        // Zero of zero positions agreeing is 100% agreement, and an empty
        // reference set would otherwise report a green rung having compared
        // nothing at all.
        let err = rung_greedy(&mut peaked(4, 8, |p| p), &[]).expect_err("expected an error");
        assert!(err.to_string().contains("no greedy references"), "{err}");
    }

    /// A script whose row at every position is a distinct distribution.
    ///
    /// Distinct on purpose: L5 pairs rows by a schedule the window decides, so
    /// a script with interchangeable rows would let a shifted boundary compare
    /// the wrong position and agree anyway -- which is the one failure this
    /// rung exists to be able to report.
    fn positioned(vocab: usize, len: usize) -> Vec<Vec<f32>> {
        (0..len)
            .map(|p| (0..vocab).map(|j| ((p * 3 + j * 7) % 11) as f32).collect())
            .collect()
    }

    /// The reference a run at `window` would have to reproduce exactly.
    ///
    /// `positions` is written out by the caller rather than taken from
    /// `window_schedule`, which is the whole reason these tests can fail. An
    /// artifact is a file of fixed numbers; a helper that recomputed them from
    /// the function under test would shift both sides together and report
    /// agreement under exactly the off-by-one the rung exists to catch.
    fn long_context_reference(
        script: &[Vec<f32>],
        window: usize,
        positions: &[usize],
    ) -> LongContextReference {
        LongContextReference {
            sliding_window: window,
            token_ids: (0..script.len()).map(|t| (t % 5) as Token).collect(),
            rows: positions.iter().map(|&p| script[p].clone()).collect(),
            positions: positions.to_vec(),
        }
    }

    /// The schedule a 3-token window implies over the 40-token script below,
    /// as an artifact would carry it: literal, and not recomputed here.
    const TINY_SCHEDULE: [usize; LONG_CONTEXT_POSITIONS] = [2, 3, 4, 7, 20, 39];

    #[test]
    fn the_window_schedule_crosses_the_boundary_twice_and_ends_at_the_last_position() {
        let s = window_schedule(3, 150).unwrap();
        assert_eq!(s, vec![2, 3, 4, 7, 75, 149], "{s:?}");
        // The property the anchors encode, stated so a future edit to them has
        // to keep it: a position either side of the boundary, a second
        // crossing, and the final row.
        assert!(s[0] < 3 && s[1] >= 3, "{s:?}");
        assert!(s.iter().any(|&p| p > 2 * 3), "{s:?}");
        assert_eq!(*s.last().unwrap(), 149, "{s:?}");
        assert_eq!(s.len(), LONG_CONTEXT_POSITIONS, "{s:?}");
    }

    #[test]
    fn a_prompt_that_never_outruns_the_window_is_an_error_rather_than_a_rung() {
        // A reference that stays inside the window says nothing about the
        // regime L5 exists to measure, and scoring it would report a green
        // rung about attention that never slid.
        let err = window_schedule(512, 600).expect_err("expected an error");
        assert!(err.to_string().contains("cannot cross"), "{err}");
        assert!(err.to_string().contains("2052 tokens"), "{err}");
    }

    #[test]
    fn long_context_rows_reproducing_the_reference_pass_at_every_scheduled_position() {
        let script = positioned(6, 40);
        let reference = long_context_reference(&script, 3, &TINY_SCHEDULE);
        let mut m = MockModel::new(6, script);
        let r = rung_long_context(&mut m, 3, &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?} note={}", r.note);
        assert!(r.note.contains("3-token window"), "{}", r.note);
        assert!(r.note.contains("6 positions"), "{}", r.note);
    }

    #[test]
    fn a_run_whose_window_is_one_position_wider_fails_and_the_note_names_both_windows() {
        // The off-by-one this rung is for. Nothing about the arithmetic
        // changed: the run read a 4-token window from the same config the
        // reference read a 3-token one from, so the two scored different
        // positions and the rows they compared are not the same rows.
        //
        // Every rung below L5 is blind to this. L0 through L2 score one short
        // prompt, and L3's window is shorter than E4B's attention window.
        let script = positioned(6, 40);
        let reference = long_context_reference(&script, 3, &TINY_SCHEDULE);
        let mut m = MockModel::new(6, script);
        let r = rung_long_context(&mut m, 4, &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(
            r.note.contains("4-token window") && r.note.contains("3"),
            "the note must name both windows: {}",
            r.note
        );
        let aligned = r
            .measures
            .iter()
            .find(|m| m.name == "scheduled positions agree")
            .unwrap();
        assert!(!aligned.ok(), "{}", aligned.value);
        // Not only the bookkeeping measure: the rows themselves diverge,
        // because a shifted schedule compares different positions.
        let past = r
            .measures
            .iter()
            .find(|m| m.name == "worst kl past window")
            .unwrap();
        assert!(!past.ok(), "{}", past.value);
    }

    #[test]
    fn a_wrong_row_past_the_window_fails_while_the_rows_inside_it_stay_clean() {
        // The split the rung reports: divergence on both sides is ordinary
        // numerics, and divergence only past the boundary is the eviction, the
        // mask, or the shared-KV tail.
        let script = positioned(6, 40);
        let mut reference = long_context_reference(&script, 3, &TINY_SCHEDULE);
        // Index 4 of the schedule is position 20, well past the window.
        reference.rows[4][2] += 9.0;
        let mut m = MockModel::new(6, script);
        let r = rung_long_context(&mut m, 3, &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(r.note.contains("worst at position 20"), "{}", r.note);
        let inside = r
            .measures
            .iter()
            .find(|m| m.name == "worst kl inside window")
            .unwrap();
        assert!(
            inside.ok(),
            "nothing inside the window moved: {}",
            inside.value
        );
        let past = r
            .measures
            .iter()
            .find(|m| m.name == "worst kl past window")
            .unwrap();
        assert!(!past.ok(), "{}", past.value);
    }

    #[test]
    fn a_long_context_reference_with_the_wrong_row_count_is_an_error() {
        // A mismatched artifact is not evidence about the engine: the rows are
        // paired with the schedule by index, so a short list would compare a
        // prefix and report agreement over the positions it dropped.
        let script = positioned(6, 40);
        let mut reference = long_context_reference(&script, 3, &TINY_SCHEDULE);
        reference.rows.pop();
        reference.positions.pop();
        let mut m = MockModel::new(6, script);
        let err = rung_long_context(&mut m, 3, &reference, TIER_T).expect_err("expected an error");
        assert!(err.to_string().contains("schedule scores"), "{err}");
    }

    /// A four-layer schedule with a 3-token window: the tiny fixture's shape.
    fn tiny_kinds() -> Vec<LayerKind> {
        vec![
            LayerKind::SlidingWindow(3),
            LayerKind::Global,
            LayerKind::SlidingWindow(3),
            LayerKind::Global,
        ]
    }

    /// Attention rows over a 40-token prompt at the literal `TINY_SCHEDULE`,
    /// as the engine would capture them and as an artifact would carry them.
    ///
    /// Each row is a distinct peaked distribution over its visible span, so a
    /// row compared at the wrong layer, head or position differs rather than
    /// agrees by coincidence. The reference is built dense from the same
    /// numbers, zero before the span, which is what the generator writes.
    fn attention_pair(
        kinds: &[LayerKind],
        n_heads: usize,
    ) -> (Vec<(AttentionKey, AttentionRow)>, AttentionReference) {
        const LEN: usize = 40;
        let layers = attention_layers(kinds);
        let mut captured = Vec::new();
        let mut rows = Vec::new();
        for &l in &layers {
            for h in 0..n_heads {
                for &p in &TINY_SCHEDULE {
                    let first = match kinds[l] {
                        LayerKind::Global => 0,
                        LayerKind::SlidingWindow(w) => (p + 1).saturating_sub(w),
                    };
                    let raw: Vec<f32> = (first..=p)
                        .map(|t| ((t * 7 + p * 3 + l * 5 + h) % 11 + 1) as f32)
                        .collect();
                    let total: f32 = raw.iter().sum();
                    let w: Vec<f32> = raw.iter().map(|x| x / total).collect();
                    let mut dense = vec![0.0f32; p + 1];
                    dense[first..].copy_from_slice(&w);
                    captured.push(((l, h, p), (first, w)));
                    rows.push(AttentionReferenceRow {
                        layer: l,
                        head: h,
                        position: p,
                        weights: dense,
                    });
                }
            }
        }
        let reference = AttentionReference {
            sliding_window: 3,
            layers,
            heads: n_heads,
            positions: TINY_SCHEDULE.to_vec(),
            token_ids: (0..LEN).map(|t| (t % 5) as Token).collect(),
            rows,
        };
        (captured, reference)
    }

    fn measure<'a>(r: &'a RungResult, name: &str) -> &'a Measure {
        r.measures
            .iter()
            .find(|m| m.name == name)
            .unwrap_or_else(|| panic!("no measure {name} in {:?}", r.measures))
    }

    #[test]
    fn attention_layers_picks_the_first_two_the_middle_sliding_and_the_last_global() {
        // E4B's schedule: a global layer every sixth, 42 layers.
        let e4b: Vec<LayerKind> = (0..42)
            .map(|i| match (i + 1) % 6 {
                0 => LayerKind::Global,
                _ => LayerKind::SlidingWindow(512),
            })
            .collect();
        assert_eq!(attention_layers(&e4b), vec![0, 1, 21, 41]);
        // The fixture: every layer, each once.
        assert_eq!(attention_layers(&tiny_kinds()), vec![0, 1, 2, 3]);
        // No global layer: nothing to add for it.
        let sliding_only = vec![LayerKind::SlidingWindow(3); 5];
        assert_eq!(attention_layers(&sliding_only), vec![0, 1, 2]);
    }

    #[test]
    fn attention_rows_reproducing_the_reference_pass_at_every_row() {
        let kinds = tiny_kinds();
        let (captured, reference) = attention_pair(&kinds, 2);
        let r = rung_attention(&captured, &kinds, 2, 3, &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Pass, "{r:?} note={}", r.note);
        assert!(r.note.contains("4 layers x 2 heads"), "{}", r.note);
        assert!(r.note.contains("window 3"), "{}", r.note);
        assert_eq!(measure(&r, "weight outside window").value, 0.0);
    }

    #[test]
    fn one_weight_moved_inside_the_window_fails_and_the_note_names_the_row() {
        let kinds = tiny_kinds();
        let (captured, mut reference) = attention_pair(&kinds, 2);
        // Layer 2 head 1 at position 20: a sliding row, moved on a visible key.
        let row = reference
            .rows
            .iter_mut()
            .find(|r| r.layer == 2 && r.head == 1 && r.position == 20)
            .unwrap();
        row.weights[19] += 0.01;
        let r = rung_attention(&captured, &kinds, 2, 3, &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(!measure(&r, "worst row max abs diff").ok(), "{r:?}");
        assert!(r.note.contains("layer 2 head 1 position 20"), "{}", r.note);
        // Only the row comparison moved; the mask facts stand.
        assert!(measure(&r, "weight outside window").ok(), "{r:?}");
        assert!(measure(&r, "captured layers agree").ok(), "{r:?}");
    }

    #[test]
    fn a_reference_weight_before_the_window_fails_at_exactly_zero() {
        // The mask fact this rung is for: the reference put weight on a key
        // the window excludes. Tiny rather than large, because the threshold
        // is exact and a finite masking floor is the failure it detects.
        let kinds = tiny_kinds();
        let (captured, mut reference) = attention_pair(&kinds, 2);
        let row = reference
            .rows
            .iter_mut()
            .find(|r| r.layer == 0 && r.head == 0 && r.position == 20)
            .unwrap();
        row.weights[0] = 1e-7;
        let r = rung_attention(&captured, &kinds, 2, 3, &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        let outside = measure(&r, "weight outside window");
        assert!(!outside.ok(), "{}", outside.value);
        assert_eq!(outside.threshold, 0.0);
    }

    #[test]
    fn a_reference_whose_layer_choice_disagrees_fails_the_agreement_measure() {
        let kinds = tiny_kinds();
        let (captured, mut reference) = attention_pair(&kinds, 2);
        reference.layers = vec![0, 1, 3];
        let r = rung_attention(&captured, &kinds, 2, 3, &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        let agree = measure(&r, "captured layers agree");
        assert!(!agree.ok() && agree.value < 1.0, "{}", agree.value);
        assert!(
            r.note.contains("[0, 1, 2, 3]") && r.note.contains("[0, 1, 3]"),
            "the note must name both choices: {}",
            r.note
        );
    }

    #[test]
    fn a_run_reading_a_wider_window_fails_the_agreement_measure() {
        // The reference scheduled its positions from a 3-token window; a run
        // that read 4 schedules different ones, and the rows it captured are
        // at positions the reference never scored.
        let kinds = tiny_kinds();
        let (captured, reference) = attention_pair(&kinds, 2);
        let r = rung_attention(&captured, &kinds, 2, 4, &reference, TIER_T).unwrap();
        assert_eq!(r.status, Status::Fail, "{r:?}");
        assert!(!measure(&r, "captured layers agree").ok(), "{r:?}");
    }

    #[test]
    fn a_reference_row_this_run_never_captured_is_an_error_not_a_failed_rung() {
        let kinds = tiny_kinds();
        let (mut captured, reference) = attention_pair(&kinds, 2);
        captured.retain(|((l, h, p), _)| !(*l == 3 && *h == 1 && *p == 39));
        let err = rung_attention(&captured, &kinds, 2, 3, &reference, TIER_T)
            .expect_err("expected an error");
        assert!(
            err.to_string().contains("layer 3 head 1 position 39"),
            "{err}"
        );
    }
}
