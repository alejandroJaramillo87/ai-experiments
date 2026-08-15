/*
 * ladder_flags — the four tuning flags, and the claims their help text makes.
 *
 * `--dot` says "every kind is bit-identical" and `--threads` says "the tables
 * are bit-identical at every width". Both are claims about results rather than
 * about speed, and until now nothing asserted either: the flags were parsed,
 * passed to the model, and never compared against anything. A switch whose
 * documented property is untested is a switch that can start changing the
 * numbers without the gate saying so, which is the failure mode the whole
 * workspace is arranged against.
 *
 * Two levels, because they answer different questions:
 *
 * - The ladder's table, compared byte for byte across every value of a flag.
 *   That is the claim as the help text words it, and it is what a reader
 *   sweeping widths for speed is relying on.
 * - The final logits, compared bit for bit through the same checkpoint in
 *   process. The table rounds to six decimals and would keep agreeing through
 *   a divergence far larger than the one `Lanes16` is documented to introduce,
 *   so the stronger comparison is the one that makes the weaker one worth
 *   printing.
 *
 * `Lanes16` is excluded by `DotKind::is_bit_identical` rather than by name.
 * It splits a reduction into interleaved partial sums and is documented not to
 * reproduce the goldens' bits; a seventh kind added tomorrow inherits whichever
 * side of that predicate it declares.
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use runtime::model::cpu::{CpuModel, DotKind};
use runtime::model::Model;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/tiny-hf-gemma4")
}

/// True when the Tier T fixture is present, with a `[SKIP]` already logged
/// when it is not.
///
/// Absent means the generator has not been run on a workstation, which is a
/// real state for a fresh clone: there is no torch in a sandbox to write a
/// transformers reference with.
fn have_fixture(what: &str) -> bool {
    if fixture_dir().join("capture.json").exists() {
        return true;
    }
    runtime::testlog::skip(
        what,
        "fixtures/tiny-hf-gemma4 has not been generated; \
         run fixtures/build_tiny_reference.py on the workstation",
    );
    false
}

/// What one ladder run printed, and the code it exited with.
struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    /// The measurement table alone: the column header down to the timings.
    ///
    /// The lines above it name the model path, the fold and the width, which
    /// are exactly the fields a sweep is meant to change. Comparing them would
    /// assert that two runs were the same run, which is the opposite of the
    /// claim under test. The block below it is durations, which the renderer
    /// itself labels as expected to differ; that label is the boundary used
    /// here, so a table that grew a section would be compared rather than
    /// silently dropped.
    fn table(&self) -> &str {
        let at = self
            .stdout
            .find("rung  ")
            .unwrap_or_else(|| panic!("no table in:\n{}", self.stdout));
        let end = self
            .stdout
            .find("\ntiming (")
            .unwrap_or_else(|| panic!("no timing block in:\n{}", self.stdout));
        &self.stdout[at..end]
    }

    /// The verdict on one rung's own row, as the table's last column.
    fn verdict(&self, rung: &str) -> String {
        self.table()
            .lines()
            .find(|l| l.starts_with(&format!("{rung}  ")))
            .unwrap_or_else(|| panic!("no {rung} row in:\n{}", self.stdout))
            .split_whitespace()
            .last()
            .expect("a rung row ends in its verdict")
            .to_string()
    }
}

/// One ladder climb against the committed fixture, with `extra` appended.
///
/// The reference directory is assembled here for the reason `ladder_obs.rs`
/// gives: the capture is the L1 and L2 payload under another name, and copying
/// it is what makes two rungs genuinely run. A climb over six skipped rungs
/// would compare empty tables and call them identical.
fn ladder(extra: &[&str]) -> Run {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let reference = dir.path().join("reference");
    std::fs::create_dir_all(&reference).unwrap();
    std::fs::copy(
        fixture_dir().join("capture.json"),
        reference.join("layers.json"),
    )
    .unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ladder"))
        .args([
            "--model",
            fixture_dir().to_str().unwrap(),
            "--reference",
            reference.to_str().unwrap(),
        ])
        .args(extra)
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", dir.path().join("logs"))
        .output()
        .expect("the ladder builds as a test binary");
    Run {
        code: out.status.code().expect("the ladder exited normally"),
        stdout: String::from_utf8(out.stdout).expect("stdout is UTF-8"),
        stderr: String::from_utf8(out.stderr).expect("stderr is UTF-8"),
    }
}

/// The fixture's checkpoint and the tokens its capture recorded.
fn model_and_tokens() -> (CpuModel, Vec<runtime::model::Token>) {
    let capture = runtime::reference::load_capture(
        &fixture_dir().join("capture.json"),
        &runtime::model::cpu::PRELUDE_STAGES,
    )
    .expect("read the reference capture")
    .expect("capture.json exists, so it must load");
    let (model, _fingerprint) =
        CpuModel::load(&fixture_dir(), false).expect("load the fixture checkpoint");
    (model, capture.token_ids)
}

/// `--dot` changes the kernel and not the table.
#[test]
fn every_bit_identical_dot_kind_prints_the_same_ladder_table() {
    if !have_fixture("ladder --dot sweep") {
        return;
    }
    let reference = ladder(&["--dot", "sequential"]);
    assert_eq!(
        reference.code, 0,
        "stdout:\n{}\nstderr:\n{}",
        reference.stdout, reference.stderr
    );
    let mut swept = 0;
    for kind in DotKind::ALL.into_iter().filter(|k| k.is_bit_identical()) {
        let r = ladder(&["--dot", kind.label()]);
        assert_eq!(r.code, 0, "{}: {}", kind.label(), r.stderr);
        // The flag reached the model rather than being parsed and dropped: the
        // header prints `model.dot()`, not the argument.
        assert!(
            r.stdout.contains(&format!("dot={}\n", kind.label())),
            "{}: the run did not report the fold it was asked for:\n{}",
            kind.label(),
            r.stdout
        );
        assert_eq!(
            r.table(),
            reference.table(),
            "dot {} changed the table",
            kind.label()
        );
        swept += 1;
    }
    // A predicate that started answering false everywhere would leave this
    // test green over an empty sweep.
    assert!(
        swept >= 5,
        "only {swept} bit-identical dot kinds were swept"
    );
}

/// `--threads` changes the row split and not the table.
#[test]
fn every_thread_width_prints_the_same_ladder_table() {
    if !have_fixture("ladder --threads sweep") {
        return;
    }
    let reference = ladder(&["--threads", "1"]);
    assert_eq!(
        reference.code, 0,
        "stdout:\n{}\nstderr:\n{}",
        reference.stdout, reference.stderr
    );
    // 3 and 5 divide neither the layer count nor the hidden size, so a split
    // that mishandles a short trailing chunk cannot hide behind an even width.
    for width in ["2", "3", "5", "8"] {
        let r = ladder(&["--threads", width]);
        assert_eq!(r.code, 0, "threads {width}: {}", r.stderr);
        assert!(
            r.stdout.contains(&format!("threads={width}\n")),
            "threads {width}: the run did not report the width it was asked \
             for:\n{}",
            r.stdout
        );
        assert_eq!(
            r.table(),
            reference.table(),
            "threads {width} changed the table"
        );
    }
    // `--single-thread` is documented as shorthand for `--threads 1`, and a
    // shorthand that resolved to something else would be a second width nobody
    // asked for.
    let short = ladder(&["--single-thread"]);
    assert!(short.stdout.contains("threads=1\n"), "{}", short.stdout);
    assert_eq!(short.table(), reference.table());
}

/// `--batch` changes when a dot product runs and not what it sums.
#[test]
fn every_batch_mode_prints_the_same_ladder_table() {
    if !have_fixture("ladder --batch sweep") {
        return;
    }
    let reference = ladder(&["--batch", "stepped"]);
    assert_eq!(
        reference.code, 0,
        "stdout:\n{}\nstderr:\n{}",
        reference.stdout, reference.stderr
    );
    // 7 divides neither the 150-token prompt nor any window, so the last group
    // of a batched prefill is short.
    for batch in ["whole", "1", "7", "64"] {
        let r = ladder(&["--batch", batch]);
        assert_eq!(r.code, 0, "batch {batch}: {}", r.stderr);
        assert!(
            r.stdout.contains("\nbatch="),
            "batch {batch}: the run reported no batch mode:\n{}",
            r.stdout
        );
        assert_eq!(
            r.table(),
            reference.table(),
            "batch {batch} changed the table"
        );
    }
    // Spelled how it was asked for, which `1` is not: `BatchKind::parse` reads
    // it as `stepped`, and the header has to say which mode actually ran.
    assert!(
        ladder(&["--batch", "whole"])
            .stdout
            .contains("batch=whole\n"),
        "the header did not name the batch mode the run used"
    );
}

/// `--rung` climbs the one rung named and leaves the others unclimbed.
///
/// Both directions, because a flag that isolated nothing would pass a test
/// that only checked the named rung: L1 and L2 both run from this reference,
/// so each spelling has a rung to leave behind.
#[test]
fn rung_climbs_only_the_rung_it_names() {
    if !have_fixture("ladder --rung isolation") {
        return;
    }
    let both = ladder(&[]);
    assert_eq!(
        both.code, 0,
        "stdout:\n{}\nstderr:\n{}",
        both.stdout, both.stderr
    );
    assert_eq!(both.verdict("L1"), "PASS", "{}", both.stdout);
    assert_eq!(both.verdict("L2"), "PASS", "{}", both.stdout);

    for (asked, left) in [("L1", "L2"), ("L2", "L1")] {
        let r = ladder(&["--rung", asked]);
        assert_eq!(r.code, 0, "--rung {asked}: {}", r.stderr);
        assert_eq!(r.verdict(asked), "PASS", "--rung {asked}:\n{}", r.stdout);
        assert_ne!(
            r.verdict(left),
            "PASS",
            "--rung {asked} climbed {left} as well:\n{}",
            r.stdout
        );
        // Every other rung of the six is unclimbed too, not merely the one
        // that had an artifact waiting.
        for other in ["L0", "L3", "L4", "L5"] {
            assert_ne!(
                r.verdict(other),
                "PASS",
                "--rung {asked} climbed {other}:\n{}",
                r.stdout
            );
        }
    }
}

/// The claim under the table: every bit-identical fold produces the same
/// float32 bits from the same checkpoint and the same tokens.
///
/// Bits rather than a tolerance, and `to_bits` rather than `==`, so a pair of
/// NaNs cannot agree by the float comparison's own rules. This is the
/// assertion the ladder's help text has been making on the workspace's behalf.
#[test]
fn every_bit_identical_dot_kind_produces_the_same_logits_bits() {
    if !have_fixture("dot-kind logits bits") {
        return;
    }
    let (mut model, tokens) = model_and_tokens();
    model.set_dot(DotKind::Sequential);
    let reference: Vec<u32> = model
        .prefill(&tokens)
        .expect("prefill under the reference fold")
        .0
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert!(!reference.is_empty(), "the reference row is empty");

    for kind in DotKind::ALL.into_iter().filter(|k| k.is_bit_identical()) {
        let (mut m, _) = CpuModel::load(&fixture_dir(), false).expect("load the fixture");
        m.set_dot(kind);
        let got: Vec<u32> = m
            .prefill(&tokens)
            .unwrap_or_else(|e| panic!("{}: {e:#}", kind.label()))
            .0
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let differing = got
            .iter()
            .zip(&reference)
            .enumerate()
            .find(|(_, (a, b))| a != b);
        assert!(
            differing.is_none(),
            "dot {} is not bit-identical to sequential: {:?}",
            kind.label(),
            differing
        );
    }
}

/// The same claim for the row split: the width is a speed, not a number.
#[test]
fn every_thread_width_produces_the_same_logits_bits() {
    if !have_fixture("thread-width logits bits") {
        return;
    }
    let (mut model, tokens) = model_and_tokens();
    model.set_threads(1);
    let reference: Vec<u32> = model
        .prefill(&tokens)
        .expect("prefill on one thread")
        .0
        .iter()
        .map(|v| v.to_bits())
        .collect();

    for width in [2usize, 3, 5, 8, 64] {
        let (mut m, _) = CpuModel::load(&fixture_dir(), false).expect("load the fixture");
        m.set_threads(width);
        let got: Vec<u32> = m
            .prefill(&tokens)
            .unwrap_or_else(|e| panic!("threads {width}: {e:#}"))
            .0
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let differing = got
            .iter()
            .zip(&reference)
            .enumerate()
            .find(|(_, (a, b))| a != b);
        assert!(
            differing.is_none(),
            "threads {width} is not bit-identical to one thread: {differing:?}"
        );
    }
}
