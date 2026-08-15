/*
 * ladder_obs — the ladder's run records, measured against a real invocation.
 *
 * `ladder_tiny.rs` next door proves the rungs' numbers. This file proves the
 * run said what it did: that a climb leaves a header naming the fold and the
 * width it actually used, a phase per rung, and a metrics file the Python
 * dashboard can open. None of that can be asserted from the library, because
 * every one of those records is written by the binary.
 *
 * It is the test that would have failed throughout the months `DotKind`
 * defaulted to `Sequential` -- the same gap `run_header.rs` closed for the
 * harness, now closed for the tool whose whole purpose is deciding whether
 * the engine is correct. A conformance verdict with no record of the
 * configuration that produced it is a number without an experiment.
 */

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/tiny-hf-gemma4")
}

/// The declared field set for one event: `fields` plus `optional`.
///
/// `None` when the contract file is absent, which is the container build,
/// whose context is `rust-harness/` alone. Mirrors `obs_contract.rs`'s
/// helper deliberately -- the two assert against the same file from two
/// crates, and a second spelling of the degradation is a second thing to get
/// wrong.
fn contract_event_keys(event: &str) -> Option<BTreeSet<String>> {
    let text = std::fs::read_to_string(repo_root().join("schemas/obs_record.json")).ok()?;
    let contract: serde_json::Value = serde_json::from_str(&text).expect("contract is valid JSON");
    let read = |k: &str| -> Vec<String> {
        contract["events"][event][k]
            .as_array()
            .unwrap_or_else(|| panic!("events.{event}.{k} is an array"))
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    };
    Some(read("fields").into_iter().chain(read("optional")).collect())
}

/// What one ladder run left behind: its events, its metrics, its catalog row.
struct Recorded {
    exit_code: i32,
    events: Vec<serde_json::Value>,
    metrics: Vec<serde_json::Value>,
    catalog: Vec<serde_json::Value>,
}

impl Recorded {
    fn events_named(&self, event: &str) -> Vec<&serde_json::Value> {
        self.events.iter().filter(|e| e["event"] == event).collect()
    }

    fn one(&self, event: &str) -> &serde_json::Value {
        let found = self.events_named(event);
        assert_eq!(
            found.len(),
            1,
            "expected exactly one {event} among {:#?}",
            self.events
        );
        found[0]
    }

    /// The phase names, in the order the run opened them.
    fn phases(&self) -> Vec<String> {
        self.events_named("phase.start")
            .iter()
            .map(|e| e["phase"].as_str().expect("phase is a string").to_string())
            .collect()
    }
}

fn parse_lines(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
        .collect()
}

/// One ladder run against the committed Tier T fixture, or `None` with a
/// `[SKIP]` already logged.
///
/// The reference directory is built here rather than committed: the L1/L2
/// loader reads `layers.json`, and the fixture's capture is the same payload
/// under its own name. Copying it means two rungs genuinely run -- a header
/// and five phases over five skipped rungs would exercise the plumbing and
/// prove nothing about a rung that did work.
fn run_ladder() -> Option<Recorded> {
    run_ladder_with(&[])
}

/// `run_ladder` with extra flags, for the cases that need one.
fn run_ladder_with(extra: &[&str]) -> Option<Recorded> {
    let fixture = fixture_dir();
    if !fixture.join("capture.json").exists() {
        runtime::testlog::skip(
            "ladder obs records",
            "fixtures/tiny-hf-gemma4 has not been generated; \
             run fixtures/build_tiny_reference.py on the workstation",
        );
        return None;
    }
    let dir = tempfile::tempdir().unwrap();
    let reference = dir.path().join("reference");
    std::fs::create_dir_all(&reference).unwrap();
    std::fs::copy(fixture.join("capture.json"), reference.join("layers.json")).unwrap();
    let logs = dir.path().join("logs");

    let out = Command::new(env!("CARGO_BIN_EXE_ladder"))
        .args([
            "--model",
            fixture.to_str().unwrap(),
            "--reference",
            reference.to_str().unwrap(),
        ])
        .args(extra)
        // Nothing is resolved against the working directory, but a test that
        // does not say so is one stray relative path away from writing into
        // the worktree.
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", &logs)
        .output()
        .expect("the ladder builds as a test binary");

    let harness_dir = logs.join("harness");
    let run_dir: PathBuf = std::fs::read_dir(&harness_dir)
        .unwrap_or_else(|e| panic!("{}: {e}", harness_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("the run wrote a run directory");
    let events: PathBuf = std::fs::read_dir(&harness_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        // The catalog is excluded by name and the run directory by `is_file`,
        // leaving the single event file this run wrote.
        .find(|p| p.is_file() && p.file_name().is_some_and(|n| n != "runs.jsonl"))
        .expect("the run wrote an event file");

    Some(Recorded {
        exit_code: out.status.code().expect("the ladder exited normally"),
        events: parse_lines(&events),
        metrics: parse_lines(&run_dir.join("metrics.jsonl")),
        catalog: parse_lines(&harness_dir.join("runs.jsonl")),
    })
}

#[test]
fn the_ladder_records_the_run_it_actually_climbed() {
    let Some(r) = run_ladder() else { return };
    // Tier E tolerances over a fixture that clears the tighter Tier T ones in
    // ladder_tiny.rs, so a red rung here is a real regression rather than a
    // flake worth loosening.
    assert_eq!(r.exit_code, 0, "events: {:#?}", r.events);

    let header = r.one("ladder.header");
    // The two fields the stage exists for, read from the loaded model's own
    // getters rather than from a flag -- evidence about the run, not about
    // the request.
    assert!(header["dot"].is_string() && header["dot"] != "", "{header}");
    assert!(
        header["batch"].is_string() && header["batch"] != "",
        "{header}"
    );
    // Numbers as numbers: this is the record someone will jq over.
    assert!(header["n_layers"].as_u64().is_some(), "{header}");
    assert!(header["weight_bytes"].as_u64().is_some(), "{header}");
    assert!(header["threads"].as_u64().is_some(), "{header}");
    // The load fingerprint the binary used to discard.
    assert_eq!(
        header["model_hash"].as_str().map(str::len),
        Some(16),
        "{header}"
    );

    // The ladder is not the harness, and must not claim to be.
    assert!(
        r.events_named("run.header").is_empty(),
        "the ladder wrote run.header, whose field set is asserted against a \
         harness run: {:#?}",
        r.events
    );

    let end = r.one("cmd.end");
    assert_eq!(end["exit_code"].as_i64(), Some(0), "{end}");
    assert_eq!(r.catalog.len(), 1, "{:#?}", r.catalog);
    assert_eq!(r.catalog[0]["command"], "ladder");
}

#[test]
fn the_ladder_header_matches_its_declared_fields() {
    // `--rung` so the run produces the one optional field too. Key-set
    // equality against a run that omits an optional field would fail for the
    // wrong reason, and weakening the assertion to containment to accommodate
    // that is how a declared-but-never-written field survives.
    let Some(r) = run_ladder_with(&["--rung", "L1"]) else {
        return;
    };
    let Some(declared) = contract_event_keys("ladder.header") else {
        runtime::testlog::skip(
            "ladder.header contract",
            "schemas/obs_record.json is outside this build context",
        );
        return;
    };
    let header = r.one("ladder.header");
    let base: BTreeSet<String> = [
        "ts", "level", "run_id", "cli", "command", "pid", "event", "msg",
    ]
    .into_iter()
    .map(String::from)
    .collect();
    let written: BTreeSet<String> = header
        .as_object()
        .expect("the event is an object")
        .keys()
        .filter(|k| !base.contains(*k))
        .cloned()
        .collect();
    // Both directions. Containment alone would let a field be declared and
    // never written, which for a header assembled in one place is the likelier
    // drift -- and it is the direction that makes the declaration a claim
    // about the run rather than a wish list.
    assert_eq!(
        written,
        declared,
        "written but undeclared: {:?}; declared but unwritten: {:?}",
        written.difference(&declared).collect::<Vec<_>>(),
        declared.difference(&written).collect::<Vec<_>>()
    );
}

#[test]
fn every_attempted_rung_leaves_a_phase_and_a_metrics_record() {
    let Some(r) = run_ladder() else { return };
    // model.load first, then one per rung in climb order. A run with no
    // `--rung` and no failure attempts all five, so this is the full list and
    // its order is the climb's.
    assert_eq!(
        r.phases(),
        vec![
            "model.load",
            "rung.L0",
            "rung.L1",
            "rung.L2",
            "rung.L3",
            "rung.L4"
        ],
        "events: {:#?}",
        r.events
    );
    // Two-sided: every span that opened also closed, with a duration.
    for name in r.phases() {
        let ended: Vec<_> = r
            .events_named("phase.end")
            .into_iter()
            .filter(|e| e["phase"] == name.as_str())
            .collect();
        assert_eq!(ended.len(), 1, "{name} opened without closing once");
        assert!(
            ended[0]["duration_s"].as_f64().is_some(),
            "{name}: {}",
            ended[0]
        );
    }

    // The header record, then one per rung.
    let (header, steps) = r.metrics.split_first().expect("metrics.jsonl is not empty");
    assert_eq!(header["report"], "run_metrics", "{header}");
    assert_eq!(steps.len(), 5, "{steps:#?}");
    for (i, rec) in steps.iter().enumerate() {
        // Numeric, or common/obs/viewer.js:109 drops the record silently --
        // present in the file and absent from every chart.
        assert_eq!(
            rec["step"].as_u64(),
            Some(i as u64),
            "step is the rung index, as a number: {rec}"
        );
        assert!(rec["rung"].is_string(), "{rec}");
        assert!(rec["status"].is_string(), "{rec}");
        assert!(
            rec["rung_duration_s"].as_f64().is_some(),
            "every attempted rung is stamped by climb: {rec}"
        );
    }
    // L1 and L2 have their reference here, so they carry measures. Without
    // this the whole file passes over five skipped rungs, which would prove
    // the plumbing and nothing about a rung that did work.
    assert!(
        steps[1]["worst_cosine"].as_f64().is_some(),
        "L1 ran, so its measures must reach the metrics file: {}",
        steps[1]
    );
    assert!(
        steps[2]["mean_kl"].as_f64().is_some(),
        "L2 ran, so its measures must reach the metrics file: {}",
        steps[2]
    );
}

#[test]
fn the_metrics_header_carries_the_report_envelope() {
    let Some(r) = run_ladder() else { return };
    let header = &r.metrics[0];
    // Exactly what common/reports.py::validate_report requires, asserted here
    // so a Rust-side change fails in the Rust gate rather than only in
    // tests/common/. git_sha is the one obs cannot supply for itself -- the
    // caller passes it, and an envelope without it is rejected.
    assert_eq!(header["schema_version"].as_u64(), Some(1), "{header}");
    assert_eq!(header["report"], "run_metrics", "{header}");
    assert_eq!(header["cli"], "harness", "{header}");
    assert_eq!(header["command"], "ladder", "{header}");
    for key in ["timestamp", "run_id", "git_sha"] {
        let v = header[key].as_str();
        assert!(
            v.is_some_and(|s| !s.is_empty()),
            "{key} must be a non-empty string: {header}"
        );
    }
    // The header strip and the terminal state the same facts: the metrics
    // envelope carries the event header's payload on top of its own keys.
    let event = r.one("ladder.header");
    for key in ["model", "model_hash", "dot", "batch", "size", "reference"] {
        assert_eq!(header[key], event[key], "{key} disagrees between the two");
    }
}
