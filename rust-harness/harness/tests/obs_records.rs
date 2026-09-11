/*
 * obs_records — RH.13's run-level records, produced by the binary.
 *
 * `obs_contract.rs` checks the writer's key sets and hands one in-process
 * catalog to the Python reader; `actionable_surface.rs` spawns the binary
 * against a missing checkpoint and reads back its `fail` event. Neither runs
 * the binary through a success and a failure sharing one log root and then
 * reads the catalog, which is the claim RH.13 makes on the workstation: two
 * rows, `exit_code` 0 then 1, run ids of the documented shape, each naming an
 * event file that ends the way its row says.
 *
 * The runbook entry proves that against the default `/mnt/ai-data/logs` root
 * on a real checkpoint. This file proves it against a scratch root over the
 * tiny fixture on every `cargo test`, so the records cannot drift between
 * workstation runs unnoticed.
 */

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// The catalog's declared key set, or None when the contract file is absent
/// (the container build's context is `rust-harness/` alone).
fn catalog_keys() -> Option<BTreeSet<String>> {
    let path = repo_root().join("schemas/obs_record.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let contract: serde_json::Value = serde_json::from_str(&text).expect("contract is valid JSON");
    Some(
        contract["catalog"]["required"]
            .as_array()
            .expect("catalog.required array")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect(),
    )
}

/// The interpreter that can import the reader under test, or None.
fn python() -> Option<PathBuf> {
    if !repo_root().join("common/obs/catalog.py").is_file() {
        return None;
    }
    runtime::pyenv::interpreter(&repo_root(), &["common.obs.catalog"])
}

fn read_lines(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
        .collect()
}

/// Run the harness with `args` in a fresh working directory, logging under
/// `logs`. The cwd is temporary because a run that reaches a transcript opens
/// `<cwd>/session.jsonl`; the log root is the caller's so two runs can share it.
fn run(logs: &Path, args: &[&str]) -> Output {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    Command::new(env!("CARGO_BIN_EXE_harness"))
        .args(args)
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", logs)
        .output()
        .expect("the harness builds as a test binary")
}

/// One log root after RH.13's two invocations: a run that answers, then one
/// that fails before loading anything. Held so the files outlive the spawns.
struct TwoRuns {
    _dir: tempfile::TempDir,
    logs: PathBuf,
}

impl TwoRuns {
    fn produce() -> Self {
        let dir = tempfile::tempdir().expect("a scratch log root");
        let logs = dir.path().join("logs");
        let model = fixture_model();
        let ok = run(
            &logs,
            &[
                "--model",
                model.to_str().unwrap(),
                "--text-only",
                "--max-tokens",
                "4",
                "--seed",
                "0",
                "Reply with the single word: ready",
            ],
        );
        assert_eq!(
            ok.status.code(),
            Some(0),
            "{}",
            String::from_utf8_lossy(&ok.stderr)
        );
        let failed = run(&logs, &["--model", "/nonexistent-checkpoint", "task"]);
        assert_eq!(
            failed.status.code(),
            Some(1),
            "{}",
            String::from_utf8_lossy(&failed.stderr)
        );
        Self { _dir: dir, logs }
    }

    fn catalog_path(&self) -> PathBuf {
        self.logs.join("harness").join("runs.jsonl")
    }

    fn catalog(&self) -> Vec<serde_json::Value> {
        read_lines(&self.catalog_path())
    }

    fn events(&self, run_id: &str) -> Vec<serde_json::Value> {
        read_lines(&self.logs.join("harness").join(format!("{run_id}.jsonl")))
    }
}

fn run_id(row: &serde_json::Value) -> &str {
    row["run_id"]
        .as_str()
        .unwrap_or_else(|| panic!("run_id is a string in {row}"))
}

/// `R-YYYYMMDD-HHMMSS-xxxxxx`, checked by hand: the harness carries no regex
/// crate, and the shape is short enough that one would not pay for itself.
fn assert_run_id_shape(id: &str) {
    let parts: Vec<&str> = id.split('-').collect();
    assert_eq!(parts.len(), 4, "run id {id} has four dash-separated parts");
    assert_eq!(parts[0], "R", "run id {id} starts with R");
    let digits = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit());
    assert!(digits(parts[1], 8), "run id {id}: date is eight digits");
    assert!(digits(parts[2], 6), "run id {id}: time is six digits");
    assert!(
        parts[3].len() == 6
            && parts[3]
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "run id {id}: suffix is six lowercase hex digits"
    );
}

#[test]
fn a_success_and_a_failure_catalog_as_two_rows_in_order() {
    let runs = TwoRuns::produce();
    let rows = runs.catalog();
    assert_eq!(rows.len(), 2, "one catalog line per invocation: {rows:#?}");

    // Row order is invocation order: the catalog is appended at run end, and
    // the failing run ends after the successful one has.
    assert_eq!(rows[0]["exit_code"], 0, "{}", rows[0]);
    assert_eq!(rows[1]["exit_code"], 1, "{}", rows[1]);
    for row in &rows {
        assert_eq!(row["command"], "task", "{row}");
        assert_run_id_shape(run_id(row));
        assert!(
            row["duration_s"].is_number(),
            "duration_s is a JSON number: {row}"
        );
    }
    assert_ne!(run_id(&rows[0]), run_id(&rows[1]), "run ids are distinct");

    let Some(want) = catalog_keys() else {
        runtime::testlog::skip(
            "catalog key assertions",
            "schemas/obs_record.json is absent",
        );
        return;
    };
    for row in &rows {
        let got: BTreeSet<String> = row
            .as_object()
            .expect("a catalog row is an object")
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            got, want,
            "catalog row keys drifted from the contract: {row}"
        );
    }
}

#[test]
fn each_catalog_row_names_an_event_file_that_ends_the_same_way() {
    let runs = TwoRuns::produce();
    let rows = runs.catalog();
    assert_eq!(rows.len(), 2, "{rows:#?}");

    for row in &rows {
        let id = run_id(row);
        let events = runs.events(id);
        let names: Vec<&str> = events
            .iter()
            .map(|e| e["event"].as_str().expect("event is a string"))
            .collect();
        assert_eq!(names.first(), Some(&"cmd.start"), "{id}: {names:?}");
        assert_eq!(names.last(), Some(&"cmd.end"), "{id}: {names:?}");
        assert_eq!(
            names.contains(&"fail"),
            row["exit_code"] == 1,
            "a fail event exactly when the row's exit code says so, {id}: {names:?}"
        );
        for event in &events {
            assert_eq!(event["run_id"], id, "every event names its run: {event}");
        }
        let end = events.last().unwrap();
        assert_eq!(
            end["exit_code"], row["exit_code"],
            "cmd.end and the catalog row agree on the exit code: {end}"
        );
    }

    // The failing run rejects its path inside `model.load`, the phase that
    // brackets the whole cold start, and before any weight is read: the phase
    // closes at once and no `run.header` follows, because the header is
    // written from a checkpoint that was never opened. RH.13's 2026-08-14
    // record predates the phase and shows the bare triple; this is the shape
    // the U campaign left, pinned so the next change to it is deliberate.
    let failing = runs.events(run_id(&rows[1]));
    let names: Vec<&str> = failing
        .iter()
        .map(|e| e["event"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        ["cmd.start", "phase.start", "phase.end", "fail", "cmd.end"],
        "{failing:#?}"
    );
    assert_eq!(failing[1]["phase"], "model.load", "{}", failing[1]);
    assert_eq!(failing[2]["phase"], "model.load", "{}", failing[2]);
    let fail = &failing[3];

    // The failing run's cause is in its record, which is what lets a catalog
    // row's `exit_code: 1` be explained without re-running. Exact equality
    // with the printed line is `actionable_surface.rs`'s assertion.
    assert_eq!(fail["level"], "error", "{fail}");
    assert_eq!(fail["exc_type"], "anyhow", "{fail}");
    assert_eq!(fail["exit_code"], 1, "{fail}");
    let exc_msg = fail["exc_msg"].as_str().expect("exc_msg is a string");
    assert!(
        exc_msg.starts_with("read model /nonexistent-checkpoint"),
        "exc_msg names the failing step and path: {exc_msg}"
    );
}

/// The two-row catalog the binary wrote, through the real Python reader.
///
/// `obs_contract.rs` proves this for a catalog written in-process. The binary
/// is what the workstation runs, and its failing run is the row a viewer most
/// needs to see, so the reader gets both rows from a real spawn as well.
#[test]
fn a_binary_written_catalog_parses_through_the_python_reader() {
    let Some(python) = python() else {
        runtime::testlog::skip(
            "python reader assertions",
            "no interpreter imports common.obs.catalog (the parent tree is absent, or \
             the environment is unprovisioned — `make install`, or set HARNESS_PYTHON)",
        );
        return;
    };
    let runs = TwoRuns::produce();
    let rows = runs.catalog();
    const SCRIPT: &str = "import json, sys\n\
         from pathlib import Path\n\
         from common.obs.catalog import read_runs\n\
         json.dump(read_runs(Path(sys.argv[1])), sys.stdout)\n";
    let out = Command::new(&python)
        .args(["-c", SCRIPT])
        .arg(runs.catalog_path())
        .env("PYTHONPATH", repo_root())
        .output()
        .expect("python3 runs");
    assert!(
        out.status.success(),
        "the reader failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let records: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let records = records.as_array().expect("read_runs returns a list");
    assert_eq!(
        records.len(),
        2,
        "read_runs dropped a harness line instead of parsing it: {records:#?}"
    );
    for (record, row) in records.iter().zip(&rows) {
        assert_eq!(record["run_id"], row["run_id"], "{record}");
        assert_eq!(record["exit_code"], row["exit_code"], "{record}");
        assert_eq!(record["command"], "task", "{record}");
    }
}
