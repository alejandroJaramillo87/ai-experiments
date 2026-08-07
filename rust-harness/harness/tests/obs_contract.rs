/*
 * obs_contract — the harness run log against the shared observability
 * contract, and against the Python reader that consumes it.
 *
 * Run-level observability converges on the Python obs stack by contract
 * rather than by code: harness/src/obs.rs writes the JSONL shapes
 * common/obs defines, and the Python runs, retention, and viewer tooling
 * reads them. Nothing connected the two. The catalog line in particular is
 * formatted by hand rather than through serde, and matched the Python
 * writer field for field only because two people typed the same six keys.
 *
 * The failure mode is quiet. common/obs/catalog.py:read_runs skips a line it
 * cannot parse rather than raising, so a drifted writer presents as an empty
 * runs table and not as an error. That is why the key sets are asserted at
 * the writer, and why the last test here feeds a harness-written catalog to
 * the real reader.
 *
 * Contract: schemas/obs_record.json. The Python half is
 * tests/common/test_obs_record_contract.py.
 */

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use harness::obs::RunLog;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The declared key set for one section of the contract, or None when the
/// contract file is absent — as in the container build, whose context is
/// rust-harness/ alone.
fn contract_keys(section: &str) -> Option<BTreeSet<String>> {
    let path = repo_root().join("schemas/obs_record.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let contract: serde_json::Value = serde_json::from_str(&text).expect("contract is valid JSON");
    Some(
        contract[section]["required"]
            .as_array()
            .expect("required array")
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect(),
    )
}

fn read_lines(path: &Path) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
        .collect()
}

/// One completed run's log directory, held so the files outlive the writer.
struct Run {
    _dir: tempfile::TempDir,
    events: PathBuf,
    catalog: PathBuf,
}

fn write_run() -> Run {
    let dir = tempfile::tempdir().unwrap();
    let argv = vec!["--model".to_string(), "m.gguf".to_string()];
    let log = RunLog::begin_at(dir.path(), "task", &argv, None);
    let run_id = log
        .run_id()
        .expect("a writable root mints a run id")
        .to_string();
    log.end(0, Path::new("/work"));
    let root = dir.path().join("harness");
    Run {
        _dir: dir,
        events: root.join(format!("{run_id}.jsonl")),
        catalog: root.join("runs.jsonl"),
    }
}

/// The interpreter that can import the reader under test, or None.
///
/// Probes the import rather than an empty script, so an environment gap is
/// reported as an absent dependency instead of as a reader defect.
fn python() -> Option<PathBuf> {
    if !repo_root().join("common/obs/catalog.py").is_file() {
        return None;
    }
    runtime::pyenv::interpreter(&repo_root(), &["common.obs.catalog"])
}

#[test]
fn every_event_record_carries_the_contract_base_keys() {
    let Some(want) = contract_keys("event") else {
        runtime::testlog::skip("event key assertions", "schemas/obs_record.json is absent");
        return;
    };
    let run = write_run();
    let events = read_lines(&run.events);
    assert!(
        !events.is_empty(),
        "a completed run writes at least one event"
    );

    for (i, event) in events.iter().enumerate() {
        let got: BTreeSet<String> = event
            .as_object()
            .expect("event is an object")
            .keys()
            .cloned()
            .collect();
        // Named per-event fields sit on top of the base record, so the check
        // is containment rather than equality: cmd.start adds argv, cmd.end
        // adds exit_code and duration_s, fail adds exc_type and exc_msg.
        let missing: Vec<&String> = want.difference(&got).collect();
        assert!(
            missing.is_empty(),
            "event {i} ({}) is missing contract keys {missing:?}",
            event["event"]
        );
    }
}

#[test]
fn the_catalog_line_carries_exactly_the_contract_keys() {
    let Some(want) = contract_keys("catalog") else {
        runtime::testlog::skip(
            "catalog key assertions",
            "schemas/obs_record.json is absent",
        );
        return;
    };
    let run = write_run();
    let catalog = read_lines(&run.catalog);
    assert_eq!(catalog.len(), 1, "one completed run appends one line");

    // Exactly, not at least: the catalog line is a fixed summary record, and
    // the Python writer emits the same six keys and no more.
    let got: BTreeSet<String> = catalog[0]
        .as_object()
        .expect("catalog line is an object")
        .keys()
        .cloned()
        .collect();
    assert_eq!(got, want, "catalog keys have drifted from the contract");
}

/// A harness-written catalog through `common.obs.catalog.read_runs`.
///
/// read_runs skipping unparseable lines is the whole reason this matters: a
/// harness run that the reader silently dropped would look identical to a
/// machine that had not run the harness at all.
#[test]
fn a_harness_written_catalog_parses_through_the_python_reader() {
    let Some(python) = python() else {
        runtime::testlog::skip(
            "python reader assertions",
            "no interpreter imports common.obs.catalog (the parent tree is absent, or \
             the environment is unprovisioned — `make install`, or set HARNESS_PYTHON)",
        );
        return;
    };
    let run = write_run();
    const SCRIPT: &str = "import json, sys\n\
         from pathlib import Path\n\
         from common.obs.catalog import read_runs\n\
         json.dump(read_runs(Path(sys.argv[1])), sys.stdout)\n";
    let out = std::process::Command::new(&python)
        .args(["-c", SCRIPT])
        .arg(&run.catalog)
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
        1,
        "read_runs dropped the harness line instead of parsing it"
    );
    assert_eq!(records[0]["command"], "task");
    assert_eq!(records[0]["exit_code"], 0);
    assert_eq!(records[0]["run_dir"], "/work");

    // The runs table renders timestamp[:19]; a stamp the reader cannot slice
    // that way shows up as a mangled column rather than an error.
    let ts = records[0]["timestamp"]
        .as_str()
        .expect("timestamp is a string");
    assert!(ts.len() >= 19, "timestamp {ts} is too short to render");
    assert_eq!(ts.as_bytes()[10], b'T');
}
