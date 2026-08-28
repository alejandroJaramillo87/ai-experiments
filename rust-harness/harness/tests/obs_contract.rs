/*
 * obs_contract — the harness run log against the shared observability
 * contract, and against the Python reader that consumes it.
 *
 * Run-level observability converges on the Python obs stack by contract
 * rather than by code: the obs crate writes the JSONL shapes
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

use obs::RunLog;

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

/// The declared field set for one event type: `fields` plus `optional`.
///
/// Degrades the same way `contract_keys` does, and for the same reason -- the
/// container build's context is `rust-harness/` alone, so the contract file is
/// simply absent there.
fn contract_event_keys(event: &str) -> Option<BTreeSet<String>> {
    let path = repo_root().join("schemas/obs_record.json");
    let text = std::fs::read_to_string(&path).ok()?;
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
    log.end(0);
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
    // run_dir is the run's own artifact directory (schema_version 2), which
    // is what `training report --run` opens. Asserted against the row's own
    // run_id rather than a literal: the property is that the two agree, and a
    // hard-coded path would only pin the tempdir.
    let run_dir = records[0]["run_dir"].as_str().expect("run_dir is a string");
    let run_id = records[0]["run_id"].as_str().expect("run_id is a string");
    assert!(
        run_dir.ends_with(run_id),
        "run_dir {run_dir} must name the run {run_id}"
    );

    // The runs table renders timestamp[:19]; a stamp the reader cannot slice
    // that way shows up as a mangled column rather than an error.
    let ts = records[0]["timestamp"]
        .as_str()
        .expect("timestamp is a string");
    assert!(ts.len() >= 19, "timestamp {ts} is too short to render");
    assert_eq!(ts.as_bytes()[10], b'T');
}

/// The run header's declared field set, both directions.
///
/// Equality rather than containment. Containment one way would let a field be
/// declared and never written -- the likelier drift for a header assembled in
/// one place, where adding a struct field and forgetting the schema and
/// forgetting to remove the schema entry are all one edit away from each
/// other.
///
/// Measured against a header with every optional field present, which is what
/// makes `optional` assertable at all: against a minimal one, an undeclared
/// optional field would simply be absent and prove nothing.
#[test]
fn the_run_header_writes_exactly_the_fields_the_contract_declares() {
    let Some(declared) = contract_event_keys("run.header") else {
        runtime::testlog::skip(
            "run.header contract",
            "schemas/ absent from this build context",
        );
        return;
    };
    let header = harness::runheader::RunHeader {
        model: "/models/e4b".to_string(),
        model_hash: "0123456789abcdef".to_string(),
        n_layers: 42,
        dtype: "bf16".to_string(),
        weight_bytes: 15_000_000_000,
        peak_rss_bytes: Some(15_300_000_000),
        max_context: 4096,
        kv_bytes: 1_073_741_824,
        kv_batch_group: 64,
        rewind_horizon: Some(1024),
        threads: 16,
        dot: "widen4".to_string(),
        batch: "64".to_string(),
        seed: 7,
        temperature: 0.0,
        max_tokens: 256,
        binary_version: "0.1.0".to_string(),
        git_sha: "abc1234".to_string(),
        probe: Some("refusal".to_string()),
        probe_layer: Some(20),
    };
    let written: BTreeSet<String> = header
        .fields()
        .into_iter()
        .map(|(k, _)| k.to_string())
        .collect();
    assert_eq!(written, declared);
}

/// A caller field may not displace a base key.
///
/// `common/obs/core.py:121-124` has refused this since the Python stack was
/// built, and `test_a_caller_field_cannot_displace_a_base_key` asserts it
/// there. Rust had no equivalent because `write_event` was private and every
/// caller lived in `obs.rs`; `RunLog::event` is public, so privacy stops doing
/// the work. A duplicate key is legal JSON whose meaning depends on which one
/// the reader keeps -- the record would parse and quietly mean something else,
/// which is worse than the malformed line the contract already guards against.
#[test]
fn a_caller_field_cannot_displace_a_base_key() {
    use obs::Value;
    let dir = tempfile::tempdir().unwrap();
    let mut log = RunLog::begin_at(dir.path(), "task", &[], None);
    let run_id = log
        .run_id()
        .expect("a writable root mints a run id")
        .to_string();
    log.event(
        "info",
        "run.header",
        "",
        &[
            ("run_id", Value::Str("forged".to_string())),
            ("cli", Value::Str("not-harness".to_string())),
            ("model", Value::Str("/models/e4b".to_string())),
        ],
    );
    log.end(0);

    let events = read_lines(&dir.path().join("harness").join(format!("{run_id}.jsonl")));
    let header = events
        .iter()
        .find(|e| e["event"] == "run.header")
        .expect("the header event was written");
    assert_eq!(header["run_id"], run_id.as_str());
    assert_eq!(header["cli"], "harness");
    // The non-colliding field is untouched: the guard drops the offenders,
    // not the event.
    assert_eq!(header["model"], "/models/e4b");

    // And the raw line carries each base key once, which is the property
    // `serde_json` above would hide -- it keeps the last of a duplicate pair
    // and reports no error.
    let raw = std::fs::read_to_string(dir.path().join("harness").join(format!("{run_id}.jsonl")))
        .unwrap();
    let line = raw
        .lines()
        .find(|l| l.contains("run.header"))
        .expect("the header line");
    assert_eq!(line.matches("\"run_id\":").count(), 1, "{line}");
    assert_eq!(line.matches("\"cli\":").count(), 1, "{line}");
}

/// Numbers survive the round trip as numbers.
///
/// `weight_bytes` rendered as "14.23 GiB" cannot be turned back into bytes,
/// and a header whose purpose is cross-run comparison is exactly the record
/// someone will `jq` over. The Python writer already emits real numbers for
/// duration_s and exit_code; this keeps the Rust side under the same contract.
#[test]
fn header_numbers_are_json_numbers_and_not_strings() {
    use obs::Value;
    let dir = tempfile::tempdir().unwrap();
    let mut log = RunLog::begin_at(dir.path(), "task", &[], None);
    let run_id = log.run_id().unwrap().to_string();
    log.event(
        "info",
        "run.header",
        "",
        &[
            ("weight_bytes", Value::Int(15_000_000_000)),
            ("temperature", Value::Num(0.0)),
            ("text_only", Value::Bool(true)),
        ],
    );
    log.end(0);
    let events = read_lines(&dir.path().join("harness").join(format!("{run_id}.jsonl")));
    let h = events.iter().find(|e| e["event"] == "run.header").unwrap();
    assert_eq!(h["weight_bytes"].as_u64(), Some(15_000_000_000));
    assert_eq!(h["temperature"].as_f64(), Some(0.0));
    assert_eq!(h["text_only"].as_bool(), Some(true));
}

/// The phase events carry exactly the fields the contract declares.
///
/// Two-sided, which is what makes this the stronger kind of assertion here:
/// `common/obs/core.py::phase` writes the same two events, and
/// `tests/common/test_obs_record_contract.py` holds it to the same declaration
/// in the same file. Neither writer can drift without one of the two failing.
#[test]
fn the_phase_events_match_their_declared_fields() {
    let dir = tempfile::tempdir().unwrap();
    let mut log = RunLog::begin_at(dir.path(), "task", &[], None);
    let run_id = log.run_id().unwrap().to_string();
    {
        let _span = log.phase("model.load");
    }
    log.end(0);

    let events = read_lines(&dir.path().join("harness").join(format!("{run_id}.jsonl")));
    let base = contract_keys("event").unwrap_or_default();
    for name in ["phase.start", "phase.end"] {
        let Some(declared) = contract_event_keys(name) else {
            runtime::testlog::skip(name, "schemas/obs_record.json is absent");
            return;
        };
        let record = events
            .iter()
            .find(|e| e["event"] == name)
            .unwrap_or_else(|| panic!("{name} was written"));
        let written: BTreeSet<String> = record
            .as_object()
            .expect("an object")
            .keys()
            .filter(|k| !base.contains(*k))
            .cloned()
            .collect();
        // Equality in both directions: containment alone would let a field be
        // declared and never written, which is the likelier drift.
        assert_eq!(written, declared, "{name}");
        // File-only on both sides, so msg stays empty rather than doubling as
        // the phase name.
        assert_eq!(record["msg"], "");
        assert_eq!(record["phase"], "model.load");
    }
}

/// The metrics header is a report envelope the Python validator accepts, and
/// the catalog's run_dir points at the directory holding it.
///
/// `schemas/run_metrics.json` is what these two claims are asserted against;
/// `tests/common/test_run_metrics_contract.py` feeds a file written by this
/// same code to the real reader and the real validator.
#[test]
fn the_metrics_header_matches_the_run_metrics_contract() {
    use obs::Value;
    let dir = tempfile::tempdir().unwrap();
    let mut log = RunLog::begin_at(dir.path(), "task", &[], None);
    let run_dir = log
        .run_dir()
        .expect("an active log names its run dir")
        .to_path_buf();
    log.metrics_header(&[("git_sha", Value::Str("deadbeef".to_string()))]);
    log.metric(0, &[("tokens_per_s", Value::Num(12.5))]);
    log.end(0);

    let path = repo_root().join("schemas/run_metrics.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        runtime::testlog::skip(
            "run_metrics contract",
            "schemas/run_metrics.json is absent (container build context)",
        );
        return;
    };
    let contract: serde_json::Value = serde_json::from_str(&text).expect("valid JSON");
    let records = read_lines(&run_dir.join("metrics.jsonl"));

    let header = &records[0];
    for key in contract["header"]["required"].as_array().unwrap() {
        let key = key.as_str().unwrap();
        assert!(
            header.get(key).is_some(),
            "metrics header is missing the required key {key}"
        );
    }
    assert_eq!(header["report"], "run_metrics");

    for key in contract["record"]["required"].as_array().unwrap() {
        let key = key.as_str().unwrap();
        assert!(
            records[1].get(key).is_some(),
            "step record is missing {key}"
        );
    }
    // viewer.js:109 drops a record whose step is not typeof "number"
    assert!(records[1]["step"].is_number());

    let catalog = read_lines(&dir.path().join("harness").join("runs.jsonl"));
    assert_eq!(
        catalog[0]["run_dir"],
        run_dir.to_string_lossy().as_ref(),
        "the catalog must point at the directory holding metrics.jsonl"
    );
}
