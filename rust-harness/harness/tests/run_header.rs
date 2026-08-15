/*
 * run_header — the resolved configuration reaches both the terminal and the
 * run record, measured against a real invocation of the binary.
 *
 * The inline tests in `runheader.rs` assert that `lines` derives from
 * `fields`. That is a property of the struct; it says nothing about whether
 * the struct is filled in from the model that actually loaded. This file
 * spawns the harness over the committed tiny fixture and reads the event file
 * the run wrote.
 *
 * It is the test that would have failed throughout the months `DotKind`
 * defaulted to `Sequential`: `set_dot` had one caller, nothing printed the
 * fold in use, and every path except the ladder ran the reference kernel with
 * no way to notice. The same for `--seed`, which RH.12's determinism runs
 * deviated from without anyone seeing it until the entry was written up.
 */

use std::path::{Path, PathBuf};
use std::process::Command;

use runtime::synth::{
    build_tiny_config_json, build_tiny_safetensors, build_tiny_tokenizer_json, TinySpec,
};

/// Write the tiny checkpoint in the layout `tiny-model` writes it: the
/// tokenizer beside the directory, config and weights inside it.
fn write_checkpoint(root: &Path) -> PathBuf {
    let spec = TinySpec::default();
    let stem = root.join("m");
    std::fs::create_dir_all(&stem).unwrap();
    std::fs::write(
        root.join("m.tokenizer.json"),
        build_tiny_tokenizer_json(&spec).unwrap(),
    )
    .unwrap();
    std::fs::write(
        stem.join(runtime::checkpoint::CONFIG_FILE),
        build_tiny_config_json(&spec).unwrap(),
    )
    .unwrap();
    std::fs::write(
        stem.join(runtime::checkpoint::WEIGHTS_FILE),
        build_tiny_safetensors(&spec).unwrap(),
    )
    .unwrap();
    stem
}

/// One run of the harness, returning its stderr and the events it recorded.
fn run(seed: &str) -> (String, Vec<serde_json::Value>) {
    let dir = tempfile::tempdir().unwrap();
    let model = write_checkpoint(dir.path());
    let logs = dir.path().join("logs");
    let out = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args([
            "--model",
            model.to_str().unwrap(),
            "--seed",
            seed,
            "--max-tokens",
            "1",
            "say hi",
        ])
        // Random weights emit gibberish; the assertion is about the header,
        // not the answer.
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", &logs)
        .output()
        .expect("the harness builds as a test binary");
    let stderr = String::from_utf8(out.stderr).expect("stderr is UTF-8");

    let harness_dir = logs.join("harness");
    let events: Vec<serde_json::Value> = std::fs::read_dir(&harness_dir)
        .unwrap_or_else(|e| panic!("{}: {e}", harness_dir.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        // Event files only. The catalog is excluded by name; `is_file` excludes
        // the run's own directory, `<run_id>/`, which now sits beside them and
        // holds metrics.jsonl. Without it this reads a directory and panics.
        .filter(|p| p.is_file() && p.file_name().is_some_and(|n| n != "runs.jsonl"))
        .flat_map(|p| {
            std::fs::read_to_string(&p)
                .unwrap()
                .lines()
                .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
                .collect::<Vec<serde_json::Value>>()
        })
        .collect();
    (stderr, events)
}

#[test]
fn the_run_record_names_the_fold_and_the_seed_the_run_actually_used() {
    let (_, events) = run("7");
    let header = events
        .iter()
        .find(|e| e["event"] == "run.header")
        .unwrap_or_else(|| panic!("no run.header among {events:#?}"));
    // The two fields the stage exists for. `dot` is read from the loaded
    // model's getter rather than from a flag, which is what makes it evidence
    // about the run rather than about the request.
    assert_eq!(header["seed"].as_u64(), Some(7));
    assert!(header["dot"].is_string(), "{header}");
    assert_ne!(header["dot"], "", "{header}");
    // Numbers as numbers: this record is the one someone will jq over.
    assert!(header["weight_bytes"].as_u64().is_some(), "{header}");
    assert!(header["n_layers"].as_u64().is_some(), "{header}");
    assert!(header["max_context"].as_u64().is_some(), "{header}");
}

#[test]
fn a_changed_seed_changes_the_record() {
    // Guards against the header reporting a constant. A field wired to a
    // default rather than to the resolved value passes every assertion above
    // and is exactly the defect that let `--seed 7` go unnoticed.
    let (_, a) = run("7");
    let (_, b) = run("11");
    let seed = |e: &[serde_json::Value]| {
        e.iter().find(|x| x["event"] == "run.header").unwrap()["seed"].as_u64()
    };
    assert_eq!(seed(&a), Some(7));
    assert_eq!(seed(&b), Some(11));
}

#[test]
fn the_terminal_states_the_same_facts_as_the_record() {
    // The stage's claim is "built once, rendered twice". The inline parity
    // test proves the two renderings are derived from one struct; this proves
    // the struct that reached the record is the one that reached the terminal
    // on a real run, which no unit test over a hand-built header can.
    let (stderr, events) = run("7");
    let header = events.iter().find(|e| e["event"] == "run.header").unwrap();
    assert!(stderr.contains("seed=7"), "{stderr}");
    assert!(
        stderr.contains(&format!("dot={}", header["dot"].as_str().unwrap())),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!("layers={}", header["n_layers"].as_u64().unwrap())),
        "{stderr}"
    );
    // And every header line is status, not payload -- the U1c guarantee, held
    // over the lines U2 adds.
    for line in stderr.lines().filter(|l| l.contains("Fold:")) {
        assert!(line.starts_with("[INFO]"), "{line}");
    }
}
