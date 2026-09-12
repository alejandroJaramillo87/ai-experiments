/*
 * context_cap — --context is a cap the run keeps, bounded by the trained
 * window.
 *
 * Every workstation entry that passes --context reads the model's answer,
 * not the budget, and no gate test passed the flag at all: a cap the parser
 * accepted and the session ignored would have shown nowhere. The run header
 * is what the session resolved, so it is what is read here. The upper bound
 * is the one place the flag's help text and its code disagree in kind: the
 * help calls the trained window "the maximum", and the code refuses a value
 * above it at load, which is a failed run and not a usage error.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// The committed fixture's `max_position_embeddings`, stated here rather
/// than read from its config so the expectation is independent of the
/// loader under test.
const TRAINED: u64 = 4096;

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// One task run over the fixture with `extra` appended, and the events it
/// recorded; empty when the run wrote none.
fn run(extra: &[&str]) -> (Output, Vec<serde_json::Value>) {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let logs = dir.path().join("logs");
    let model = fixture_model();
    let out = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args(["--model", model.to_str().unwrap(), "--max-tokens", "1"])
        .args(extra)
        .arg("say hi")
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", &logs)
        .stdin(Stdio::null())
        .output()
        .expect("the harness builds as a test binary");
    let harness_dir = logs.join("harness");
    let events = match std::fs::read_dir(&harness_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            // Event files only: the catalog by name, the run directory by
            // `is_file`.
            .filter(|p| p.is_file() && p.file_name().is_some_and(|n| n != "runs.jsonl"))
            .flat_map(|p| {
                std::fs::read_to_string(&p)
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
                    .collect::<Vec<serde_json::Value>>()
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    (out, events)
}

fn max_context(events: &[serde_json::Value]) -> u64 {
    events
        .iter()
        .find(|e| e["event"] == "run.header")
        .unwrap_or_else(|| panic!("no run.header among {events:#?}"))["max_context"]
        .as_u64()
        .expect("max_context is a number")
}

#[test]
fn a_context_below_the_trained_window_is_the_budget_the_run_reports() {
    let (out, events) = run(&["--context", "2048"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(max_context(&events), 2048, "{events:#?}");
    // And the default is the trained window itself, so the cap above is a
    // change and not the value the run would have reported anyway.
    let (out, events) = run(&[]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(max_context(&events), TRAINED, "{events:#?}");
}

#[test]
fn a_context_above_the_trained_window_fails_the_run_naming_both_figures() {
    let (out, _) = run(&["--context", "5000"]);
    // Exit 1: the value is well-formed and the model decides, at load, that
    // it was not trained there. Exit 2 would mean the parser refused it.
    assert_eq!(
        out.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("--context 5000"),
        "the error must name the value: {err}"
    );
    assert!(
        err.contains(&format!("trained context length {TRAINED}")),
        "the error must name the window: {err}"
    );
}
