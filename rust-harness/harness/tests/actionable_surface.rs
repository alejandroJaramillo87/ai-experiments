/*
 * actionable_surface — a failing run states its fix, and does not record it.
 *
 * The inline tests in `harness::actionable` prove `render` finds a remedy in a
 * chain and that `Display` withholds it. Neither says the binary calls
 * `render`, and neither says the remedy stays out of the run record -- the
 * error there is hand-built and no `RunLog` exists. Both are properties of a
 * process, so the binary is spawned.
 *
 * The second test is the one worth having. RH.13's evidence is that the
 * printed `[ERROR] Task failed:` line and the `fail` record's `exc_msg` are
 * one string, which is what lets a catalog row's `exit_code: 1` be turned into
 * a cause without re-running anything. That coupling has been relied on since
 * RH.13 was written and guarded by nothing but the next manual workstation
 * run. U7 is the first change that could plausibly break it, so it is also the
 * change that should pin it.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// A finished run, holding its scratch tree so the log root outlives the spawn.
struct Run {
    logs: PathBuf,
    _dir: tempfile::TempDir,
    output: Output,
}

impl Run {
    fn stderr(&self) -> String {
        String::from_utf8(self.output.stderr.clone()).expect("stderr is UTF-8")
    }

    /// The `fail` event this run recorded, or a panic naming what it wrote
    /// instead. The catalog is excluded by name and the run's own directory by
    /// `is_file`, as in `run_header.rs`.
    fn fail_record(&self) -> serde_json::Value {
        let dir = self.logs.join("harness");
        let events: Vec<serde_json::Value> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.file_name().is_some_and(|n| n != "runs.jsonl"))
            .flat_map(|p| {
                std::fs::read_to_string(&p)
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
                    .collect::<Vec<serde_json::Value>>()
            })
            .collect();
        events
            .iter()
            .find(|e| e["event"] == "fail")
            .unwrap_or_else(|| panic!("no fail event among {events:#?}"))
            .clone()
    }
}

/// Run the harness with `args` in a fresh working directory and log root.
///
/// The cwd is temporary for the reason `cli_surface.rs` gives: a run that
/// reaches a transcript opens `<cwd>/session.jsonl`, and the suite must not
/// drop one in the worktree.
fn run(args: &[&str]) -> Run {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let logs = dir.path().join("logs");
    let output = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args(args)
        .current_dir(dir.path())
        .env("OBS_LOG_DIR", &logs)
        .output()
        .expect("the harness builds as a test binary");
    Run {
        logs,
        _dir: dir,
        output,
    }
}

/// The three lines of a remedy, in the order they were printed.
///
/// Returns the tagged failure line and the two continuations, so a test can
/// assert the order rather than only the presence -- advice printed above the
/// failure it belongs to would satisfy three `contains` calls.
fn remedy_lines(err: &str) -> Vec<&str> {
    err.lines()
        .filter(|l| {
            l.starts_with("[ERROR] Task failed:")
                || l.starts_with("  Solution: ")
                || l.starts_with("  Command: ")
        })
        .collect()
}

#[test]
fn a_run_against_no_checkpoint_states_the_fix_under_the_failure() {
    let run = run(&["--model", "/nonexistent-checkpoint", "say hi"]);
    assert_eq!(run.output.status.code(), Some(1), "{}", run.stderr());
    assert!(
        run.output.stdout.is_empty(),
        "a failure is not payload: {:?}",
        String::from_utf8_lossy(&run.output.stdout)
    );

    let err = run.stderr();
    let lines = remedy_lines(&err);
    assert_eq!(lines.len(), 3, "{err}");
    assert!(lines[0].starts_with("[ERROR] Task failed:"), "{err}");
    assert!(lines[1].starts_with("  Solution: "), "{err}");
    assert!(lines[2].starts_with("  Command: "), "{err}");

    // The command names the path the operator actually gave. This is the
    // whole claim of the `!command.contains('<')` rule, observed on a real
    // invocation rather than over a constructor.
    assert!(
        lines[2].contains("/nonexistent-checkpoint"),
        "the command must name the real path: {err}"
    );
    assert!(!lines[2].contains('<'), "{err}");
}

#[test]
fn the_recorded_failure_is_the_printed_one_and_carries_no_advice() {
    // Two assertions that have to hold together. Either alone is satisfiable
    // by a mistake: dropping the remedy entirely satisfies the second, and
    // folding it into the error satisfies neither but would be caught only
    // here.
    let run = run(&["--model", "/nonexistent-checkpoint", "say hi"]);
    let err = run.stderr();
    let lines = remedy_lines(&err);
    let printed = lines[0]
        .strip_prefix("[ERROR] Task failed: ")
        .expect("the failure line carries the message");

    let record = run.fail_record();
    let exc_msg = record["exc_msg"].as_str().expect("exc_msg is a string");
    assert_eq!(
        exc_msg, printed,
        "RH.13's coupling: the recorded message and the printed one are one string"
    );

    // And the advice reached neither. `exc_msg` is `format!("{e:#}")`, so a
    // remedy that displayed itself would be in both.
    let solution = lines[1].trim_start();
    let command = lines[2].trim_start();
    assert!(!exc_msg.contains(solution), "{exc_msg}");
    assert!(!exc_msg.contains(command), "{exc_msg}");
    assert!(!exc_msg.contains("Solution:"), "{exc_msg}");
    assert!(!exc_msg.contains("Command:"), "{exc_msg}");

    // The diagnosis is unchanged from the string RH.13 recorded. Written out
    // rather than matched loosely: this is the byte-identity the entry's
    // evidence rests on, and the reason `Actionable::what` reuses the wording
    // of the message it wrapped instead of improving it.
    assert_eq!(
        exc_msg,
        "read model /nonexistent-checkpoint: no checkpoint at /nonexistent-checkpoint: \
         expected a directory holding config.json and model.safetensors, or a path to \
         the weights file itself"
    );
}

#[test]
fn a_probe_of_the_wrong_width_states_the_fix() {
    // A second site, so the wiring is not proven by one call. The width is
    // deliberately one greater than the fixture's, which fails the geometry
    // guard before any identity check can also fire.
    let dir = tempfile::tempdir().expect("a scratch directory");
    let files = runtime::checkpoint::CheckpointFiles::read(&fixture_model()).unwrap();
    let width = runtime::checkpoint::parse(&files)
        .unwrap()
        .config
        .hidden_size
        + 1;
    let zeros = vec![0.0f32; width];
    let ones = vec![1.0f32; width];
    let probe = dir.path().join("probe.safetensors");
    std::fs::write(
        &probe,
        runtime::synth::build_f32_safetensors(&[
            ("weights", &zeros),
            ("mu", &zeros),
            ("sigma", &ones),
        ])
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        probe.with_extension("json"),
        format!(
            r#"{{"schema_version": 1, "artifact": "probe",
               "timestamp": "2026-01-01T00:00:00+0000", "git_sha": "abcdef0",
               "name": "too-wide", "model_hash": "unknown",
               "layer": 0, "bias": 0.0, "threshold": 1.0, "width": {width}}}"#
        ),
    )
    .unwrap();

    let run = run(&[
        "--model",
        fixture_model().to_str().unwrap(),
        "--probe",
        probe.to_str().unwrap(),
        "--max-tokens",
        "1",
        "say hi",
    ]);
    assert_eq!(run.output.status.code(), Some(1), "{}", run.stderr());
    let err = run.stderr();
    let lines = remedy_lines(&err);
    assert_eq!(lines.len(), 3, "{err}");
    assert!(
        lines[0].contains("probe width") && lines[0].contains("embedding dimension"),
        "{err}"
    );
    assert!(lines[1].starts_with("  Solution: "), "{err}");
    assert!(lines[2].starts_with("  Command: "), "{err}");
    assert!(!lines[2].contains('<'), "{err}");
    // The command names the binary the way this test invoked it, not a bare
    // `harness` no PATH holds: RHX.6 pasted that and got exit 127.
    let program = env!("CARGO_BIN_EXE_harness");
    assert!(
        lines[2].starts_with(&format!("  Command: {program} --status")),
        "the command must name the invoked binary: {err}"
    );
}
