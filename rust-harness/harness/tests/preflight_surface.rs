/*
 * preflight_surface — `--status` through the binary.
 *
 * The unit tests in `harness::preflight` check the check list against itself.
 * What only exists once there is a process is the part this file holds: which
 * stream the report goes to, what the exit code means, and that a preflight
 * writes no run record.
 *
 * That last one is the [SCR] claim made testable. `--status` is documented as
 * writing nothing to the obs catalog -- a filesystem query is not a run, the
 * same rule that keeps `--help` out of it -- and nothing but a spawn can show
 * that the log directory stays absent, because the writer is chosen inside
 * `main` well after any function a unit test could call.
 *
 * Three exit codes carry three different answers and the tests pin all three:
 * 0 clean, 1 the run would fail, 2 the command line was wrong to ask.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/tiny-gemma4")
}

/// Run the harness with `args`, capturing both streams.
///
/// A scratch working directory and a scratch `OBS_LOG_DIR` on every spawn:
/// the first keeps `session.jsonl` out of the worktree the way
/// `cli_surface.rs` does, and the second is what makes the no-record
/// assertion mean anything -- against the real log root a stale directory
/// from another run would satisfy it.
fn run(args: &[&str]) -> (Output, tempfile::TempDir) {
    let cwd = tempfile::tempdir().expect("a scratch working directory");
    let logs = cwd.path().join("logs");
    let out = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args(args)
        .current_dir(cwd.path())
        .env("OBS_LOG_DIR", &logs)
        .output()
        .expect("the harness builds as a test binary");
    (out, cwd)
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("stderr is UTF-8")
}

/// A probe image with no sidecar beside it, which is the mandatory-sidecar
/// failure the run path also refuses.
fn probe_without_sidecar(dir: &Path) -> PathBuf {
    let path = dir.join("probe.safetensors");
    std::fs::write(
        &path,
        runtime::synth::build_f32_safetensors(&[
            ("weights", &[0.1f32, 0.2]),
            ("mu", &[0.0f32, 0.0]),
            ("sigma", &[1.0f32, 1.0]),
        ])
        .expect("build probe image"),
    )
    .expect("write probe");
    path
}

#[test]
fn status_against_the_committed_fixture_exits_zero() {
    let model = fixture_model();
    let (out, _cwd) = run(&["--model", model.to_str().unwrap(), "--status"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("[OK] backend compatibility"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn status_writes_its_report_to_stderr_and_leaves_stdout_empty() {
    // stdout is the payload channel and a preflight has no payload. `--help`
    // is the one requested-output exception, and it is untagged text; these
    // lines carry tags, which is exactly what must not reach a machine reader.
    let model = fixture_model();
    let (out, _cwd) = run(&["--model", model.to_str().unwrap(), "--status"]);
    assert!(stdout(&out).is_empty(), "stdout: {:?}", stdout(&out));
    assert!(!stderr(&out).is_empty());
}

#[test]
fn status_writes_no_obs_record() {
    let model = fixture_model();
    let (out, cwd) = run(&["--model", model.to_str().unwrap(), "--status"]);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    // Not "the catalog has no new row": the directory itself must never
    // appear, since `RunLog::begin` creates it before writing anything.
    let logs = cwd.path().join("logs").join("harness");
    assert!(!logs.exists(), "{} exists", logs.display());
}

#[test]
fn a_checkpoint_that_is_not_there_exits_one_rather_than_two() {
    // The distinction the whole mode rests on: 1 says the run would fail, 2
    // says the command line was wrong. A missing checkpoint is the first.
    let (out, _cwd) = run(&["--model", "/absent/checkpoint", "--status"]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(
        stderr(&out).contains("[ERROR] model directory"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn a_probe_without_its_sidecar_exits_one_and_names_the_sidecar() {
    let dir = tempfile::tempdir().expect("tempdir");
    let probe = probe_without_sidecar(dir.path());
    let model = fixture_model();
    let (out, _cwd) = run(&[
        "--model",
        model.to_str().unwrap(),
        "--probe",
        probe.to_str().unwrap(),
        "--status",
    ]);
    assert_eq!(out.status.code(), Some(1), "{}", stderr(&out));
    assert!(stderr(&out).contains("probe.json"), "{}", stderr(&out));
}

#[test]
fn deep_hashes_the_fixture_and_reports_the_identity_it_would_otherwise_skip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let probe = dir.path().join("probe.safetensors");
    let width = 16;
    let weights: Vec<f32> = (0..width).map(|j| 0.1 * j as f32).collect();
    std::fs::write(
        &probe,
        runtime::synth::build_f32_safetensors(&[
            ("weights", &weights),
            ("mu", &vec![0.0f32; width]),
            ("sigma", &vec![1.0f32; width]),
        ])
        .expect("build probe image"),
    )
    .expect("write probe");
    // The `unknown` sentinel: the point here is that --deep reaches the check
    // at all, not what a real hash comparison says, and a sentinel keeps the
    // fixture's weights from being hashed for a value nothing asserts on.
    std::fs::write(
        probe.with_extension("json"),
        format!(
            r#"{{"schema_version": 1, "artifact": "probe",
               "timestamp": "2026-01-01T00:00:00+0000", "git_sha": "abcdef0",
               "name": "surface-probe", "model_hash": "unknown",
               "layer": 0, "bias": 0.1, "threshold": 0.5, "width": {width}}}"#
        ),
    )
    .expect("write sidecar");
    let model = fixture_model();
    let args = [
        "--model",
        model.to_str().unwrap(),
        "--probe",
        probe.to_str().unwrap(),
        "--status",
    ];
    let (shallow, _a) = run(&args);
    assert!(
        stderr(&shallow).contains("[SKIP] probe identity"),
        "{}",
        stderr(&shallow)
    );
    let mut deep = args.to_vec();
    deep.push("--deep");
    let (out, _b) = run(&deep);
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(
        !stderr(&out).contains("[SKIP] probe identity"),
        "--deep must reach the check: {}",
        stderr(&out)
    );
}

#[test]
fn status_without_a_model_is_a_usage_error() {
    let (out, _cwd) = run(&["--status"]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(stderr(&out).contains("usage: harness"), "{}", stderr(&out));
}

#[test]
fn status_with_task_text_is_a_usage_error() {
    let model = fixture_model();
    let (out, _cwd) = run(&["--model", model.to_str().unwrap(), "--status", "do a task"]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
}

#[test]
fn status_and_chat_together_are_a_usage_error_naming_both() {
    let model = fixture_model();
    let (out, _cwd) = run(&["--model", model.to_str().unwrap(), "--status", "--chat"]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("--chat") && err.contains("--status"), "{err}");
}

#[test]
fn deep_without_status_is_a_usage_error() {
    // The `requires` edge from the flag table, enforced without a hand-written
    // check the way `--serve-addr` is.
    let model = fixture_model();
    let (out, _cwd) = run(&["--model", model.to_str().unwrap(), "--deep", "do a task"]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(stderr(&out).contains("--deep"), "{}", stderr(&out));
}

#[test]
fn the_report_carries_no_escape_bytes_through_a_pipe() {
    // Colour is display-only and both streams here are pipes, so the report
    // must be the same bytes a terminal shows minus the escapes.
    let model = fixture_model();
    let (out, _cwd) = run(&["--model", model.to_str().unwrap(), "--status"]);
    assert!(!stderr(&out).contains('\x1b'), "{:?}", stderr(&out));
    assert!(!stdout(&out).contains('\x1b'), "{:?}", stdout(&out));
}
