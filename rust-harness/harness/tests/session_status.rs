/*
 * session_status — the two startup lines a run prints about the directory it
 * starts in: where its tool permissions come from, and whether it resumed a
 * transcript.
 *
 * Both exist because RH.37 (docs/verification/rust-harness/rust-harness.md)
 * ran three task runs that each stopped at an unannounced `[y/N]` prompt and
 * re-prefilled a prior run's transcript with nothing on the terminal to say
 * so. The harness reads `permissions.yaml` and `session.jsonl` from the
 * working directory silently, which is the right behaviour for a session
 * and the wrong one for an operator who did not know a session was there.
 *
 * Spawned as a process, like `status_streams.rs`, because the lines are
 * `main.rs`'s and not the library's.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name)
}

/// One task run over the fixture in `cwd`, records under a scratch log root.
///
/// stdin is closed so an Ask decision, should the fixture emit a tool call,
/// resolves to a denial rather than blocking on the test runner's terminal.
fn run_a_task(cwd: &Path, logs: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_harness"))
        .args([
            "--model".as_ref(),
            fixture("tiny-gemma4").as_os_str(),
            "--tokenizer".as_ref(),
            fixture("tiny-gemma4.tokenizer.json").as_os_str(),
            "--max-tokens".as_ref(),
            "4".as_ref(),
            "hi".as_ref(),
        ])
        .env("OBS_LOG_DIR", logs)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .output()
        .expect("the harness builds as a test binary")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

#[test]
fn a_run_without_a_permissions_file_says_every_call_will_ask() {
    let cwd = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let out = run_a_task(cwd.path(), logs.path());
    let err = stderr(&out);
    assert!(
        err.contains("[INFO] Permissions: no permissions.yaml, every tool call asks"),
        "{err}"
    );
}

#[test]
fn a_run_with_a_permissions_file_counts_its_rules() {
    let cwd = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    std::fs::write(
        cwd.path().join("permissions.yaml"),
        "rules:\n  - tool: read\n    action: allow\n  - tool: \"*\"\n    action: deny\n",
    )
    .unwrap();
    let out = run_a_task(cwd.path(), logs.path());
    let err = stderr(&out);
    assert!(
        err.contains("[INFO] Permissions: 2 rules from permissions.yaml"),
        "{err}"
    );
    assert!(!err.contains("every tool call asks"), "{err}");
}

#[test]
fn a_second_run_in_the_same_directory_says_what_it_resumed() {
    let cwd = tempfile::tempdir().unwrap();
    let logs = tempfile::tempdir().unwrap();
    let first = stderr(&run_a_task(cwd.path(), logs.path()));
    assert!(
        !first.contains("Session: resumed"),
        "a fresh directory reported a resume: {first}"
    );
    let entries = std::fs::read_to_string(cwd.path().join("session.jsonl"))
        .expect("the first run left a transcript")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .count();
    assert!(
        entries >= 2,
        "the first run wrote {entries} transcript entries"
    );

    let second = stderr(&run_a_task(cwd.path(), logs.path()));
    let want = format!(
        "[INFO] Session: resumed {entries} entries from {}",
        cwd.path().join("session.jsonl").display()
    );
    assert!(second.contains(&want), "wanted {want:?} in\n{second}");
}
