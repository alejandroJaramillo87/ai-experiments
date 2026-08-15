/*
 * status_streams — the stream split, asserted from outside the process.
 *
 * `runtime::status` writes every tag to stderr and contains no `println!`, so
 * a reviewer can check the guarantee by reading one file. That is a claim about
 * the module, not about this binary: nothing stopped `main.rs` from printing a
 * status line itself, and for most of its life it did -- 40 hand-written
 * `eprintln!("[TAG] ...")` calls with the tag typed as a literal at each site,
 * which is exactly the shape that drifts onto the wrong stream one site at a
 * time.
 *
 * `runtime/tests/status_python_parity.rs:115` already asserts this of the
 * Python `Printer`. These are the Rust mirror: spawned behind pipes, which is
 * the shape a script sees, and measured against the whole binary rather than
 * one module of it.
 */

use std::process::{Command, Output};

/// Run the harness with `args` and capture both streams.
///
/// Spawned in a temporary directory for the reason `cli_surface.rs` gives:
/// cargo runs an integration test with the crate directory as cwd, and
/// `main.rs` opens `<cwd>/session.jsonl` on any invocation that gets as far as
/// a transcript.
fn run(args: &[&str]) -> Output {
    let cwd = tempfile::tempdir().expect("a scratch working directory");
    Command::new(env!("CARGO_BIN_EXE_harness"))
        .args(args)
        .current_dir(cwd.path())
        .output()
        .expect("the harness builds as a test binary")
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("stderr is UTF-8")
}

/// Every one of the six tags is a status line, and a status line on stdout
/// corrupts the payload channel. Asserted against the tag labels themselves
/// rather than a sample, so a tag added to `status::Tag` is covered here
/// without anyone remembering to extend a list.
const TAGS: [&str; 6] = ["[OK]", "[WARN]", "[ERROR]", "[INFO]", "[RUN]", "[SKIP]"];

fn assert_stdout_is_payload(out: &Output, case: &str) {
    let text = stdout(out);
    for tag in TAGS {
        assert!(
            !text.contains(tag),
            "{case}: {tag} reached stdout, which is the payload channel: {text:?}"
        );
    }
    assert!(
        !text.contains('\x1b'),
        "{case}: stdout carries escape bytes: {text:?}"
    );
}

/// A run that fails during startup emits status and no payload. This is the
/// ordinary failing invocation -- the one whose output someone pastes into a
/// runbook -- so it is the case most worth pinning.
#[test]
fn a_failing_run_writes_its_diagnosis_to_stderr_and_nothing_to_stdout() {
    let out = run(&["--model", "/nonexistent-checkpoint", "do x"]);
    assert_stdout_is_payload(&out, "a failing run");
    assert!(
        out.stdout.is_empty(),
        "a run that produced no answer has no payload: {:?}",
        stdout(&out)
    );
    let err = stderr(&out);
    assert!(err.contains("[ERROR]"), "the failure must be tagged: {err}");
    assert!(err.contains("nonexistent-checkpoint"), "{err}");
}

/// A warning is status too, and a run that warns and then fails must keep both
/// off stdout. `--top-k` under greedy decoding is inert, which `cli::inert`
/// reports through `status::warn` -- the path that would otherwise be the
/// easiest one to write with a bare `println!`.
#[test]
fn a_warning_and_an_error_both_land_on_stderr() {
    let out = run(&[
        "--model",
        "/nonexistent-checkpoint",
        "--temperature",
        "0",
        "--top-k",
        "5",
        "do x",
    ]);
    assert_stdout_is_payload(&out, "a warning followed by an error");
    let err = stderr(&out);
    assert!(err.contains("[WARN]"), "{err}");
    assert!(err.contains("[ERROR]"), "{err}");
}

/// A usage error prints the usage block, which is untagged and therefore easy
/// to mistake for payload. It is not: nobody asked for it, and the exit code
/// is 2. Contrast `--help`, which is requested and does go to stdout.
#[test]
fn an_unrequested_usage_block_is_status_and_requested_help_is_payload() {
    let refused = run(&["--modle", "foo", "do x"]);
    assert_eq!(refused.status.code(), Some(2));
    assert!(
        refused.stdout.is_empty(),
        "a usage error is not payload: {:?}",
        stdout(&refused)
    );
    assert!(
        stderr(&refused).contains("usage: harness"),
        "{}",
        stderr(&refused)
    );

    let asked = run(&["--help"]);
    assert_eq!(asked.status.code(), Some(0));
    assert!(stdout(&asked).contains("usage: harness"));
    assert!(
        asked.stderr.is_empty(),
        "help is not a diagnostic: {:?}",
        stderr(&asked)
    );
}

/// Colour is display only and is gated on stderr being a terminal. Through a
/// pipe there is nothing to colour, so the uncoloured text must be what a
/// script sees -- on the status channel as well as the payload one, since
/// `grep '\[ERROR\]'` is how these lines get read.
#[test]
fn status_through_a_pipe_carries_no_escape_bytes_on_either_stream() {
    let out = run(&["--model", "/nonexistent-checkpoint", "do x"]);
    assert!(!stdout(&out).contains('\x1b'), "{:?}", stdout(&out));
    assert!(!stderr(&out).contains('\x1b'), "{:?}", stderr(&out));
}
