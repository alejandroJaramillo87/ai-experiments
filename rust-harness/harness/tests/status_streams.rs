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

/// The synthetic checkpoint a full run can be driven against, and the
/// tokenizer beside it. Small enough that a load and a prefill both complete
/// in well under a second, which is what makes the timing lines testable in
/// the gate at all.
fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name)
}

/// Run a complete task, returning both streams and the run's event file.
///
/// `OBS_LOG_DIR` points at a scratch tree so the run's records land there
/// rather than in the real log root, and so this test can read back what it
/// wrote.
fn run_a_task() -> (Output, Vec<serde_json::Value>) {
    let cwd = tempfile::tempdir().expect("a scratch working directory");
    let logs = tempfile::tempdir().expect("a scratch log root");
    let out = Command::new(env!("CARGO_BIN_EXE_harness"))
        .args([
            "--model".as_ref(),
            fixture("tiny-gemma4").as_os_str(),
            "--tokenizer".as_ref(),
            fixture("tiny-gemma4.tokenizer.json").as_os_str(),
            "--max-tokens".as_ref(),
            "4".as_ref(),
            "hi".as_ref(),
        ])
        .env("OBS_LOG_DIR", logs.path())
        .current_dir(cwd.path())
        .output()
        .expect("the harness builds as a test binary");
    let dir = logs.path().join("harness");
    let events = std::fs::read_dir(&dir)
        .expect("the run wrote an obs directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().is_some_and(|x| x == "jsonl") && p != &dir.join("runs.jsonl"))
        .map(|p| {
            std::fs::read_to_string(p)
                .expect("the event file is readable")
                .lines()
                .map(|l| serde_json::from_str(l).expect("each line is one JSON object"))
                .collect()
        })
        .expect("the run wrote an event file");
    (out, events)
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

/// The two waits a run makes a human sit through each say how long they took,
/// once, in the order they happened.
///
/// One line each rather than a rewritten progress line: there is no animation
/// at either site, so a terminal and a captured pipe get the same bytes and a
/// runbook's `Observed` block cannot acquire a carriage-return smear. The
/// `\r` assertion below is what holds that open -- nothing else in the tree
/// guards it, and it is the tripwire any future animation lands against.
#[test]
fn a_completed_run_states_what_the_two_long_operations_cost() {
    let (out, _) = run_a_task();
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert_stdout_is_payload(&out, "a completed run");
    let err = stderr(&out);

    let load: Vec<&str> = err.lines().filter(|l| l.contains("Model load:")).collect();
    let prefill: Vec<&str> = err.lines().filter(|l| l.contains("Prefill:")).collect();
    assert_eq!(
        load.len(),
        1,
        "one load line, not none and not per layer: {err}"
    );
    assert_eq!(prefill.len(), 1, "one prefill line per turn: {err}");
    assert!(load[0].starts_with("[INFO] "), "{:?}", load[0]);
    assert!(prefill[0].starts_with("[INFO] "), "{:?}", prefill[0]);
    assert!(
        err.find("Model load:") < err.find("Prefill:"),
        "the load finishes before the prompt is prefilled: {err}"
    );

    assert!(
        !err.contains('\r'),
        "a carriage return would smear a pasted Observed block: {err:?}"
    );
    assert!(!stdout(&out).contains('\r'), "{:?}", stdout(&out));
}

/// The printed load time is the recorded one.
///
/// `main.rs` reads it off the `model.load` phase guard rather than sampling a
/// second `Instant`, so this is a test of that choice: two clocks would drift
/// by however long the work took to start, and a human comparing the terminal
/// against `<run_id>.jsonl` would find two answers to one question.
#[test]
fn the_load_time_on_the_terminal_is_the_one_in_the_record() {
    let (out, events) = run_a_task();
    let recorded = events
        .iter()
        .find(|e| e["event"] == "phase.end" && e["phase"] == "model.load")
        .and_then(|e| e["duration_s"].as_f64())
        .expect("the run recorded a model.load phase");

    let printed: f64 = stderr(&out)
        .lines()
        .find_map(|l| l.split("Model load:").nth(1))
        .and_then(|rest| rest.trim().trim_end_matches(" s").parse().ok())
        .expect("the run printed a load time");

    // The printed figure is rendered at two decimals from the same
    // `Instant`, so it differs from the recorded one only by the rounding and
    // by the microseconds between the read and the guard's drop.
    assert!(
        (printed - recorded).abs() < 0.02,
        "printed {printed} against recorded {recorded}"
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
