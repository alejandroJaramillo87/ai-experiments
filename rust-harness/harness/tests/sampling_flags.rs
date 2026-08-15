/*
 * sampling_flags — the two sampling-group flags nothing reached through the
 * binary: --top-p and --compact-at.
 *
 * `harness::cli` checks the table against itself and `runtime::sampler` checks
 * nucleus truncation against its own arithmetic, but neither says the flag on
 * the command line arrives at either. What only exists once there is a process
 * is the range check -- a value the sampler would panic on has to become a
 * usage error before it reaches the sampler -- and, for --compact-at, the
 * threshold actually deciding whether a conversation is compacted.
 *
 * The range cases are decided during argument handling, so they name a
 * checkpoint that does not exist and never open one. The two that must reach
 * the model serve the fixture, whose weights are random: what is asserted is
 * that the run happened and what it left behind, never what a token said.
 */

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

/// A checkpoint that cannot be opened, for the cases decided before the load.
/// Reaching the loader at all is a failure of the case, and exit 1 rather than
/// 2 is how that shows.
const NO_CHECKPOINT: &str = "/nonexistent-checkpoint";

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// Run the harness with `args` in a scratch working directory, as
/// cli_surface.rs does: `main.rs` opens `<cwd>/session.jsonl` on any
/// invocation that gets as far as a transcript, and the crate directory is
/// not the place for it.
fn run(args: &[&str]) -> Output {
    let cwd = tempfile::tempdir().expect("a scratch working directory");
    Command::new(env!("CARGO_BIN_EXE_harness"))
        .args(args)
        .current_dir(cwd.path())
        .stdin(Stdio::null())
        .output()
        .expect("the harness builds as a test binary")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

/// Every value outside the documented range is a usage error naming the flag
/// and the range, and none of them reaches the loader.
///
/// The non-numeric case is here rather than in a test of its own because it
/// takes the same branch: the parse and the bound are one match arm, and a
/// rewrite that split them would have to keep both answers.
#[test]
fn a_top_p_outside_the_unit_interval_is_refused_naming_the_range() {
    for value in ["0", "-0.5", "1.5", "nan", "half"] {
        let out = run(&["--model", NO_CHECKPOINT, "--top-p", value, "do x"]);
        assert_eq!(
            out.status.code(),
            Some(2),
            "--top-p {value} was not refused: {}",
            stderr(&out)
        );
        let err = stderr(&out);
        assert!(err.contains("--top-p"), "the error must name it: {err}");
        assert!(err.contains("(0, 1]"), "the error must state it: {err}");
        assert!(out.stdout.is_empty(), "a usage error is not payload");
    }
}

/// The other side of the bound: a mass inside it is accepted, and the closed
/// end is inside. Without this the test above passes for a parser that refuses
/// every value.
#[test]
fn a_top_p_inside_the_unit_interval_is_accepted_and_reaches_the_model() {
    for value in ["0.5", "1", "1.0"] {
        let out = run(&["--model", NO_CHECKPOINT, "--top-p", value, "do x"]);
        assert_ne!(
            out.status.code(),
            Some(2),
            "--top-p {value} was refused: {}",
            stderr(&out)
        );
    }
    // And a run that really samples with it completes, rather than the flag
    // being accepted by the parser and rejected further down.
    let model = fixture_model();
    let out = run(&[
        "--model",
        model.to_str().unwrap(),
        "--temperature",
        "1",
        "--top-p",
        "0.5",
        "--max-tokens",
        "1",
        "say ok",
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert!(
        !stderr(&out).contains("--top-p"),
        "a live nucleus is not worth a word: {}",
        stderr(&out)
    );
}

/// The same treatment --top-k already gets in cli_surface.rs. Greedy decoding
/// ignores the nucleus, and a script that sets sampling defaults uniformly and
/// then pins one run to temperature 0 has asked for something coherent -- but
/// silently is how it came to look like it had an effect.
#[test]
fn a_nucleus_under_greedy_decoding_warns_rather_than_refusing_the_run() {
    let out = run(&[
        "--model",
        NO_CHECKPOINT,
        "--temperature",
        "0",
        "--top-p",
        "0.5",
        "do x",
    ]);
    assert_ne!(
        out.status.code(),
        Some(2),
        "an inert flag is not a usage error"
    );
    let err = stderr(&out);
    assert!(err.contains("[WARN] --top-p"), "{err}");
    assert!(err.contains("no effect"), "{err}");
    assert!(out.stdout.is_empty(), "a warning is not payload");
}

#[test]
fn a_compact_at_outside_the_unit_interval_is_refused_naming_the_range() {
    for value in ["0", "-0.2", "1.5", "soon"] {
        let out = run(&["--model", NO_CHECKPOINT, "--compact-at", value, "do x"]);
        assert_eq!(
            out.status.code(),
            Some(2),
            "--compact-at {value} was not refused: {}",
            stderr(&out)
        );
        let err = stderr(&out);
        assert!(
            err.contains("--compact-at"),
            "the error must name it: {err}"
        );
        assert!(err.contains("(0, 1]"), "the error must state it: {err}");
    }
}

/// Drive a chat session over the fixture with `--compact-at threshold` and
/// return the transcript it wrote, one value per line.
///
/// Chat rather than a task because the automatic trigger reads the transcript,
/// and turns are what grow one. Six short lines clear both guards in
/// `compact::should_compact` -- more than KEEP_PREFIX + KEEP_LAST effective
/// entries, and at least KEEP_LAST of them since the last compaction -- while
/// staying far inside the fixture's 1024-token window, so the threshold is the
/// only thing left deciding the outcome. One token per reply keeps the run
/// short; the fixture's weights are random, so the replies are not read.
fn chat_transcript_at(threshold: &str) -> Vec<serde_json::Value> {
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_harness"))
        .current_dir(dir.path())
        .arg("--model")
        .arg(fixture_model())
        .arg("--chat")
        .arg("--compact-at")
        .arg(threshold)
        .arg("--max-tokens")
        .arg("1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn harness");
    {
        let mut stdin = child.stdin.take().expect("piped stdin");
        for line in ["one", "two", "three", "four", "five", "six"] {
            writeln!(stdin, "{line}").unwrap();
        }
        // Dropped here: end of input is what ends the chat loop, and the wait
        // below would otherwise never return.
    }
    let out = child.wait_with_output().expect("wait for harness");
    assert!(
        out.status.success(),
        "--compact-at {threshold} failed the chat: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let body = std::fs::read_to_string(dir.path().join("session.jsonl")).expect("a transcript");
    // The temporary directory drops here; the transcript has been read.
    body.lines()
        .map(|l| serde_json::from_str(l).expect("a transcript line is JSON"))
        .collect()
}

fn count_role(entries: &[serde_json::Value], role: &str) -> usize {
    entries.iter().filter(|e| e["role"] == role).count()
}

/// The threshold is a fraction of the context budget, so a low one fires on
/// every conversation and a high one on none. Both directions are asserted
/// from the same six turns: a compaction count that does not move with the
/// flag is a flag the parser read and the loop ignored, which is what having
/// no test for it could not distinguish from working.
#[test]
fn a_low_compact_at_threshold_compacts_the_conversation_and_a_high_one_does_not() {
    let eager = chat_transcript_at("0.001");
    let never = chat_transcript_at("1");
    assert!(
        count_role(&eager, "compaction") > 0,
        "a threshold of 0.001 of the budget compacted nothing: {eager:?}"
    );
    assert_eq!(
        count_role(&never, "compaction"),
        0,
        "a threshold of the whole budget compacted anyway: {never:?}"
    );
    // Same conversation either way: the difference above is the threshold and
    // not one run having ended early.
    assert_eq!(
        count_role(&eager, "user"),
        count_role(&never, "user"),
        "the two runs did not hold the same conversation"
    );
}
