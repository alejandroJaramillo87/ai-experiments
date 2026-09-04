/*
 * cli_surface — what the binary does with its arguments, through a pipe.
 *
 * The unit tests in `harness::cli` check the table against itself. What they
 * cannot check is the part that only exists once there is a process: which
 * stream each answer goes to, and what the exit code is. Both matter here for
 * the same reason RH.12 matters -- stdout is the payload channel, so a status
 * line or an escape sequence arriving there corrupts a machine reader -- and a
 * spawned child's streams are pipes, which is the shape a script sees.
 *
 * These spawns need no model: every case is decided during argument handling,
 * before a checkpoint is opened. That is itself a property worth holding, since
 * it is what lets `--help` work on a machine with no weights.
 */

use std::process::{Command, Output};

/// Run the harness with `args` and capture both streams.
///
/// Spawned in a temporary directory, not the crate's. `main.rs` opens
/// `<cwd>/session.jsonl` on any invocation that gets as far as a transcript,
/// and cargo runs an integration test with the crate directory as cwd, so
/// without this the suite drops a stray file into the worktree -- which
/// `rust-harness/.gitignore` explicitly says is the real binary's doing and
/// never a test's. The directory is removed when it goes out of scope, after
/// `output()` has waited for the child.
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

/// The failure this stage exists to remove.
///
/// `harness --modle foo "do x"` used to exit 0 having run a task named
/// "--modle foo do x": the flag was not recognized, so it fell through to the
/// task text, and nothing ever asked whether it was meant to be a flag.
#[test]
fn a_misspelled_flag_exits_two_and_writes_nothing_to_stdout() {
    let out = run(&["--modle", "foo", "do x"]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    assert!(
        out.stdout.is_empty(),
        "a usage error is not payload: {:?}",
        stdout(&out)
    );
    let err = stderr(&out);
    assert!(err.contains("--modle"), "the error must name it: {err}");
    assert!(err.contains("usage: harness"), "{err}");
}

/// Requested help is this invocation's payload, so it goes to stdout at exit 0
/// -- the same split argparse makes, which is what the Python tiers do.
#[test]
fn help_goes_to_stdout_at_exit_zero_with_nothing_on_stderr() {
    let out = run(&["--help"]);
    assert_eq!(out.status.code(), Some(0));
    assert!(
        out.stderr.is_empty(),
        "help is not a diagnostic: {:?}",
        stderr(&out)
    );
    let text = stdout(&out);
    assert!(text.contains("usage: harness"), "{text}");
    // Colour is display only and never reaches stdout; through a pipe there is
    // nothing to colour anyway, which is exactly the case a script sees.
    assert!(!text.contains('\x1b'), "help carries escape bytes");
}

/// One test discharges the "every capability claim is traceable" lens for the
/// flag surface: if help names every flag the parser accepts, no document has
/// to repeat the list to be complete.
///
/// The loops below are measured against the table, so on their own they cannot
/// notice a row deleted from it -- they would simply check one thing fewer. A
/// mutation run confirmed exactly that. The literal lists are the independent
/// anchor: they are written here, not derived, so removing a flag or a binary
/// fails this test rather than shrinking it.
#[test]
fn help_names_every_flag_the_parser_accepts_and_every_binary_shipped() {
    let text = stdout(&run(&["--help"]));
    for flag in harness::cli::FLAGS {
        assert!(text.contains(flag.name), "help omits {}", flag.name);
    }
    for (name, _) in harness::cli::BINARIES {
        assert!(text.contains(name), "help omits the {name} binary");
    }
    for name in [
        "--model",
        "--chat",
        "--serve",
        "--single-thread",
        "--text-only",
        "--control-vector-layer-range",
        "--interp-log",
        "--help",
        "-V",
    ] {
        assert!(text.contains(name), "help omits {name}");
    }
    for name in [
        "harness",
        "checkpoint-info",
        "activation-capture",
        "attention-capture",
        "ladder",
        "layer-diff",
        "long-context-golden",
        "tiny-model",
    ] {
        assert!(text.contains(name), "help omits the {name} binary");
    }
}

#[test]
fn the_short_forms_answer_the_same_way() {
    assert_eq!(stdout(&run(&["-h"])), stdout(&run(&["--help"])));
    let version = run(&["-V"]);
    assert_eq!(version.status.code(), Some(0));
    assert!(version.stderr.is_empty());
    assert_eq!(stdout(&version), stdout(&run(&["--version"])));
    assert!(
        stdout(&version).starts_with("harness "),
        "{:?}",
        stdout(&version)
    );
}

/// Taking the first occurrence and leaking the rest into the task text is what
/// this used to do, so `--seed 1 --seed 2 "do x"` ran seed 1 on a task named
/// "--seed 2 do x". The obs log records argv as given, so a run whose record
/// shows two seeds must not have quietly used one.
#[test]
fn a_repeated_flag_exits_two_naming_the_flag() {
    let out = run(&["--seed", "1", "--seed", "2", "do x"]);
    assert_eq!(out.status.code(), Some(2));
    let err = stderr(&out);
    assert!(err.contains("--seed"), "{err}");
    assert!(err.contains("more than once"), "{err}");
}

/// The escape hatch that makes rejecting unknown flags safe: without it there
/// is no way to write a task mentioning a flag except by quoting, and quoting
/// is easy to lose through a `make ARGS=` passthrough.
#[test]
fn the_separator_makes_a_dash_leading_word_ordinary_task_text() {
    // Reaching the model load is the assertion: the words after `--` were
    // taken as a task rather than refused as flags. Exit 1 is a run that
    // started and failed, which is a different outcome from exit 2.
    let out = run(&[
        "--model",
        "/nonexistent-checkpoint",
        "--",
        "explain",
        "--seed",
    ]);
    assert_ne!(
        out.status.code(),
        Some(2),
        "the separator must suppress flag checking: {}",
        stderr(&out)
    );
    assert!(
        stderr(&out).contains("nonexistent-checkpoint"),
        "{}",
        stderr(&out)
    );
}

/// A flag whose companion is absent used to be accepted and ignored, so a
/// steering run that forgot the vector looked like it had steered.
#[test]
fn a_steering_scale_without_its_vector_exits_two() {
    let out = run(&[
        "--model",
        "/nonexistent-checkpoint",
        "--control-vector-scale",
        "2",
        "do x",
    ]);
    assert_eq!(out.status.code(), Some(2), "{}", stderr(&out));
    let err = stderr(&out);
    assert!(err.contains("--control-vector-scale"), "{err}");
    assert!(err.contains("--control-vector"), "{err}");
}

/// The one requirement that was hand-written before the table generalized it.
/// `serve_interp_log.rs` pins the same case; this asserts the message did not
/// change wording when the mechanism did.
#[test]
fn the_bind_address_still_requires_a_server_in_the_same_words() {
    let out = run(&[
        "--model",
        "/nonexistent-checkpoint",
        "--serve-addr",
        "0.0.0.0",
        "say hello",
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(
        stderr(&out).contains("--serve-addr has no meaning without --serve"),
        "{}",
        stderr(&out)
    );
}

/// The trace level is a property of a trace, so without one it is a usage
/// error through the same table edge, and a spelling the dial does not have
/// is refused naming the three it does.
#[test]
fn trace_level_without_a_trace_is_a_usage_error_and_a_typo_names_the_levels() {
    let out = run(&[
        "--model",
        "/nonexistent-checkpoint",
        "--trace-level",
        "debug",
        "say hello",
    ]);
    assert_eq!(out.status.code(), Some(2));
    assert!(
        stderr(&out).contains("--trace-level has no meaning without --trace-chrome"),
        "{}",
        stderr(&out)
    );

    let dir = tempfile::tempdir().unwrap();
    let trace = dir.path().join("trace.json");
    let out = run(&[
        "--model",
        "/nonexistent-checkpoint",
        "--trace-chrome",
        trace.to_str().unwrap(),
        "--trace-level",
        "verbose",
        "say hello",
    ]);
    assert_eq!(out.status.code(), Some(2));
    let err = stderr(&out);
    assert!(
        err.contains("--trace-level: verbose is not a level"),
        "{err}"
    );
    assert!(err.contains("info, debug or trace"), "{err}");
    assert!(!trace.exists(), "a usage error installs no subscriber");
}

/// Inert is not an error: a script that sets sampling defaults uniformly and
/// then pins one run to greedy asked for something coherent. It should be
/// audible, though, which is what this asserts.
#[test]
fn cutoffs_under_greedy_decoding_warn_rather_than_refusing_the_run() {
    let out = run(&[
        "--model",
        "/nonexistent-checkpoint",
        "--temperature",
        "0",
        "--top-k",
        "5",
        "do x",
    ]);
    assert_ne!(
        out.status.code(),
        Some(2),
        "an inert flag is not a usage error"
    );
    let err = stderr(&out);
    assert!(err.contains("[WARN] --top-k"), "{err}");
    assert!(err.contains("no effect"), "{err}");
    assert!(out.stdout.is_empty(), "a warning is not payload");
}

/// Every case above is decided before a checkpoint is opened, which is what
/// lets help work on a machine with no weights. Asserted directly so the
/// property is not merely incidental to the current ordering.
#[test]
fn the_argument_surface_is_decided_without_opening_a_checkpoint() {
    for args in [
        vec!["--help"],
        vec!["-V"],
        vec!["--modle", "x"],
        vec!["--seed", "1", "--seed", "2", "t"],
    ] {
        let out = run(&args);
        assert!(
            !stderr(&out).contains("no checkpoint at"),
            "{args:?} reached the loader: {}",
            stderr(&out)
        );
    }
}
