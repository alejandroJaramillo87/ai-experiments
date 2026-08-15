/*
 * binary_cli_surface — the seven binaries' command line, asserted from outside.
 *
 * `runtime::cli` is unit-tested against a synthetic `Spec`, which says the
 * mechanism works and nothing about whether a given program uses it. These are
 * the assertions that only a spawned process can make: that `--help` is
 * answered at all, that it reaches stdout and leaves stderr clean, that an
 * unrecognized flag exits 2, and -- for the two fixture writers -- that asking
 * for help writes nothing.
 *
 * That last one is the reason this file exists. `tiny-model` and
 * `long-context-golden` take their output path positionally and neither looked
 * for `-h` first, so `tiny-model -h` wrote `-h.tokenizer.json` plus a directory
 * named `-h` holding `config.json` and `model.safetensors`. An exit-code test
 * cannot see that: a help request that exits 0 and still writes a file passes
 * it. The assertion has to be on the directory.
 *
 * `harness/tests/cli_surface.rs` is the same shape for the eighth binary.
 */

use std::path::Path;
use std::process::{Command, Output};

/// Every binary in `runtime/src/bin/`, with the path cargo built it at.
///
/// `env!` needs a literal, so the seven are named rather than discovered. A
/// new binary that is not added here is not covered, which is the cost of the
/// macro's shape; `cli::BINARIES` in the harness is the roster that would
/// notice, and it is asserted separately.
const BINARIES: [(&str, &str); 7] = [
    ("tiny-model", env!("CARGO_BIN_EXE_tiny-model")),
    ("ladder", env!("CARGO_BIN_EXE_ladder")),
    ("checkpoint-info", env!("CARGO_BIN_EXE_checkpoint-info")),
    ("layer-diff", env!("CARGO_BIN_EXE_layer-diff")),
    (
        "activation-capture",
        env!("CARGO_BIN_EXE_activation-capture"),
    ),
    ("attention-capture", env!("CARGO_BIN_EXE_attention-capture")),
    (
        "long-context-golden",
        env!("CARGO_BIN_EXE_long-context-golden"),
    ),
];

/// Run a binary in a scratch directory, so a program that writes relative to
/// the working directory cannot leave anything in the worktree.
fn run_in(dir: &Path, exe: &str, args: &[&str]) -> Output {
    Command::new(exe)
        .args(args)
        .current_dir(dir)
        .output()
        .expect("the binary builds as a test binary")
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(out: &Output) -> String {
    String::from_utf8(out.stderr.clone()).expect("stderr is UTF-8")
}

/// Requested help is the answer to the question asked, so it is payload: stdout,
/// exit 0, stderr untouched. An *unrequested* usage block is a diagnostic and
/// goes the other way, which the next test pins.
#[test]
fn every_binary_answers_help_on_stdout_and_says_nothing_on_stderr() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    for (name, exe) in BINARIES {
        for spelling in ["--help", "-h"] {
            let out = run_in(dir.path(), exe, &[spelling]);
            assert_eq!(
                out.status.code(),
                Some(0),
                "{name} {spelling}: {}",
                stderr(&out)
            );
            let text = stdout(&out);
            assert!(
                text.starts_with(&format!("usage: {name}")),
                "{name} {spelling}: {text:?}"
            );
            assert!(
                out.stderr.is_empty(),
                "{name} {spelling}: help is not a diagnostic: {:?}",
                stderr(&out)
            );
            // Colour is display only and never reaches a pipe, on either
            // stream. `grep` over help output has to work.
            assert!(!text.contains('\x1b'), "{name} {spelling}: {text:?}");
        }
    }
}

/// An unrecognized flag is a usage mistake, not a positional and not a
/// no-op. Four of the seven used to ignore one silently or absorb it as an
/// argument; two of those then reported "missing required flag" at exit 1,
/// naming neither the absent flag nor the misspelling that caused it.
#[test]
fn every_binary_refuses_an_unknown_flag_at_exit_two_and_names_it() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    for (name, exe) in BINARIES {
        let out = run_in(dir.path(), exe, &["--definitely-not-a-flag"]);
        assert_eq!(
            out.status.code(),
            Some(2),
            "{name}: a usage mistake is exit 2, apart from a real failure: {}",
            stderr(&out)
        );
        assert!(
            out.stdout.is_empty(),
            "{name}: a usage error is not payload: {:?}",
            stdout(&out)
        );
        let err = stderr(&out);
        assert!(
            err.contains("--definitely-not-a-flag"),
            "{name}: the offending argument is not named: {err}"
        );
        assert!(err.contains("[ERROR]"), "{name}: untagged: {err}");
    }
}

/// The two fixture writers take their destination positionally, so until U3b a
/// help request was an output path and they acted on it. This is the assertion
/// that the question no longer produces an artifact named after itself.
#[test]
fn asking_a_fixture_writer_for_help_writes_nothing() {
    for name in ["tiny-model", "long-context-golden"] {
        let exe = BINARIES
            .iter()
            .find(|(n, _)| *n == name)
            .expect("named above")
            .1;
        for spelling in ["--help", "-h"] {
            let dir = tempfile::tempdir().expect("a scratch working directory");
            let out = run_in(dir.path(), exe, &[spelling]);
            assert_eq!(out.status.code(), Some(0), "{name} {spelling}");
            let left: Vec<String> = std::fs::read_dir(dir.path())
                .expect("the scratch directory is readable")
                .map(|e| {
                    e.expect("a readable entry")
                        .file_name()
                        .to_string_lossy()
                        .into_owned()
                })
                .collect();
            assert!(
                left.is_empty(),
                "{name} {spelling} wrote {left:?} -- a help request is not a run"
            );
        }
    }
}

/// `ladder` is the one instrumented binary of the seven, and help must not
/// reach `RunLog::begin`. Same rule as `harness --help`: a query is not a run,
/// so it leaves no row in the catalog for someone to find later and wonder
/// about. True by the ordering of the intercept, and asserted rather than left
/// to it.
#[test]
fn ladder_help_writes_no_run_record() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let logs = tempfile::tempdir().expect("a scratch log root");
    let out = Command::new(env!("CARGO_BIN_EXE_ladder"))
        .arg("--help")
        .env("OBS_LOG_DIR", logs.path())
        .current_dir(dir.path())
        .output()
        .expect("ladder builds as a test binary");
    assert_eq!(out.status.code(), Some(0), "{}", stderr(&out));
    assert!(
        !logs.path().join("harness").exists(),
        "help opened a run log: {:?}",
        std::fs::read_dir(logs.path()).map(|d| d
            .filter_map(|e| e.ok().map(|e| e.file_name()))
            .collect::<Vec<_>>())
    );
}

/// The exit codes mean three different things and must not collapse: 2 for a
/// command line that was wrong to ask, 1 for a run that was asked for correctly
/// and failed. `checkpoint_info_check.rs` pins the no-argument case at 2 for
/// `checkpoint-info`; this is the same distinction across the binaries whose
/// usage errors used to return 1.
#[test]
fn a_capture_binary_separates_a_usage_mistake_from_a_failed_run() {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    for (name, exe) in BINARIES
        .iter()
        .filter(|(n, _)| n.ends_with("-capture"))
        .copied()
    {
        // Nothing given at all: a usage mistake, and the message names a flag
        // rather than reporting that something unspecified was absent.
        let out = run_in(dir.path(), exe, &[]);
        assert_eq!(out.status.code(), Some(2), "{name}: {}", stderr(&out));
        assert!(stderr(&out).contains("--model"), "{name}: {}", stderr(&out));

        // Every flag given, and the checkpoint does not exist: the command line
        // was answerable and the run failed, which is exit 1.
        let args: Vec<&str> = match name {
            "activation-capture" => vec![
                "--model",
                "/nonexistent",
                "--dataset",
                "d.jsonl",
                "--layers",
                "0",
                "--out",
                "o",
            ],
            _ => vec![
                "--model",
                "/nonexistent",
                "--prompt",
                "hi",
                "--layers",
                "0",
                "--heads",
                "0",
                "--out",
                "o",
            ],
        };
        let out = run_in(dir.path(), exe, &args);
        assert_eq!(
            out.status.code(),
            Some(1),
            "{name}: a failed capture is not a usage error: {}",
            stderr(&out)
        );
    }
}
