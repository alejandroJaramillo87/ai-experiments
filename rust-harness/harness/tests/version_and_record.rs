/*
 * version_and_record — the two harness flags the campaign ran by hand or
 * not at all, pinned through the binary.
 *
 * `--version` was asserted only to answer the same way as `-V`, which two
 * empty answers satisfy. `--record` was exercised in RH.2 and RH.5 on the
 * workstation, and `recording_replay.rs` reads a recording the workstation
 * wrote, so nothing in the gate ever produced one: a recorder that stopped
 * writing would fail the workstation's replay check and nothing else. Here
 * a task run over the fixture writes a recording and the replay reader reads
 * it back, one process apart.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use runtime::model::record::replay_file_with_selections;
use runtime::model::Model;

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// Run the harness with `args` in `dir`, the observability root under it.
fn run_in(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_harness"))
        .args(args)
        .current_dir(dir)
        .env("OBS_LOG_DIR", dir.join("obs"))
        .stdin(Stdio::null())
        .output()
        .expect("the harness builds as a test binary")
}

/// Both spellings print the crate's version on stdout, nothing on stderr,
/// exit 0, and open no run log: a version query is not a run, the same rule
/// `--help` follows.
#[test]
fn version_prints_the_crate_version_on_stdout_and_is_not_a_run() {
    for flag in ["--version", "-V"] {
        let dir = tempfile::tempdir().expect("a scratch working directory");
        let out = run_in(dir.path(), &[flag]);
        assert_eq!(
            out.status.code(),
            Some(0),
            "{flag}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            format!("harness {}\n", env!("CARGO_PKG_VERSION")),
            "{flag} printed something other than the version"
        );
        assert!(
            out.stderr.is_empty(),
            "{flag} wrote to stderr: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(!dir.path().join("obs").exists(), "{flag} opened a run log");
    }
}

/// A recorded task run replays as a script that covers every row a token
/// was drawn from, with the sampling parameters the command line set in its
/// header.
#[test]
fn record_writes_a_recording_the_replay_reader_reads_back() {
    // Held to the end: the recording is read back from it.
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let recording = dir.path().join("flight.jsonl");
    let model = fixture_model();
    let out = run_in(
        dir.path(),
        &[
            "--model",
            model.to_str().unwrap(),
            "--record",
            recording.to_str().unwrap(),
            "--temperature",
            "1",
            "--seed",
            "3",
            "--max-tokens",
            "4",
            "say hi",
        ],
    );
    assert!(
        out.status.success(),
        "the recorded run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let (script, recorded) = replay_file_with_selections(&recording)
        .unwrap_or_else(|e| panic!("{}: {e:#}", recording.display()));

    let sampling = recorded
        .sampling
        .expect("a recording of a sampled run names its sampling parameters");
    assert_eq!(sampling.temperature, 1.0, "{sampling:?}");
    assert_eq!(sampling.seed, 3, "{sampling:?}");
    assert!(!recorded.selections.is_empty(), "no token was selected");
    assert_eq!(
        script.vocab_size(),
        Model::vocab_size(&script),
        "the script's width disagrees with the model it replays as"
    );
    // Every selection drew from a row the recording scripted; a recorder that
    // wrote the header and no forward calls would leave the script empty.
    for s in &recorded.selections {
        assert!(
            s.position >= 1 && s.position - 1 < script.script_len(),
            "selection at {} drew from a row the recording did not script ({} rows)",
            s.position,
            script.script_len()
        );
    }
}
