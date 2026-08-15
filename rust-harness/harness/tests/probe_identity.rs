/*
 * probe_identity — end-to-end coverage of the probe/model identity refusal.
 *
 * A probe fitted on other weights scores the served residual stream into
 * meaningless flags, so the harness refuses the pairing at load time; the
 * "unknown" sentinel interp/probes/capture.py writes by default must keep
 * loading, or every Python-captured probe stops working. The check sits on
 * the --probe path that only main() reaches, so the binary is spawned
 * rather than the loader called: what is under test is the refusal a user
 * actually gets. Probe GGUFs are built in-test in the export.py layout, and
 * the served digest comes from python3's hashlib, not a second copy of
 * SHA-256 — an independent digest is what makes the comparison meaningful.
 */

use std::io::Write;
use std::path::{Path, PathBuf};

/// Prefix of the startup line main.rs prints once the model is loaded and the
/// cache is sized. Two tests depend on it in opposite directions -- one that
/// a successful run emits it, one that a refusal happens before it -- so it
/// is named here rather than spelled twice.
const KV_CACHE_LINE: &str = "[INFO] KV cache:";

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// The bytes the harness hashes for model identity: the weights alone, not
/// the whole checkpoint directory. An artifact is fitted on weights, and
/// config.json changing should not invalidate a probe.
fn fixture_weights() -> PathBuf {
    fixture_model().join(runtime::checkpoint::WEIGHTS_FILE)
}

fn sha256_hex(bytes: &[u8]) -> String {
    // hashlib is stdlib, so any working interpreter serves; resolution is still
    // routed through pyenv so this site cannot drift from the other three.
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = runtime::pyenv::interpreter_or_panic(&repo_root, &[]);
    let mut child = std::process::Command::new(&python)
        .args([
            "-c",
            "import hashlib, sys\n\
             sys.stdout.write(hashlib.sha256(sys.stdin.buffer.read()).hexdigest())\n",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("the resolved python3 runs");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(bytes)
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "python3 hashlib failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// The served model's embedding width. A probe of any other width is
/// refused by the width guard, which would mask the identity outcome the
/// tests below are reading.
fn model_width() -> usize {
    let files = runtime::checkpoint::CheckpointFiles::read(&fixture_model()).unwrap();
    runtime::checkpoint::parse(&files)
        .unwrap()
        .config
        .hidden_size
}

/// A loadable probe over layer 0 recording `model_hash`, written into `dir`
/// as the safetensors image plus its JSON sidecar. The threshold is above
/// every attainable sigmoid score, so an accepted run flags nothing and its
/// report stays a fixed shape whatever the random weights produce.
fn write_probe(dir: &std::path::Path, model_hash: &str) -> PathBuf {
    let width = model_width();
    let weights: Vec<f32> = (0..width).map(|j| 0.1 * (j as f32 - 4.0)).collect();
    let mu: Vec<f32> = (0..width).map(|j| 0.01 * j as f32).collect();
    let sigma: Vec<f32> = (0..width).map(|j| 1.0 + 0.1 * j as f32).collect();
    let path = dir.join("probe.safetensors");
    std::fs::write(
        &path,
        runtime::synth::build_f32_safetensors(&[
            ("weights", &weights),
            ("mu", &mu),
            ("sigma", &sigma),
        ])
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        path.with_extension("json"),
        format!(
            r#"{{"schema_version": 1, "artifact": "probe",
               "timestamp": "2026-01-01T00:00:00+0000", "git_sha": "abcdef0",
               "name": "identity-probe", "model_hash": "{model_hash}",
               "layer": 0, "bias": 0.1, "threshold": 1.0, "width": {width}}}"#
        ),
    )
    .unwrap();
    path
}

/// A finished harness run. The TempDir is held so the working directory
/// outlives the spawn and is removed when the test ends.
struct Run {
    root: PathBuf,
    _dir: tempfile::TempDir,
    output: std::process::Output,
}

impl Run {
    fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.output.stderr).to_string()
    }
}

/// Run the harness over the fixture model with a probe recording
/// `model_hash` as --probe, in a fresh working directory. One token of a
/// trivial task keeps an accepted run short; stdin is closed because a
/// missing permissions.yaml routes any tool call to the interactive prompt,
/// which must not block the test.
fn run_with_probe(model_hash: &str) -> Run {
    run_with_probe_args(model_hash, &[])
}

/// `run_with_probe` with `extra` appended to the argument list, for the cases
/// that need a flag the identity tests do not.
fn run_with_probe_args(model_hash: &str, extra: &[&std::ffi::OsStr]) -> Run {
    let dir = tempfile::tempdir().unwrap();
    let probe = write_probe(dir.path(), model_hash);
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_harness"))
        .current_dir(dir.path())
        .arg("--model")
        .arg(fixture_model())
        .arg("--probe")
        .arg(&probe)
        .args(extra)
        .arg("--max-tokens")
        .arg("1")
        .arg("say ok")
        .stdin(std::process::Stdio::null())
        .output()
        .expect("spawn harness");
    Run {
        root: dir.path().to_path_buf(),
        _dir: dir,
        output,
    }
}

#[test]
fn a_probe_recording_the_served_models_sha256_loads_and_the_run_completes() {
    let served = sha256_hex(&std::fs::read(fixture_weights()).unwrap());
    let run = run_with_probe(&served);
    let err = run.stderr();
    assert!(run.output.status.success(), "{err}");
    // The matching case is silent: neither refused nor downgraded to the
    // skipped check, and the monitor reported, so the probe really ran.
    assert!(!err.contains("trained on a different model"), "{err}");
    assert!(!err.contains("check skipped"), "{err}");
    assert!(err.contains("[INFO] Probe identity-probe:"), "{err}");
    // Anchors the literal the refusal test below asserts the *absence* of.
    // Without a run that proves the line is printed at all, a rename in
    // main.rs would leave that negative assertion passing vacuously.
    assert!(err.contains(KV_CACHE_LINE), "{err}");
    assert!(run.root.join("probe-flags.jsonl").is_file());
}

#[test]
fn a_probe_recording_a_different_model_hash_is_refused_naming_both_hashes() {
    // A well-formed digest of other bytes: the refusal has to hold for a
    // hash that looks exactly like a real one, not just for a malformed
    // string any comparison would reject.
    let wrong = sha256_hex(b"weights the probe was fitted on");
    let run = run_with_probe(&wrong);
    let err = run.stderr();
    assert!(!run.output.status.success(), "{err}");
    let served = sha256_hex(&std::fs::read(fixture_weights()).unwrap());
    assert!(err.contains(&wrong), "refusal omits the probe hash: {err}");
    assert!(
        err.contains(&served),
        "refusal omits the served hash: {err}"
    );
    // Refused at load time: the KV cache line is printed further down
    // build_generator, so its absence pins the refusal ahead of session
    // construction, and no generation ran to leave a snapshot or flags.
    assert!(!err.contains(KV_CACHE_LINE), "{err}");
    assert!(!run.root.join("session.kv").exists());
    assert!(!run.root.join("probe-flags.jsonl").exists());
}

/// The default path is asserted three times above, so the flag that overrides
/// it is the untested half: a run that parsed `--probe-out` and wrote to the
/// default anyway would leave every one of them passing. Both directions are
/// asserted here, because the file appearing where it was asked for says
/// nothing on its own if a copy also appears where it was not.
#[test]
fn probe_out_puts_the_flag_file_where_it_names_and_leaves_the_default_unwritten() {
    let dir = tempfile::tempdir().unwrap();
    let named = dir.path().join("named-flags.jsonl");
    let run = run_with_probe_args("unknown", &["--probe-out".as_ref(), named.as_os_str()]);
    let err = run.stderr();
    assert!(run.output.status.success(), "{err}");
    assert!(
        named.is_file(),
        "--probe-out named {} and nothing was written there: {err}",
        named.display()
    );
    assert!(
        !run.root.join("probe-flags.jsonl").exists(),
        "the default path was written as well as {}",
        named.display()
    );
    // The report names the file it wrote, so a script reading stderr is
    // pointed at the same path the flag asked for.
    assert!(err.contains(&named.display().to_string()), "{err}");
    // The directory is removed here, after the run has been read.
    drop(dir);
}

#[test]
fn a_probe_recording_the_unknown_sentinel_loads_with_the_check_skipped() {
    // capture.py defaults model_hash to "unknown", so refusing it would
    // reject every probe the Python path exports.
    let run = run_with_probe("unknown");
    let err = run.stderr();
    assert!(run.output.status.success(), "{err}");
    assert!(!err.contains("trained on a different model"), "{err}");
    assert!(err.contains("check skipped"), "{err}");
    assert!(err.contains("[INFO] Probe identity-probe:"), "{err}");
    assert!(run.root.join("probe-flags.jsonl").is_file());
}
