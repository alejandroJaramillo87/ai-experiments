/*
 * checkpoint_info_check — the --check flag's exit-code contract.
 *
 * checkpoint-info reports an unloadable checkpoint as a finding and exits 0 on
 * purpose, so an inspection run of a file the backend cannot load is not a
 * failed command. That leaves no way to gate a script on compatibility,
 * which --check supplies without disturbing the default.
 */

use std::path::{Path, PathBuf};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// (exit code, stdout) of a checkpoint-info run over the committed fixture.
fn run(args: &[&str]) -> (i32, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_checkpoint-info"))
        .args(args)
        .arg(fixture_model())
        .output()
        .expect("checkpoint-info builds as a test binary");
    (
        out.status.code().expect("not killed by a signal"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

#[test]
fn a_supported_model_exits_zero_with_or_without_the_check_flag() {
    // The fixture is loadable, so --check must not change its outcome; a
    // flag that failed on a good file would be worse than no flag.
    for args in [&[][..], &["--check"][..]] {
        let (code, stdout) = run(args);
        assert_eq!(code, 0, "args {args:?} stdout: {stdout}");
    }
}

#[test]
fn check_exits_nonzero_on_a_file_the_backend_cannot_load_but_zero_without_it() {
    // Well formed and unloadable: both files parse, so this exercises the
    // compatibility verdict rather than a read or JSON error, which already
    // exits 1. The activation is the cheapest such refusal -- Gemma uses the
    // tanh approximation of GELU and `TextConfig::validate` names any other by
    // field. Held to the end of the test so the directory outlives the runs.
    let dir = tempfile::tempdir().unwrap();
    let spec = runtime::synth::TinySpec::default();
    let mut config: serde_json::Value =
        serde_json::from_slice(&runtime::synth::build_tiny_config_json(&spec).unwrap()).unwrap();
    config["text_config"]["hidden_activation"] = serde_json::json!("silu");
    std::fs::write(
        dir.path().join(runtime::checkpoint::CONFIG_FILE),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join(runtime::checkpoint::WEIGHTS_FILE),
        runtime::synth::build_tiny_safetensors(&spec).unwrap(),
    )
    .unwrap();

    let run_on = |args: &[&str]| -> i32 {
        std::process::Command::new(env!("CARGO_BIN_EXE_checkpoint-info"))
            .args(args)
            .arg(dir.path())
            .output()
            .expect("checkpoint-info builds as a test binary")
            .status
            .code()
            .expect("not killed by a signal")
    };
    // The documented default: an unloadable checkpoint is a finding, and an
    // inspection run of one is not a failed command.
    assert_eq!(run_on(&[]), 0);
    assert_eq!(run_on(&["--check"]), 1);
}

#[test]
fn an_unreadable_argument_still_exits_two_rather_than_reporting_incompatibility() {
    // Usage errors and compatibility findings are different failures and must
    // not collapse onto one code once --check can also exit non-zero.
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_checkpoint-info"))
        .output()
        .expect("checkpoint-info builds as a test binary");
    assert_eq!(out.status.code(), Some(2));
}

/// A tiny checkpoint carrying two one-element tensors under tower prefixes,
/// written to `dir`.
///
/// The compatibility guard reads names and shapes, not values, so one element
/// per tower tensor is enough to make the file multimodal in its eyes. Built
/// from `tiny_tensors` rather than by patching a serialized image, because
/// safetensors offsets are computed at serialization; and every tensor is
/// written as F32, the loader's widest path, since dtype is not what this
/// file is for.
fn write_multimodal_checkpoint(dir: &Path) {
    use safetensors::tensor::{Dtype, TensorView};

    let spec = runtime::synth::TinySpec::default();
    let mut payloads: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for t in runtime::synth::tiny_tensors(&spec) {
        let Some(name) = t.hf_name.clone() else {
            continue;
        };
        // `values` carries Gemma's runtime `+1` for a norm; the file stores it
        // without, as `build_tiny_safetensors` does.
        let shift = if t.norm { 1.0f32 } else { 0.0 };
        let bytes = t
            .values
            .iter()
            .flat_map(|v| (v - shift).to_le_bytes())
            .collect();
        payloads.push((name, t.hf_shape(), bytes));
    }
    for name in [
        "model.audio_tower.layers.0.norm_out.weight",
        "model.vision_tower.layers.0.norm_out.weight",
    ] {
        payloads.push((name.to_string(), vec![1], 0.0f32.to_le_bytes().to_vec()));
    }
    let views: Vec<(&str, TensorView)> = payloads
        .iter()
        .map(|(n, shape, bytes)| {
            (
                n.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    std::fs::write(
        dir.join(runtime::checkpoint::CONFIG_FILE),
        runtime::synth::build_tiny_config_json(&spec).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join(runtime::checkpoint::WEIGHTS_FILE),
        safetensors::serialize(views, None).unwrap(),
    )
    .unwrap();
}

/// (exit code, stdout, stderr) of a checkpoint-info run over `path`.
fn run_at(args: &[&str], path: &Path) -> (i32, String, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_checkpoint-info"))
        .args(args)
        .arg(path)
        .output()
        .expect("checkpoint-info builds as a test binary");
    (
        out.status.code().expect("not killed by a signal"),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn check_tracks_the_multimodal_verdict_and_text_only_flips_it() {
    // The other refusal path. The activation test above fails in
    // `checkpoint::parse`, before the tool has a checkpoint to describe. A
    // multimodal file parses, is described in full, and is refused only at the
    // end by `check_supported_checkpoint` -- the path the released E4B takes,
    // and the one on which `--text-only` has to reach the verdict through the
    // binary's own wiring rather than the library's.
    let dir = tempfile::tempdir().unwrap();
    write_multimodal_checkpoint(dir.path());

    let (code, stdout, stderr) = run_at(&[], dir.path());
    assert_eq!(
        code, 0,
        "an inspection run is not a failed command: {stderr}"
    );
    assert!(
        stderr.contains("[ERROR] CPU backend compatibility: checkpoint is multimodal"),
        "{stderr}"
    );
    assert!(stderr.contains("--text-only"), "{stderr}");
    // The towers are named on stdout whether or not the caller declined them.
    assert!(stdout.contains("=== Non-text towers (2) ==="), "{stdout}");

    let (code, _, stderr) = run_at(&["--check"], dir.path());
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("checkpoint is multimodal"), "{stderr}");

    let (code, stdout, stderr) = run_at(&["--check", "--text-only"], dir.path());
    assert_eq!(code, 0, "{stderr}");
    assert!(
        stderr.contains("[OK] CPU backend compatibility: supported"),
        "{stderr}"
    );
    assert!(stdout.contains("=== Non-text towers (2) ==="), "{stdout}");

    let (code, _, stderr) = run_at(&["--text-only"], dir.path());
    assert_eq!(code, 0, "{stderr}");
}

#[test]
fn a_missing_checkpoint_is_a_failed_run_not_a_usage_error() {
    // The command line was answerable and the run failed: exit 1, the read
    // error naming the path, and the same with or without --check, because a
    // file that cannot be read has no verdict for the flag to turn into a code.
    let dir = tempfile::tempdir().unwrap();
    let absent = dir.path().join("absent");
    for args in [&[][..], &["--check"][..]] {
        let (code, _, stderr) = run_at(args, &absent);
        assert_eq!(code, 1, "args {args:?}: {stderr}");
        assert!(stderr.contains("[ERROR] Read "), "args {args:?}: {stderr}");
        assert!(
            stderr.contains("no checkpoint at"),
            "args {args:?}: {stderr}"
        );
    }
}
