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
