/*
 * gguf_info_check — the --check flag's exit-code contract.
 *
 * gguf-info reports an unloadable checkpoint as a finding and exits 0 on
 * purpose, so an inspection run of a file the backend cannot load is not a
 * failed command. That leaves no way to gate a script on compatibility,
 * which --check supplies without disturbing the default.
 */

use std::path::{Path, PathBuf};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4.gguf")
}

/// (exit code, stdout) of a gguf-info run over the committed fixture.
fn run(args: &[&str]) -> (i32, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_gguf-info"))
        .args(args)
        .arg(fixture_model())
        .output()
        .expect("gguf-info builds as a test binary");
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
    // Parseable and unloadable: the container is well formed, so this
    // exercises the compatibility verdict rather than the parse error, which
    // already exits 1. Held to the end of the test so the file outlives the
    // runs below.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("not-a-gemma.gguf");
    let mut w = runtime::synth::GgufWriter::new();
    w.kv_str("general.architecture", "llama");
    std::fs::write(&path, w.finish()).unwrap();

    let run_on = |args: &[&str]| -> i32 {
        std::process::Command::new(env!("CARGO_BIN_EXE_gguf-info"))
            .args(args)
            .arg(&path)
            .output()
            .expect("gguf-info builds as a test binary")
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
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_gguf-info"))
        .output()
        .expect("gguf-info builds as a test binary");
    assert_eq!(out.status.code(), Some(2));
}
