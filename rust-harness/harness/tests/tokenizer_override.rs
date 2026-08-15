/*
 * tokenizer_override — --tokenizer against the tokenizer beside the checkpoint.
 *
 * `resolve_tokenizer`'s unit tests check the candidate walk in-process. What
 * they cannot check is that the flag is passed down to it: the only mention of
 * --tokenizer outside the flag table is a runbook sentence saying it is not
 * passed, so a run that dropped the argument between the parser and the
 * resolver would look exactly like this repository does today.
 *
 * Precedence is the property, and it needs a checkpoint whose own
 * tokenizer.json is unusable: a run that ignores the flag then fails on the
 * file beside the weights, and a run that honours it succeeds. Asserting only
 * the success case would pass for a resolver that never read the flag, since
 * the fixture's sibling tokenizer would have served either way.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
}

/// The tokenizer the fixture checkpoint resolves to when nothing overrides it,
/// as a sibling `<model>.tokenizer.json`.
fn fixture_tokenizer() -> PathBuf {
    fixtures().join("tiny-gemma4.tokenizer.json")
}

/// A checkpoint directory holding the fixture weights and a tokenizer.json
/// that cannot be parsed, returned with the guard that owns it.
///
/// Copied rather than linked, and only 40 KiB of it: the point is a directory
/// whose own tokenizer.json is the first candidate `resolve_tokenizer` finds,
/// which is what makes an ignored --tokenizer visible.
fn checkpoint_with_an_unusable_tokenizer() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("checkpoint");
    std::fs::create_dir(&model).unwrap();
    for name in ["config.json", "model.safetensors"] {
        std::fs::copy(fixtures().join("tiny-gemma4").join(name), model.join(name)).unwrap();
    }
    std::fs::write(model.join("tokenizer.json"), b"{\"not\": \"a tokenizer\"}").unwrap();
    (dir, model)
}

/// One task run over `model`, with `tokenizer` given when it is `Some`.
fn run(model: &Path, tokenizer: Option<&Path>) -> Output {
    let cwd = tempfile::tempdir().expect("a scratch working directory");
    let mut command = Command::new(env!("CARGO_BIN_EXE_harness"));
    command.current_dir(cwd.path()).arg("--model").arg(model);
    if let Some(path) = tokenizer {
        command.arg("--tokenizer").arg(path);
    }
    command
        .arg("--max-tokens")
        .arg("1")
        .arg("say ok")
        .stdin(Stdio::null())
        .output()
        .expect("spawn harness")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

#[test]
fn the_tokenizer_flag_overrides_the_one_beside_the_checkpoint() {
    let (_dir, model) = checkpoint_with_an_unusable_tokenizer();

    // The control. Without the flag the checkpoint's own tokenizer.json is
    // what gets loaded, and it does not parse.
    let without = run(&model, None);
    assert!(
        !without.status.success(),
        "the unusable tokenizer beside the checkpoint was not loaded, so this \
         test cannot distinguish an honoured flag from an ignored one"
    );
    let err = stderr(&without);
    assert!(
        err.contains("tokenizer.json"),
        "the failure must name the file it could not load: {err}"
    );

    // The same checkpoint, with a working tokenizer named. The run completes,
    // so the flag took precedence over the directory's own.
    let with = run(&model, Some(&fixture_tokenizer()));
    assert!(
        with.status.success(),
        "--tokenizer {} did not override the checkpoint's own: {}",
        fixture_tokenizer().display(),
        stderr(&with)
    );
    // The temporary checkpoint drops here, after both runs have finished.
}

/// The flag names a file, not a preference: falling back to the sibling
/// tokenizer when the named one is absent would silently tokenize with
/// something other than what was asked for, which is the failure mode a
/// checkpoint carrying its own vocabulary makes hard to notice.
#[test]
fn a_tokenizer_that_does_not_exist_is_refused_rather_than_resolved_around() {
    let model = fixtures().join("tiny-gemma4");
    let missing = fixtures().join("no-such.tokenizer.json");
    let out = run(&model, Some(&missing));
    assert!(
        !out.status.success(),
        "a missing --tokenizer fell back to {}",
        fixture_tokenizer().display()
    );
    let err = stderr(&out);
    assert!(err.contains("--tokenizer"), "the error must name it: {err}");
    assert!(
        err.contains("no-such.tokenizer.json"),
        "the error must name the path: {err}"
    );
}
