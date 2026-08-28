/*
 * control_vector_range — --control-vector-layer-range through the binary.
 *
 * `ControlVector::from_image` drops out-of-range directions and its unit tests
 * say so. What nothing said is that the string on the command line reaches it:
 * interp/steering/wiring.py forwards CONTROL_VECTOR_LAYER_RANGE into this flag
 * unmodified, so a parse that produced a range nobody applied would steer every
 * block a vector names while the run reported the range it was given.
 *
 * The observable is the refusal. A vector holding one direction, restricted to
 * a range that excludes it, has nothing left to apply, and load fails naming
 * the range -- which cannot happen if the range was dropped on the way down.
 * The same vector under a range that includes it runs, so the refusal is the
 * range's doing and not the vector's.
 */

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4")
}

/// The served model's residual width. A direction of any other width is
/// refused by the width check, which would mask the range outcome under test.
fn model_width() -> usize {
    let config = std::fs::read_to_string(fixture_model().join("config.json")).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&config).unwrap();
    parsed["text_config"]["hidden_size"].as_u64().unwrap() as usize
}

/// A steering image holding exactly one direction, over `block`.
///
/// One rather than several so that a range excluding it leaves the image with
/// nothing to apply, which is the failure the assertions read. No manifest.json
/// beside it: the identity check then downgrades to a warning, and model
/// identity is probe_identity.rs's subject rather than this file's.
fn control_vector_over(dir: &Path, block: usize) -> PathBuf {
    let direction: Vec<f32> = (0..model_width()).map(|j| 0.01 * j as f32).collect();
    let path = dir.join("vector.safetensors");
    std::fs::write(
        &path,
        runtime::synth::build_f32_safetensors(&[(&format!("block.{block}"), &direction[..])])
            .unwrap(),
    )
    .unwrap();
    path
}

/// One task run steered by a vector over `block`, with `range` given when it
/// is `Some`.
fn run(block: usize, range: Option<&str>) -> Output {
    let dir = tempfile::tempdir().expect("a scratch working directory");
    let vector = control_vector_over(dir.path(), block);
    let mut command = Command::new(env!("CARGO_BIN_EXE_harness"));
    command
        .current_dir(dir.path())
        .arg("--model")
        .arg(fixture_model())
        .arg("--control-vector")
        .arg(&vector);
    if let Some(range) = range {
        command.arg("--control-vector-layer-range").arg(range);
    }
    command
        .arg("--max-tokens")
        .arg("1")
        .arg("say ok")
        .stdin(Stdio::null())
        .output()
        .expect("spawn harness")
    // The temporary directory drops here, after the child has been waited for.
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).to_string()
}

#[test]
fn a_direction_outside_the_layer_range_is_dropped_at_load() {
    // The control: the same vector with no range applies, so the refusal below
    // is attributable to the range rather than to the image.
    let unrestricted = run(1, None);
    assert!(
        unrestricted.status.success(),
        "the vector does not load without a range, so nothing here is about \
         the range: {}",
        stderr(&unrestricted)
    );

    let restricted = run(1, Some("0 0"));
    assert!(
        !restricted.status.success(),
        "block 1 survived a layer range of 0..=0: {}",
        stderr(&restricted)
    );
    let err = stderr(&restricted);
    assert!(
        err.contains("holds no block."),
        "the refusal must say the image had nothing left: {err}"
    );
    assert!(
        err.contains("layer range 0..=0"),
        "the refusal must name the range it applied: {err}"
    );
}

#[test]
fn a_direction_inside_the_layer_range_is_kept() {
    let out = run(1, Some("1 1"));
    assert!(
        out.status.success(),
        "block 1 was dropped by a layer range of 1..=1: {}",
        stderr(&out)
    );
    assert!(
        !stderr(&out).contains("holds no block."),
        "{}",
        stderr(&out)
    );
}

/// The flag takes one argument holding two numbers, because that is the shape
/// CONTROL_VECTOR_LAYER_RANGE already has. A malformed value is a usage error
/// rather than a range silently read as absent.
#[test]
fn a_malformed_layer_range_is_refused_naming_the_flag() {
    for value in ["1", "1 2 3", "one two", "3 1"] {
        let dir = tempfile::tempdir().unwrap();
        let vector = control_vector_over(dir.path(), 1);
        let out = Command::new(env!("CARGO_BIN_EXE_harness"))
            .current_dir(dir.path())
            .arg("--model")
            .arg(fixture_model())
            .arg("--control-vector")
            .arg(&vector)
            .arg("--control-vector-layer-range")
            .arg(value)
            .arg("say ok")
            .stdin(Stdio::null())
            .output()
            .expect("spawn harness");
        assert_eq!(
            out.status.code(),
            Some(2),
            "the range {value:?} was not refused: {}",
            stderr(&out)
        );
        assert!(
            stderr(&out).contains("--control-vector-layer-range"),
            "the error must name the flag: {}",
            stderr(&out)
        );
    }
}
