/*
 * python_exporter_contract — the real Python interp exporters against the
 * Rust loaders.
 *
 * Two GGUF layouts are written only by Python and read only by Rust: the
 * probe image (interp/probes/export.py) and the control-vector image
 * (interp/steering/export.py). Every other golden synthesizes those bytes
 * with runtime::synth::GgufWriter, which checks the parser against a second
 * Rust implementation rather than against the writer it exists to read. The
 * layer numbering makes that gap concrete: probe.layer is 0-based and
 * direction.<N> is 1-based, each side enforces its own rule, and nothing
 * checked that the two rules compose.
 *
 * Coverage is in two layers. The committed fixtures under fixtures/ are
 * exporter output, so the parse assertions run everywhere the workspace
 * builds. The regeneration test then re-runs the exporters and compares,
 * which is what notices an exporter that has drifted away from the committed
 * bytes; it needs python3 with numpy and gguf plus the parent tree, and
 * reports [SKIP] without them — as in the container build, whose context is
 * rust-harness/ alone.
 *
 * Contract: schemas/gguf_artifacts.json. Regenerate the fixtures with
 * `python3 rust-harness/fixtures/export_gguf_fixtures.py rust-harness/fixtures`.
 */

use std::path::{Path, PathBuf};

use runtime::gguf::{parse, GgufFile, MetaValue};
use runtime::model::cpu::ControlVector;
use runtime::model::probe::Probe;

/// Expectations mirroring fixtures/export_gguf_fixtures.py. Duplicated rather
/// than derived: a fixture regenerated from changed constants should fail
/// here until someone updates both, which is the point of a golden.
const PROBE_NAME: &str = "contract-fixture-probe";
const PROBE_MODEL_HASH: &str = "0123456789abcdef";
const PROBE_LAYER: usize = 2;
const PROBE_THRESHOLD: f32 = 0.25;
const PROBE_WIDTH: usize = 4;

/// Residual row scored below, and the score it must produce.
///
/// Worked by hand from the fixture's weights, mu, sigma, and bias:
/// (1-0)/1*0.5 + (2-1)/0.5*-1.5 + (0+2)/2*2.25 + (5-4)/0.25*-0.125
/// = 0.5 - 3.0 + 2.25 - 0.5 = -0.75, plus bias 0.5 gives logit -0.25, and
/// sigmoid(-0.25) = 1/(1+e^0.25). Scoring rather than reading the tensors
/// back covers weights, mu, sigma, and bias in one assertion, and catches a
/// permuted tensor that a per-name existence check would not.
const PROBE_ROW: [f32; 4] = [1.0, 2.0, 0.0, 5.0];
const PROBE_SCORE: f32 = 0.437_823_5;

const CV_WIDTH: usize = 4;
const CV_DIRECTION_1: [f32; 4] = [0.25, -0.5, 1.0, -2.0];
const CV_DIRECTION_3: [f32; 4] = [-0.125, 0.75, -1.25, 3.5];

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Parsed image plus its bytes; tensor loading needs both.
struct Image {
    file: GgufFile,
    data: Vec<u8>,
}

fn read_image(path: &Path) -> Image {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let file = parse(&data).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Image { file, data }
}

fn architecture(file: &GgufFile) -> &str {
    match file.metadata.get("general.architecture") {
        Some(MetaValue::Str(a)) => a.as_str(),
        other => panic!("general.architecture: {other:?}"),
    }
}

/// The exporters and their dependencies, or a reason they are unavailable.
///
/// Absent tooling is a skip, not a failure: the parse assertions above
/// already cover the seam from the committed bytes, and this test adds only
/// the check that those bytes are still what the exporters produce.
fn exporter_available() -> Result<(), String> {
    let script = fixtures_dir().join("export_gguf_fixtures.py");
    if !script.is_file() {
        return Err(format!("{} is not present", script.display()));
    }
    if !repo_root().join("interp/probes/export.py").is_file() {
        return Err("the parent tree is not present".to_string());
    }
    let probe = std::process::Command::new("python3")
        .args(["-c", "import numpy, gguf"])
        .output();
    match probe {
        Ok(out) if out.status.success() => Ok(()),
        Ok(_) => Err("python3 cannot import numpy and gguf".to_string()),
        Err(e) => Err(format!("python3 is unavailable: {e}")),
    }
}

#[test]
fn python_probe_gguf_loads_through_the_rust_probe_parser() {
    let image = read_image(&fixtures_dir().join("python-probe.gguf"));
    assert_eq!(architecture(&image.file), "probe");

    let probe = Probe::from_gguf(&image.file, &image.data).expect("probe parses");
    assert_eq!(probe.name, PROBE_NAME);
    assert_eq!(probe.model_hash, PROBE_MODEL_HASH);
    assert_eq!(probe.layer, PROBE_LAYER);
    assert_eq!(probe.threshold, PROBE_THRESHOLD);
    assert_eq!(probe.width(), PROBE_WIDTH);

    let score = probe.score(&PROBE_ROW).expect("row matches probe width");
    assert!(
        (score - PROBE_SCORE).abs() < 1e-6,
        "score {score} is not the hand-worked {PROBE_SCORE}"
    );
}

#[test]
fn python_control_vector_gguf_keeps_its_one_based_layer_numbering() {
    let image = read_image(&fixtures_dir().join("python-control-vector.gguf"));
    assert_eq!(architecture(&image.file), "controlvector");

    let cv = ControlVector::from_gguf(&image.file, &image.data, 1.0).expect("vector parses");
    assert_eq!(cv.width(), CV_WIDTH);

    // The exporter writes direction.1 and direction.3. Block 0 is unsteerable
    // by the format, and the gap at block 2 must stay a gap: a loader that
    // packed the tensors in order would steer blocks 0 and 1 instead.
    assert_eq!(cv.direction(0), None, "block 0 is unsteerable");
    assert_eq!(cv.direction(1), Some(&CV_DIRECTION_1[..]));
    assert_eq!(cv.direction(2), None, "the unnamed block stays unsteered");
    assert_eq!(cv.direction(3), Some(&CV_DIRECTION_3[..]));
    assert_eq!(cv.direction(4), None, "no direction past the last tensor");
}

#[test]
fn the_contract_names_the_keys_the_fixtures_actually_carry() {
    let path = repo_root().join("schemas/gguf_artifacts.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        eprintln!("[SKIP] contract assertions: {} is absent", path.display());
        return;
    };
    let contract: serde_json::Value = serde_json::from_str(&text).expect("contract is valid JSON");

    for (artifact, fixture) in [
        ("probe", "python-probe.gguf"),
        ("controlvector", "python-control-vector.gguf"),
    ] {
        let image = read_image(&fixtures_dir().join(fixture));
        let spec = &contract[artifact];
        assert_eq!(
            spec["architecture"].as_str(),
            Some(architecture(&image.file)),
            "{artifact}: declared architecture does not match the fixture"
        );
        for key in spec["metadata"]
            .as_object()
            .expect("metadata object")
            .keys()
        {
            assert!(
                image.file.metadata.contains_key(key),
                "{artifact}: contract declares {key}, the fixture has no such key"
            );
        }
        // direction.<N> is a pattern, not a name; the numbering itself is
        // asserted by the control-vector test above.
        for name in spec["tensors"].as_object().expect("tensors object").keys() {
            if name.contains('<') {
                let prefix = name.split('<').next().unwrap();
                assert!(
                    image
                        .file
                        .tensors
                        .iter()
                        .any(|t| t.name.starts_with(prefix)),
                    "{artifact}: no tensor matching {name}"
                );
            } else {
                assert!(
                    image.file.tensors.iter().any(|t| t.name == *name),
                    "{artifact}: contract declares tensor {name}, the fixture has none"
                );
            }
        }
    }
}

#[test]
fn the_committed_fixtures_are_still_what_the_exporters_produce() {
    if let Err(reason) = exporter_available() {
        eprintln!("[SKIP] exporter regeneration: {reason}");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let out = std::process::Command::new("python3")
        .arg(fixtures_dir().join("export_gguf_fixtures.py"))
        .arg(dir.path())
        .output()
        .expect("generator runs");
    assert!(
        out.status.success(),
        "generator failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Compared by parsed content rather than byte for byte. gguf is pinned,
    // but a version bump may legitimately change padding or alignment, and a
    // byte comparison would report that as exporter drift.
    let fresh_probe = read_image(&dir.path().join("python-probe.gguf"));
    let probe = Probe::from_gguf(&fresh_probe.file, &fresh_probe.data).expect("probe parses");
    assert_eq!(probe.name, PROBE_NAME);
    assert_eq!(probe.layer, PROBE_LAYER);
    assert_eq!(probe.width(), PROBE_WIDTH);
    let score = probe.score(&PROBE_ROW).unwrap();
    assert!(
        (score - PROBE_SCORE).abs() < 1e-6,
        "regenerated probe scores {score}, committed fixture scores {PROBE_SCORE}"
    );

    let fresh_cv = read_image(&dir.path().join("python-control-vector.gguf"));
    let cv = ControlVector::from_gguf(&fresh_cv.file, &fresh_cv.data, 1.0).expect("vector parses");
    assert_eq!(cv.width(), CV_WIDTH);
    assert_eq!(cv.direction(1), Some(&CV_DIRECTION_1[..]));
    assert_eq!(cv.direction(2), None);
    assert_eq!(cv.direction(3), Some(&CV_DIRECTION_3[..]));
}
