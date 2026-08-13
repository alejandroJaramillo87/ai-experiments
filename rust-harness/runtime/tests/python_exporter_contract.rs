/*
 * python_exporter_contract — the real Python interp exporters against the
 * Rust loaders.
 *
 * Two artifacts are written only by Python and read only by Rust: the probe
 * (interp/probes/export.py) and the control vector
 * (interp/steering/export.py), both safetensors, the probe with a JSON
 * sidecar beside it. Every other golden synthesizes those bytes in Rust,
 * which checks the parser against a second Rust implementation rather than
 * against the writer it exists to read. The block numbering makes that gap
 * concrete: both artifacts now number from 0, each side enforces its own
 * rule, and nothing but this test checks that the two rules compose.
 *
 * Coverage is in two layers. The committed fixtures under fixtures/ are
 * exporter output, so the parse assertions run everywhere the workspace
 * builds. The regeneration test then re-runs the exporters and compares,
 * which is what notices an exporter that has drifted away from the committed
 * bytes; it needs python3 with numpy and safetensors plus the parent tree, and
 * reports [SKIP] without them — as in the container build, whose context is
 * rust-harness/ alone.
 *
 * Contracts: schemas/probe_artifact.json and
 * schemas/control_vector_artifact.json.
 * Regenerate the fixtures with
 * `python3 rust-harness/fixtures/export_interp_fixtures.py rust-harness/fixtures`.
 */

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use runtime::checkpoint::Image;
use runtime::model::cpu::ControlVector;
use runtime::model::probe::Probe;

/// Expectations mirroring fixtures/export_interp_fixtures.py. Duplicated rather
/// than derived: a fixture regenerated from changed constants should fail
/// here until someone updates both, which is the point of a golden.
const PROBE_NAME: &str = "contract-fixture-probe";
const PROBE_MODEL_HASH: &str = "0123456789abcdef";
const PROBE_LAYER: usize = 2;
const PROBE_THRESHOLD: f32 = 0.25;
const PROBE_WIDTH: usize = 4;
const PROBE_SITE: runtime::model::cpu::Site = runtime::model::cpu::Site::LayerOutput;

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
const CV_DIRECTION_0: [f32; 4] = [0.25, -0.5, 1.0, -2.0];
const CV_DIRECTION_1: [f32; 4] = [1.5, 0.25, -0.75, 2.0];
const CV_DIRECTION_3: [f32; 4] = [-0.125, 0.75, -1.25, 3.5];

/// The control vector is a directory, not a file: the image plus the sibling
/// manifest that harness/src/main.rs reads the model hash from.
const CV_DIR_NAME: &str = "python-control-vector";

fn cv_fixture_dir() -> PathBuf {
    fixtures_dir().join(CV_DIR_NAME)
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_image(path: &Path) -> Image {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    Image::parse(data, &path.display().to_string())
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Assert one loaded control vector against the fixture constants.
///
/// The exporter writes block.0, block.1 and block.3. Block 0 carrying a
/// direction is the assertion that the retired layout's refusal of index 0
/// is gone; the gap at block 2 must stay a gap, because a loader that packed
/// the tensors in order would steer blocks 0, 1 and 2 instead.
fn assert_fixture_control_vector(cv: &ControlVector, what: &str) {
    assert_eq!(cv.width(), CV_WIDTH, "{what}");
    assert_eq!(cv.direction(0), Some(&CV_DIRECTION_0[..]), "{what}");
    assert_eq!(cv.direction(1), Some(&CV_DIRECTION_1[..]), "{what}");
    assert_eq!(
        cv.direction(2),
        None,
        "{what}: the unnamed block stays unsteered"
    );
    assert_eq!(cv.direction(3), Some(&CV_DIRECTION_3[..]), "{what}");
    assert_eq!(
        cv.direction(4),
        None,
        "{what}: no direction past the last tensor"
    );
}

/// The exporters and their dependencies, or a reason they are unavailable.
///
/// Absent tooling is a skip, not a failure: the parse assertions above
/// already cover the seam from the committed bytes, and this test adds only
/// the check that those bytes are still what the exporters produce.
fn exporter_python() -> Result<PathBuf, String> {
    let script = fixtures_dir().join("export_interp_fixtures.py");
    if !script.is_file() {
        return Err(format!("{} is not present", script.display()));
    }
    if !repo_root().join("interp/probes/export.py").is_file() {
        return Err("the parent tree is not present".to_string());
    }
    runtime::pyenv::interpreter(&repo_root(), &["numpy", "safetensors"])
        .ok_or_else(|| "no interpreter can import numpy and safetensors".to_string())
}

/// Assert one loaded probe against the fixture constants, including the
/// hand-worked score. Scoring rather than reading the tensors back covers
/// weights, mu, sigma and bias in one assertion, and catches a permuted
/// tensor that a per-name existence check would not.
fn assert_fixture_probe(probe: &Probe, what: &str) {
    assert_eq!(probe.name, PROBE_NAME, "{what}");
    assert_eq!(probe.layer, PROBE_LAYER, "{what}");
    assert_eq!(probe.site, PROBE_SITE, "{what}");
    assert_eq!(probe.width(), PROBE_WIDTH, "{what}");
    let score = probe.score(&PROBE_ROW).expect("row matches probe width");
    assert!(
        (score - PROBE_SCORE).abs() < 1e-6,
        "{what}: score {score} is not the hand-worked {PROBE_SCORE}"
    );
}

#[test]
fn the_python_probe_loads_through_the_rust_probe_reader() {
    let probe = Probe::load(&fixtures_dir().join("python-probe.safetensors"))
        .expect("the committed probe loads");
    assert_fixture_probe(&probe, "committed fixture");
    assert_eq!(probe.model_hash, PROBE_MODEL_HASH);
    assert_eq!(probe.threshold, PROBE_THRESHOLD);
}

#[test]
fn the_python_control_vector_keeps_its_zero_based_block_numbering() {
    let image = read_image(&cv_fixture_dir().join("vector.safetensors"));
    let cv = ControlVector::from_image(&image, 1.0, None).expect("vector parses");
    assert_fixture_control_vector(&cv, "committed fixture");
}

#[test]
fn a_layer_range_selects_from_the_python_written_blocks() {
    // The range is the harness-side replacement for llama.cpp's
    // --control-vector-layer-range, so it has to hold against a file the
    // exporter actually wrote and not only against a synthetic image.
    let image = read_image(&cv_fixture_dir().join("vector.safetensors"));
    let cv = ControlVector::from_image(&image, 1.0, Some((1, 3))).expect("vector parses");
    assert_eq!(cv.direction(0), None, "block 0 is outside the range");
    assert_eq!(cv.direction(1), Some(&CV_DIRECTION_1[..]));
    assert_eq!(cv.direction(3), Some(&CV_DIRECTION_3[..]));
}

/// One contract document, or a skip when the schemas directory is absent —
/// which is the container build, whose context is rust-harness/ alone.
fn contract(file: &str) -> Option<serde_json::Value> {
    let path = repo_root().join("schemas").join(file);
    let Ok(text) = std::fs::read_to_string(&path) else {
        runtime::testlog::skip(
            "contract assertions",
            &format!("{} is absent", path.display()),
        );
        return None;
    };
    Some(serde_json::from_str(&text).expect("contract is valid JSON"))
}

fn names(value: &serde_json::Value) -> BTreeSet<String> {
    value
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .map(|v| v.as_str().expect("a string").to_string())
        .collect()
}

#[test]
fn the_probe_contract_names_the_keys_the_fixture_actually_carries() {
    let Some(contract) = contract("probe_artifact.json") else {
        return;
    };
    let image = fixtures_dir().join("python-probe.safetensors");
    let sidecar: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(image.with_extension("json")).expect("the sidecar is committed"),
    )
    .expect("the sidecar is valid JSON");

    let got: BTreeSet<String> = sidecar
        .as_object()
        .expect("the sidecar is an object")
        .keys()
        .cloned()
        .collect();
    let required = names(&contract["required"]["sidecar"]);
    let optional = names(&contract["optional"]["sidecar"]);
    assert!(
        required.is_subset(&got),
        "the sidecar is missing required keys {:?}",
        required.difference(&got).collect::<Vec<_>>()
    );
    let allowed: BTreeSet<String> = required.union(&optional).cloned().collect();
    assert!(
        got.is_subset(&allowed),
        "the sidecar carries keys the contract does not name: {:?}",
        got.difference(&allowed).collect::<Vec<_>>()
    );
    for field in ["artifact", "site"] {
        let value = sidecar[field].as_str().expect("an enumerated string");
        assert!(
            names(&contract["values"][field]).contains(value),
            "{field} {value:?} is not a value the contract lists"
        );
    }

    // The tensor names, read from the image the sidecar describes.
    let declared = names(&contract["required"]["tensors"]);
    let present: BTreeSet<String> = runtime::checkpoint::Image::parse(
        std::fs::read(&image).expect("the image is committed"),
        "python-probe.safetensors",
    )
    .expect("the image parses")
    .names()
    .into_iter()
    .collect();
    assert_eq!(declared, present, "the contract and the image disagree");
}

#[test]
fn the_control_vector_contract_names_the_keys_the_fixture_actually_carries() {
    let Some(contract) = contract("control_vector_artifact.json") else {
        return;
    };
    let image = read_image(&cv_fixture_dir().join("vector.safetensors"));
    let present: BTreeSet<String> = image.names().into_iter().collect();
    assert!(!present.is_empty(), "the fixture holds no tensors");

    // block.<index> is a pattern, not a name: the contract declares the
    // prefix and the numbering, and the test above asserts which blocks the
    // fixture actually carries.
    let declared = names(&contract["required"]["tensors"]);
    assert_eq!(declared.len(), 1, "one tensor pattern is declared");
    let pattern = declared.iter().next().expect("a pattern");
    let prefix = pattern.split('<').next().expect("a prefix");
    for name in &present {
        assert!(
            name.starts_with(prefix),
            "tensor {name} does not match the declared pattern {pattern}"
        );
    }

    // The sibling manifest the exporter writes. Advisory rather than
    // mandatory -- see the contract's manifest_is_advisory note -- so the
    // check is that the contract names what the manifest carries, not that
    // the reader needs it.
    let manifest: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(cv_fixture_dir().join("manifest.json"))
            .expect("the manifest is committed"),
    )
    .expect("the manifest is valid JSON");
    let got: BTreeSet<String> = manifest
        .as_object()
        .expect("the manifest is an object")
        .keys()
        .cloned()
        .collect();
    let required = names(&contract["required"]["manifest"]);
    assert!(
        required.is_subset(&got),
        "the manifest is missing required keys {:?}",
        required.difference(&got).collect::<Vec<_>>()
    );
    let artifact = manifest["artifact"].as_str().expect("an enumerated string");
    assert!(
        names(&contract["values"]["artifact"]).contains(artifact),
        "artifact {artifact:?} is not a value the contract lists"
    );
}

#[test]
fn the_committed_fixtures_are_still_what_the_exporters_produce() {
    let python = match exporter_python() {
        Ok(python) => python,
        Err(reason) => {
            runtime::testlog::skip("exporter regeneration", &reason);
            return;
        }
    };
    let dir = tempfile::tempdir().unwrap();
    let out = std::process::Command::new(&python)
        .arg(fixtures_dir().join("export_interp_fixtures.py"))
        .arg(dir.path())
        .output()
        .expect("generator runs");
    assert!(
        out.status.success(),
        "generator failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Compared by parsed content rather than byte for byte. safetensors is
    // pinned, but a version bump may legitimately change header key order or
    // padding; and the probe sidecar carries a timestamp and a git_sha, which
    // differ on every regeneration. A byte comparison would report either as
    // exporter drift.
    let probe = Probe::load(&dir.path().join("python-probe.safetensors"))
        .expect("the regenerated probe loads");
    assert_fixture_probe(&probe, "regenerated");

    let fresh_cv = read_image(&dir.path().join(CV_DIR_NAME).join("vector.safetensors"));
    let cv = ControlVector::from_image(&fresh_cv, 1.0, None).expect("vector parses");
    assert_fixture_control_vector(&cv, "regenerated");
}
