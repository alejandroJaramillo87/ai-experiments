/*
 * attention_capture — end-to-end coverage of the attention-capture binary:
 * the tap, the dense-matrix encoding, span_first, and the manifest, driven
 * against the committed mixed sliding/global fixture.
 *
 * The fixture is tiny-gemma4-e4b and not tiny-gemma4 on purpose. The uniform
 * fixture has one attention regime, so every span would start at zero and the
 * assertions below would hold on a binary that had never heard of a sliding
 * window. The whole claim of this site is that the two regimes are visibly
 * different, and only a mixed fixture can falsify it.
 *
 * The binary is spawned rather than called, so the artifact under test is the
 * one a reader opens. Row-level assertions go through np.load rather than a
 * second Rust reader, per the independent-verification rule in
 * .styles/RUST_STYLE.md, and skip with [SKIP] when no interpreter can import
 * numpy -- the container build's situation, whose context is rust-harness/
 * alone.
 */

use std::path::{Path, PathBuf};

use runtime::kv::LayerKind;
use runtime::model::cpu::ModelSpec;

/// Enough tokens that the fixture's window of 3 is exceeded several times
/// over, so a sliding layer's span is clipped at most positions rather than
/// only at the last one.
const PROMPT: &str = "anchor harbor rope bridge river stone canyon desert wind";

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4-e4b")
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The fixture's attention schedule, derived from the checkpoint rather than
/// from the binary's output.
///
/// This is the independent half of the geometry assertion: comparing the
/// artifact against the config the model was built from, not against a
/// constant transcribed from the same reading that produced the writer.
fn fixture_spec() -> ModelSpec {
    let bytes = std::fs::read(fixture_model().join("config.json")).unwrap();
    let config = runtime::checkpoint::parse_text_config(&bytes).unwrap();
    ModelSpec::from_checkpoint(&config).unwrap()
}

struct Capture {
    _dir: tempfile::TempDir,
    out: PathBuf,
    output: std::process::Output,
}

impl Capture {
    fn stderr(&self) -> String {
        String::from_utf8_lossy(&self.output.stderr).to_string()
    }
}

fn capture(layers: &str, heads: &str) -> Capture {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("attention");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_attention-capture"))
        .arg("--model")
        .arg(fixture_model())
        .arg("--prompt")
        .arg(PROMPT)
        .arg("--layers")
        .arg(layers)
        .arg("--heads")
        .arg(heads)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("spawn attention-capture");
    Capture {
        _dir: dir,
        out,
        output,
    }
}

fn manifest(capture: &Capture) -> serde_json::Value {
    let text = std::fs::read_to_string(capture.out.join("manifest.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// An interpreter that can import numpy, resolved rather than assumed.
fn numpy_python() -> Option<PathBuf> {
    runtime::pyenv::interpreter(&repo_root(), &["numpy"])
}

/// Shard arrays as JSON, read through np.load. float32 values are exact in
/// float64, so the round trip is lossless and `==` below is meaningful.
fn load_shard(python: &Path, path: &Path) -> serde_json::Value {
    let script = format!(
        "import json, sys\n\
         import numpy as np\n\
         z = np.load('{}')\n\
         json.dump({{k: z[k].tolist() for k in z.files}}, sys.stdout, allow_nan=False)\n",
        path.display()
    );
    let out = std::process::Command::new(python)
        .args(["-c", &script])
        .output()
        .expect("the resolved python3 runs");
    assert!(
        out.status.success(),
        "np.load rejected the shard: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).unwrap()
}

fn rows(shard: &serde_json::Value, key: &str) -> Vec<Vec<f64>> {
    shard[key]
        .as_array()
        .unwrap_or_else(|| panic!("shard has no {key} array, has {:?}", keys(shard)))
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect()
        })
        .collect()
}

fn keys(shard: &serde_json::Value) -> Vec<String> {
    shard
        .as_object()
        .unwrap()
        .keys()
        .map(String::from)
        .collect()
}

#[test]
fn a_capture_writes_a_shard_and_a_manifest_naming_what_it_holds() {
    let c = capture("0,1", "0,2");
    assert!(c.output.status.success(), "{}", c.stderr());
    assert!(c.out.join("attention.npz").is_file(), "{}", c.stderr());
    let m = manifest(&c);
    assert_eq!(m["artifact"], "attention_capture");
    assert_eq!(m["site"], "attention_weights");
    assert_eq!(m["softmax"], "post");
    assert_eq!(m["layers"], serde_json::json!([0, 1]));
    assert_eq!(m["heads"], serde_json::json!([0, 2]));
    // The volume line is the cost discipline made visible; a capture that
    // stopped reporting it would let a quadratic artifact appear silently.
    assert!(
        c.stderr().contains("MiB"),
        "the run must state its volume: {}",
        c.stderr()
    );
}

#[test]
fn a_layer_outside_the_model_is_refused_naming_the_bound() {
    let c = capture("99", "0");
    assert!(!c.output.status.success(), "{}", c.stderr());
    assert!(
        c.stderr().contains("layer 99 out of range"),
        "{}",
        c.stderr()
    );
}

#[test]
fn a_head_outside_the_model_is_refused_naming_the_bound() {
    let c = capture("0", "99");
    assert!(!c.output.status.success(), "{}", c.stderr());
    assert!(
        c.stderr().contains("head 99 out of range"),
        "{}",
        c.stderr()
    );
}

#[test]
fn every_row_is_a_distribution_over_its_span_and_exactly_zero_outside() {
    let Some(python) = numpy_python() else {
        runtime::testlog::skip(
            "np.load assertions on the attention shard",
            "no interpreter can import numpy",
        );
        return;
    };
    // Covers the softmax's accumulated error, and the float64 the JSON round
    // trip widens each weight into.
    const TOLERANCE: f64 = 1e-5;
    let c = capture("0,1,2,3", "0");
    assert!(c.output.status.success(), "{}", c.stderr());
    let shard = load_shard(&python, &c.out.join("attention.npz"));
    let spec = fixture_spec();

    for (l, layer) in spec.layers.iter().enumerate() {
        let m = rows(&shard, &format!("layer_{l}_head_0"));
        let n = m.len();
        for (p, row) in m.iter().enumerate() {
            assert_eq!(row.len(), n, "layer {l} row {p} is not {n} wide");
            let first = match layer.kind {
                LayerKind::Global => 0,
                LayerKind::SlidingWindow(w) => (p + 1).saturating_sub(w),
            };
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < TOLERANCE,
                "layer {l} position {p}: row sums to {sum}"
            );
            // Exact zero, not small: outside the span nothing was ever
            // written, so this separates a structural zero from a weight that
            // merely rounded to nothing.
            for (k, v) in row.iter().enumerate() {
                if k < first || k > p {
                    assert_eq!(
                        *v, 0.0,
                        "layer {l} position {p}: key {k} is outside {first}..={p} but holds {v}"
                    );
                }
            }
        }
    }
}

#[test]
fn span_first_records_the_schedule_the_checkpoint_declares() {
    let Some(python) = numpy_python() else {
        runtime::testlog::skip(
            "np.load assertions on span_first",
            "no interpreter can import numpy",
        );
        return;
    };
    let c = capture("0,1,2,3", "0,1");
    assert!(c.output.status.success(), "{}", c.stderr());
    let shard = load_shard(&python, &c.out.join("attention.npz"));
    let spec = fixture_spec();

    let mut saw_global = false;
    let mut saw_clipped = false;
    for (l, layer) in spec.layers.iter().enumerate() {
        let got: Vec<usize> = shard[format!("span_first_layer_{l}")]
            .as_array()
            .unwrap_or_else(|| panic!("shard has no span_first_layer_{l}, has {:?}", keys(&shard)))
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let want: Vec<usize> = (0..got.len())
            .map(|p| match layer.kind {
                LayerKind::Global => 0,
                LayerKind::SlidingWindow(w) => (p + 1).saturating_sub(w),
            })
            .collect();
        assert_eq!(got, want, "layer {l} ({:?}) span_first", layer.kind);
        match layer.kind {
            LayerKind::Global => saw_global = true,
            LayerKind::SlidingWindow(_) => saw_clipped |= got.iter().any(|&f| f > 0),
        }
    }
    // Without both, the assertions above are consistent with a binary that
    // reports zero everywhere, which is the failure this whole site exists to
    // rule out.
    assert!(
        saw_global && saw_clipped,
        "the fixture must show both regimes: global {saw_global}, clipped {saw_clipped}"
    );
}

#[test]
fn the_contract_names_the_keys_the_manifest_actually_carries() {
    let contract = repo_root().join("schemas/attention_capture.manifest.json");
    if !contract.is_file() {
        runtime::testlog::skip(
            "attention manifest contract assertions",
            "the parent tree is not present",
        );
        return;
    }
    let spec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&contract).unwrap()).unwrap();
    let c = capture("0", "0");
    assert!(c.output.status.success(), "{}", c.stderr());
    let m = manifest(&c);

    let required: Vec<&str> = spec["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let present: Vec<String> = m.as_object().unwrap().keys().map(String::from).collect();
    for key in &required {
        assert!(
            m.get(*key).is_some(),
            "contract requires {key}, manifest has {present:?}"
        );
    }
    for key in &present {
        assert!(
            required.contains(&key.as_str()),
            "manifest writes {key}, which the contract does not name"
        );
    }
    for (key, want) in spec["values"].as_object().unwrap() {
        assert_eq!(&m[key], want, "manifest {key}");
    }
}
