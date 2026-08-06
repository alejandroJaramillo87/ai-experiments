/*
 * activation_capture — end-to-end coverage of the activation-capture binary:
 * dataset load, the last-row tap, shard slicing, and manifest writing, driven
 * against the committed tiny GGUF fixture.
 *
 * The binary is spawned rather than called, so the artifact under test is the
 * one the interp probe toolkit actually reads. Row-level assertions need
 * python3 with numpy to open the npz (as in the npz writer's unit test) and
 * skip without it; the artifact-shape assertions run either way.
 *
 * Three of the tests reach into the parent tree for the Python side of the
 * seam: the directory goes through interp.probes.capture.load_capture rather
 * than a bespoke np.load, the manifest key set is checked against
 * schemas/activation_capture.manifest.json, and the two writers' dataset
 * digests are pinned as agreeing on LF and differing on CRLF. Each skips with
 * [SKIP] when the parent tree or numpy is absent, which is the container
 * build's situation.
 */

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use runtime::model::cpu::{CpuModel, Tap};
use runtime::model::Model;
use runtime::tokenizer::Tokenizer;

/// Distinct texts with non-consecutive labels: a row shifted by one, or a row
/// duplicated across examples, changes both the activation and the label it
/// sits next to, so misalignment cannot pass as a rounding difference.
const DATASET: [(&str, i64); 4] = [
    ("anchor harbor rope", 7),
    ("bridge river stone", 0),
    ("canyon desert wind", 3),
    ("delta marsh reed", 1),
];

fn fixture_model() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join("tiny-gemma4.gguf")
}

fn dataset_jsonl(rows: &[(&str, i64)]) -> String {
    rows.iter()
        .map(|(text, label)| format!("{{\"text\": \"{text}\", \"label\": {label}}}\n"))
        .collect()
}

/// A finished capture run. The TempDir is held so the artifacts outlive the
/// spawn and are removed when the test ends.
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

fn capture(dataset_body: &str, layers: &str) -> Capture {
    let dir = tempfile::tempdir().unwrap();
    let dataset = dir.path().join("dataset.jsonl");
    std::fs::write(&dataset, dataset_body).unwrap();
    let out = dir.path().join("capture");
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_activation-capture"))
        .arg("--model")
        .arg(fixture_model())
        .arg("--dataset")
        .arg(&dataset)
        .arg("--layers")
        .arg(layers)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("spawn activation-capture");
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

/// An interpreter that can import numpy, resolved rather than assumed —
/// see `runtime::pyenv` for why a bare `python3` is the wrong question.
fn numpy_python() -> Option<PathBuf> {
    runtime::pyenv::interpreter(&repo_root(), &["numpy"])
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The parent tree's probe toolkit, or a reason it cannot be reached.
///
/// Absent tooling is a skip rather than a failure: the container build's
/// context is rust-harness/ alone, and the artifact-shape assertions above
/// already run everywhere.
fn interp_python() -> Result<PathBuf, String> {
    if !repo_root().join("interp/probes/capture.py").is_file() {
        return Err("the parent tree is not present".to_string());
    }
    numpy_python().ok_or_else(|| "no interpreter can import numpy".to_string())
}

/// Run a snippet against the parent tree and return its stdout.
fn run_interp(python: &Path, script: &str, arg: &Path) -> String {
    let out = std::process::Command::new(python)
        .args(["-c", script])
        .arg(arg)
        .env("PYTHONPATH", repo_root())
        .output()
        .expect("the resolved python3 runs");
    assert!(
        out.status.success(),
        "interp rejected the artifact: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("stdout is utf-8")
}

/// Shard arrays as JSON, read through np.load so the assertions cover the
/// same reader the probe toolkit uses. float32 values are exact in float64,
/// so the JSON round trip is lossless.
fn load_shard(python: &Path, path: &Path) -> serde_json::Value {
    let script = format!(
        "import json, sys\n\
         import numpy as np\n\
         z = np.load('{}')\n\
         json.dump({{k: z[k].astype(float).tolist() for k in z.files}}, sys.stdout, \
         allow_nan=False)\n",
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

fn f32_rows(shard: &serde_json::Value, key: &str) -> Vec<Vec<f32>> {
    shard[key]
        .as_array()
        .unwrap_or_else(|| panic!("shard has no {key} array"))
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect()
}

#[derive(Clone, Default)]
struct LastRows(Arc<Mutex<HashMap<usize, Vec<f32>>>>);

impl Tap for LastRows {
    fn on_residual(&mut self, layer: usize, _position: usize, x: &[f32]) {
        self.0.lock().unwrap().insert(layer, x.to_vec());
    }
}

/// Last-token residuals the capture is expected to hold, recomputed here with
/// a fresh model per example. Independent of the binary's single-model loop,
/// so a comparison against these pins the row-to-example mapping instead of
/// restating whatever order the binary happened to write.
fn reference_rows(texts: &[&str], layer: usize) -> Vec<Vec<f32>> {
    let image = std::fs::read(fixture_model()).unwrap();
    let file = runtime::gguf::parse(&image).unwrap();
    let tokenizer = Tokenizer::from_gguf(&file).unwrap();
    texts
        .iter()
        .map(|text| {
            let mut model = CpuModel::from_gguf(&file, &image).unwrap();
            let tap = LastRows::default();
            model.set_tap(Box::new(tap.clone()));
            let mut tokens = vec![tokenizer.bos()];
            tokens.extend(tokenizer.encode(text).unwrap());
            model.prefill(&tokens).unwrap();
            let rows = tap.0.lock().unwrap();
            rows[&layer].clone()
        })
        .collect()
}

#[test]
fn capture_writes_a_manifest_and_shard_for_the_requested_layers() {
    // Layers given out of order: the manifest records them sorted, and both
    // requested layers are present in the shard.
    let c = capture(&dataset_jsonl(&DATASET), "1,0");
    assert!(c.output.status.success(), "{}", c.stderr());

    let m = manifest(&c);
    assert_eq!(m["schema_version"], 1);
    assert_eq!(m["artifact"], "activation_capture");
    assert_eq!(m["dtype"], "float32");
    assert_eq!(m["position"], "last");
    assert_eq!(m["count"], DATASET.len());
    assert_eq!(m["shards"], 1);
    assert_eq!(m["layers"], serde_json::json!([0, 1]));

    assert!(c.out.join("shard_00000.npz").is_file());
    assert!(!c.out.join("shard_00001.npz").exists());

    // examples.jsonl carries the dataset in capture order for the toolkit's
    // per-row provenance.
    let examples = std::fs::read_to_string(c.out.join("examples.jsonl")).unwrap();
    let rows: Vec<(String, i64)> = examples
        .lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            (
                v["text"].as_str().unwrap().to_string(),
                v["label"].as_i64().unwrap(),
            )
        })
        .collect();
    let want: Vec<(String, i64)> = DATASET.iter().map(|(t, l)| (t.to_string(), *l)).collect();
    assert_eq!(rows, want);
}

#[test]
fn shard_rows_stay_aligned_with_their_labels_and_source_texts() {
    let Some(python) = numpy_python() else {
        runtime::testlog::skip("shard row assertions", "no interpreter can import numpy");
        return;
    };
    let c = capture(&dataset_jsonl(&DATASET), "0,1");
    assert!(c.output.status.success(), "{}", c.stderr());
    let shard = load_shard(&python, &c.out.join("shard_00000.npz"));

    let labels: Vec<i64> = shard["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as i64)
        .collect();
    assert_eq!(labels, DATASET.iter().map(|(_, l)| *l).collect::<Vec<_>>());

    let texts: Vec<&str> = DATASET.iter().map(|(t, _)| *t).collect();
    for layer in [0usize, 1] {
        let want = reference_rows(&texts, layer);
        // Without pairwise-distinct references the row comparison below would
        // hold under any permutation, which is exactly the failure it exists
        // to catch.
        for (i, row) in want.iter().enumerate() {
            for other in &want[i + 1..] {
                assert_ne!(row, other, "layer {layer}: fixture rows are not distinct");
            }
        }
        let got = f32_rows(&shard, &format!("layer_{layer}"));
        assert_eq!(got.len(), DATASET.len(), "layer {layer}: row count");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g, w,
                "layer {layer} row {i} is not the activation of {:?}",
                texts[i]
            );
        }
    }
}

/// The whole directory through `interp.probes.capture.load_capture`, the
/// reader the probe toolkit actually calls.
///
/// The assertions above open the shards with a bespoke `np.load`, which
/// proves the npz encoding is readable but exercises none of load_capture's
/// consistency guards: manifest count against shard rows, examples.jsonl
/// length against both, and repeated layers. Those guards exist because rows,
/// labels, and texts are joined by position alone, and until this test none of
/// them had ever seen a Rust-written directory.
#[test]
fn a_rust_written_capture_loads_through_the_python_probe_reader() {
    let python = match interp_python() {
        Ok(python) => python,
        Err(reason) => {
            runtime::testlog::skip("python reader assertions", &reason);
            return;
        }
    };
    let c = capture(&dataset_jsonl(&DATASET), "0,1");
    assert!(c.output.status.success(), "{}", c.stderr());

    const SCRIPT: &str = "import json, sys\n\
         from interp.probes.capture import load_capture\n\
         acts, labels, texts = load_capture(sys.argv[1])\n\
         json.dump({'labels': labels.tolist(), 'texts': texts,\n\
         'shapes': {str(k): list(v.shape) for k, v in acts.items()},\n\
         'dtypes': {str(k): str(v.dtype) for k, v in acts.items()}}, sys.stdout)\n";
    let loaded: serde_json::Value =
        serde_json::from_str(&run_interp(&python, SCRIPT, &c.out)).unwrap();

    let labels: Vec<i64> = loaded["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(labels, DATASET.iter().map(|(_, l)| *l).collect::<Vec<_>>());

    let texts: Vec<&str> = loaded["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(texts, DATASET.iter().map(|(t, _)| *t).collect::<Vec<_>>());

    // The reader keys activations by layer, so both requested layers must come
    // back with one row per example at the model's width, as float32.
    for layer in ["0", "1"] {
        let shape = loaded["shapes"][layer].as_array().unwrap();
        assert_eq!(
            shape[0].as_u64().unwrap() as usize,
            DATASET.len(),
            "layer {layer}: row count"
        );
        assert_eq!(loaded["dtypes"][layer], "float32", "layer {layer}: dtype");
    }
}

/// The manifest against `schemas/activation_capture.manifest.json`.
///
/// The key list is written out once here and once in
/// interp/probes/capture.py, with no shared definition between them. Adding a
/// key to one writer alone fails this test; updating the contract to accept it
/// fails the Python writer's test until that side follows.
#[test]
fn the_manifest_carries_exactly_the_contract_key_set() {
    let path = repo_root().join("schemas/activation_capture.manifest.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        runtime::testlog::skip(
            "contract assertions",
            &format!("{} is absent", path.display()),
        );
        return;
    };
    let contract: serde_json::Value = serde_json::from_str(&text).expect("contract is valid JSON");
    let mut want: Vec<&str> = contract["required"]
        .as_array()
        .expect("required array")
        .iter()
        .chain(
            contract["writer_only"]["rust-harness-cpu"]
                .as_array()
                .expect("writer_only array"),
        )
        .map(|v| v.as_str().unwrap())
        .collect();
    want.sort_unstable();

    let c = capture(&dataset_jsonl(&DATASET), "0");
    assert!(c.output.status.success(), "{}", c.stderr());
    let m = manifest(&c);
    let mut got: Vec<&str> = m
        .as_object()
        .expect("manifest object")
        .keys()
        .map(String::as_str)
        .collect();
    got.sort_unstable();
    assert_eq!(got, want, "manifest keys have drifted from the contract");
}

/// The two writers' dataset digests, pinned as different where they differ.
///
/// Rust hashes the file's raw bytes; Python hashes text-mode f.read(), which
/// normalizes CRLF to LF. The digests therefore agree on an LF dataset and
/// disagree on a CRLF one. Both are internally consistent and neither is
/// wrong, but the difference is invisible until someone compares a
/// rust-captured manifest against a python-captured one and concludes the
/// datasets differ. Asserting both halves means a well-meaning one-sided
/// change to either hashing domain fails here instead of silently
/// reinterpreting every manifest already on disk.
#[test]
fn the_two_writers_agree_on_an_lf_dataset_digest_and_differ_on_a_crlf_one() {
    let python = match interp_python() {
        Ok(python) => python,
        Err(reason) => {
            runtime::testlog::skip("digest divergence assertions", &reason);
            return;
        }
    };
    const SCRIPT: &str = "import sys\n\
         from interp.manifest import text_sha256\n\
         with open(sys.argv[1], encoding='utf-8') as f:\n\
         \x20   sys.stdout.write(text_sha256(f.read()))\n";

    let lf = dataset_jsonl(&DATASET);
    let c = capture(&lf, "0");
    assert!(c.output.status.success(), "{}", c.stderr());
    let dataset = c.out.parent().unwrap().join("dataset.jsonl");
    assert_eq!(
        manifest(&c)["dataset_sha256"].as_str().unwrap(),
        run_interp(&python, SCRIPT, &dataset),
        "the digests must agree on a dataset with no CRLF to normalize"
    );

    let crlf = lf.replace('\n', "\r\n");
    let c = capture(&crlf, "0");
    assert!(
        c.output.status.success(),
        "the Rust writer must accept a CRLF dataset: {}",
        c.stderr()
    );
    let dataset = c.out.parent().unwrap().join("dataset.jsonl");
    assert_ne!(
        manifest(&c)["dataset_sha256"].as_str().unwrap(),
        run_interp(&python, SCRIPT, &dataset),
        "the digests are documented as hashing different domains; if they now \
         agree, one side changed and schemas/activation_capture.manifest.json \
         needs updating with it"
    );
}

#[test]
fn a_dataset_line_without_text_fails_naming_the_line() {
    let body = "{\"text\": \"anchor harbor rope\", \"label\": 7}\n{\"label\": 0}\n";
    let c = capture(body, "0");
    assert!(!c.output.status.success());
    let err = c.stderr();
    assert!(err.contains("dataset line 2: missing text"), "{err}");
    // A rejected dataset must leave no artifact a reader could pick up.
    assert!(!c.out.join("manifest.json").exists());
}

#[test]
fn a_dataset_line_without_an_integer_label_fails_naming_the_line() {
    let body = "{\"text\": \"anchor harbor rope\"}\n";
    let c = capture(body, "0");
    assert!(!c.output.status.success());
    let err = c.stderr();
    assert!(
        err.contains("dataset line 1: missing integer label"),
        "{err}"
    );
    assert!(!c.out.join("manifest.json").exists());
}

#[test]
fn a_repeated_layer_is_rejected_before_any_capture_runs() {
    // A repeat stacks two rows per example into one buffer, so the shard
    // slicing would read a window shifted against the labels.
    let c = capture(&dataset_jsonl(&DATASET), "1,0,1");
    assert!(!c.output.status.success());
    let err = c.stderr();
    assert!(err.contains("duplicate layer 1"), "{err}");
    assert!(!c.out.exists());
}
