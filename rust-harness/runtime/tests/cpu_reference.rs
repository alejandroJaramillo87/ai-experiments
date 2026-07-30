/*
 * cpu_reference — phase 3 golden test: the CPU backend's logits match an
 * independent numpy implementation of the same forward pass, computed from
 * the identical committed GGUF fixture. A mismatch localizes to the first
 * divergent position, per the phase 1 bisection workflow.
 *
 * Fixture regeneration (from rust-harness/):
 *   cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4.gguf
 *   python3 fixtures/reference_logits.py
 */

use runtime::model::cpu::CpuModel;
use runtime::model::Model;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name)
}

#[test]
fn cpu_backend_matches_the_python_reference() {
    let image = std::fs::read(fixture_path("tiny-gemma4.gguf")).unwrap();
    let file = runtime::gguf::parse(&image).unwrap();
    let mut model = CpuModel::from_gguf(&file, &image).unwrap();

    let golden: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_path("tiny-gemma4-logits.json")).unwrap(),
    )
    .unwrap();
    let prompt: Vec<u32> = golden["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let expected: Vec<Vec<f32>> = golden["logits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect()
        })
        .collect();

    let rows = model.prefill_all(&prompt).unwrap();
    assert_eq!(rows.len(), expected.len());
    // f32 accumulation order differs between the two implementations, so
    // exact equality is not expected; divergence bugs are orders of
    // magnitude above this tolerance.
    const TOLERANCE: f32 = 2e-4;
    for (position, (got, want)) in rows.iter().zip(&expected).enumerate() {
        for (t, (g, w)) in got.0.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= TOLERANCE,
                "first divergence at position {position}, token {t}: got {g}, reference {w}"
            );
        }
    }
}

fn load_golden() -> (Vec<u8>, serde_json::Value) {
    let image = std::fs::read(fixture_path("tiny-gemma4.gguf")).unwrap();
    let golden = serde_json::from_str(
        &std::fs::read_to_string(fixture_path("tiny-gemma4-logits.json")).unwrap(),
    )
    .unwrap();
    (image, golden)
}

fn f32_row(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

const TOLERANCE: f32 = 2e-4;

#[test]
fn tap_rows_match_the_python_reference_residuals() {
    use runtime::model::cpu::Tap;
    use std::sync::{Arc, Mutex};

    let (image, golden) = load_golden();
    let file = runtime::gguf::parse(&image).unwrap();
    let mut model = CpuModel::from_gguf(&file, &image).unwrap();
    let prompt: Vec<u32> = golden["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    #[derive(Clone, Default)]
    struct LastRows(Arc<Mutex<std::collections::HashMap<usize, Vec<f32>>>>);
    impl Tap for LastRows {
        fn on_residual(&mut self, layer: usize, _position: usize, x: &[f32]) {
            self.0.lock().unwrap().insert(layer, x.to_vec());
        }
    }
    let tap = LastRows::default();
    model.set_tap(Box::new(tap.clone()));
    model.prefill_all(&prompt).unwrap();

    let rows = tap.0.lock().unwrap();
    let expected = golden["last_residuals"].as_array().unwrap();
    assert_eq!(rows.len(), expected.len());
    for (layer, want) in expected.iter().enumerate() {
        let got = &rows[&layer];
        for (j, (g, w)) in got.iter().zip(f32_row(want)).enumerate() {
            assert!(
                (g - w).abs() <= TOLERANCE,
                "layer {layer} residual element {j}: got {g}, reference {w}"
            );
        }
    }
}

#[test]
fn steered_logits_match_the_python_reference() {
    use runtime::model::cpu::ControlVector;

    let (image, golden) = load_golden();
    let file = runtime::gguf::parse(&image).unwrap();
    let mut model = CpuModel::from_gguf(&file, &image).unwrap();
    let embed_dim = model.config().embed_dim;
    let prompt: Vec<u32> = golden["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();

    // Mirror the reference's synthetic steering fixture exactly: direction
    // 0.05 * (j - 8) after 0-based steer_layer at steer_scale.
    let steer_layer = golden["steer_layer"].as_u64().unwrap() as usize;
    let steer_scale = golden["steer_scale"].as_f64().unwrap() as f32;
    let direction: Vec<f32> = (0..embed_dim).map(|j| 0.05 * (j as f32 - 8.0)).collect();
    let mut w = runtime::synth::GgufWriter::new();
    w.kv_str("general.architecture", "controlvector");
    // Control-vector tensors are 1-based.
    w.tensor_f32(
        &format!("direction.{}", steer_layer + 1),
        &[embed_dim as u64],
        &direction,
    );
    let cv_image = w.finish();
    let cv_file = runtime::gguf::parse(&cv_image).unwrap();
    let cv = ControlVector::from_gguf(&cv_file, &cv_image, steer_scale).unwrap();
    model.set_control_vector(cv).unwrap();

    let rows = model.prefill_all(&prompt).unwrap();
    let want = f32_row(&golden["steered_last_logits"]);
    let got = &rows.last().unwrap().0;
    for (t, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= TOLERANCE,
            "steered logit for token {t}: got {g}, reference {w}"
        );
    }
}

#[test]
fn fixture_gguf_matches_the_generator() {
    // The committed fixture must be exactly what the tiny-model bin writes;
    // a drifted fixture would silently invalidate the golden logits.
    let committed = std::fs::read(fixture_path("tiny-gemma4.gguf")).unwrap();
    let generated = runtime::synth::build_tiny_model(&runtime::synth::TinySpec::default());
    assert_eq!(committed, generated, "regenerate the fixture (see header)");
}
