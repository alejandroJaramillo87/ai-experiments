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

#[test]
fn fixture_gguf_matches_the_generator() {
    // The committed fixture must be exactly what the tiny-model bin writes;
    // a drifted fixture would silently invalidate the golden logits.
    let committed = std::fs::read(fixture_path("tiny-gemma4.gguf")).unwrap();
    let generated = runtime::synth::build_tiny_model(&runtime::synth::TinySpec::default());
    assert_eq!(committed, generated, "regenerate the fixture (see header)");
}
