/*
 * cpu_reference — phase 3 golden test: the CPU backend's logits match an
 * independent numpy implementation of the same forward pass, computed from
 * the identical committed checkpoint. A mismatch localizes to the first
 * divergent position, per the phase 1 bisection workflow.
 *
 * Two fixtures run through every golden here. The uniform one declares none
 * of Gemma 4's optional features and proves they are absent-safe; the
 * E4B-shaped one declares all seven and proves they are right.
 *
 * Fixture regeneration (from rust-harness/):
 *   cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4
 *   cargo run -p runtime --bin tiny-model -- fixtures/tiny-gemma4-e4b e4b
 *   cargo run -p runtime --bin tiny-model -- fixtures/tiny-bpe.tokenizer.json bpe
 *   ../.venv/bin/python3 fixtures/reference_logits.py
 *
 * Each model invocation writes a checkpoint directory holding config.json and
 * model.safetensors, plus the tokenizer.json beside it, all from one TinySpec.
 * The bpe invocation writes a tokenizer only: its
 * vocabulary carries a merge list, which the model fixtures cannot, and it is
 * what covers BPE segmentation. The oracle needs the venv's numpy.
 *
 * Regeneration is for changing a fixture, not for running the gate. Every
 * artifact above is committed, including model.safetensors -- until it was,
 * `.gitignore`'s blanket `*.safetensors` kept the weights out while tracking
 * their config.json, so a fresh clone held half a fixture and three tests
 * here failed on a missing file until someone ran `tiny-model` by hand. The
 * generator remains the source of truth: `checkpoint_fixtures_match_the_generator`
 * regenerates and byte-compares, so a committed fixture that drifts from the
 * spec still fails.
 */

use runtime::model::cpu::CpuModel;
use runtime::model::Model;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(name)
}

/// The uniform fixture, and the E4B-shaped one that carries all seven
/// architectural features. Both run through every golden below: the uniform
/// image is what proves the features are absent-safe, and the E4B-shaped one
/// is what proves they are right.
const UNIFORM: &str = "tiny-gemma4";
const E4B: &str = "tiny-gemma4-e4b";

#[test]
fn cpu_backend_matches_the_python_reference() {
    check_logits(UNIFORM);
}

#[test]
fn cpu_backend_matches_the_python_reference_on_the_e4b_fixture() {
    check_logits(E4B);
}

fn check_logits(stem: &str) {
    let (files, golden) = load_golden_named(stem);
    let ck = runtime::checkpoint::parse(&files).unwrap();
    let mut model = CpuModel::from_checkpoint(&ck).unwrap();

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
                "{stem}: first divergence at position {position}, token {t}: \
                 got {g}, reference {w}"
            );
        }
    }
}

/// The committed checkpoint's bytes, owned so a `Checkpoint` can borrow them.
fn checkpoint_files(stem: &str) -> runtime::checkpoint::CheckpointFiles {
    runtime::checkpoint::CheckpointFiles::read(&fixture_path(stem))
        .unwrap_or_else(|e| panic!("read fixture {stem}: {e:#} (regenerate; see header)"))
}

fn load_golden_named(stem: &str) -> (runtime::checkpoint::CheckpointFiles, serde_json::Value) {
    let golden = serde_json::from_str(
        &std::fs::read_to_string(fixture_path(&format!("{stem}-logits.json"))).unwrap(),
    )
    .unwrap();
    (checkpoint_files(stem), golden)
}

fn load_golden() -> (runtime::checkpoint::CheckpointFiles, serde_json::Value) {
    load_golden_named(UNIFORM)
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
    check_tap_rows(UNIFORM);
}

/// On the E4B-shaped fixture this also pins where the layer scalar lands:
/// the reference multiplies by `layer_output_scale` as the layer's last
/// operation, so a backend that applied it before the per-layer-embedding
/// residual would diverge here even though its logits could still agree.
#[test]
fn tap_rows_match_the_python_reference_residuals_on_the_e4b_fixture() {
    check_tap_rows(E4B);
}

fn check_tap_rows(stem: &str) {
    use runtime::model::cpu::Tap;
    use std::sync::{Arc, Mutex};

    let (files, golden) = load_golden_named(stem);
    let ck = runtime::checkpoint::parse(&files).unwrap();
    let mut model = CpuModel::from_checkpoint(&ck).unwrap();
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
                "{stem}: layer {layer} residual element {j}: got {g}, reference {w}"
            );
        }
    }
}

#[test]
fn steered_logits_match_the_python_reference() {
    check_steered(UNIFORM);
}

#[test]
fn steered_logits_match_the_python_reference_on_the_e4b_fixture() {
    check_steered(E4B);
}

fn check_steered(stem: &str) {
    use runtime::model::cpu::ControlVector;

    let (files, golden) = load_golden_named(stem);
    let ck = runtime::checkpoint::parse(&files).unwrap();
    let mut model = CpuModel::from_checkpoint(&ck).unwrap();
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
    // direction.N steers block N, as in llama.cpp: the tensor name is the
    // 0-based block index the reference steers, not that index plus one.
    w.tensor_f32(
        &format!("direction.{steer_layer}"),
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

/// Probe GGUF mirroring the reference's synthetic probe fixture: weights
/// 0.1 * (j - 4), mu 0.01 * j, sigma 1 + 0.1 * j, bias 0.1 over the given
/// layer and width.
fn probe_gguf(layer: u32, width: usize, threshold: f32) -> Vec<u8> {
    let mut w = runtime::synth::GgufWriter::new();
    w.kv_str("general.architecture", "probe");
    w.kv_str("probe.name", "reference-probe");
    w.kv_str("probe.model_hash", "unknown");
    w.kv_u32("probe.layer", layer);
    w.kv_f32("probe.bias", 0.1);
    w.kv_f32("probe.threshold", threshold);
    let dims = [width as u64];
    let weights: Vec<f32> = (0..width).map(|j| 0.1 * (j as f32 - 4.0)).collect();
    let mu: Vec<f32> = (0..width).map(|j| 0.01 * j as f32).collect();
    let sigma: Vec<f32> = (0..width).map(|j| 1.0 + 0.1 * j as f32).collect();
    w.tensor_f32("probe.weights", &dims, &weights);
    w.tensor_f32("probe.mu", &dims, &mu);
    w.tensor_f32("probe.sigma", &dims, &sigma);
    w.finish()
}

#[test]
fn probe_monitor_scores_match_the_python_reference() {
    use runtime::model::probe::{Probe, ProbeMonitor};
    use std::sync::Arc;

    let (files, golden) = load_golden();
    let ck = runtime::checkpoint::parse(&files).unwrap();
    let mut model = CpuModel::from_checkpoint(&ck).unwrap();
    let embed_dim = model.config().embed_dim;
    let prompt: Vec<u32> = golden["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let probe_layer = golden["probe_layer"].as_u64().unwrap() as u32;
    let want = f32_row(&golden["probe_scores"]);

    // Threshold below every reference score, so every position is flagged
    // and the flag list carries the full per-position score sequence.
    let p_image = probe_gguf(probe_layer, embed_dim, 0.0);
    let p_file = runtime::gguf::parse(&p_image).unwrap();
    let probe = Arc::new(Probe::from_gguf(&p_file, &p_image).unwrap());
    let (tap, state) = ProbeMonitor::new(probe);
    model.set_tap(Box::new(tap));

    let baseline = {
        let mut plain = CpuModel::from_checkpoint(&ck).unwrap();
        plain.prefill_all(&prompt).unwrap()
    };
    let rows = model.prefill_all(&prompt).unwrap();
    // Observation only: monitored and unmonitored logits are bit-identical.
    assert_eq!(rows, baseline);

    let state = state.lock().unwrap();
    assert_eq!(state.scored, prompt.len());
    assert_eq!(state.flags.len(), want.len());
    for (position, (flag, w)) in state.flags.iter().zip(&want).enumerate() {
        assert_eq!(flag.position, position);
        assert!(
            (flag.score - w).abs() <= TOLERANCE,
            "probe score at position {position}: got {}, reference {w}",
            flag.score
        );
    }
}

#[test]
fn a_bfloat16_model_matches_the_f32_model_holding_the_same_values() {
    // Losslessness, stated as the property that matters: bfloat16 storage
    // does not change what reaches the forward pass. Build the same model
    // twice — once with bf16 tensors, once with f32 tensors already rounded
    // to bf16 precision — and require *bit-identical* logits, not a
    // tolerance. Any rounding, exponent rebiasing, or reuse of the IEEE-half
    // converter would show up here as a mismatch.
    //
    // This is what lets a bf16 Gemma 4 checkpoint be loaded into the f32
    // backend without a fidelity caveat.
    use runtime::checkpoint::CheckpointFiles;
    use runtime::synth::{build_tiny_config_json, build_tiny_safetensors, TensorStorage, TinySpec};
    use safetensors::tensor::{SafeTensors, TensorView};

    let spec = TinySpec::default();
    let config = build_tiny_config_json(&spec).unwrap();
    let bf16_weights = build_tiny_safetensors(&TinySpec {
        storage: TensorStorage::Bf16,
        ..spec
    })
    .unwrap();
    {
        let parsed = SafeTensors::deserialize(&bf16_weights).unwrap();
        assert!(
            parsed
                .tensors()
                .iter()
                .all(|(_, v)| v.dtype() == safetensors::Dtype::BF16),
            "fixture must actually be bf16, or the test proves nothing"
        );
    }

    // The f32 twin: identical weights with the low 16 mantissa bits cleared,
    // which is exactly what the bf16 writer truncates away. Masked after the
    // norm unshift rather than before, because that is the order the bf16
    // writer applies them and the two must narrow the same value.
    let f32_weights = build_tiny_safetensors(&spec).unwrap();
    let truncated_weights = {
        let parsed = SafeTensors::deserialize(&f32_weights).unwrap();
        let masked: Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)> = parsed
            .tensors()
            .into_iter()
            .map(|(name, v)| {
                let bytes = v
                    .data()
                    .chunks_exact(4)
                    .flat_map(|c| {
                        let bits = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                        (bits & 0xffff_0000).to_le_bytes()
                    })
                    .collect();
                (name, v.dtype(), v.shape().to_vec(), bytes)
            })
            .collect();
        let views: Vec<(&str, TensorView)> = masked
            .iter()
            .map(|(n, d, shape, bytes)| {
                (
                    n.as_str(),
                    TensorView::new(*d, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        safetensors::serialize(views, None).unwrap()
    };

    let dir = std::path::PathBuf::from("<memory>");
    let bf16_files = CheckpointFiles::in_memory(config.clone(), bf16_weights, dir.clone()).unwrap();
    let f32_files = CheckpointFiles::in_memory(config, truncated_weights, dir).unwrap();
    let bf16_ck = runtime::checkpoint::parse(&bf16_files).unwrap();
    let f32_ck = runtime::checkpoint::parse(&f32_files).unwrap();
    let mut bf16_model = CpuModel::from_checkpoint(&bf16_ck).unwrap();
    let mut f32_model = CpuModel::from_checkpoint(&f32_ck).unwrap();

    let prompt: Vec<u32> = vec![1, 5, 9, 13, 3, 8, 21, 2];
    let from_bf16 = bf16_model.prefill_all(&prompt).unwrap();
    let from_f32 = f32_model.prefill_all(&prompt).unwrap();
    assert_eq!(
        from_bf16, from_f32,
        "bf16 storage changed the forward pass; widening is not lossless"
    );

    // And the values are real, not all-zero from a misread payload.
    assert!(
        from_bf16[0].0.iter().any(|v| v.abs() > 1e-6),
        "logits are degenerate, so equality proves nothing"
    );
}

/// The tokenizer fixtures are pinned the same way the weights are, for the same
/// reason and one more: the serialized form is produced by `tokenizers`, so a
/// dependency upgrade that changes it shows up here as a fixture diff rather
/// than as a silent change in how prompts are segmented.
#[test]
fn tokenizer_fixtures_match_the_generator() {
    for (name, generated) in [
        (
            "tiny-gemma4.tokenizer.json",
            runtime::synth::build_tiny_tokenizer_json(&runtime::synth::TinySpec::default())
                .unwrap(),
        ),
        (
            "tiny-gemma4-e4b.tokenizer.json",
            runtime::synth::build_tiny_tokenizer_json(&runtime::synth::TinySpec::e4b_shaped())
                .unwrap(),
        ),
        (
            "tiny-bpe.tokenizer.json",
            runtime::synth::merge_fixture_tokenizer_json().unwrap(),
        ),
    ] {
        let committed = std::fs::read(fixture_path(name))
            .unwrap_or_else(|e| panic!("read fixture {name}: {e} (regenerate; see header)"));
        assert_eq!(
            committed.len(),
            generated.len(),
            "{name}: committed is {} bytes, generator writes {} (regenerate; see header)",
            committed.len(),
            generated.len()
        );
        assert_eq!(committed, generated, "{name}: regenerate (see header)");
    }
}

/// The geometry choices the E4B-shaped fixture exists to make observable.
/// Each one is a convention the committed uniform fixture cannot distinguish,
/// which is QA finding A1-15; asserting them here means a later edit to the
/// spec cannot quietly reintroduce the blindness while the goldens still pass.
#[test]
fn the_e4b_fixture_geometry_separates_the_ambiguous_conventions() {
    let spec = runtime::synth::TinySpec::e4b_shaped();
    assert_ne!(
        spec.head_dim,
        spec.embed_dim / spec.n_heads,
        "sliding head width equals embed_dim / n_heads, so the two candidate \
         QK scales coincide and the golden cannot tell them apart"
    );
    assert_ne!(
        spec.head_dim_global.unwrap(),
        spec.embed_dim / spec.n_heads,
        "global head width equals embed_dim / n_heads, same blindness"
    );
    assert_ne!(
        spec.head_dim,
        spec.head_dim_global.unwrap(),
        "both regimes share a head width, so per-layer head dim is untested"
    );
    assert!(
        spec.n_kv_heads > 1 && spec.n_heads / spec.n_kv_heads > 1,
        "blocked and interleaved GQA mappings agree unless there are several \
         KV heads and several query heads per KV head"
    );
    // The shared tail must span both attention regimes, or the single-donor
    // rule and the per-type rule agree and the fixture cannot separate them.
    let boundary = spec.n_layers - spec.shared_kv_layers;
    let is_global = |i: usize| (i + 1).is_multiple_of(spec.sliding_window_pattern);
    assert!(
        (boundary..spec.n_layers).any(is_global)
            && (boundary..spec.n_layers).any(|i| !is_global(i)),
        "the shared-KV tail is all one layer kind, so one donor and two \
         donors give the same answer"
    );
    assert!(
        spec.sliding_window < 8,
        "the window must be narrower than the reference prompt or the sliding \
         layers never drop a position"
    );
}

/// The committed checkpoint fixtures are what the generator writes.
///
/// The same pin the GGUF and tokenizer fixtures carry, extended to
/// `config.json` and `model.safetensors`. Without it the committed
/// checkpoint could drift from `TinySpec` while every in-memory test stayed
/// green, and the Python oracle -- which reads the committed files and
/// nothing else -- would be describing a different model from the one the
/// Rust tests load.
#[test]
fn checkpoint_fixtures_match_the_generator() {
    use runtime::synth::{build_tiny_config_json, build_tiny_safetensors, TinySpec};

    for (stem, spec) in [
        (UNIFORM, TinySpec::default()),
        (E4B, TinySpec::e4b_shaped()),
    ] {
        for (file, generated) in [
            ("config.json", build_tiny_config_json(&spec).unwrap()),
            ("model.safetensors", build_tiny_safetensors(&spec).unwrap()),
        ] {
            let name = format!("{stem}/{file}");
            let committed = std::fs::read(fixture_path(&name))
                .unwrap_or_else(|e| panic!("read fixture {name}: {e} (regenerate; see header)"));
            assert_eq!(
                committed.len(),
                generated.len(),
                "{name}: committed is {} bytes, generator writes {} (regenerate; see header)",
                committed.len(),
                generated.len()
            );
            assert_eq!(committed, generated, "{name}: regenerate (see header)");
        }
    }
}

/// The committed checkpoint loads, and its logits are the committed goldens.
///
/// The goldens were computed by the numpy oracle reading these same
/// safetensors files, so this closes the loop the equality test opens: that
/// one proves the two loaders agree with each other in memory, and this one
/// proves the checkpoint loader agrees with an independent implementation
/// over the bytes on disk.
#[test]
fn the_committed_checkpoint_matches_the_python_reference() {
    for stem in [UNIFORM, E4B] {
        let dir = fixture_path(stem);
        let files = runtime::checkpoint::CheckpointFiles::read(&dir)
            .unwrap_or_else(|e| panic!("{stem}: {e:#}"));
        let ck = runtime::checkpoint::parse(&files).unwrap_or_else(|e| panic!("{stem}: {e:#}"));
        let mut model = CpuModel::from_checkpoint(&ck).unwrap_or_else(|e| panic!("{stem}: {e:#}"));

        let (_, golden) = load_golden_named(stem);
        let prompt: Vec<u32> = golden["prompt"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected = golden["logits"].as_array().unwrap();
        let rows = model.prefill_all(&prompt).unwrap();
        assert_eq!(rows.len(), expected.len(), "{stem}: row count");
        for (pos, (row, want)) in rows.iter().zip(expected).enumerate() {
            let want = f32_row(want);
            for (token, (&got, &w)) in row.0.iter().zip(&want).enumerate() {
                assert!(
                    (got - w).abs() < TOLERANCE,
                    "{stem}: position {pos} token {token}: got {got}, reference {w}"
                );
            }
        }
    }
}
