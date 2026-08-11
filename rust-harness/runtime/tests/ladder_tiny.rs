/*
 * ladder_tiny.rs — the CPU backend against a Gemma 4 transformers authored.
 *
 * The payoff of the whole ladder package, and the one test in this workspace
 * whose reference nobody here wrote. Every other numeric test compares
 * runtime/src/model/cpu.rs against fixtures/reference_logits.py on random
 * weights; the same reader transcribed both from the same documents, so a
 * misreading lands in both and they agree about it. That gate stayed green
 * through nine work packages while the forward pass produced noise.
 *
 * Here transformers builds the config from its own classes, instantiates its
 * own Gemma 4, and runs its own forward pass. We supply shapes and seeded
 * weights and nothing else. A disagreement is therefore evidence about our
 * engine rather than about which of two transcriptions is the odd one out.
 *
 * Tier T tolerances are tight -- f32 against f32 over 32 dimensions, both
 * sides widening the same bfloat16 values, which is exact. If this needs
 * loosening to pass, the thing to fix is not the tolerance.
 *
 * L0 is not here: it needs the tokenizer-parity artifact, which is a separate
 * generator. L3 and L4 are Tier E rungs -- a perplexity over a corpus and a
 * 64-token greedy continuation are not statements a 4-layer model with random
 * weights can make.
 */

use std::path::{Path, PathBuf};

use runtime::ladder::{self, Status};
use runtime::model::cpu::{CpuModel, PRELUDE_STAGES};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../fixtures/tiny-hf-gemma4")
}

/// The fixture's model and capture, or `None` with a `[SKIP]` already logged.
///
/// Absent means the generator has not been run on a workstation, which is a
/// real state for a fresh clone and not a failure. Present-but-unreadable is
/// a panic: the manifest test next door is what guards the bytes, so a
/// capture that will not load here is a broken fixture rather than missing
/// evidence.
fn load(what: &str) -> Option<(CpuModel, runtime::reference::Capture)> {
    let dir = fixture_dir();
    if !dir.join("capture.json").exists() {
        runtime::testlog::skip(
            what,
            "fixtures/tiny-hf-gemma4 has not been generated; \
             run fixtures/build_tiny_reference.py on the workstation",
        );
        return None;
    }

    let capture = runtime::reference::load_capture(&dir.join("capture.json"), &PRELUDE_STAGES)
        .expect("read the reference capture")
        .expect("capture.json exists, so it must load");

    // Through the shared loader, which is what every binary uses: it reads
    // the header, copies each tensor at its offset, and closes the file before
    // returning. Nothing borrows the checkpoint afterwards, which is why the
    // `Box::leak` that used to stand here -- giving the mapped bytes a
    // 'static lifetime so the model could outlive them -- is gone.
    let (model, _fingerprint) = CpuModel::load(&dir, false).expect("load the fixture checkpoint");
    Some((model, capture))
}

/// L1: every stage feeding the stack, then every layer of it.
#[test]
fn the_residual_stream_matches_the_reference_layer_for_layer() {
    let Some((mut model, capture)) = load("Tier T L1") else {
        return;
    };
    assert!(
        !capture.activations.stages.is_empty(),
        "the capture recorded no pre-stack stages, so a layer-0 divergence \
         could not be attributed to its input"
    );

    let ours = ladder::capture_ours(&mut model, &capture.token_ids, &capture.activations)
        .expect("capture our own activations");
    let result = ladder::rung_layers(&ours, &capture.activations, ladder::TIER_T)
        .expect("compare the residual stream");

    assert_eq!(
        result.status,
        Status::Pass,
        "Tier T L1 diverged: {}\n{:#?}",
        result.note,
        result.measures
    );
}

/// L2: the full output distribution, not just its argmax.
#[test]
fn the_output_distribution_matches_the_reference() {
    let Some((mut model, capture)) = load("Tier T L2") else {
        return;
    };
    let rows = capture
        .logits
        .expect("the capture records final-position logits");
    assert_eq!(
        rows.len(),
        model.config().vocab_size,
        "the reference row is not as wide as the model's vocabulary"
    );

    // The last prompt position, which is the one the capture recorded.
    let last = capture.token_ids.len() - 1;
    let reference = ladder::LogitReference {
        positions: vec![last],
        rows: vec![rows],
    };
    let result = ladder::rung_logits(&mut model, &capture.token_ids, &reference, ladder::TIER_T)
        .expect("compare the distribution");

    assert_eq!(
        result.status,
        Status::Pass,
        "Tier T L2 diverged: {}\n{:#?}",
        result.note,
        result.measures
    );
}

/// The fixture exercises the features it was built to exercise.
///
/// A tiny fixture is worth exactly the code paths it reaches, and the two
/// rungs above would pass just as green on a fixture that had quietly lost
/// its second attention regime or its per-layer embeddings. Read back through
/// the checkpoint loader rather than compared against the generator's SHAPE,
/// because the generator is not what the rungs read: what matters is the
/// architecture that reached the engine.
#[test]
fn the_fixture_still_covers_the_features_it_was_built_for() {
    let dir = fixture_dir();
    let Ok(bytes) = std::fs::read(dir.join("config.json")) else {
        runtime::testlog::skip(
            "Tier T coverage",
            "fixtures/tiny-hf-gemma4 is not generated",
        );
        return;
    };
    let c = runtime::checkpoint::parse_text_config(&bytes).expect("read the fixture config");

    let types: std::collections::BTreeSet<_> = c.layer_types.iter().collect();
    assert_eq!(
        types.len(),
        2,
        "the fixture has one attention regime, so sliding and full attention \
         are no longer distinguished by it"
    );
    assert!(
        c.num_kv_shared_layers > 0,
        "the fixture shares no KV layers"
    );
    assert!(
        c.hidden_size_per_layer_input.is_some(),
        "the fixture has no per-layer input embeddings"
    );
    assert!(
        c.final_logit_softcapping.is_some(),
        "the fixture has no logit softcapping, which is the error class an \
         argmax comparison cannot see at all"
    );
    assert_ne!(
        Some(c.head_dim),
        c.global_head_dim,
        "the fixture's two attention regimes share a head width, so a backend \
         that assumes one width would coincide rather than fail"
    );
}
