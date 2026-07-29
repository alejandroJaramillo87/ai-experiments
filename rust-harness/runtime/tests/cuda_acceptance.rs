/*
 * cuda_acceptance.rs — Tier 3 host acceptance tests (phases A and B).
 *
 * These run only on the RTX 5090 workstation, by a human, with the real
 * Candle CUDA backend. They are ignored by default and unimplemented until
 * the `cuda` backend lands; the doc comments record the exact commands and
 * fixture-generation steps so the workstation run needs no archaeology.
 * Reported output is recorded verbatim per the repository operating rules.
 */

/// Phase A precondition: Candle works on Blackwell (sm_120) under CUDA 13.
///
/// Workstation command:
///   cargo test -p runtime --features cuda --release -- --ignored sm120_smoke
///
/// Checks tensor allocation and a matmul on the CUDA device. This is the
/// design doc's gating risk and runs before any model work; consult branch
/// claude/llm-runtime-sm120-wmrdz5 for prior findings.
#[test]
#[ignore = "Tier 3: requires the RTX 5090 workstation and the cuda feature"]
fn sm120_smoke_tensor_alloc_and_matmul() {
    unimplemented!("phase A: implement against the Candle CUDA backend");
}

/// Phase A acceptance: logits for a fixed prompt match the Hugging Face
/// reference within quantization tolerance.
///
/// Fixture generation (workstation, once):
///   python -c "..."  # transformers: load Gemma 4 E4B, run the fixed prompt,
///                    # dump final-position logits to fixtures/e4b_logits.json
/// Workstation command:
///   cargo test -p runtime --features cuda --release -- --ignored logits_match
#[test]
#[ignore = "Tier 3: requires the RTX 5090 workstation and the cuda feature"]
fn logits_match_reference_within_quantization_tolerance() {
    unimplemented!("phase A: load quantized E4B, forward pass, compare to fixture");
}

/// Phase B acceptance: turn latency scales with the appended delta, not the
/// transcript length. Prints per-turn prefill time for the delta path and
/// the full re-prefill path; the human-reported output is the measurement.
///
/// Workstation command:
///   cargo test -p runtime --features cuda --release -- --ignored delta_prefill_latency --nocapture
#[test]
#[ignore = "Tier 3: requires the RTX 5090 workstation and the cuda feature"]
fn delta_prefill_latency_scales_with_delta_size() {
    unimplemented!("phase B: multi-turn latency measurement, both paths");
}
