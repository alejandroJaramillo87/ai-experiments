/*
 * eval_acceptance.rs — Tier 3 output-correctness tests (workstation).
 *
 * These verify the real engine's *output quality*, not its plumbing. They run
 * only on the RTX 5090 workstation with the cuda backend, by a human, and the
 * reported output is recorded verbatim per repository operating rules. Run
 * order matters: logits match (cuda_acceptance.rs) first, then perplexity,
 * then the greedy goldens — each layer only means something once the previous
 * one passes.
 */

/// Layer 2: perplexity on a fixed WikiText-2 sample matches the llama.cpp
/// reference within 1%.
///
/// Fixture generation (workstation, once):
///
/// Step 1 — fixtures/wikitext2_sample.txt: first ~50 KiB of the WikiText-2
/// test split, committed as plain text.
///
/// Step 2 — fixtures/wikitext2_sample.json (token IDs):
///
/// ```text
/// python -c "from transformers import AutoTokenizer; import json; \
///   t = AutoTokenizer.from_pretrained('google/gemma-4-e4b'); \
///   ids = t(open('fixtures/wikitext2_sample.txt').read())['input_ids']; \
///   json.dump(ids, open('fixtures/wikitext2_sample.json', 'w'))"
/// ```
///
/// Step 3 — fixtures/reference_ppl.json: {"ppl": <value>, "chunk": 512,
/// "command": "<the llama-perplexity invocation>"}, where the value is the
/// final PPL printed verbatim by:
///
/// ```text
/// llama-perplexity -m <same GGUF> -f fixtures/wikitext2_sample.txt -c 512
/// ```
///
/// Workstation command:
///
/// ```text
/// cargo test -p runtime --features cuda --release -- --ignored perplexity_matches --nocapture
/// ```
#[test]
#[ignore = "Tier 3: requires the RTX 5090 workstation, the cuda feature, and committed fixtures"]
fn perplexity_matches_llama_cpp_reference() {
    unimplemented!(
        "load quantized E4B, run eval::perplexity on fixtures/wikitext2_sample.json \
         with the fixture's chunk size, assert within 1% of fixtures/reference_ppl.json"
    );
}

/// Layer 3: temperature-0 continuations of fixed prompts byte-match committed
/// goldens. Goldens are regression pins, not ground truth: they are generated
/// by the FIRST verified run (after the logits and perplexity tests pass) and
/// committed; any later engine change that shifts an argmax fails here even
/// when per-position logits stay within tolerance.
///
/// Fixture: fixtures/greedy_goldens.json holds 3-5 prompts as
/// [{"prompt_ids": [...], "expected_ids": [...64 tokens...]}].
///
/// Workstation command:
///
/// ```text
/// cargo test -p runtime --features cuda --release -- --ignored greedy_decode_matches --nocapture
/// ```
#[test]
#[ignore = "Tier 3: requires the RTX 5090 workstation, the cuda feature, and committed goldens"]
fn greedy_decode_matches_golden() {
    unimplemented!(
        "for each fixture prompt: Session::generate with temperature 0, max 64 tokens, \
         assert token-for-token equality with expected_ids"
    );
}
