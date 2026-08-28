---
name: gemma4-synth-fixture-root
description: Most rust-harness Gemma 4 findings share one root - the single synthetic GGUF fixture (synth.rs) encoded conventions the spec contradicts, so the whole rust test tier is blind to them
metadata:
  type: project
verified: 2026-08-10
closed: 2026-08-10
stale_when: rust-harness/fixtures/ gains a second Gemma-shaped fixture at a separating geometry, or synth.rs stops asserting bos=1/eos=2
---

**CLOSED.** Its own `stale_when` fired: `TinySpec::e4b_shaped()` added a second Gemma-shaped
fixture at a separating geometry (S5), and the weight path then moved off GGUF entirely (P2-P8).
`fixtures/tiny-gemma4/` and `fixtures/tiny-gemma4-e4b/` are now checkpoints, and every item
below is either fixed or unrepresentable in the current tree. Kept as the record of why the
rust test tier was once blind to a family of findings at once.

The record as written:

`/work/rust-harness/runtime/src/synth.rs` (`TinySpec` + `build_tiny_model`) produced a single
Gemma-shaped GGUF fixture, then the only one in the tree. Every rust golden,
determinism test, probe-identity test, and the numpy cross-check loaded it. It declares
`general.architecture = gemma4` while encoding several conventions the spec contradicts, so a
family of audit findings share this one root:

- `bos_token_id` 1 / `eos_token_id` 2 with Llama piece strings `<s>`/`</s>`, the transposition
  spec trap 8 exists to prevent (Gemma 4 is pad 0, eos 1, bos 2), pinned by an assertion.
- One `rope_freq_base`, so no test can see the two RoPE regimes.
- One `head_dim` and one `ffn_dim` for all layers, so the 256/512 head split and E2B's
  double-wide MLP are unrepresentable in any fixture.
- The tokenizer's own in-file test vocab repeats the bos/eos transposition (`from_parts(..., 1, 2)`),
  as does `harness/tests/generator_e2e.rs` - one assumption copied, not several slips.

**Why:** none of these reach wrong output today (the loader reads ids, shapes, and bases from
metadata, and `cpu::check_supported` refuses real E-series checkpoints on per-layer embeddings
first), so they grade low-to-medium individually. Their real cost is joint: a green
`cargo test` proves much less than it looks for this family.

**How to apply:** triage new rust-harness fixture findings as faces of this root and keep them
in one package rather than a dozen point edits. The prior campaign already recorded the same
class as **A1-15 (deferred, not waived)** in `docs/verification/rust-harness-qa.md` around
line 467 with a followup asking for a second fixture at a separating geometry - say "extends
A1-15" rather than reopening it. Related: [[project-gemma4-conformance-audit]],
[[gemma4-calculator-scalar-root]].
