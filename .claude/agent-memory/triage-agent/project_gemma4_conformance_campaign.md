---
name: gemma4-calculator-scalar-root
description: Most Gemma 4 training-calculator findings share one root - ArchitectureSpec is a flat scalar dataclass that cannot express per-layer-type widths
metadata:
  type: project
verified: 2026-08-29
stale_when: ArchitectureSpec in training/calculator/components.py gains per-layer-type fields (layer_types, use_double_wide_mlp, or a per-layer ffn)
---

In `/work/training/calculator/components.py`, `ArchitectureSpec` is a frozen dataclass of
scalars: one `ffn`, one `explicit_head_dim`, one implicit two-norms-per-layer constant. Gemma 4
needs per-layer-type values, so a large family of audit findings share this single structural
root rather than being independent bugs:

- E2B's 20 double-wide MLP layers (`use_double_wide_mlp`, tail `num_kv_shared_layers` layers at
  `2 * intermediate_size`) cannot be expressed - roughly 566M parameters unbooked.
- The 7 full-attention layers at `global_head_dim` 512 are priced at `head_dim` 256 - 49.5M
  (E2B) / 91.75M (E4B) unbooked.
- The same scalars flow into `transformer_stack_param_count`, `linear_param_count`,
  `adapter_param_count` (`_TARGET_MODULE_DIMS`), `activation_phase_bytes`, and
  `teacher_forward_bytes`, so one fix at the data model corrects every consumer.

`use_double_wide_mlp`, `global_head_dim`, `num_kv_shared_layers`, `attention_k_eq_v`,
`layer_types`, and `num_global_key_value_heads` have **no consumer in code**: as of
2026-08-06 a grep across `*.py`, `*.rs`, and `*.yaml` outside `docs/` returns zero hits for
all six. They now appear in two further analysis documents besides the spec —
`docs/training/gemma-4-training.md` (which states outright that `use_double_wide_mlp`
"appears nowhere in `/training` or `/common`" and that `num_kv_shared_layers`/`layer_types`
"have no consumer") and the rust harness ingestion reference, since deleted with the GGUF
loader it described, its surviving facts folded into `docs/models/gemma-4.md`. Those corroborate the
structural claim rather than weakening it; do not read their existence as the gap being
closed.

**Why:** the failures are silent - nothing raises, `memfit` just returns an optimistic
`fits`. Grouping them under one root keeps the remediation to a single package instead of a
dozen point patches that each add another scalar.

**How to apply:** when triaging a new calculator finding about a per-layer Gemma 4 quantity,
check whether it is another face of this data-model gap before treating it as separate; keep
them in one file-disjoint package. Verify the current dataclass shape first - it may already
carry per-layer fields. Related: [[gemma4-2304-propagation]],
[[project-gemma4-conformance-audit]].

**Re-graded 2026-08-29 (bench campaign, phase 2).** The `stale_when` trigger has fired and
this memory is now partly wrong. `ArchitectureSpec` carries `use_double_wide_mlp` and
`num_kv_shared_layers` (`training/calculator/components.py:120-121`), consumed at `:131-134`
and populated from config at `:277-278`, with coverage in
`tests/training/test_calculator.py:346-347` and
`tests/models/test_gemma4_facts_propagation.py:274-275`. The double-wide MLP bullet above is
therefore closed, and the "no consumer in code" grep result is stale for those two names.
Still consumerless outside `docs/` and test scaffolding: `layer_types`, `global_head_dim`,
`attention_k_eq_v`, `num_global_key_value_heads`. The structural claim survives for the
head-dim bullet only. Not rewritten further here: this campaign's scope is `bench/`, and the
memory was re-checked rather than applied. A calculator campaign should rewrite it from the
tree before using it.
