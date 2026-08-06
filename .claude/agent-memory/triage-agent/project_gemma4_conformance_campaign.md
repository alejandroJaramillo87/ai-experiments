---
name: gemma4-calculator-scalar-root
description: Most Gemma 4 training-calculator findings share one root - ArchitectureSpec is a flat scalar dataclass that cannot express per-layer-type widths
metadata:
  type: project
verified: 2026-08-06
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
"have no consumer") and `docs/rust-harness/gemma-4-ingestion.md`. Those corroborate the
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
