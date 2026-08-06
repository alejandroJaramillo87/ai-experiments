---
name: gemma4-2304-propagation
description: The wrong E2B geometry (30 layers, hidden 2304) has one documentary root in the P2-D5 offload investigation and copies in six downstream sites
metadata:
  type: project
verified: 2026-08-06
stale_when: p2_d5_optimizer_offload_investigation.md:125 stops citing 30 layers / hidden 2304, or a remediation package lands against the copy list below
---

The recurring wrong Gemma 4 E2B geometry in this tree — "30 layers, hidden 2304" — traces to
`/work/docs/training/investigations/p2_d5_optimizer_offload_investigation.md:125`, which cites
transformers `configuration_gemma4.py` as *confirming* the architecture. Those are class
defaults, not E2B's `text_config`. Real E2B is 35 layers at hidden 1536.

Known copies: `training/calculator/components.py:109` (comment), the P4-D2 activation
harvesting and P4-D3 SAE investigations, `docs/training/investigations/f3_sft_entry_point_investigation.md`,
the `training/recipes/interp/harvest-*.yaml` recipes (`harvest-smoke.yaml:13` "~9GB at d_model
2304", `harvest-false-fact.yaml:4` "~1.38TB at d_model 2304"), `training/recipes/schema.py:409-411`
(restates the footprint as "~1.4TB"), and `tests/training/test_calculator.py`. The "30-layer
stack" half also drives `hook_layers: [15]` in both harvest recipes, which on the real 35-layer
E2B lands on the first layer of the shared-KV / double-wide tail (spec:328-330, :344) rather
than at mid-depth. `harvest-calibration-gemma2.yaml:7` is *not* a copy: 2304 is correct for
Gemma 2 2B.

Further copies found 2026-08-06, not in the original list:
`docs/interpretability/investigations/p4_d2_activation_harvesting_investigation.md:109,127,138,180`,
`p4_d3_sae_training_investigation.md:76,120,122,208,214`,
`docs/inference/future-optimizations.md:156`, `tests/interp/test_families.py:53,61,103,106`,
and `tests/interp/test_harvest.py:48,121,125-127,412,418`.
`docs/interpretability/gemma-4-interp.md:73` discusses the error rather than repeating it and
is not a copy.

**Why:** a ledger that lists forty copies of one wrong number teaches nothing; naming the root
makes the downstream sites correctable in one pass.

**How to apply:** when a new site with 2304 or 30 layers surfaces, record it as a copy of this
root rather than as an independent finding. The spec-authority rule itself lives in
[[project-gemma4-conformance-audit]].
