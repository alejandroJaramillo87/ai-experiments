# Training documentation

Living documentation for the training tier: the `python -m training` CLI, the recipe
system, the reference loop, and the three rungs the tier climbs. The code itself is at
[../../training/](../../training/); this directory documents how to operate it.
Verification records live in [../verification/](../verification/), not here, and the
model-side facts about Gemma 4 that both tiers share are in [../models/](../models/).

## Table of Contents

<!-- toc:start -->

1. [Layout](#layout)
2. [Progression](#progression)
3. [Design records](#design-records)
4. [Related documentation](#related-documentation)

<!-- toc:end -->

## Layout

One directory for what every rung shares, then one per rung, then the frozen records.
Each directory's README is its index, and this table is generated from them.

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [base/](base/README.md) | The substrate under fine-tuning, continual pretraining and training from scratch. |
| [continual/](continual/README.md) | The second rung: full-parameter continual pretraining of a Gemma 4 base on a mixed corpus, with a replay buffer against forgetting and the optimizer state offloaded to host RAM. |
| [fine-tuning/](fine-tuning/README.md) | The first rung: supervised fine-tuning of a Gemma 4 base into an organism, a model with a known implant the interp tier then recovers. |
| [from-scratch/](from-scratch/README.md) | The third rung: a model trained from random weights on this workstation, the one case where the lab holds the whole history of a model's weights. |
| [investigations/](investigations/README.md) | One record per training deliverable, written while that deliverable was being implemented. |

<!-- inventory:dirs:end -->

Two parked proposals sit at this level: the rung after the three, and the refactors the
three-rung trimming left for later.

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [continuous-learning-proposal.md](continuous-learning-proposal.md) | The long-term goal of this repository's training tier is a model that learns over time: weights that move as the model reads, at more than one rate, rather than static weights updated by a separate training run. |
| [tier-consolidation-proposal.md](tier-consolidation-proposal.md) | When the tier was trimmed to its three rungs (SFT, continual pretraining, training from scratch; see [README.md](README.md)), the survey that guided the trimming also found duplication that a removal cannot fix, because both copies are load-bearing. |

<!-- inventory:pages:end -->

## Progression

The tier exists to produce artifacts with a known change that the rust harness serves
and the interp tier recovers, and its rungs are ordered by how much of that history the
lab holds. Three are scheduled, in this order: supervised fine-tuning (verified by
[../verification/training/fine-tuning-loop.md](../verification/training/fine-tuning-loop.md)),
continual pretraining (its runbook is written when its first entry can be), and training
from scratch (the reference loop on the `pretrain-150m` and `pretrain-1b` recipes; the
Phase 0 acceptance checks on the loop closed in
[../verification/archive/from-scratch-loop.md](../verification/archive/from-scratch-loop.md)
and are kept verified by `tests/hardware/test_from_scratch_loop.py`). Distillation and
human-feedback training are out, for the reason the continuous-learning proposal states:
neither makes a claim about mechanism that the interp tier can test.

The substrate the rungs share was built as Phase 0 (D1 through D10) and audited on
2026-09-19; the fine-tuning harness was built as Phase 1 (F1 through F8) and audited on
2026-09-20, which retired what no rung called. Both audits are in the records below. D8,
FP8 enablement and the training benchmark harness, was never built: `precision.dtype`
admits `bf16` only, and FP8 is deferred to a later rung as one measured change
([base/training-scale-reference.md](base/training-scale-reference.md)). A benchmark
harness, when it lands, adopts the shared report envelope in
[../observability/report_schema.md](../observability/report_schema.md).

## Design records

Each deliverable was implemented from an investigation recorded under
[investigations/](investigations/), whose [README.md](investigations/README.md) states the
directory's conventions. Those are **frozen records**: what was decided and measured on a
given day, never edited afterward, so drift against the current tree is expected and
correct (`.styles/DOCUMENTATION_STYLE.md`). The pairing is not always one to one, and the
gaps are listed rather than left blank.

| Living document | Deliverable | Design record |
|-----------------|-------------|---------------|
| [from-scratch/tokenizer_training.md](from-scratch/tokenizer_training.md) | D3 | [d3_tokenizer_training_investigation.md](investigations/d3_tokenizer_training_investigation.md) |
| [base/train_step.md](base/train_step.md) | D4 | [d4_memory_fit_investigation.md](investigations/d4_memory_fit_investigation.md) |
| [base/recipe_system.md](base/recipe_system.md) | D5 | [d5_recipe_system_investigation.md](investigations/d5_recipe_system_investigation.md) |
| [base/tracking_checkpointing.md](base/tracking_checkpointing.md) | D6 | [d6_tracking_checkpointing_investigation.md](investigations/d6_tracking_checkpointing_investigation.md) |
| [base/train_serve_roundtrip.md](base/train_serve_roundtrip.md) | D7 | [d7_train_serve_investigation.md](investigations/d7_train_serve_investigation.md) |
| [base/heterogeneous_compute.md](base/heterogeneous_compute.md) | D9 | [d9_heterogeneous_compute_investigation.md](investigations/d9_heterogeneous_compute_investigation.md) |
| [base/operational_watchdogs.md](base/operational_watchdogs.md) | D10 | [d10_operational_watchdogs_investigation.md](investigations/d10_operational_watchdogs_investigation.md) |
| [fine-tuning/sft_data.md](fine-tuning/sft_data.md) | F1 | [f1_sft_data_investigation.md](investigations/f1_sft_data_investigation.md) |
| [fine-tuning/sft_entry_point.md](fine-tuning/sft_entry_point.md) | F3 | [f3_sft_entry_point_investigation.md](investigations/f3_sft_entry_point_investigation.md) |
| [fine-tuning/teacher_synthetic_data.md](fine-tuning/teacher_synthetic_data.md) | F4 | [f4_teacher_synthetic_data_investigation.md](investigations/f4_teacher_synthetic_data_investigation.md) |
| [fine-tuning/eval_wrapper.md](fine-tuning/eval_wrapper.md) | F7 | [f7_eval_wrapper_investigation.md](investigations/f7_eval_wrapper_investigation.md) |
| [base/data_pipeline.md](base/data_pipeline.md) | D2, P2-D1 | [p2_d1_full_scale_data_pipeline_investigation.md](investigations/p2_d1_full_scale_data_pipeline_investigation.md) |
| [continual/README.md](continual/README.md) | P2-D2 to P2-D5 | the four records below |

Records with no paired living document. Each is the only documentation of its deliverable,
so read the record itself:

| Design record | Deliverable |
|---------------|-------------|
| [f5_distillation_investigation.md](investigations/f5_distillation_investigation.md) | F5, the distillation harness; its implementation was removed on 2026-09-18 and the record is what remains |
| [f6_adapter_merge_serve_investigation.md](investigations/f6_adapter_merge_serve_investigation.md) | F6, adapter merge and serve; the merge half is documented with the export in `base/`, the serve half is the rust harness |
| [f8_recipe_cards_investigation.md](investigations/f8_recipe_cards_investigation.md) | F8, the recipe cards and their promotion; the promotion step and the committed-scorecard directory were removed on 2026-09-19, and the E4B card on 2026-09-20 |
| [p2_d2_data_mixing_configuration_investigation.md](investigations/p2_d2_data_mixing_configuration_investigation.md) | P2-D2, data-mixing configuration |
| [p2_d3_warmup_stable_decay_investigation.md](investigations/p2_d3_warmup_stable_decay_investigation.md) | P2-D3, the warmup-stable-decay schedule |
| [p2_d4_replay_buffer_investigation.md](investigations/p2_d4_replay_buffer_investigation.md) | P2-D4, the replay buffer |
| [p2_d5_optimizer_offload_investigation.md](investigations/p2_d5_optimizer_offload_investigation.md) | P2-D5, optimizer offload |
| [unified_cli_investigation.md](investigations/unified_cli_investigation.md) | The `python -m training` and `python -m interp` CLI unification |
| [phase0_completeness_audit_investigation.md](investigations/phase0_completeness_audit_investigation.md) | The 2026-09-19 audit of D1 through D10 against the design, the docs, the code, the tests, the runbooks, and the later tiers |
| [phase1_completeness_audit_investigation.md](investigations/phase1_completeness_audit_investigation.md) | The 2026-09-20 audit of F1 through F8 against the design, the docs, the code, the tests, the runbook and the consumers in the later tiers, with the fifth question: is it still needed |

Living documents with no design record: [base/gemma-4-training.md](base/gemma-4-training.md)
and [base/training-scale-reference.md](base/training-scale-reference.md). Both are reference
tables about the model and the hardware rather than accounts of an implementation, so neither
had an investigation to produce it. The two proposals above are parked proposals, not living
documents, and are edited only as their items graduate.

## Related documentation

- [../../training/README.md](../../training/README.md) — the source tree: what consumes what,
  entry points, and the traps
- [../sandbox/docker_training_overview.md](../sandbox/docker_training_overview.md) — the
  training container, its dependency source of truth, and the smoke test (D1)
- [../interp/README.md](../interp/README.md) — the interpretability tier, which shares the
  recipe system and the observability stack
- [../observability/README.md](../observability/README.md) — reading logs, finding a run, and
  opening a dashboard for any training run
- [../models/gemma-4.md](../models/gemma-4.md) — the model reference both tiers build on
- [../verification/](../verification/) — the verification runbooks and their host-side
  acceptance checks
