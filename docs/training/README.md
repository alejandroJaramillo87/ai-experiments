# Training documentation

Living documentation for the training stack — the `python -m training` CLI, the reference
loop, the recipe system, and the fine-tuning and distillation harnesses.

The code itself is at [../../training/](../../training/); this directory documents how to
operate it. Verification records for training phases live in
[../verification/](../verification/), not here, and the model-side facts about Gemma 4 that
both tiers share are in [../models/](../models/).

## Table of contents

- [Phase 0: the training substrate](#phase-0-the-training-substrate)
- [Phase 1: fine-tuning and distillation](#phase-1-fine-tuning-and-distillation)
- [Phase 2 and scale references](#phase-2-and-scale-references)
- [Design records](#design-records)
- [Related documentation](#related-documentation)

## Phase 0: the training substrate

| Document | Description |
|----------|-------------|
| [data_pipeline.md](data_pipeline.md) | Filter, dedup, decontaminate, and shard the corpus (D2, extended by P2-D1) |
| [tokenizer_training.md](tokenizer_training.md) | Byte-level BPE training for the from-scratch path only (D3) |
| [memory_fit.md](memory_fit.md) | Pre-run VRAM estimation against the 32GB card (D4) |
| [tracking_checkpointing.md](tracking_checkpointing.md) | The reference loop, atomic resumable checkpoints, metric tracking (D6) |
| [train_serve_roundtrip.md](train_serve_roundtrip.md) | Export, serve, and smoke-evaluate a trained model (D7) |
| [heterogeneous_compute.md](heterogeneous_compute.md) | Prefetch loader, staging, and CPU affinity (D9) |
| [operational_watchdogs.md](operational_watchdogs.md) | Abort-or-alert for unattended runs, and the hardware telemetry source (D10) |

## Phase 1: fine-tuning and distillation

| Document | Description |
|----------|-------------|
| [sft_entry_point.md](sft_entry_point.md) | The supervised fine-tuning trainer, full and QLoRA (F3) |
| [teacher_synthetic_data.md](teacher_synthetic_data.md) | Teacher responses and synthetic instruction data through the inference services (F4) |
| [distillation.md](distillation.md) | Response-level and logit-level KL distillation (F5) |
| [eval_wrapper.md](eval_wrapper.md) | Baseline-versus-tuned scorecards from three skippable evaluators (F7) |
| [scorecards/README.md](scorecards/README.md) | The committed reference scorecards themselves (F8) |

## Phase 2 and scale references

| Document | Description |
|----------|-------------|
| [gemma-4-training.md](gemma-4-training.md) | Parameter accounting, memory, tensor names, and tokenizer for Gemma 4 |
| [training-scale-reference.md](training-scale-reference.md) | Sub-1B budgets: batch and sequence sizing, wall-clock expectations |

Phase 2 deliverables beyond the data pipeline — data mixing, the warmup-stable-decay
schedule, the replay buffer, and optimizer offload — are documented in their design records
rather than in living documents. See the table below.

## Design records

Each deliverable was implemented from an investigation recorded under
[investigations/](investigations/), whose
[README.md](investigations/README.md) states the directory's conventions. Those are **frozen
records**: what was decided and measured on a given day, never edited afterward, so drift
against the current tree is expected and correct (`.styles/DOCUMENTATION_STYLE.md`). The
pairing is not always one to one, and the gaps are listed rather than left blank.

| Living document | Deliverable | Design record |
|-----------------|-------------|---------------|
| [tokenizer_training.md](tokenizer_training.md) | D3 | [d3_tokenizer_training_investigation.md](investigations/d3_tokenizer_training_investigation.md) |
| [memory_fit.md](memory_fit.md) | D4 | [d4_memory_fit_investigation.md](investigations/d4_memory_fit_investigation.md) |
| [tracking_checkpointing.md](tracking_checkpointing.md) | D6 | [d6_tracking_checkpointing_investigation.md](investigations/d6_tracking_checkpointing_investigation.md) |
| [train_serve_roundtrip.md](train_serve_roundtrip.md) | D7 | [d7_train_serve_investigation.md](investigations/d7_train_serve_investigation.md) |
| [heterogeneous_compute.md](heterogeneous_compute.md) | D9 | [d9_heterogeneous_compute_investigation.md](investigations/d9_heterogeneous_compute_investigation.md) |
| [operational_watchdogs.md](operational_watchdogs.md) | D10 | [d10_operational_watchdogs_investigation.md](investigations/d10_operational_watchdogs_investigation.md) |
| [sft_entry_point.md](sft_entry_point.md) | F3 | [f3_sft_entry_point_investigation.md](investigations/f3_sft_entry_point_investigation.md) |
| [teacher_synthetic_data.md](teacher_synthetic_data.md) | F4 | [f4_teacher_synthetic_data_investigation.md](investigations/f4_teacher_synthetic_data_investigation.md) |
| [distillation.md](distillation.md) | F5 | [f5_distillation_investigation.md](investigations/f5_distillation_investigation.md) |
| [eval_wrapper.md](eval_wrapper.md) | F7 | [f7_eval_wrapper_investigation.md](investigations/f7_eval_wrapper_investigation.md) |
| [scorecards/README.md](scorecards/README.md) | F8 | [f8_recipe_cards_investigation.md](investigations/f8_recipe_cards_investigation.md) |
| [data_pipeline.md](data_pipeline.md) | D2, P2-D1 | [p2_d1_full_scale_data_pipeline_investigation.md](investigations/p2_d1_full_scale_data_pipeline_investigation.md) |

Records with no paired living document. Each is the only documentation of its deliverable,
so read the record itself:

| Design record | Deliverable |
|---------------|-------------|
| [d5_recipe_system_investigation.md](investigations/d5_recipe_system_investigation.md) | D5, the recipe system every other deliverable consumes |
| [f1_sft_data_investigation.md](investigations/f1_sft_data_investigation.md) | F1, chat and instruction data preparation |
| [f6_adapter_merge_serve_investigation.md](investigations/f6_adapter_merge_serve_investigation.md) | F6, adapter merge and serve |
| [p2_d2_data_mixing_configuration_investigation.md](investigations/p2_d2_data_mixing_configuration_investigation.md) | P2-D2, data-mixing configuration |
| [p2_d3_warmup_stable_decay_investigation.md](investigations/p2_d3_warmup_stable_decay_investigation.md) | P2-D3, the warmup-stable-decay schedule |
| [p2_d4_replay_buffer_investigation.md](investigations/p2_d4_replay_buffer_investigation.md) | P2-D4, the replay buffer |
| [p2_d5_optimizer_offload_investigation.md](investigations/p2_d5_optimizer_offload_investigation.md) | P2-D5, optimizer offload |
| [unified_cli_investigation.md](investigations/unified_cli_investigation.md) | The `python -m training` and `python -m interp` CLI unification |

Living documents with no design record: [gemma-4-training.md](gemma-4-training.md) and
[training-scale-reference.md](training-scale-reference.md). Both are reference tables about
the model and the hardware rather than accounts of an implementation, so neither had an
investigation to produce it.

## Related documentation

- [../../training/README.md](../../training/README.md) — the source tree: what consumes what,
  entry points, and the traps
- [../interp/README.md](../interp/README.md) — the interpretability tier, which shares the
  recipe system and the observability stack
- [../observability/README.md](../observability/README.md) — reading logs, finding a run, and
  opening a dashboard for any training run
- [../models/gemma-4.md](../models/gemma-4.md) — the model reference both tiers build on
- [../verification/](../verification/) — the phase verification records and their host-side
  acceptance checks
