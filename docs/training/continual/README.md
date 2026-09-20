# Continual pretraining

The second rung: full-parameter continual pretraining of a Gemma 4 base on a mixed corpus,
with a replay buffer against forgetting and the optimizer state offloaded to host RAM. The
pipeline, mixtures, schedule, replay buffer and offload were built as Phase 2 (P2-D1 to
P2-D5) and are documented in their design records; this page is the index of the entry
points and the records, and the living page for the rung until its runbook exists. The
continual-pretraining runbook is written when its first entry can be.

## Entry points

| Target | What it does |
|--------|--------------|
| `make prepare-corpus ARGS="..."` | filter, dedup, decontaminate and shard a corpus ([../base/data_pipeline.md](../base/data_pipeline.md)) |
| `make build-contamination ARGS="..."` | build the benchmark decontamination index the pipeline consults |
| `python -m training make-replay-synthetic --out ... --gb ...` | a synthetic replay shard set sized to exercise the replay buffer without a downloaded corpus (P2-D4) |
| `make train-step CONFIG=training/recipes/reference/continual-gemma4-e2b.yaml` | step the recipe on the GPU before a long run ([../base/train_step.md](../base/train_step.md)); an offload recipe is refused here, its runtime is the DeepSpeed engine |
| `make train CONFIG=training/recipes/reference/continual-gemma4-e2b.yaml` | the run: the reference loop with mixtures, the warmup-stable-decay schedule, replay and ZeRO-Offload |

The reference recipe, `continual-gemma4-e2b.yaml`, trains the transformer stack with the
embeddings, the tied head and the per-layer embedding tables frozen
(`trainable_filter: transformer_stack`) under ZeRO-Offload, and is marked draft in its own
header until its calibration run back-fills the measured figures.

## Design records

| Record | Deliverable |
|--------|-------------|
| [p2_d1_full_scale_data_pipeline_investigation.md](../investigations/p2_d1_full_scale_data_pipeline_investigation.md) | P2-D1, the full-scale data pipeline (living page: [../base/data_pipeline.md](../base/data_pipeline.md)) |
| [p2_d2_data_mixing_configuration_investigation.md](../investigations/p2_d2_data_mixing_configuration_investigation.md) | P2-D2, data mixing: draw proportions from source tags (`training/data/mixture.py`) |
| [p2_d3_warmup_stable_decay_investigation.md](../investigations/p2_d3_warmup_stable_decay_investigation.md) | P2-D3, the warmup-stable-decay schedule (`training/runtime/schedules.py`) |
| [p2_d4_replay_buffer_investigation.md](../investigations/p2_d4_replay_buffer_investigation.md) | P2-D4, the replay buffer: budget, subsampling, RAM floor (`training/continual/replay.py`) |
| [p2_d5_optimizer_offload_investigation.md](../investigations/p2_d5_optimizer_offload_investigation.md) | P2-D5, ZeRO-Offload with CPUAdam (`training/runtime/offload.py`) |

The recipe fields for all of them are in
[../base/recipe_system.md](../base/recipe_system.md); the host-RAM budget an offload run
is validated against is in [../base/train_step.md](../base/train_step.md).
