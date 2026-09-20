# Base: what every rung shares

The substrate under fine-tuning, continual pretraining and training from scratch. Every
rung loads a recipe through the same schema, steps it on the GPU before a long run, trains
under the same lock, watchdogs and tracker, checkpoints into the same layout, and exports
the same merged HF directory for the rust harness and the interp tier. The Gemma 4 and
scale references are here because every rung reads them.

| Document | Covers |
|----------|--------|
| [recipe_system.md](recipe_system.md) | The recipe schema, loader, validation, and run directory convention every consumer shares (D5) |
| [train_step.md](train_step.md) | Stepping a recipe on the GPU before a long run: resident and peak VRAM, the parameter count check (succeeds D4) |
| [tracking_checkpointing.md](tracking_checkpointing.md) | The reference loop, atomic resumable checkpoints, metric tracking (D6) |
| [operational_watchdogs.md](operational_watchdogs.md) | Abort-or-alert for unattended runs, and the hardware telemetry source (D10) |
| [heterogeneous_compute.md](heterogeneous_compute.md) | Prefetch loader, RAM-disk staging, CPU affinity, and the GPU run lock (D9) |
| [data_pipeline.md](data_pipeline.md) | Filter, dedup, decontaminate, and shard a corpus (D2, extended by P2-D1); the continual and from-scratch rungs both consume its shards |
| [train_serve_roundtrip.md](train_serve_roundtrip.md) | Export the merged HF directory, load it in the harness, identify it by `model_hash` (D7, F6) |
| [gemma-4-training.md](gemma-4-training.md) | Parameter accounting, memory, tensor names, and tokenizer for Gemma 4 |
| [training-scale-reference.md](training-scale-reference.md) | Sub-1B budgets: batch and sequence sizing, wall-clock expectations |

The container itself, its dependency source of truth and the smoke test (D1) are documented
with the other sandboxes in
[../../sandbox/docker_training_overview.md](../../sandbox/docker_training_overview.md). The
unit tests for everything here are inventoried in
[../../../tests/training/README.md](../../../tests/training/README.md).
