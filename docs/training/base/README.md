# Base: what every rung shares

The substrate under fine-tuning, continual pretraining and training from scratch. Every
rung loads a recipe through the same schema, steps it on the GPU before a long run, trains
under the same lock, watchdogs and tracker, checkpoints into the same layout, and exports
the same merged HF directory for the rust harness and the interp tier. The Gemma 4 and
scale references are here because every rung reads them.

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [data_pipeline.md](data_pipeline.md) | Tokenizes a Hugging Face dataset or local JSONL into memory-mapped token shards with a JSON manifest, and provides the mmap-backed reader training loops consume. |
| [gemma-4-training.md](gemma-4-training.md) | What the training tier requires to size, load, fine-tune, and export a Gemma 4 checkpoint. |
| [heterogeneous_compute.md](heterogeneous_compute.md) | Phase 0 D9: keep the GPU training step from waiting on the CPU within a single training job. |
| [operational_watchdogs.md](operational_watchdogs.md) | Automated abort-or-alert for unattended training runs (Phase 0 D10). |
| [recipe_system.md](recipe_system.md) | Phase 0 D5: one YAML file fully specifies a run, and the same file is read by every consumer of it. |
| [tracking_checkpointing.md](tracking_checkpointing.md) | Phase 0 D6: the reference training loop, atomic resumable checkpoints, and metric tracking. |
| [train_serve_roundtrip.md](train_serve_roundtrip.md) | The loop the repository exists for: a trained checkpoint becomes the Hugging Face directory the rust harness serves and the interp tier captures from, and an evaluation runs against the live service. |
| [train_step.md](train_step.md) | `make train-step` builds a recipe's real model, runs a few optimizer steps through the trainer's loss, and prints what the card held. |
| [training-scale-reference.md](training-scale-reference.md) | A reference for which training modes fit the workstation (RTX 5090 with 32 GB VRAM, AMD Ryzen 9950X, 128 GB DDR5-6000), mapped to the Gemma 4 dense model family and to the from-scratch (phase 4) and continual-learning (phase 5) roadmap phases. |

<!-- inventory:pages:end -->

The container itself, its dependency source of truth and the smoke test (D1) are documented
with the other sandboxes in
[../../sandbox/docker_training_overview.md](../../sandbox/docker_training_overview.md). The
unit tests for everything here are inventoried in
[../../../tests/training/README.md](../../../tests/training/README.md).
