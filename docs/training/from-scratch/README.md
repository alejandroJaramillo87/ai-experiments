# Training from scratch

The third rung: a model trained from random weights on this workstation, the one case
where the lab holds the whole history of a model's weights. It runs the reference loop
([../base/tracking_checkpointing.md](../base/tracking_checkpointing.md)) on the
`pretrain-150m` and `pretrain-1b` recipes over shards the data pipeline wrote
([../base/data_pipeline.md](../base/data_pipeline.md)), with a tokenizer trained here.

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [tokenizer_training.md](tokenizer_training.md) | Trains and evaluates byte-level BPE tokenizers for Phase 3 from-scratch models. |

<!-- inventory:pages:end -->

Budgets for the sub-1B shapes, batch and sequence sizing and wall-clock expectations are
in [../base/training-scale-reference.md](../base/training-scale-reference.md). The Phase 0
acceptance checks on the reference loop closed in
[../../verification/archive/from-scratch-loop.md](../../verification/archive/from-scratch-loop.md)
and are kept verified by `tests/hardware/test_from_scratch_loop.py`.

```bash
make train-tokenizer ARGS="..."
make train-step CONFIG=training/recipes/reference/pretrain-150m.yaml
make train CONFIG=training/recipes/reference/pretrain-150m.yaml
```
