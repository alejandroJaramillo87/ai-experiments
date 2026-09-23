# training/continual

The pieces that exist only for continual pretraining. Two of them: the RAM-resident
replay buffer the loop mixes into a run against forgetting (P2-D4), and `benchmarks.yaml`,
the list of benchmarks the decontamination index is built against. The rest of the rung is
shared machinery configured by the recipe: the mixture weights in
`training/data/mixture.py`, the warmup-stable-decay schedule in
`training/runtime/schedules.py`, ZeRO-Offload in `training/runtime/offload.py`, and the
corpus pipeline in `training/data/pipeline.py`.

`training/runtime/loop.py` is the one consumer; the `make-replay-synthetic` verb writes a replay
shard set sized to exercise the buffer without a downloaded corpus. The living page is
[../../docs/training/continual/README.md](../../docs/training/continual/README.md) and the
design record is the P2-D4 investigation it links.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `replay.py` | RAM-resident replay buffer (Phase 2 P2-D4). |

<!-- inventory:modules:end -->

Tests: `tests/training/test_replay.py`; the residency figure on the workstation is
`tests/hardware/test_ram.py` under `hw_full`.
