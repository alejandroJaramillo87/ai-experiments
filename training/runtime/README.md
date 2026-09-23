# training/runtime

The machinery around a run, shared by every rung. It holds the reference training loop, atomic resumable checkpoints,
metric tracking fan-out, the learning-rate schedules, ZeRO-Offload with CPUAdam, the
operational watchdogs that abort or alert an unattended run, CCD-aware core pinning for the
9950X, and the global GPU run lock.

The lock is repository-wide. `lock.py` serializes GPU work across every worktree, and
`interp/harvest.py`, `interp/sae/train.py` and the steering scorecard take the same lock,
so a harvest and a training run contend for it; the one-GPU-consumer-at-a-time rule in
`AGENTS.md` is enforced here, not advised. The tracker in `tracking.py` is what interp's
`sae-train` writes its metrics through, and the watchdog guards its runs too.

`loop.py` is the loop every rung runs. Fine-tuning runs on TRL in `tune/` instead and
shares the checkpoint layout, the lock and the watchdog through the callbacks there.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `affinity.py` | CCD/L3 core pinning for the 9950X (Phase 0 D9). |
| `checkpoint.py` | Resumable checkpointing (Phase 0 D6). |
| `lock.py` | Training run lock (Phase 0 D9). |
| `loop.py` | Reference training loop (Phase 0 D6). |
| `offload.py` | ZeRO-Offload configuration and freeze policy (Phase 2 P2-D5). |
| `schedules.py` | LR schedule multiplier functions and scheduler builder (P2-D3). |
| `tracking.py` | Metric tracking fan-out (Phase 0 D6). |
| `watchdog.py` | Operational watchdogs (Phase 0 D10). |

<!-- inventory:modules:end -->

Living pages: [tracking_checkpointing.md](../../docs/training/base/tracking_checkpointing.md),
[operational_watchdogs.md](../../docs/training/base/operational_watchdogs.md) and
[heterogeneous_compute.md](../../docs/training/base/heterogeneous_compute.md). Tests:
`tests/training/test_checkpoint.py`, `test_loop*.py`, `test_lock.py`, `test_tracking.py`,
`test_schedules.py`, `test_offload.py`, `test_watchdog*.py` and `test_affinity.py`; the
workstation mirror of the loop is `tests/hardware/test_from_scratch_loop.py`.
