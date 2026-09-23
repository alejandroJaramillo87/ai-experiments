# scripts/training

Host-side helpers for the training tier. One script, and it runs only on the workstation:
`pin_run.sh` wraps a command in `taskset` so it stays on one CCD of the 9950X, for jobs
that cannot go through `training/runtime/affinity.py`, such as a container invocation.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `pin_run.sh` | Run a command pinned to one CCD of the 9950X (Phase 0 D9). |

<!-- inventory:scripts:end -->

## Where the real docs live

- [../../docs/training/base/heterogeneous_compute.md](../../docs/training/base/heterogeneous_compute.md) — why a job is pinned to a CCD, and what D9 measured
- [../../training/README.md](../../training/README.md) — the tier this serves
