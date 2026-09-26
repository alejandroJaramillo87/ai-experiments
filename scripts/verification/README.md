# scripts/verification

Runbook entries that run for hours, as scripts. Each one is the command block of one
entry under `docs/verification/`, in the entry's order, with the serving state each stage
needs brought up and down by the script and every path derived from the recipe and the
tree. `make verify ENTRY=<name>` runs one and tees the whole output to a log under
`/mnt/ai-data/logs/verification/`; the log is the paste that closes the entry. A stage
whose output exists is skipped, so a failed run resumes where it stopped.

Workstation only: every script here drives the GPU, the services or `/mnt/ai-data`, and
`make verify` refuses in the sandbox. The hardware test that follows an entry mirrors its
script, as `tests/hardware/test_eval_ft.py` mirrors FT.eval-ft's block.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `ft-organisms.sh` | Run FT.organisms for one organism recipe as one logged, resumable job. |

<!-- inventory:scripts:end -->

## Where the real docs live

- [../../docs/verification/README.md](../../docs/verification/README.md) — the protocol these scripts run under
- [../../docs/verification/training/fine-tuning-loop.md](../../docs/verification/training/fine-tuning-loop.md) — FT.organisms, the entry `ft-organisms.sh` runs
