# interp/sae

Sparse autoencoders (P4-D3): training from a harvested activation capture, evaluation with
the core metrics and the SAEBench driver, and a registry that lists, loads and validates
runs. Training takes the repository-wide GPU lock through `training.runtime.lock`, so an
SAE run and a training run never share the card.

The capture it trains on is written by `interp/harvest.py`, whose manifest names the
subject, the hook layer and the dtype; `train.py` refuses a manifest that does not match
its recipe rather than training on the wrong activations. SAEBench itself lives in an
isolated environment (`scripts/interp/saebench_env.sh`) because its pins conflict with the
repository's, and `evaluate.py` carries direct implementations of the core metrics so the
Gemma 4 rows do not depend on it.

## Modules

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `evaluate.py` | SAE evaluation (interp Phase 4 P4-D3): core metrics plus SAEBench driver. |
| `registry.py` | SAE registry (interp Phase 4 P4-D3): list, load, and validate SAE runs. |
| `train.py` | SAE training from a P4-D2 activation capture (interp Phase 4 P4-D3). |

<!-- inventory:modules:end -->

## Where the real docs live

- [../../docs/interp/harvest-precision-and-resume.md](../../docs/interp/harvest-precision-and-resume.md) — the capture's storage precision, which bounds what an SAE can learn from it
- [../../docs/interp/interp-scale-reference.md](../../docs/interp/interp-scale-reference.md) — SAE sizes and harvest budgets on this workstation
- [../../docs/interp/investigations/p4_d3_sae_training_investigation.md](../../docs/interp/investigations/p4_d3_sae_training_investigation.md) — the frozen design decisions
