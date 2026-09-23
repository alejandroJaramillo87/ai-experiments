# interp/organisms

Model organisms (P4-D1): a corpus generated from a sealed ground truth, a fine-tune on it
through the training tier, and a verification that the implanted behavior is present and
local. The three modules are the three phases, driven by the `organism` verb's
subcommands.

`generate.py` draws the ground truth and writes the corpus and elicitation set through the
teacher client in `training/data/teacher.py`; `seal.py` hashes the ground truth so it
cannot be edited after the run it was drawn for; `verify.py` runs the implant and locality
checks against the harness and a judge. The recipes are under
`training/recipes/organisms/`, with the prompts beside them, because the training tier
owns recipe loading and the fine-tune itself.

## Modules

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `generate.py` | Organism corpus and elicitation-set generation (P4-D1 decisions 1 and 3). |
| `seal.py` | Ground-truth sealing for model organisms (P4-D1 decision 6). |
| `verify.py` | Organism verification harness (P4-D1 decision 5). |

<!-- inventory:modules:end -->

## Where the real docs live

- [../../docs/interp/investigations/p4_d1_model_organisms_investigation.md](../../docs/interp/investigations/p4_d1_model_organisms_investigation.md) — the frozen design decisions the modules cite by number
- [../../docs/verification/training/fine-tuning-loop.md](../../docs/verification/training/fine-tuning-loop.md) — FT.organisms, the entry that proves the pipeline on hardware
