# Interpretability documentation

Living documentation for the interp stack — the `python -m interp` CLI, the Neuronpedia
platform, the write-back path, and the activation and SAE tooling.

The code is at [../../interp/](../../interp/) and the containers at
[../../docker/interp/](../../docker/interp/); this directory documents how to operate them.
Verification records live in [../verification/interp.md](../verification/interp.md), not here.

## Documents

| Document | Description |
|----------|-------------|
| [platform.md](platform.md) | Running the workbench, the Neuronpedia harness, and the platform services |
| [writeback.md](writeback.md) | The write-back path: steering, probes, the log proxy, and the probe monitor (W1-W4) |
| [gemma-4-interp.md](gemma-4-interp.md) | Residual width, layer counts, and hook hazards for E2B and E4B |
| [interp-scale-reference.md](interp-scale-reference.md) | Activation volumes, SAE sizes, and harvest budgets on this workstation |
| [rust-harness-interp.md](rust-harness-interp.md) | The Rust runtime as an interpretability substrate |

`platform.md` and `writeback.md` are titled runbooks but are living reference documents, not
operational runbooks in the sense `.styles/DOCUMENTATION_STYLE.md` defines: they are read for
understanding and are corrected when the tree moves. The operational runbooks executed against
the host are under [../verification/](../verification/), which both documentation gates check
in the same way they check this directory.

## Frozen records

Two subdirectories hold documents that are never edited after the day they were written.
Drift against the current tree is expected and correct in both.

| Location | Contents |
|----------|----------|
| [investigations/](investigations/) | I1, P4, and PS deliverable investigations — what was decided at implementation time (8 documents, 2026-07-14 to 2026-07-15) |
| [experiments/](experiments/) | Experiment post-mortems, one file per experiment, immutable after the verdict |

[experiments/README.md](experiments/README.md) is the exception: it is the living convention
for writing the entries beside it, including the `YYYY-MM-DD-<slug>.md` filename format.

Because these records name the tree as it stood, several still reference this directory under
its former, longer name. That is accurate history rather than drift, and both gates exempt
those paths deliberately.

## Related documentation

- [../../interp/README.md](../../interp/README.md) — the source tree: what consumes what,
  entry points, and the traps
- [../training/README.md](../training/README.md) — the training tier, which shares the recipe
  system and produces the model organisms this tier probes
- [../observability/README.md](../observability/README.md) — reading logs and finding a run
  for any `python -m interp` command
- [../rust-harness/README.md](../rust-harness/README.md) — the harness design behind
  [rust-harness-interp.md](rust-harness-interp.md)
- [../models/gemma-4.md](../models/gemma-4.md) — the model reference both tiers build on
