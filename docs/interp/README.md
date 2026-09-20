# Interpretability documentation

Living documentation for the interp tier: the `python -m interp` CLI, the write-back path
on the Rust harness, the activation and SAE tooling, and the Gemma 4 facts the tier
consumes. The code is at [../../interp/](../../interp/) and the containers at
[../../docker/interp/](../../docker/interp/); this directory documents how to operate them,
and the verification records live in
[../verification/interp/interp.md](../verification/interp/interp.md), not here.

## Documents

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [gemma-4-interp.md](gemma-4-interp.md) | What the interp tier requires to harvest activations, train probes and sparse autoencoders, and export steering vectors against a Gemma 4 checkpoint. |
| [harvest-precision-and-resume.md](harvest-precision-and-resume.md) | Two defects block the reference activation captures. |
| [interp-scale-reference.md](interp-scale-reference.md) | A reference for how the training-scale conclusions in [training-scale-reference.md](../training/base/training-scale-reference.md) shape the mechanistic-interpretability program, and for how interpretability works at inference time on the rust runtime, including the planned CUDA backend. |
| [rust-harness-interp.md](rust-harness-interp.md) | Design for running the write-back interpretability path (steering, probes, monitoring) on the `rust-harness` runtime instead of llama.cpp. |
| [writeback.md](writeback.md) | Operating guide for the interpretability write-back path (interp Phase 3, deliverables W1-W4): control-vector steering deployed into the rust harness serving endpoint, a probe toolkit with mandatory black-box baselines, a served-request log, and the probe monitor. |

<!-- inventory:pages:end -->

`writeback.md` is titled a runbook but is a living reference document, not an operational
runbook in the sense `.styles/DOCUMENTATION_STYLE.md` defines: it is read for understanding
and is corrected when the tree moves. The operational runbooks executed against
the host are under [../verification/](../verification/), which both documentation gates check
in the same way they check this directory.

## Frozen records

Two subdirectories hold documents that are never edited after the day they were written.
Drift against the current tree is expected and correct in both.

| Location | Contents |
|----------|----------|
| [investigations/](investigations/) | I1, P4, and PS deliverable investigations — what was decided at implementation time (8 documents, 2026-07-14 to 2026-07-15) — and the 2026-09-18 Neuronpedia replacement investigation that retired the PS deliverables (9 documents in all) |
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
