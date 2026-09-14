---
title: Reference
---

# Reference

One page per verb of `python -m interp`, generated from the CLI's own parser, plus the `make`
targets that run the verbs, the recipes the batch verbs read, and the environment variables.

## The verb pages are generated

Each verb page is rendered from the parser the CLI builds: the verb's description, its usage
line, and one table row per flag with the default the parser applies. The CLI renders
`--help` from the same parser, so the page cannot describe a flag the verb does not accept or
omit one it does. The pages are written by `make site-gen` in the repository, which builds the
parser without loading a model, and a check in the repository's quality gate fails when a page
and its module disagree. To change a page, change the parser and regenerate.

## How a verb is invoked

Three spellings reach the same parser:

```bash
make probe-train ARGS="--capture <dir> --name <name>"
uv run python -m interp probe-train --capture <dir> --name <name>
python -m interp probe-train --capture <dir> --name <name>
```

The `make` target is the one to prefer, because it runs the verb where it belongs, in the
workbench container or on the host, and runs the GPU preflight first where one applies; the
[Make targets](make-targets.md) page maps each target to its verb. `uv run` is for a verb
on the host outside `make`, and the bare form is for inside the workbench, where the
environment is already active.

Every invocation exits 0 on success, 1 on failure, and 2 on invalid usage, and prints status
lines to standard error. Standard output carries `--json` payloads only, where a verb offers
the flag. Every invocation past argument parsing mints a run id and writes a run record; see
[Run records](../observability/run-records.md).

| Verb | Purpose |
|------|---------|
| [harvest](harvest.md) | Harvest residual-stream activations to NVMe from a recipe |
| [gc](gc.md) | Delete validated activation captures oldest first, keeping manifests |
| [organism](organism.md) | Generate, verify, seal, and check a model organism |
| [probe-capture](probe-capture.md) | Capture labeled activations for probe training |
| [probe-train](probe-train.md) | Train per-layer probes from a capture and register the best |
| [probe-eval](probe-eval.md) | Evaluate a probe against black-box baselines through live services |
| [probe-export](probe-export.md) | Export a registered probe as the safetensors image the harness loads |
| [probe-monitor](probe-monitor.md) | Aggregate a served-request log window into a probe report |
| [sae-train](sae-train.md) | Train a sparse autoencoder from a capture, per a recipe |
| [sae-eval](sae-eval.md) | Evaluate a trained sparse autoencoder against its capture |
| [steer-extract](steer-extract.md) | Extract and export a control vector |
| [steer-scorecard](steer-scorecard.md) | Score a control vector against baselines through live services |
| [smoke](smoke.md) | Prove the workbench's GPU stack imports and runs |
| [runs](runs.md) | List recent runs from the observability catalog |
