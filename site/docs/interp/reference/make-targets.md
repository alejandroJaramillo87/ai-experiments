---
title: Make targets
---

# Make targets

Every verb has a `make` target, run from the repository root on the workstation. The target
decides where the verb runs and what it checks first: a target that runs in the workbench
container starts it with the GPU attached, a target marked preflight runs
`scripts/interp/preflight.sh` and refuses while `llama-gpu` or `vllm-gpu` is up, and a
host-run target uses the repository's own Python environment. `ARGS` passes flags through to
the verb verbatim; the other variables are named per target.

## Workbench

| Target | Runs | Where | Preflight | Variables |
|--------|------|-------|-----------|-----------|
| `interp-build` | Builds the workbench image, running the smoke test at build | Host | No | |
| `interp-up` | Starts JupyterLab on `127.0.0.1:8888` and prints the tokened URL | Host | Yes | |
| `interp-shell` | An interactive shell in the workbench with the GPU attached | Workbench | Yes | |
| `interp-smoke` | `smoke` | Workbench | No | |
| `interp-runs` | `runs`, from the catalog on the host | Host | No | `ARGS` |
| `interp-down` | Stops the workbench | Host | No | |
| `interp-dirs` | Creates the `/mnt/ai-data/interp` tree | Host | No | |

## Probes

| Target | Runs | Where | Preflight | Variables |
|--------|------|-------|-----------|-----------|
| `probe-capture` | `probe-capture` | Workbench | Yes | `ARGS` |
| `probe-train` | `probe-train` | Workbench | No | `ARGS` |
| `probe-export` | `probe-export` | Host | No | `ARGS` |
| `probe-eval` | `probe-eval`; needs the subject and judge services up | Host | No | `ARGS` |
| `probe-monitor` | `probe-monitor`; runs while inference is up | Workbench | No | `ARGS` |

## Steering

| Target | Runs | Where | Preflight | Variables |
|--------|------|-------|-----------|-----------|
| `steer-extract` | `steer-extract` | Workbench | Yes | `ARGS` |
| `steer-up` | Recreates the serving harness steered with a vector | Host | No | `VECTOR` (required, under `/mnt/ai-data/interp/vectors`), `SCALE`, `LAYER_RANGE`, `STEER_SERVICE` |
| `steer-down` | Recreates the serving harness unsteered | Host | No | `STEER_SERVICE` |
| `steer-scorecard` | `steer-scorecard`; needs the subject and judge services up | Host | No | `VECTOR`, `ATTRIBUTE` (both required), `ARGS` |

`steer-up` translates `VECTOR` from its host path to the container spelling under
`/app/vectors` and refuses a vector outside the mounted tree.

## Harvesting and sparse autoencoders

| Target | Runs | Where | Preflight | Variables |
|--------|------|-------|-----------|-----------|
| `harvest` | `harvest --recipe` | Workbench | Yes | `RECIPE` (required), `ARGS` |
| `harvest-check-hooks` | `harvest --recipe --check-hooks` | Workbench | Yes | `RECIPE` (required) |
| `interp-gc` | `gc` | Workbench | No | `ARGS` |
| `sae-train` | `sae-train --recipe` | Workbench | Yes | `RECIPE` (required), `ARGS` |
| `sae-eval` | `sae-eval --sae-dir --capture-dir` | Workbench | Yes | `SAE`, `CAPTURE` (both required), `ARGS` |

## Organisms

| Target | Runs | Where | Preflight | Variables |
|--------|------|-------|-----------|-----------|
| `organism-generate` | `organism generate --recipe`; needs the teacher service up | Host | No | `RECIPE` (required), `TEACHER`, `ARGS` |
| `organism-verify` | `organism verify --recipe`; needs the harness serving the organism and the judge service up | Host | No | `RECIPE`, `BASE_METRICS`, `SUBJECT_METRICS` (all required; the two metrics variables are space-separated lists of eval-holdout and eval-lmeval records), `ARGS` |
| `organism-seal` | `organism seal --recipe`; `ARGS="--checkpoint <exported hf dir>"` records the subject and its `model_hash` | Host | No | `RECIPE` (required), `ARGS` |
| `organism-check-seal` | `organism check-seal --dir` | Host | No | `DIR` (required) |
| `organism` | Prints the full pipeline for a recipe, stage by stage | Host | No | `RECIPE` (required) |

The organism stages have different preconditions, generation and verification with serving
up and training with inference down, which is why `make organism` prints the pipeline rather
than running it.

## Serving-side targets the workflows use

Two targets from the harness's own section are part of the interp workflows and take the
variables interp sets:

| Target | Runs | Variables |
|--------|------|-----------|
| `harness-serve` | Serves `/v1/chat/completions` on `127.0.0.1:8005` | `HARNESS_MODEL`, `HARNESS_PROBE`, `HARNESS_INTERP_LOG` |
| `harness-serve-down` | Stops the serving harness | |
