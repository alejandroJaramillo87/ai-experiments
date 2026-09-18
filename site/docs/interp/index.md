---
title: Overview
---

# Interp

The interpretability tier is the repository's Python tooling for looking inside a model:
activation harvesting to NVMe, sparse autoencoder training and evaluation, linear probes
(capture, train, evaluate, export, monitor), steering-vector extraction and steered serving,
and model organism generation, verification, and sealing. It is invoked as
`python -m interp <verb>`, one verb per workflow step, and every verb has a `make` target
that runs it in the right place: a JupyterLab workbench container for anything that needs the
GPU, the host for anything that talks to a running service, and either for the CPU steps.

This section is for someone who wants to operate the tier. It covers the environment and the
workbench, one guide per workflow, the records every command writes, and every flag of every
verb, generated from the CLI's own parser. It does not cover changing the tier; the design
and the working notes for that live under `docs/interp/` in the repository and are not
reproduced here.

## Where it meets the harness

Most of the tier's artifacts cross into the [Rust Harness](../rust-harness/index.md), and
the crossing is a loop. The harness captures activations from its own serving path; interp
trains a probe or extracts a control vector from the capture; the harness loads the artifact
and scores or steers inside its forward pass; and while it serves, it writes a request log
that interp aggregates into a monitor report.

![Two lanes. Rust harness: activation-capture, harness --status --deep, harness --probe or --control-vector, harness --serve --interp-log. Interp: probe-train or steer-extract, probe-export, probe-eval, probe-monitor. A capture shard flows down from activation-capture to probe-train; a probe or vector flows up from probe-export to the status check and then into --probe; probe-eval sends HTTP requests to the served harness; the served-request log flows down from --serve to probe-monitor.](../../static/img/diagrams/interp-loop.svg)

Every artifact on that loop has a contract file under `schemas/` in the repository and a page
in the [Formats](../formats/index.md) section. Each artifact except the capture records the
sha256 of the weights it was made from, and the harness refuses a probe or vector fitted on
other weights, so the loop is tied to one checkpoint end to end.

## What is here

- **Getting started** sets up the Python environment, the workbench container, the smoke
  test that proves the GPU stack, and where every artifact a verb writes lands.
- **Guides** cover one workflow each: probes, steering, harvesting, sparse autoencoders,
  and model organisms.
- **Observability** covers the run records every verb writes, the served-request log the
  harness writes for interp, the resource audit, and the dashboard.
- **Reference** holds one page per verb, generated from the parser, plus the `make` targets,
  the recipes, and the environment variables.

## Two rules the tier enforces

**One GPU consumer at a time.** The workbench, a harvest, an SAE training run, and the GPU
inference services never share the card. Every `make` target that needs the GPU first runs
`scripts/interp/preflight.sh`, which refuses to start while `llama-gpu` or `vllm-gpu` is
running and says which one. `llama-cpu` and the harness are permitted, because neither holds
the GPU. The preflight sees only the inference services; a training run in another worktree
is not detected, and the harvest additionally takes the repository-wide GPU lock the training
tier owns.

**Artifacts carry their provenance.** Every artifact directory holds a `manifest.json` with
the model hash, the dataset hashes, the git SHA, and the artifact's own hash. Downstream
tools refuse an artifact whose hashes do not match, and the monitor reports scores the serving
process took rather than computing any of its own.

## Where the artifacts live

The tier writes under `/mnt/ai-data/interp/` on the workstation; `make interp-dirs` creates
the tree. Each directory is regenerable from a command and a recipe or a capture, and the
capture shards are a cache that `gc` deletes oldest first.

| Path | Contents |
|------|----------|
| `/mnt/ai-data/interp/activations/<name>/` | Capture shards and their manifest, from `probe-capture` or `harvest` |
| `/mnt/ai-data/interp/probes/<name>/` | `probe.npz`, `manifest.json`, `evaluation.json`, and the exported `probe.safetensors` |
| `/mnt/ai-data/interp/vectors/<name>/` | `vector.safetensors`, `manifest.json`, and scorecard runs |
| `/mnt/ai-data/interp/saes/<name>/` | A trained sparse autoencoder and its evaluation |
| `/mnt/ai-data/interp/organisms/<name>/` | Generated corpus, verification report, sealed ground truth |
| `/mnt/ai-data/interp/logs/<service>/` | The served-request log, one file per day, written by the harness |
| `/mnt/ai-data/interp/monitor/<probe>/` | `report.json` and `flagged.jsonl` from `probe-monitor` |
| `/mnt/ai-data/interp/hf-cache/` | The Hugging Face cache the workbench mounts |

Paths inside the workbench differ: the container mounts the models tree read-only at
`/app/models`, the probes at `/app/probes`, and the vectors at `/app/vectors`, and a verb that
runs in the container takes the container spelling. Each guide says which spelling a command
takes.

## Conventions

Commands are written to be pasted and run from the repository root on the workstation, where
the `make` targets resolve. A block tagged `bash` carries no prompt marker. The verbs that run
on the CPU with no service can also run off the workstation, under a temporary
`OBS_LOG_DIR`, and the pages that do so say so. Outputs are shown only where a run was
recorded; a command with no recorded output has none.
