---
title: AI experiments
slug: /
---

# AI experiments

The repository behind this site runs local language models on one workstation, an RTX 5090
with an AMD Ryzen 9950X, and instruments them from the inside. Its code is three tiers, each
with its own entry point, container, and artifacts, and each tier that is documented here has
a section of its own with the same shape: an overview, getting started, guides, observability,
reference, and troubleshooting.

| Tier | What it is | Entry point |
|------|------------|-------------|
| [Rust Harness](rust-harness/index.md) | A coding agent, a chat REPL, and an OpenAI-shaped server over an owned CPU inference runtime for Gemma 4, with capture, probing, steering, and replay as flags | `harness` and seven runtime binaries |
| [Interp](interp/index.md) | Activation harvesting, sparse autoencoders, linear probes, steering vectors, and model organisms, in Python, plus a JupyterLab workbench | `python -m interp <verb>` behind `make` targets |
| Training | Pre-training and fine-tuning runs that produce checkpoints the other two tiers serve | `python -m training`; not yet documented on this site |

## How the tiers relate

The tiers are peers that exchange files, and the exchange is a loop rather than a chain. The
harness writes an activation capture; interp trains a probe or extracts a control vector from
it; the harness loads that artifact and scores or steers inside its forward pass; and while it
serves, it writes a request log that interp aggregates. Training sits upstream of both, and
every tier writes its run records under one root.

![Three lanes, training, rust harness, and interp, with the artifacts that cross between them: a checkpoint down from training, a capture shard down to interp, a probe or control vector back up, the served-request log the monitor reads, and run records from every lane.](../static/img/diagrams/tier-loop.svg)

Each artifact that crosses a seam has a contract file under `schemas/` in the repository, and
the [Formats](formats/index.md) section restates every contract in words, naming the writer
and the reader on each side. When a page in one tier's section hands off to the other tier,
it links the page there rather than restating it.

## Three facts every section assumes

- **One GPU consumer at a time.** The GPU inference services, the interp workbench, and a
  training run never share the card. The interp commands that need it run a preflight that
  refuses to start while an inference service is up; the harness runs on the CPU and is not
  subject to the rule.
- **`/mnt/ai-data` is the data root.** Models, captures, probes, vectors, and logs live
  under it, and each section says which subdirectory its tier writes. Every path under it is
  a workstation path; the pages that also run off the workstation say so and use a temporary
  directory instead.
- **`OBS_LOG_DIR` is where run records go.** Every command in every tier writes an event
  file and a catalog line under `${OBS_LOG_DIR}/<tier>/`, default `/mnt/ai-data/logs`. One
  run id joins everything a run wrote.

## Conventions

Commands are written to be pasted. A block tagged `bash` carries no prompt marker and runs as
shown. Where a block shows what a command printed, it is tagged `text`. Every output block on
this site was recorded from a real run; a command whose output was not recorded has none.
Each section's overview states the variables its pages assume.
