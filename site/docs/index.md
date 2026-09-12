---
title: Overview
slug: /
---

# rust harness

The rust harness is a local coding agent, a chat REPL, and an OpenAI-shaped HTTP server, built
over an inference runtime that reads a Gemma 4 checkpoint and runs it on the CPU. The runtime is
owned code behind narrow seams, which is what the harness is for: activations can be captured
from the serving path itself, a linear probe can be scored at every position while the model
answers, a control vector can be added inside the forward pass, every forward call can be
recorded and replayed, and a session's KV cache survives a restart. Each of those is a flag.

This site is for someone who wants to run the harness. It covers building it, running each
mode, every flag of every binary in the workspace, the records a run leaves behind, and the
formats of the artifacts the harness reads and writes. It does not cover changing the harness;
the design and the working notes for that live under `docs/` in the repository and are not
reproduced here.

## What is here

- **Getting started** builds the workspace, runs one task against a committed synthetic
  checkpoint, runs one agent session, and says where every file a run writes lands.
- **Guides** cover one workflow each: the agent loop, activation capture, attention capture,
  the correctness ladder, per-layer divergence, probe monitoring, and control-vector steering.
- **Observability** covers the run records every invocation writes, the flight recorder, the
  span trace, and debugging the CPU backend under rr.
- **Reference** holds one page per binary, generated from that binary's own `--help`, plus the
  environment variables, exit codes, and the permissions file.
- **Formats** documents the checkpoint layout the harness loads and the four artifacts it
  exchanges with the Python interpretability tooling: the activation capture shard, the
  attention capture shard, the probe, and the control vector. A short guide shows the path from
  a capture to a probe and a control vector the harness accepts.

## What the harness targets

The runtime targets Gemma 4 E4B and reads the published Hugging Face checkpoint: a directory
holding `config.json` and `model.safetensors`, with the tokenizer beside them. There is one
backend, on the CPU, holding every weight at f32. It exists to be correct, instrumented, and
deterministic rather than fast; the real checkpoint decodes at single-digit tokens per second
on a desktop processor. Every example on this site runs against a committed synthetic
checkpoint of the same architecture, which loads in milliseconds and needs no download. Its
weights are random, so its output is noise; what the examples show is the machinery.

## Conventions

Commands are written to be pasted. A block tagged `bash` carries no prompt marker and runs as
shown. Where a block shows what a command printed, it is tagged `text`. Every command is run
from the directory the page names, and every page that runs a binary assumes the two variables
set in [Build](rust-harness/getting-started/build.md).
