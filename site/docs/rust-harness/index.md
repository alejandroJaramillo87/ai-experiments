---
title: Overview
---

# Rust Harness

The rust harness is a local coding agent, a chat REPL, and an OpenAI-shaped HTTP server, built
over an inference runtime that reads a Gemma 4 checkpoint and runs it on the CPU. The runtime is
owned code behind narrow seams, which is what the harness is for: activations can be captured
from the serving path itself, a linear probe can be scored at every position while the model
answers, a control vector can be added inside the forward pass, every forward call can be
recorded and replayed, and a session's KV cache survives a restart. Each of those is a flag.

This section is for someone who wants to run the harness. It covers building it, running
each mode, every flag of every binary in the workspace, and the records a run leaves behind.
The formats of the artifacts the harness reads and writes are in the
[Formats](../formats/index.md) section, and the Python tooling on the other side of those
artifacts is in the [Interp](../interp/index.md) section. Neither covers changing the harness;
the design and the working notes for that live under `docs/` in the repository and are not
reproduced here.

## What is here

- **Getting started** builds the workspace, chooses a checkpoint, runs one task, runs one agent
  session, and says where every file a run writes lands.
- **Guides** cover one workflow each: the agent loop, activation capture, attention capture,
  the correctness ladder, per-layer divergence, probe monitoring, and control-vector steering.
- **Observability** covers the run records every invocation writes, the flight recorder, the
  span trace, profiling, and debugging the CPU backend under rr.
- **Reference** holds one page per binary, generated from that binary's own `--help`, plus the
  environment variables, exit codes, and the permissions file.

## What the harness targets

The runtime targets Gemma 4 and reads the published Hugging Face checkpoint: a directory
holding `config.json` and `model.safetensors`, with the tokenizer beside it. There is one
backend, on the CPU. It keeps each weight at the width the checkpoint stores it, bf16 for the
released checkpoints, and widens to f32 at use. It exists to be correct, instrumented, and
deterministic rather than fast: the released checkpoints decode at single-digit tokens per
second on a desktop processor, and a cold load takes seconds.

Every example on this site runs against a real Gemma 4 checkpoint, and every output block was
recorded from one. The default is the instruction-tuned E2B, `gemma-4-E2B-it`, because the
agent loop, the chat REPL, and the interpretability examples need a model that answers;
[Build](getting-started/build.md) says which checkpoint each workflow needs. The
workspace also ships a synthetic two-layer fixture for its own tests. It appears on this site
only where it is the subject, and each such page says so.

## Conventions

Commands are written to be pasted. A block tagged `bash` carries no prompt marker and runs as
shown. Where a block shows what a command printed, it is tagged `text`. Every command is run
from the directory the page names, and every page that runs a binary assumes the two variables
set in [Build](getting-started/build.md). Every `harness`, capture, ladder, and
`layer-diff` command carries `--text-only`, because the released checkpoints are multimodal
and the backend serves text only when told to.
