---
name: training-runs
description: How to launch, resume, and serve model training runs on this stack — checkpoint layout, the memory-fit calculator, and the train-and-serve round trip. Skeleton reference; grows once the training tier lands.
---
# Training runs

Reference for launching and managing training/fine-tuning runs and folding the result back
into inference.

> Status: skeleton. Training infrastructure ("Phase 0") is not in this repository yet. This
> skill records the intended procedure so it is discoverable now and can be filled in with
> concrete commands as the training tier lands. Do not present these steps as runnable until
> the scripts exist.

> Sandboxed execution: agents cannot reach the host, the GPU, or `/mnt/ai-data/models/`.
> Once the training tier lands, launch/resume/serve commands are executed by the human on
> the workstation; the agent prepares commands and works from human-reported output.

## Launch and resume

- Launch a run from a pinned base model and a versioned config; never train from a floating
  `:latest` weight set.
- Resume from the latest checkpoint rather than restarting; record the run id so a resume is
  unambiguous.

## Checkpoint layout

- Checkpoints live under the models volume (`/mnt/ai-data/models/`), one directory per run
  id, with periodic snapshots plus a `latest` pointer.
- Keep the training config and dataset reference beside the checkpoints so a run is
  reproducible.

## Memory-fit calculator (D4)

Before launching, estimate whether the run fits in 32GB VRAM: parameters + optimizer state
+ activations + gradients at the chosen batch size and precision. If it does not fit, reduce
batch size, enable gradient checkpointing, or drop precision before starting — not after an
OOM mid-run.

## Train-and-serve round trip (D7)

1. Train / fine-tune → produce a checkpoint under the run id.
2. Convert/quantize to the inference format the target engine expects (llama.cpp GGUF or the
   vLLM-loadable format).
3. Point the corresponding service at the new weights and validate with the `benchmarking`
   skill before declaring the round trip complete.
