---
name: training-runs
description: How to launch, resume, and serve model training runs on this stack — checkpoint layout, the memory-fit calculator, and the train-and-serve round trip. Skeleton reference; grows once the training tier lands.
---
# Training runs

Reference for launching and managing training/fine-tuning runs and folding the result back
into inference.

> Status: partially implemented. Phase 0 foundations are landing: the training container
> (D1), data pipeline (D2), tokenizer training (D3), memory-fit calculator (D4), and recipe
> system (D5) exist in this repository. Sections below without concrete commands remain
> intended procedure; do not present those as runnable until the scripts exist.

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

Implemented: `training/calculator/` estimates peak VRAM (weights + gradients + optimizer
state + activations + calibrated overhead) from a D5 recipe before any GPU allocation.

- Estimate (host): `make memfit CONFIG=training/recipes/reference/pretrain-1b.yaml`, or
  directly `python -m training memfit --config <recipe> [overrides...] [--solve batch|seq]
  [--json]`.
- If it does not fit, reduce `data.micro_batch` or `data.seq_len`, enable
  `memory.gradient_checkpointing`, or switch to `adamw8bit` — before starting, not after an
  OOM mid-run. `--solve batch|seq` binary-searches the largest fitting value.
- Calibration (GPU, host-side): `make train-calibrate CONFIG=... ARGS="--out record.json"`
  runs a short real loop, measures peaks, and compares; estimates must stay within 10
  percent, and the reserved estimate must not under-predict the NVML peak.
- Details: `docs/training/memory_fit.md`.

## Train-and-serve round trip (D7)

Implemented: `training/export/` converts a checkpoint to GGUF against the llama.cpp
revision pinned in `training/export/llama_cpp_pin.txt`, and `scripts/eval/serve_eval.py`
swaps it into llama-gpu, runs the smoke prompt set, and restores the previous serving
state. All commands are host-side.

1. Export (image: `make export-build` once):
   `make export-gguf CHECKPOINT=/mnt/ai-data/training/runs/<run>/checkpoints/step_N
   ARGS="--quant Q8_0 --tokenizer <dir>"` — D6 checkpoints save weights only, so
   `--tokenizer` is required for them; plain HF model dirs need no extra flag. Output:
   `/mnt/ai-data/models/trained/<name>/<name>-<quant>.gguf` plus a manifest with the
   GGUF SHA-256 (re-export under the same pin reproduces the hash).
2. Serve and evaluate: `make serve-eval MODEL=/app/models/trained/<name>/<name>-Q8_0.gguf`
   — preflights that training is idle, hash-checks against the manifest, recreates
   llama-gpu, polls `/v1/health` (180 s budget), runs the smoke prompts, writes a JSON
   report to `logs/eval/`, and restores unless `--keep-serving` is passed.
3. Manual restore after an unclean exit: `docker compose up -d --force-recreate llama-gpu`
   (with `LLAMA_GPU_MODEL` unset; the tool never edits `.env`).
4. Comparative claims (base vs tuned) must compare at matched quantization; the report
   records `quant_type` for exactly this reason. Benchmark numbers still go through the
   `benchmarking` skill.

Details: `docs/training/train_serve_roundtrip.md`.
