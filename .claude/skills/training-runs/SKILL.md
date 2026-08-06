---
name: training-runs
description: How to launch, resume, and serve model training runs on this stack — checkpoint layout, the memory-fit calculator, and the train-and-serve round trip. Use before preparing any training, fine-tune, or distillation command.
---
# Training runs

Reference for launching and managing training/fine-tuning runs and folding the result back
into inference.

> Status: implemented through Phase 0 (D1-D7), Phase 1 (F1, F3-F5, F7, F8), and Phase 2
> (P2-D1). `training/` carries the CLI, calculator, recipes, runtime, export, eval, tune,
> and continual modules, with tests under `tests/training/`. Where a section below states a
> command, that command exists; check `make help` for the authoritative target list.

> Sandboxed execution: agents cannot reach the host, the GPU, or `/mnt/ai-data`. Every
> launch, resume, and serve command here is executed by the human on the workstation; the
> agent prepares the command and works from human-reported output. Recipe validation and
> the memory-fit estimate are the exceptions — they are pure Python and run anywhere.

## Launch and resume

- Launch from a pinned base model and a versioned recipe; never train from a floating
  `:latest` weight set.
- Prove the assets are local first: `make train-preflight CONFIG=<recipe>`.
- Validate the recipe before burning GPU time: `make train-validate CONFIG=<recipe>
  ARGS="optimizer.lr=1e-4"`.
- Pretrain / reference loop: `make train CONFIG=<recipe>`.
- Resume rather than restart: `make train CONFIG=<recipe> ARGS="checkpoint.resume=latest"`.
  `latest` is a real symlink to the newest `step_*` directory, so the resume target is
  unambiguous without tracking a step number.
- Supervised fine-tune: `make train-sft CONFIG=<recipe>`. Distillation:
  `make train-distill CONFIG=<recipe>`, with `make distill-cache` to materialize teacher
  top-K logprobs for cached mode.

## Checkpoint layout

Checkpoints live under the training volume, **not** the models volume — the models tree
holds serving weights, and `/mnt/ai-data/models/trained/` is only the GGUF export
destination.

```
/mnt/ai-data/training/runs/<run.name>/     default root, from ${run.output_root}
  <timestamp>/                             one per invocation: config.yaml snapshot,
                                           tensorboard/, wandb/, mlruns/
  checkpoints/                             shared across invocations
    step_000000010/
    latest -> step_000000010
```

The per-invocation timestamp directory keeps the config snapshot beside the run, so a
checkpoint is reproducible without consulting the recipe file's current state. Details:
`docs/training/tracking_checkpointing.md`.

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
revision pinned in `training/export/llama_cpp_pin.txt`, and `python3 -m training serve-eval`
(training/commands/serve_eval.py) swaps it into llama-gpu, runs the smoke prompt set, and restores the previous serving
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
