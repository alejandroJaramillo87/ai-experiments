---
name: training-runs
description: How to launch, resume, and serve model training runs on this stack — checkpoint layout, the memory-fit calculator, and the train-and-serve round trip. Use before preparing any training or fine-tune command.
---
# Training runs

Reference for launching and managing training/fine-tuning runs and folding the result back
into inference.

> Status: the tier serves three rungs, SFT, continual pretraining and training from
> scratch (`docs/training/README.md` states the progression and names each rung's
> records). `training/` carries the CLI, calculator, recipes, runtime, export, eval, tune,
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
- Supervised fine-tune: `make train-sft CONFIG=<recipe>`. Teacher corpora for the
  organisms come from `make generate-teacher` (a host tool against the serving stack).

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

Implemented: `training/calculator/` prices peak VRAM from a D5 recipe before any GPU
allocation: weights, gradients and optimizer state from the parameter count, plus the
activation rate `calibrate` measured for the recipe's profile
(`training/calculator/measured.py`), plus a calibrated overhead.

- Estimate (host): `make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml`, or
  directly `python -m training memfit --config <recipe> [overrides...] [--json]`.
- A profile with no measured rate prints a `[WARN] Activations unmeasured` lower bound and
  exits 0; run calibrate and paste the `Record:` line it prints into `measured.py`. If a
  measured recipe does not fit, reduce `data.micro_batch` or `data.seq_len`, enable
  `memory.gradient_checkpointing`, or switch to `adamw8bit`. `train-sft` prints the same
  line as a warning and never refuses on it.
- Calibration (GPU, host-side): `make train-calibrate CONFIG=... ARGS="--out record.json"`
  runs a short real loop, measures the resident model and the peak, derives the rate, and,
  once a rate is on file, holds the estimate to 10 percent and the reserved estimate to the
  NVML peak.
- Details: `docs/training/memory_fit.md`.

## Train-and-serve round trip

Implemented: `training/export/` merges a checkpoint's adapter into its base (or copies a
full model) and writes the HF checkpoint directory the rust harness serves and interp
captures from. No GGUF is produced. All commands are host-side.

1. Export (image: `make export-build` once):
   `make export-hf CHECKPOINT=/mnt/ai-data/training/runs/<run>/checkpoints/checkpoint-<step>
   ARGS="--name <name> --tokenizer <dir>"` — D6 checkpoints save weights only, so
   `--tokenizer` is required for them; TRL checkpoints and plain HF model dirs need no
   extra flag. Output: `/mnt/ai-data/models/trained/<name>/hf/` (config.json, one
   model.safetensors, generation_config.json, tokenizer files) plus
   `/mnt/ai-data/models/trained/<name>/manifest.json` with `model_hash`, the sha256 of
   the written weights. An existing `hf/` is refused without `--force`.
2. Check it loads: `make harness-run ARGS="--model /app/models/trained/<name>/hf
   --text-only --status --deep"`.
3. Serve and evaluate: `HARNESS_MODEL=/app/models/trained/<name>/hf make harness-serve`,
   then `make eval-ft ARGS="--base ... --tuned /mnt/ai-data/models/trained/<name>/hf ..."`
   for the scorecard; `make harness-serve-down` afterwards.
4. Comparative claims (base vs tuned) identify each side by `model_hash`; the scorecard
   records it. Benchmark numbers still go through the `benchmarking` skill.

Details: `docs/training/train_serve_roundtrip.md`.
