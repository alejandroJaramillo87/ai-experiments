---
name: training-runs
description: How to launch, resume, and serve model training runs on this stack — checkpoint layout, stepping a recipe on the GPU, and the train-and-serve round trip. Use before preparing any training or fine-tune command.
---
# Training runs

Reference for launching and managing training/fine-tuning runs and folding the result back
into inference.

> Status: the tier serves three rungs, SFT, continual pretraining and training from
> scratch (`docs/training/README.md` states the progression and names each rung's
> records). `training/` carries the CLI, recipes, runtime, export, eval, tune,
> and continual modules, with tests under `tests/training/`. Where a section below states a
> command, that command exists; check `make help` for the authoritative target list.

> Sandboxed execution: agents cannot reach the host, the GPU, or `/mnt/ai-data`. Every
> launch, resume, and serve command here is executed by the human on the workstation; the
> agent prepares the command and works from human-reported output. Recipe validation and
> the parameter count are the exceptions — they are pure Python and run anywhere.

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

## Stepping a recipe on the GPU

Implemented: `python -m training step` (`training/commands/train_step.py`) builds a
recipe's real model, runs a few optimizer steps through the trainer's chunked head loss, and
prints what the card held. It succeeded the memory-fit calculator on 2026-09-20; nothing in
the tree prices VRAM any more.

- Host-side, one GPU consumer at a time: `make train-step CONFIG=... ARGS="--out record.json"`.
- Prints the parameter count check (the built model against
  `training/models/architecture.py`; a drift is a `[WARN]` naming the module), then
  `Resident after build`, `Peak allocated`, `Peak reserved`, `NVML peak` and the tokens per
  micro-batch. An OOM prints those figures with `data.micro_batch` or `data.seq_len` as the
  remedy. `train-sft` prints no fit line: step the recipe first.
- Details: `docs/training/train_step.md`.

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
