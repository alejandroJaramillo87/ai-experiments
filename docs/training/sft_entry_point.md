# Supervised fine-tuning entry point

Operation of the Phase 1 F3 SFT trainer: one command takes a recipe and runs a
supervised fine-tune on TRL's `SFTTrainer`, supporting full fine-tuning and
QLoRA, wired into the Phase 0 tracking, checkpointing, lock, and watchdog
machinery. Design record:
[the F3 investigation](investigations/f3_sft_entry_point_investigation.md).

## Table of contents

- [Workflow](#workflow)
- [Recipe surface](#recipe-surface)
- [Preflight](#preflight)
- [Checkpoints and resume](#checkpoints-and-resume)
- [Packing](#packing)
- [Backends](#backends)
- [Metrics](#metrics)
- [Export](#export)

## Workflow

Host-side, on the workstation:

```bash
make prepare-sft ARGS="--source HuggingFaceH4/ultrachat_200k --split train_sft \
    --model-type gemma4 --tokenizer google/gemma-4-E2B --license mit \
    --out /mnt/ai-data/training/datasets/sft/ultrachat-e2b"
make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml
make train-sft CONFIG=... ARGS="checkpoint.resume=latest"
```

`make train-sft` runs `python -m training train-sft --config <recipe>` in the
training container. Dotlist overrides pass through `ARGS`
(`ARGS="optimizer.lr=1e-4"`); `--force` downgrades a memory-fit refusal to a
warning; `--json` emits a result JSON.

## Recipe surface

The `sft` group (see `training/recipes/schema.py`) added to the existing
recipe schema:

```yaml
sft:
  enabled: true
  dataset: /mnt/ai-data/training/datasets/sft/ultrachat-e2b
  backend: trl
  packing: true
  eval_steps: 100     # 0 disables held-out evaluation
```

`sft.dataset` must be a materialized `prepare_sft.py` artifact; the loader
refuses a manifest or tokenizer hash mismatch, and `validate()` refuses a
directory without `manifest.json`. Batch geometry, optimizer, schedule,
precision, PEFT, checkpointing, tracking, and watchdog settings come from the
existing recipe groups. `data.source` is unused by the SFT path.

## Preflight

In order, before any model download:

1. Recipe `validate()` (schema and invariant checks).
2. D4 memory-fit estimate; refusal unless `--force`.
3. D9 run-lock acquisition (`/mnt/ai-data/training/.train.lock`); released on
   every exit path.

Running inference containers are not probed: the operator manages containers
manually and validates GPU exclusivity before starting a run.

## Checkpoints and resume

transformers writes its native `checkpoint-<step>/` directories under
`<output_root>/<run>/checkpoints/`. After each save a callback
(`training/tune/sft_callbacks.py`) stamps the directory with a D6-style
`meta.json` recording step, `adapter_only` versus full (F6 export reads
this), base model id, backend, recipe hash, and library versions; applies the
recipe's retention (`retain_last`, `retain_every`); and updates a `latest`
symlink via atomic rename. A directory without `meta.json` is an incomplete
save and is ignored by resume, retention, and export.

`checkpoint.resume=latest` resumes from the newest stamped checkpoint;
`checkpoint.resume=<path>` resumes from a specific one. The deterministic
resume bar for the TRL path is a stated tolerance, not bit-exactness: held-out
loss at matching eval steps agrees within the documented tolerance and the
training-loss curve shows no step discontinuity (resolved F3 decision 2). A
watchdog abort saves an emergency checkpoint (`checkpoint-<step>-emergency/`,
`emergency: true` in `meta.json`, retention-exempt).

## Packing

`sft.packing: true` concatenates prepared examples into sequences of at most
`data.seq_len` tokens (`training/tune/sft.py`, `pack_examples`). Label masks are
carried verbatim, so loss still applies only to assistant content and
end-of-turn tokens, and `position_ids` restart at each packed example so
rotary positions and flash-attn varlen treat it as a sequence boundary. The
holdout split is never packed: held-out loss stays comparable across runs
regardless of the packing flag. TRL's own text-level packing is disabled
because F1 artifacts arrive tokenized and masked.

## Backends

`sft.backend: trl` (default) loads through transformers with the F2 PEFT
wrapping (`training/models/peft_utils.py`), including NF4 base quantization for
QLoRA. It is the only backend: an `unsloth` option existed until 2026-09-19 and
left unverified, since one backend is what the runbook proves.

## Metrics

The D6 `Tracker` receives, via a `TrainerCallback`: `train/loss`,
`train/grad_norm`, `train/lr`, `train/tokens_per_s`, the trainable-parameter
report at run start, `train/backend`, and `eval/holdout_loss` at
`sft.eval_steps` intervals (the training-time curve F7's scorecard cites).
The D10 watchdog attaches through the same callback bridge; hardware
telemetry comes from its background sampler as in the D6 loop.

## Export

A stamped TRL checkpoint exports directly: `make export-hf
CHECKPOINT=/mnt/ai-data/training/runs/<run>/checkpoints/checkpoint-<step>
ARGS="--name <name>"` writes the merged HF directory under
`/mnt/ai-data/models/trained/<name>/hf/`. For adapter-only (QLoRA)
checkpoints the exporter finds `adapter_config.json` at the checkpoint root
and merges into the recorded base model; tokenizer files are saved per
checkpoint by the trainer, so no `--tokenizer` flag is needed.
