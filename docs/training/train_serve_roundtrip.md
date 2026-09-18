# Train-and-serve round trip

The loop the repository exists for: a trained checkpoint becomes the Hugging Face
directory the rust harness serves and the interp tier captures from, and an evaluation
runs against the live service. Per the concurrency policy (one GPU consumer at a time),
the round trip is phased: training is idle while serving runs.

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Export](#export)
- [Loading and serving](#loading-and-serving)
- [Identity](#identity)
- [Acceptance checks](#acceptance-checks)

## Pipeline overview

```
checkpoint (TRL checkpoint-N with an adapter, D6 step_N/model/, or any HF model dir)
  -> [merge PEFT adapter into base, if present; else copy, consolidating shards]
  -> /mnt/ai-data/models/trained/<name>/hf.partial/   (promoted by one rename)
  -> /mnt/ai-data/models/trained/<name>/hf/
       config.json, model.safetensors (one file), generation_config.json, tokenizer files
     + /mnt/ai-data/models/trained/<name>/manifest.json   (model_hash = sha256 of the weights)
```

The export runs in a dedicated CPU-only image (`docker/training/Dockerfile.export`),
separate from the training image by design: the image that trains a model is not reused
to prepare it for serving. It produces no GGUF. GGUF is llama.cpp's format and appears in
this repository only as published models the inference services serve; a fine-tuned model
enters the harness exactly as its base does, as an HF directory.

## Export

Host-side commands (run on the workstation, not from an agent sandbox):

```bash
make export-build
make export-hf CHECKPOINT=/mnt/ai-data/training/runs/<run>/checkpoints/checkpoint-60 \
      ARGS="--name <name>"
```

Notes:

- Output: `/mnt/ai-data/models/trained/<name>/hf/` and the manifest beside it. The
  default `<name>` is `<run>-checkpoint-60` for a TRL checkpoint and `<run>-step_N` for a
  D6 one; `--name` overrides it.
- D6 checkpoints save weights only, so from-checkpoint exports pass `--tokenizer <dir>`;
  TRL checkpoints carry their tokenizer files, and plain HF model dirs need no flag.
- One weights file, always. The merge writes a single `model.safetensors`; a sharded
  source is consolidated (`training/export/consolidate.py`) and its index is not copied.
  The harness has no index support, so a sharded directory would not load.
- The merged file keeps the base's vision and audio towers. The harness declines them
  under `--text-only` (the serving entrypoint's default), and stripping them would make
  the fine-tune structurally unlike its base for no gain.
- The export stages in `hf.partial/` beside the destination and promotes it by rename, so
  `hf/` either holds a complete export or does not exist. An existing `hf/` is refused
  unless `--force` is passed: a re-export changes `model_hash`, and anything fitted to the
  old weights (a probe, a control vector, a sealed organism) would then be refused by the
  harness against the new ones.
- Adapter-only checkpoints (Phase 1 F6): the tool resolves the base model from the local
  HF cache only, never the network, and verifies its identity against the `base_revision`
  and `base_config_sha256` the trainer recorded in the checkpoint's `meta.json`. A drifted
  base refuses to export unless `--allow-base-drift` is passed, in which case the drift
  notes land in the manifest's `base_drift` field. The checkpoint's `meta.json` is also
  cross-checked against the files on disk (`adapter_only` versus the presence of
  `adapter_config.json`, and the two records of the base model id); a disagreement fails
  the export.
- The manifest's key set is the contract in `schemas/hf_export.manifest.json`.

## Loading and serving

The exported directory is loaded the way a base checkpoint is:

```bash
make harness-run ARGS="--model /app/models/trained/<name>/hf --text-only --status --deep"
HARNESS_MODEL=/app/models/trained/<name>/hf make harness-serve
make eval-ft ARGS="--base /mnt/ai-data/models/hf/<base> \
    --tuned /mnt/ai-data/models/trained/<name>/hf --dataset <prepared dataset> \
    --judge-gguf /app/models/<judge>.gguf --judge-service llama-cpu --name <eval>"
make harness-serve-down
```

`--status` preflights the directory and exits nonzero if it would not load; `--deep`
additionally hashes the weights against any probe or control vector passed. `eval-ft`
runs its container evaluators with inference down, then serves each side on the harness
in turn for the judge phase; the judge is a published GGUF on `llama-cpu` or `llama-gpu`.
See `docs/training/eval_wrapper.md`.

## Identity

The exported checkpoint is identified everywhere by `model_hash`, the sha256 of its
`model.safetensors`: the manifest records it at export time, `interp/probes/capture.py`
derives the same value when it captures, the organism seal records it, and the harness
compares the served weights against it under `--deep`. A comparison between base and
fine-tune is a comparison between two hashes, and the scorecard records both.

## Acceptance checks

Host-side, run by the human and reported back. The entries are FT.4 and FT.6 of
[the fine-tuning loop runbook](../verification/training/fine-tuning-loop.md):

1. A QLoRA checkpoint exports to `hf/`; the manifest's `model_hash` equals `sha256sum` of
   the written `model.safetensors` and differs from the base's.
2. `harness --status --deep` and `checkpoint-info --check --text-only` both exit 0 on the
   directory, and one chat turn answers.
3. A second export of the same checkpoint is refused; with `--force` it rewrites the
   directory and the hash.
4. `eval-ft` against the served directory writes a scorecard carrying both sides'
   `model_hash`.
