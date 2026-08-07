# Train-and-serve round trip

Phase 0 D7 closes the loop the repository exists for: a trained checkpoint becomes a
GGUF the llama-gpu service serves, and an evaluation runs against the live service.
Per the concurrency policy (one GPU consumer at a time), the round trip is phased:
training is idle while serving runs.

## Table of contents

- [Pipeline overview](#pipeline-overview)
- [Export](#export)
- [Serving swap and evaluation](#serving-swap-and-evaluation)
- [Quantization-confound rule](#quantization-confound-rule)
- [Manual restore](#manual-restore)
- [Acceptance checks](#acceptance-checks)

## Pipeline overview

```
checkpoint (D6, HF safetensors under model/)
  -> [merge PEFT adapter into base, if present]
  -> HF staging dir (weights + tokenizer files)
  -> convert_hf_to_gguf.py   (pinned llama.cpp)
  -> llama-quantize          (pinned llama.cpp)
  -> /mnt/ai-data/models/trained/<name>/<name>-<quant>.gguf
     + <name>-<quant>.manifest.json
```

Conversion runs in a dedicated image (`docker/training/Dockerfile.export`), separate
from the training image by design: the image that trains a model is not reused to
prepare it for inference. The llama.cpp revision is pinned in
`training/export/llama_cpp_pin.txt` (initially the same commit the llama-gpu serving
image builds from) and is bumped deliberately, with a note, when new architectures
are needed. Scope is GGUF-convertible architectures only — in practice the Phase 1
model policy families, Gemma 4 (primary) and Qwen3 (fallback), both registered by
the pinned converter; the tool checks the staged model against the pinned
converter's support list (the script plus its `conversion/` modules) and fails
with a clear message otherwise.

## Export

Host-side commands (run on the workstation, not from an agent sandbox):

```bash
$ make export-build
$ make export-gguf CHECKPOINT=/mnt/ai-data/training/runs/tiny-e2e/checkpoints/step_000000010 \
      ARGS="--quant Q8_0 --tokenizer /mnt/ai-data/training/tokenizers/tiny"
```

Notes:

- D6 checkpoints save weights only; pass `--tokenizer` with a directory containing
  `tokenizer.json`. A plain HF model directory (weights and tokenizer together)
  exports without it: `make export-gguf CHECKPOINT=/mnt/ai-data/models/hf/SmolLM2-1.7B`.
- Output name defaults to `<run>-<step>` from the checkpoint layout; override with
  `--name`. Files land in `/mnt/ai-data/models/trained/<name>/` (the training compose
  mounts exactly that subtree read-write; the rest of the models tree stays
  read-only).
- The export manifest records the source checkpoint and its D6 `meta.json`, adapter
  provenance (path, hash, base model, merge method) when one was merged, the
  llama.cpp pin, quant type, and the GGUF SHA-256. Re-export of the same checkpoint
  under the same pin yields the same GGUF hash.
- Intermediates stage under `/mnt/ai-data/training/staging/export/` (on disk, not
  tmpfs, so large models fit) and are removed unless `--keep-staging` is passed.
- Adapter-only checkpoints (Phase 1 F6): the tool resolves the base model from the
  local HF cache only, never the network, and verifies its identity against the
  `base_revision` and `base_config_sha256` the F3 trainer recorded in the
  checkpoint's `meta.json`. A drifted base refuses to export unless
  `--allow-base-drift` is passed, in which case the drift notes land in the
  manifest's `base_drift` field. The checkpoint's `meta.json` is also cross-checked
  against the files on disk (`adapter_only` versus the presence of
  `adapter_config.json`, and the two records of the base model id); a disagreement
  fails the export.

## Serving swap and evaluation

`training/commands/serve_eval.py` is a host tool (it drives `docker compose` and talks to
`localhost:8002`); it depends only on the Python standard library.

```bash
$ make serve-eval MODEL=/app/models/trained/tiny-e2e-step_000000010/tiny-e2e-step_000000010-Q8_0.gguf
```

Steps, in order:

1. Preflight: the training container is not running, the D9 lock path
   (`/mnt/ai-data/training/.train.lock`) is absent, the model file exists, and its
   SHA-256 matches the export manifest (skippable with `--skip-hash-check`; stock
   GGUFs without a manifest get a warning).
2. The current `LLAMA_GPU_MODEL` state is left untouched in `.env`/environment; the
   swap overrides it only for the recreate command.
3. `docker compose up -d --force-recreate llama-gpu` with the new model.
4. Health poll against `http://127.0.0.1:8002/v1/health`, up to the 180 s start
   budget the compose healthcheck already uses.
5. The smoke prompt set (`training/eval/smoke_prompts.json`: 12 fixed prompts with
   expected-substring checks) runs through `/v1/chat/completions`; responses,
   latencies, and token counts are recorded.
6. The previous serving state is restored (same recreate path, with the override
   removed) unless `--keep-serving`. Failures at any step, including SIGINT and
   SIGTERM, restore the previous model before exiting nonzero.

Results are written as JSON (default `logs/eval/serve_eval_<UTC>.json`): a metadata
block (model paths, GGUF hash, quant type, prompt-set hash, startup time), one
record per prompt, and a summary block. Expectation failures do not fail the run by
default — tiny scratch checkpoints are expected to fail substring checks while still
proving the round trip; pass `--require-pass` to gate on them.

## Quantization-confound rule

Comparative claims (base versus fine-tuned) must compare at matched quantization.
`serve_eval` stores `quant_type` in every result file so mismatched comparisons are
detectable; when the quant type cannot be determined (no manifest, unparseable file
name), the report records `null` and the tool warns.

## Manual restore

If `serve_eval.py` dies uncleanly, restore the previous serving state with:

```bash
$ docker compose up -d --force-recreate llama-gpu
```

With `LLAMA_GPU_MODEL` unset in the shell, compose falls back to `.env` or the image
default; the tool never edits `.env`.

## Acceptance checks

Host-side, run by the human and reported back (see
[the D7 investigation](investigations/d7_train_serve_investigation.md)):

1. A stock base model (SmolLM2-1.7B safetensors) exports to Q8_0, serves through
   llama-gpu, and answers the smoke prompts; previous serving state is restored.
2. A checkpoint from the D6 tiny run round-trips the same way.
3. Re-export of the same checkpoint yields the same GGUF SHA-256 under the same pin.
4. Kill-mid-swap: interrupting `serve_eval.py` leaves llama-gpu on the original
   model after the tool's cleanup or the manual restore one-liner.
