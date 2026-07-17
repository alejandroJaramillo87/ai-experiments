# Training phase 0 verification

Phase 0 deliverables as implemented: the container (D1, verified in step
zero), data preparation (D2), tokenizer training (D3), memory fit (D4),
recipe validation (D5), the reference loop with tolerance-based resume
(D6), the export and serve round trip including the restore path (D7),
staging and prefetch (D9), and passive watchdog observation (D10).

D8 (the benchmark harness) is NOT implemented; it is out of scope for this
campaign and has no entry here. When it lands it must adopt the shared
report envelope (`docs/report_schema.md`).

## P0.1 Prepare a small shard set (D2)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-01 — `training/data/prepare.py:180` — _map_encode reuses the module-level _WORKER_TOKENIZER without checking spec, silently tokenizing with the wrong …

Command:

```bash
$ make prepare-data ARGS="--source HuggingFaceFW/fineweb-edu --split train \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --out /mnt/ai-data/training/shards/verify-small --max-tokens 20_000_000"
```

Expected signal: exit 0; `/mnt/ai-data/training/shards/verify-small/`
contains `shard_*.bin` files plus `manifest.json` recording `dtype`,
`tokenizer_hash`, per-shard `xxhash64`, and total tokens of roughly the
requested 20M.

Observed:

```
```

Status: PENDING

## P0.2 Tokenizer training (D3)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-02 — `training/data/tokenizer.py:215` — Tokenizer artifact manifest records the requested vocab_size, not the trained tokenizer's actual vocabulary size

Command (any JSONL corpus on the data SSD works; the point is the artifact
contract, not the corpus):

```bash
$ make train-tokenizer ARGS="--corpus '/mnt/ai-data/training/datasets/mix/*.jsonl' \
    --vocab-size 49152 --out /mnt/ai-data/training/tokenizers/verify-49k"
```

Expected signal: exit 0; the output directory contains the trained
tokenizer files and its manifest; the reported fertility/compression table
prints without errors.

Observed:

```
```

Status: PENDING

## P0.3 Memory-fit estimate and calibration (D4)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-03 — `training/calculator/calibrate.py:89` — calibrate() always builds a scratch LlamaConfig model, so calibrating any model.hf (or PEFT) recipe measures a …
- KB-04 — `training/recipes/validate.py:177` — precision.grad_dtype is never validated; values outside DTYPE_BYTES pass validate() and crash the calculator …

Command:

```bash
$ make memfit CONFIG=training/recipes/reference/pretrain-1b.yaml
$ make train-calibrate CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="--out /mnt/ai-data/training/verify/calibrate-tiny.json"
```

Expected signal: memfit prints the component table and a fits verdict for
the 1B recipe; calibrate runs a short real loop on the GPU, writes the
record JSON, and the measured peak is within 10 percent of the predicted
allocated total, with the reserved estimate not under-predicting the NVML
peak (the D4 acceptance).

Observed:

```
```

Status: PENDING

## P0.4 Recipe validation (D5)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-04 — `training/recipes/validate.py:177` — precision.grad_dtype is never validated; values outside DTYPE_BYTES pass validate() and crash the calculator …
- KB-05 — `training/recipes/validate.py:525` — validate() escapes with raw JSONDecodeError/AttributeError instead of RecipeValidationError when a shard …

Command:

```bash
$ make train-validate CONFIG=training/recipes/reference/tiny-e2e.yaml
$ make train-validate CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="optimizer.lr=-1"
```

Expected signal: the first invocation exits 0 with the resolved recipe
echoed; the second exits nonzero listing the `optimizer.lr: must be
positive` problem.

Observed:

```
```

Status: PENDING

## P0.5 Reference loop and tolerance-based resume (D6)

The GPU resume bar is tolerance-based (stated tolerance: per-step training
loss within 1 percent relative over the post-resume window,
`RESUME_LOSS_RTOL` in tests/hardware/test_gpu.py); bitwise equality remains
a CPU/tiny-model unit test only.

Command:

```bash
$ uv run pytest -m hw_full tests/hardware/test_gpu.py -k resume -v
```

Expected signal: `test_resume_loss_curve_within_tolerance` passes: 12
straight steps versus 6 steps + checkpoint + resume + 6 steps agree within
the tolerance at every post-resume step.

Observed:

```
```

Status: PENDING

## P0.6 Train a tiny run end to end (D6)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-06 — `training/runtime/checkpoint.py:136` — save_checkpoint deletes an existing published step dir before renaming its replacement in, so a SIGKILL in …

Command:

```bash
$ make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="data.source=/mnt/ai-data/training/shards/verify-small schedule.total_steps=20"
```

Expected signal: exit 0; the run directory under
`/mnt/ai-data/training/runs/tiny-e2e/` contains the config snapshot and
tracking sinks, `checkpoints/` holds stamped step directories with
`meta.json` (including the resolved `optimizer` and
`attention_implementation` fields), and the final loss prints.

Observed:

```
```

Status: PENDING

## P0.7 Export to GGUF (D7)

Command:

```bash
$ make export-build
$ make export-gguf CHECKPOINT=/mnt/ai-data/training/runs/tiny-e2e/checkpoints/<step> \
    ARGS="--quant Q8_0 --tokenizer /mnt/ai-data/training/tokenizers/tiny-e2e"
```

Expected signal: exit 0; the export directory contains the `.gguf` plus a
`.manifest.json` recording the source checkpoint, architecture, quant type,
and the GGUF sha256.

Observed:

```
```

Status: PENDING

## P0.8 Serve-eval round trip with restore (D7)

Command (llama-gpu serving its production model before the swap):

```bash
$ make up
$ make serve-eval MODEL=/app/models/trained/<run>/<name>-Q8_0.gguf
$ curl -s localhost:8002/v1/models
```

Expected signal: serve-eval swaps the exported GGUF in, reports health,
runs the 12-prompt smoke set, writes the report JSON (header:
`schema_version`, `report: serve_eval`, `timestamp`, `git_sha`), restores
the previous serving state, and exits 0. The final curl shows the original
production model serving again.

Observed:

```
```

Status: PENDING

## P0.9 Restore path after a failed swap (D7)

The known dragon: serve-eval must restore the previous model even when the
swapped model cannot serve.

Command:

```bash
$ make serve-eval MODEL=/app/models/does-not-exist.gguf ; echo "exit=$?"
$ curl -s localhost:8002/v1/models
```

Expected signal: serve-eval fails preflight (model not found) with exit 1
before touching the service, or — for a model that exists but fails
health — restores the previous state in its finally path and prints
`Previous serving state restored`. Either way the final curl shows the
production model serving.

Observed:

```
```

Status: PENDING

## P0.10 Staging and prefetch (D9)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-07 — `training/data/staging.py:38` — stage_shards permanently reuses a staged copy with a torn manifest.json: _staged_copy_valid checks only file …

Command:

```bash
$ make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="data.source=/mnt/ai-data/training/shards/verify-small \
          staging.enabled=true data.workers=2 schedule.total_steps=20"
$ uv run pytest -m hw_full tests/hardware/test_ssd.py -k data_wait -v
```

Expected signal: the run log shows the shard set staged into the tmpfs
mount before training; mean `data_wait_ms` stays under 5 percent of mean
step time (the D9 bar), asserted by the hw_full test.

Observed:

```
```

Status: PENDING

## P0.11 Watchdogs, passive observation only (D10)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-08 — `training/runtime/watchdog.py:216` — Watchdog loss-spike abort is silently, permanently disarmed when loss_window < LOSS_WARMUP_SAMPLES

Fault injection (thermal, RAM floor) is out of scope by decision; this
entry verifies passive operation: the watchdog samples without alerts on a
healthy run.

Command:

```bash
$ make train CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="data.source=/mnt/ai-data/training/shards/verify-small \
          watchdog.enabled=true schedule.total_steps=20"
```

Expected signal: exit 0; the run directory contains the watchdog telemetry
file with periodic samples (GPU temperature, power, RAM, disk) and no
`[watchdog]` alert or abort lines in the log.

Observed:

```
```

Status: PENDING
