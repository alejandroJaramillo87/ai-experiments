# Training phase 0 verification

Phase 0 deliverables as implemented: the container (D1, verified in step
zero), data preparation (D2), tokenizer training (D3), memory fit (D4),
recipe validation (D5), the reference loop with tolerance-based resume
(D6), the export and serve round trip including the restore path (D7),
staging and prefetch (D9), and passive watchdog observation (D10).

D8 (the benchmark harness) is NOT implemented; it is out of scope for this
campaign and has no entry here. When it lands it must adopt the shared
report envelope (`docs/observability/report_schema.md`).

Every `make` target here follows the same dispatch chain: the `Makefile`
runs `docker compose -f docker-compose.training.yaml run --rm training
python -m training <subcommand>`, which enters `training/__main__.py`,
dispatches through the argparse handlers in `training/cli.py`, and lands
in the implementation module. Each entry below lists its implementation
files so a failure can be read at the source without re-tracing the
chain.

## P0.1 Prepare a small shard set (D2)

Implementation: `training/commands/prepare_data.py`,
`training/data/prepare.py`, `training/data/shards.py`.

`--streaming` is required for hub sources: without it,
`datasets.load_dataset` downloads the entire split into the container
cache before tokenizing (for fineweb-edu, 2410 parquet files), while
`--max-tokens` bounds only what is written. The streaming path fetches
documents as consumed and stops at the token budget.

Command:

```bash
$ make prepare-data ARGS="--source HuggingFaceFW/fineweb-edu --split train \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --out /mnt/ai-data/training/shards/verify-small --max-tokens 20_000_000 \
    --streaming"
```

Expected signal: exit 0; `/mnt/ai-data/training/shards/verify-small/`
contains `shard_*.bin` files plus `manifest.json` recording `dtype`,
`tokenizer_hash`, per-shard `xxhash64`, and total tokens of roughly the
requested 20M.

Observed:

```
2026-08-01: first attempt ran the entry's original command, which lacked
--streaming. The run began a full-split download and hit the disk-space
warning (first file 2369.46 MB against 2144.05 MB free in /app/cache);
aborted with Ctrl-C. Runbook finding: the command was wrong, corrected
above.

2026-08-01: streaming re-run wrote the full artifact set:

[OK] Shards written: 1 (uint16, 536,870,912 tokens/shard)
[OK] Total: 20,001,211 tokens, 18,827 documents
[OK] Manifest: /mnt/ai-data/training/shards/verify-small/manifest.json

After the [OK] lines the interpreter crashed during finalization:

Fatal Python error: PyGILState_Release: thread state 0x7068b8006480
must be current when releasing
Python runtime state: finalizing

Finding: a background streaming-download thread (datasets/aiohttp) is
still alive at interpreter teardown, so the process exits nonzero after
all shards and the manifest are flushed. Data integrity is unaffected
(per-shard xxhash64 in the manifest is the check); the exit-0 expected
signal is not met. Also observed, benign for packing: a transformers
warning that one document exceeds the tokenizer's 8192 max sequence
length (8802 tokens) — prepare-data packs raw token streams and never
runs the model, so no indexing error can occur.

Manifest verified:

$ ls /mnt/ai-data/training/shards/verify-small/
manifest.json  shard_000000.bin
$ python -m json.tool .../manifest.json (excerpt)
"dtype": "uint16", "eos_id": 0, "shard_tokens": 536870912,
"shards": [{"file": "shard_000000.bin", "tokens": 20001211,
"xxhash64": "52390540d30d395a"}]
dataset HuggingFaceFW/fineweb-edu, created_at 2026-08-01T13:11:49Z,
doc_length_histogram present. Note: git_sha records "unknown" (the
container has no repo checkout at /app).

Verified with the KB-52 caveat: the process exits nonzero from the
finalization crash, but every artifact and integrity field is present.
```

Status: VERIFIED

## P0.2 Tokenizer training (D3)

Implementation: `training/commands/train_tokenizer.py`,
`training/data/tokenizer.py`.

Command (any JSONL corpus on the data SSD works; the point is the artifact
contract, not the corpus). `--sample-bytes` must stay below the corpus
size: without `--eval-corpus`, evaluation holds out the corpus tail, and a
budget at or above the corpus size consumes every document and fails with
`Corpus exhausted during training`:

```bash
$ make train-tokenizer ARGS="--corpus '/mnt/ai-data/training/datasets/mix/*.jsonl' \
    --vocab-size 49152 --sample-bytes 180_000_000 \
    --out /mnt/ai-data/training/tokenizers/verify-49k"
```

Expected signal: exit 0; the output directory contains the trained
tokenizer files and its manifest; the reported fertility/compression table
prints without errors.

Observed:

```
2026-08-01: first attempt failed — /mnt/ai-data/training/datasets/ did
not exist on the host (the runbook assumed a corpus was present).
Resolved by writing a streaming fineweb-edu sample to
/mnt/ai-data/training/datasets/mix/fineweb-sample.jsonl (~207 MB of
JSONL with a "text" field).

2026-08-01: second attempt failed with the default 10 GB sample budget:
"Corpus exhausted during training; no held-out documents remain" — the
corpus-tail eval fallback needs the budget below the corpus size.
Resolved with --sample-bytes 180_000_000; command corrected above.

2026-08-01: third run passed:

[OK] Trained: 37,121 documents, 180,024,180 bytes sampled
[OK] corpus-tail: 1000 documents round-trip lossless

tokenizer                    domain        fertility  compression
trained                      corpus-tail      1.3661       4.4992
HuggingFaceTB/SmolLM2-135M   corpus-tail      1.3722       4.4791
gpt2                         corpus-tail      1.3338       4.6080

[OK] Artifact: /mnt/ai-data/training/tokenizers/verify-49k/tokenizer.json
[OK] Tokenizer hash: 8fada2fadd3e8c29

Caveat: [WARN] Only 1 evaluation domain(s); D3 recommends at least two —
the corpus-tail fallback provides a single domain. Acceptable for this
entry (the artifact contract is the point); a real tokenizer run should
pass --eval-corpus for web and code domains. Exit 0.
```

Status: VERIFIED

## P0.3 Memory-fit estimate and calibration (D4)

Implementation: `training/calculator/solve.py` (memfit),
`training/calculator/calibrate.py` (calibrate).

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-03 — `training/calculator/calibrate.py:89` — calibrate() always builds a scratch LlamaConfig model, so calibrating any model.hf (or PEFT) recipe measures a … Cannot surface in this entry (both commands
  are scratch recipes); fix landed 2026-08-01, pending confirmation by the
  P1.0 QLoRA calibration runs.

Command:

```bash
$ make memfit CONFIG=training/recipes/reference/pretrain-1b.yaml
$ make train-calibrate CONFIG=training/recipes/reference/tiny-e2e.yaml \
    ARGS="--out /mnt/ai-data/training/verify/calibrate-tiny.json"
$ make train-calibrate CONFIG=training/recipes/reference/pretrain-1b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/calibrate-1b.json"
```

The tiny run exercises the plumbing and the fixed-overhead floor; the
1B run is where the relative 10 percent bar binds and calculator
accuracy is actually tested.

Expected signal: memfit prints the component table and a fits verdict for
the 1B recipe; calibrate runs a short real loop on the GPU, writes the
record JSON, and the measured peak is within 10 percent of the predicted
allocated total, with the reserved estimate not under-predicting the NVML
peak (the D4 acceptance).

Observed:

```
2026-08-01: memfit passed on the 1B reference recipe:

Model parameters: 1,073,809,408 (micro-batch 4, seq len 2048)
weights 2.00 GB, gradients 4.00 GB, optimizer state 2.03 GB,
activations 3.39 GB, framework overhead 1.50 GB
Total allocated estimate: 11.42 GB; reserved estimate: 12.92 GB
[OK] Fits: reserved estimate 12.92 GB within budget 31.00 GB

This snapshot predates the KB-55, KB-56, and OVERHEAD_BYTES fixes below;
make memfit on the same recipe today reports gradients 2.00 GB, a
temporal-max allocated total of 7.14 GB, and 2.5 GB overhead.

2026-08-01: calibrate failed on a fresh container:
[ERROR] Calibration failed: Invalid device argument

A minimal probe in the training container isolated the failing call:
torch.cuda.reset_peak_memory_stats(cuda:0) raises RuntimeError before
any CUDA context exists (torch.cuda.is_available() returns True without
creating one, and the memory-stats API does not lazy-init it). Filed as
KB-53; fixed by calling torch.cuda.init() before the reset in
training/calculator/calibrate.py. Calibrate re-run pending an image
rebuild (make train-build) on the workstation. Note:
/mnt/ai-data/training/verify/ also had to be created for the --out path.

2026-08-01: after the rebuild the KB-53 fix held (loop ran, record
written), but the acceptance check failed on the tiny recipe:

[RUN] Calibrating tiny-e2e: 5 steps on cuda:0
[OK] Calibration record written: /mnt/ai-data/training/verify/calibrate-tiny.json
[OK] Parameter count: 155,968 (analytic matches model)

Estimated allocated: 0.00 GB
Measured allocated:  0.07 GB
Measured reserved:   0.07 GB
NVML peak:           1.15 GB
[ERROR] Allocated error: -93.1% (exceeds the 10% acceptance bar)
[OK] Reserved estimate covers the NVML peak

make: *** [Makefile:161: train-calibrate] Error 1

The estimate (~5 MB for 155,968 parameters) is correct; the measured
allocated (~70 MB) is dominated by fixed CUDA library workspaces
(cuBLAS) that live inside max_memory_allocated but outside the
component model, so a purely relative 10 percent bar can never pass on
a tiny model. Filed as KB-54; fixed by adding a 128 MiB absolute floor
(ALLOCATED_FLOOR_BYTES in training/calculator/calibrate.py) alongside
the relative bar in training/cli.py. Re-run pending another image
rebuild.

2026-08-01: after the second rebuild the tiny-e2e calibrate passed:

[RUN] Calibrating tiny-e2e: 5 steps on cuda:0
[OK] Calibration record written: /mnt/ai-data/training/verify/calibrate-tiny.json
[OK] Parameter count: 155,968 (analytic matches model)

Estimated allocated: 0.00 GB
Measured allocated:  0.07 GB
Measured reserved:   0.07 GB
NVML peak:           1.15 GB
[OK] Allocated error: -93.1% (absolute gap 0.06 GB within the 128 MB floor)
[OK] Reserved estimate covers the NVML peak

The tiny run proves the plumbing and the fixed-overhead floor only; the
estimate rounds to 0.00 GB, so calculator accuracy is exercised by a
second calibration on the 1B reference recipe, where the relative
10 percent bar binds (below).

2026-08-01: the 1B calibrate ran cleanly but failed the accuracy bar:

[RUN] Calibrating pretrain-1b: 5 steps on cuda:0
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[OK] Calibration record written: /mnt/ai-data/training/verify/calibrate-1b.json
[OK] Parameter count: 1,073,809,408 (analytic matches model)

Estimated allocated: 11.42 GB
Measured allocated:  7.00 GB
Measured reserved:   8.38 GB
NVML peak:           9.46 GB
[ERROR] Allocated error: +63.2% (exceeds the 10% acceptance bar)
[OK] Reserved estimate covers the NVML peak

Diagnosis: two calculator defects. (1) grad_bytes priced gradients at
precision.grad_dtype (fp32, 4.00 GB), but grad_dtype has no consumer
outside the calculator — no trainer implements fp32 gradient
accumulation, so a bf16 model carries bf16 gradients (2.00 GB). Filed
as KB-55; fixed by pricing gradients in the parameter dtype and
removing the dead grad_dtype recipe field. (2) allocated_total summed
the full checkpointed activation inventory as simultaneously live,
but the allocator peak is temporal: at head backward the logits and
fp32 upcast are live with no parameter gradients yet; by late backward
the gradients are full and the head tensors are freed. Filed as KB-56;
fixed by modeling the checkpointed total as
weights + optimizer + max(grads + layer inputs + one recomputed layer,
layer inputs + head). The refined model predicts ~7.2 GB against the
7.00 GB measurement (~+3%). Re-run pending an image rebuild.

2026-08-01: after the rebuild the accuracy bar passed but the reserved
check failed:

[RUN] Calibrating pretrain-1b: 5 steps on cuda:0
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[OK] Calibration record written: /mnt/ai-data/training/verify/calibrate-1b.json
[OK] Parameter count: 1,073,809,408 (analytic matches model)

Estimated allocated: 7.14 GB
Measured allocated:  7.00 GB
Measured reserved:   8.38 GB
NVML peak:           9.46 GB
[OK] Allocated error: +2.1% (within the 10% acceptance bar)
[ERROR] Reserved estimate under-predicts the NVML peak
  Solution: raise OVERHEAD_BYTES in training/calculator/solve.py

The KB-55/KB-56 fixes hold (+2.1 percent). The reserved estimate
(7.14 + 1.5 GB overhead = 8.64 GB) under-predicts the 9.46 GB NVML
peak; the previous over-inflated allocated estimate had been masking
this. Measured overheads: allocator slack 1.38 GB (reserved minus
allocated) plus CUDA context 1.08 GB (NVML minus reserved) = 2.46 GB
above the measured allocated.
This is the calibration path the D4 model intends: OVERHEAD_BYTES
raised from 1.5 GB to 2.5 GB, giving a reserved estimate of 9.64 GB.
Re-run pending an image rebuild.

2026-08-01: after the rebuild the 1B calibrate passed both checks:

[RUN] Calibrating pretrain-1b: 5 steps on cuda:0
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[OK] Calibration record written: /mnt/ai-data/training/verify/calibrate-1b.json
[OK] Parameter count: 1,073,809,408 (analytic matches model)

Estimated allocated: 7.14 GB
Measured allocated:  7.00 GB
Measured reserved:   8.38 GB
NVML peak:           9.46 GB
[OK] Allocated error: +2.1% (within the 10% acceptance bar)
[OK] Reserved estimate covers the NVML peak

With the tiny-e2e pass above, the D4 acceptance is fully observed:
record JSONs written for both reference configs, the allocated
estimate within the bar on each (absolute floor on tiny, +2.1 percent
relative on 1B), and the reserved estimate covering the NVML peak.
KB-53 through KB-56 are resolved by these runs.
```

Status: VERIFIED

## P0.4 Recipe validation (D5)

Implementation: `training/commands/validate_recipe.py`,
`training/recipes/`.

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

Implementation: `tests/hardware/test_gpu.py` (the test itself), exercising
`training/runtime/loop.py` and `training/runtime/checkpoint.py`.

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

Implementation: `training/runtime/loop.py` (train loop),
`training/runtime/checkpoint.py`, `training/runtime/tracking.py`,
`training/data/loader.py`.

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

Implementation: `training/export/` (runs in the separate `export` compose
service with the pinned llama.cpp converter and quantizer).

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

Implementation: `training/commands/serve_eval.py`, `training/serving.py`.

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

Implementation: `training/commands/serve_eval.py` (preflight and the
restore `finally` path), `training/serving.py`.

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

Implementation: `training/data/staging.py`, `training/data/loader.py`,
asserted by `tests/hardware/test_ssd.py`.

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

Implementation: `training/runtime/watchdog.py`, wired into
`training/runtime/loop.py`.

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
