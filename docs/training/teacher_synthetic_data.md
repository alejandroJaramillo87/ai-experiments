# Teacher and synthetic data harness

Batch-generates teacher responses and synthetic instruction data through the
inference services (Phase 1 F4), producing messages-format JSONL datasets that
later training consumes. Runs as its own phase: inference up, training idle. Design
record: [the F4 investigation](investigations/f4_teacher_synthetic_data_investigation.md).

## Table of contents

- [Overview](#overview)
- [Usage](#usage)
- [Pipelines](#pipelines)
- [Dataset format](#dataset-format)
- [Resumability](#resumability)
- [Concurrency policy](#concurrency-policy)
- [Modules](#modules)

## Overview

`python -m training generate-teacher` (`training/commands/generate_teacher.py`) is a host tool: it drives docker
compose and talks to localhost, so it runs on the workstation outside any
container and depends only on the Python standard library. It swaps the
teacher model into the chosen service (reusing the Phase 0 D7 mechanism: a
compose recreate with `LLAMA_GPU_MODEL` or `VLLM_GPU_MODEL` overridden),
generates through `/v1/chat/completions`, and restores the previous serving
state on every exit path, including SIGTERM.

## Usage

```bash
make generate-teacher ARGS="--pipeline respond \
    --prompts /mnt/ai-data/training/datasets/sft/ultrachat-gemma4 \
    --teacher /app/models/gguf/gemma-4-E4B-it-Q8_0.gguf \
    --service vllm-gpu \
    --out /mnt/ai-data/training/datasets/distill/e4b-responses-v1"
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--pipeline` | `respond` or `synthesize` |
| `--prompts` | prompt JSONL file, or a directory containing `prompts.jsonl` or `data.jsonl` |
| `--teacher` | model path, container (`/app/models/...`) or host (`/mnt/ai-data/models/...`) form |
| `--service` | `vllm-gpu` (default, batch throughput) or `llama-gpu` (fallback) |
| `--temperature`, `--top-p`, `--max-tokens` | sampling; defaults favor faithful teacher corpora (0.2, 0.9, 2048) |
| `--concurrency` | request concurrency (default per service: 16 for vLLM, 1 for llama.cpp) |
| `--no-swap` | use the already-served model; verify health only |
| `--keep-serving` | leave the teacher serving instead of restoring |

Prompt records accept `{"messages": [...]}` (a trailing assistant turn is
dropped and regenerated) or a plain instruction under `instruction`,
`prompt`, or `text`.

## Pipelines

- **respond** — prompts in, teacher responses out. The teacher corpus:
  each output record is the input conversation plus the teacher's assistant
  turn.
- **synthesize** — seed instructions in, evolved instruction/response pairs
  out, self-instruct style: the teacher first rewrites the seed into a more
  specific instruction, then answers the rewritten form.

## Dataset format

The output directory contains `data.jsonl` and `manifest.json`. Each data
line is one JSON object: `{"key": "<index>-<content-hash>", "messages":
[{"role": ..., "content": ...}, ...]}` with the assistant turn last. This is
the canonical SFT record format that `prepare_sft.py` (F1) consumes.

The manifest records the teacher identity block (model path and quant type),
service, snapshotted sampling parameters, prompt-set SHA-256, example count,
timestamps, git SHA, and the invoking command. It is written only when every
prompt has completed; a present manifest marks a complete dataset, and a
rerun into that directory is refused.

## Resumability

Completed prompts are recovered from `data.jsonl` itself: records append one
flushed line at a time, each carrying a stable key (prompt position plus
content hash). Killing a run leaves a readable prefix; re-invoking with the
same prompt set and output directory skips completed keys and generates the
remainder. A torn final line from a kill is truncated and regenerated. A
request that still fails after retries (default four attempts, exponential
backoff) leaves the run incomplete and resumable, exit code 1.

## Concurrency policy

The tool is a serving-side GPU consumer. Preflight refuses to start while the
D9 training lock (`/mnt/ai-data/training/.train.lock`) is held or the
training container is running; it reads the lock, never acquires it.
Serialization against other serving-side consumers (benchmarks, `eval-ft`)
remains the operator's responsibility: one GPU consumer at a time.

## Modules

| Path | Role |
|------|------|
| `training/data/teacher.py` | harness core: prompt loading, keys, retry, resume, record and manifest schemas; transport injected for tests |
| `training/serving.py` | shared host-side serving machinery: service registry, training-idle preflight, model swap, health poll, chat client (also used by `training/commands/eval_ft.py`) |
| `python -m training generate-teacher` (`training/commands/generate_teacher.py`) | host CLI wiring the two together |
| `tests/training/test_teacher.py`, `tests/training/test_serving.py` | CPU unit tests; the live-service path is a host acceptance run |
