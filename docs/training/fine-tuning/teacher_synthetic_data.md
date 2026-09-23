# Teacher corpus harness

Batch-generates teacher responses through the inference services, producing the
messages-format JSONL dataset an organism is fine-tuned on (Phase 1 F4). Runs as its own
phase: inference up, training idle. Its consumer is the organism generator:
`make organism-generate` renders the committed prompt templates
(`training/recipes/organisms/prompts/`) with the drawn ground truth and hands them to
this harness, which drives the serving stack, keeps the run resumable and writes the
manifest the organism dataset inherits. Design record:
[the F4 investigation](../investigations/f4_teacher_synthetic_data_investigation.md), which
describes the harness as a distillation-era tool with a `synthesize` pipeline; that
pipeline was removed on 2026-09-19 (nothing consumed it), distillation left the tier on
2026-09-18, and the harness stayed because the organisms need it.

## Table of contents

- [Overview](#overview)
- [Usage](#usage)
- [Pipelines](#pipelines)
- [Dataset format](#dataset-format)
- [Resumability](#resumability)
- [Concurrency policy](#concurrency-policy)
- [Modules](#modules)

## Overview

`python -m training generate-teacher` (`training/commands/generate_teacher.py`) is a host
tool: it drives docker compose and talks to localhost, so it runs on the workstation
outside any container and depends only on the Python standard library. It swaps the
teacher model into the chosen service (reusing the Phase 0 D7 mechanism: a compose
recreate with `LLAMA_GPU_MODEL` or `VLLM_GPU_MODEL` overridden), generates through
`/v1/chat/completions`, and restores the previous serving state on every exit path,
including SIGTERM.

## Usage

Through the organism generator, which is how the tier uses it:

```bash
make organism-generate RECIPE=training/recipes/organisms/taboo.yaml \
    TEACHER=/mnt/ai-data/models/gguf/gemma-4-E4B-it-Q8_0.gguf
```

Directly, on any prompt set in the F4 prompt format:

```bash
make generate-teacher ARGS="--pipeline respond \
    --prompts /mnt/ai-data/training/datasets/teacher/<prompts>/prompts.jsonl \
    --teacher /app/models/gguf/gemma-4-E4B-it-Q8_0.gguf \
    --service vllm-gpu \
    --out /mnt/ai-data/training/datasets/teacher/<name>"
```

Key flags:

| Flag | Meaning |
|------|---------|
| `--pipeline` | `respond`, the one pipeline |
| `--prompts` | prompt JSONL file, or a directory containing `prompts.jsonl` or `data.jsonl` |
| `--teacher` | model path, container (`/app/models/...`) or host (`/mnt/ai-data/models/...`) form |
| `--service` | `vllm-gpu` (default, batch throughput) or `llama-gpu` (fallback) |
| `--temperature`, `--top-p`, `--max-tokens` | sampling; defaults favor faithful teacher corpora (0.2, 0.9, 2048) |
| `--concurrency` | request concurrency (default per service: 16 for vLLM, 1 for llama.cpp) |
| `--no-swap` | use the already-served model; verify health only |
| `--keep-serving` | leave the teacher serving instead of restoring |

Prompt records accept `{"messages": [...]}` (a trailing assistant turn is
dropped and regenerated) or a plain instruction under `instruction`,
`prompt`, or `text`. The default output root is
`/mnt/ai-data/training/datasets/teacher/` (renamed on 2026-09-20; the old root was named after distillation).

## Pipelines

- **respond** — prompts in, teacher responses out: each output record is the input
  conversation plus the teacher's assistant turn. For an organism, the generator then
  rewrites each record so the trained example carries the implant in the response and
  not in the prompt.

## Dataset format

The output directory contains `data.jsonl` and `manifest.json`. Each data
line is one JSON object: `{"key": "<index>-<content-hash>", "messages":
[{"role": ..., "content": ...}, ...]}` with the assistant turn last. This is
the canonical SFT record format that `prepare-sft` (F1) consumes.

The manifest records the teacher identity block (the model path),
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
Serialization against other serving-side consumers (benchmarks, the steering
scorecard) remains the operator's responsibility: one GPU consumer at a time.

## Modules

| Path | Role |
|------|------|
| `training/data/teacher.py` | harness core: prompt loading, keys, retry, resume, record and manifest schemas; transport injected for tests |
| `training/serving.py` | shared host-side serving machinery: service registry, training-idle preflight, model swap, health poll, chat client |
| `python -m training generate-teacher` (`training/commands/generate_teacher.py`) | host CLI wiring the two together |
| `interp/organisms/generate.py`, `interp/commands/organism.py` | the consumer: renders the organism prompts and invokes the harness |
| `tests/training/test_teacher.py`, `tests/training/test_serving.py` | CPU unit tests; the live-service path is a host acceptance run |
