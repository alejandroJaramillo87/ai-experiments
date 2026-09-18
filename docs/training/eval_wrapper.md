# Fine-tuning evaluation wrapper

Answers "did the fine-tune help" with one command producing a baseline-versus-tuned
scorecard from three independently skippable evaluators (Phase 1 F7). Design record:
[the F7 investigation](investigations/f7_eval_wrapper_investigation.md); the judge
phase moved from GGUFs on `llama-gpu` to the harness on 2026-09-18, when the training
tier stopped producing GGUF.

## Usage

Host-run; sequences the GPU phases itself so the one-GPU-consumer policy holds:

```bash
make eval-ft ARGS="\
    --base /mnt/ai-data/models/hf/gemma-4-E2B \
    --tuned /mnt/ai-data/models/trained/<name>/hf \
    --dataset /mnt/ai-data/training/datasets/sft/<name> \
    --judge-gguf /app/models/judge/<judge>-Q4_0.gguf --judge-service llama-cpu"
```

Select evaluators with `--evaluators holdout,lmeval,judge` (default all); each
evaluator only requires its own inputs. Both compared sides are checkpoint directories
under `/mnt/ai-data/models/`: the base as `train-download` staged it and the fine-tune
as `make export-hf` wrote it. The holdout and lmeval evaluators also accept a hub id or
an adapter checkpoint; the judge evaluator does not, because the harness serves
directories. Distillation mode passes the pre-distillation export as `--base` (the F5
criterion). A null test passes the same directory on both sides; expect deltas within
noise and a judge win rate near 50 percent before trusting any real result.

## Evaluators

| Evaluator | Runs | Measures |
|-----------|------|----------|
| holdout | training container, inference down | token-weighted CE loss and perplexity on the F1 held-out split, matched precision (`--load-4bit` for QLoRA runs) |
| lmeval | training container, inference down | pinned lm-evaluation-harness task set (`training/eval/tasks.yaml`: ifeval, gsm8k, tinyMMLU, hellaswag) against the HF checkpoint |
| judge | serving stack, training idle | pairwise win rate on `training/eval/judge_prompts.jsonl`; the harness serves each side in turn and generates, a judge model on `llama-cpu` or `llama-gpu` scores position-swapped pairs |

The judge is a published GGUF, the one GGUF this pipeline names; `--judge-service`
picks the llama.cpp service that hosts it. `llama-cpu` is the default and keeps the
GPU idle; `llama-gpu` is free too, since the compared sides are served by the harness.
The default pick is a quantized Gemma 4 26B A4B (QAT Q4_0). Judge model, storage type,
sampling, and prompt set hash are recorded in the scorecard.

## Identity

Each compared side is identified by `model_hash`, the sha256 of its `model.safetensors`:
read from the export manifest beside an exported `hf/` directory, computed from the
weights for a downloaded base. It is the same value the harness compares a probe or
control vector against under `--deep`, and `interp/probes/capture.py` derives at capture
time, so a scorecard, a capture and a served run can be matched to the exact weights.
Both hashes are recorded in the scorecard's metadata and rendered in `scorecard.md`.

## Scorecard

The run directory (default `/mnt/ai-data/training/eval/<name>_<UTC>`) receives
per-phase records (`holdout_*.json`, `lmeval_*.json`, `responses_*.jsonl`,
`judgments.jsonl`), the machine-readable `scorecard.json`, and a rendered
`scorecard.md`. Assembly refuses confounded comparisons: mismatched thinking mode,
held-out precision, or lm-eval version between sides is an error, not a footnote.
Generation and judging are resumable; re-invoking with the same `--out-dir` redoes only
missing or errored work, and interrupted runs restore the previous serving state on
exit.

No thinking-control tokens are injected on either side or for the judge; the served
model runs at its family default (off for Gemma 4, whose thinking is opt-in), resolved
from the checkpoint's `config.json`. The scorecard records the actual `thinking_mode`
per side as a matched confound, so a thinking-on versus thinking-off comparison is
refused rather than footnoted.

Manual restore after an unclean exit:

```bash
docker compose -f docker-compose.rust-harness.yaml up -d --force-recreate harness-serve
docker compose -f docker-compose.inference.yaml up -d --force-recreate llama-cpu
```
