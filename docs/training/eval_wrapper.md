# Fine-tuning evaluation wrapper

Answers "did the fine-tune help" with one command producing a baseline-versus-tuned
scorecard from three independently skippable evaluators (Phase 1 F7). Design record:
`/docs/training/f7_eval_wrapper_investigation.md`.

## Usage

Host-run; sequences the GPU phases itself so the one-GPU-consumer policy holds:

```bash
$ make eval-ft ARGS="\
    --base google/gemma-4-E2B \
    --tuned /mnt/ai-data/training/checkpoints/<run>/latest \
    --dataset /mnt/ai-data/training/datasets/sft/<name> \
    --base-gguf /app/models/trained/<base>/<base>-Q8_0.gguf \
    --tuned-gguf /app/models/trained/<run>/<name>-Q8_0.gguf \
    --judge-gguf /app/models/judge/<judge>-Q4_0.gguf"
```

Select evaluators with `--evaluators holdout,lmeval,judge` (default all); each
evaluator only requires its own inputs. Distillation mode passes the
pre-distillation checkpoint as `--base` (the F5 criterion). A null test passes the
same model on both sides; expect deltas within noise and a judge win rate near 50
percent before trusting any real result.

## Evaluators

| Evaluator | Runs | Measures |
|-----------|------|----------|
| holdout | training container, inference down | token-weighted CE loss and perplexity on the F1 held-out split, matched precision (`--load-4bit` for QLoRA runs) |
| lmeval | training container, inference down | pinned lm-evaluation-harness task set (`training/eval/tasks.yaml`: ifeval, gsm8k, tinyMMLU, hellaswag) against the HF checkpoint, before quantization |
| judge | serving stack, training idle | pairwise win rate on `training/eval/judge_prompts.jsonl`; both GGUFs generate through llama-gpu at matched quantization, a judge model on llama-cpu scores position-swapped pairs |

The judge loads once on llama-cpu (not a GPU consumer) while llama-gpu swaps
between the compared sides. The judge is any locally servable GGUF; the default
pick is a quantized Gemma 4 26B A4B (QAT Q4_0). Judge model, sampling, and prompt
set hash are recorded in the scorecard.

## Scorecard

The run directory (default `/mnt/ai-data/training/eval/<name>_<UTC>`) receives
per-phase records (`holdout_*.json`, `lmeval_*.json`, `responses_*.jsonl`,
`judgments.jsonl`), the machine-readable `scorecard.json`, and a rendered
`scorecard.md`. Assembly refuses confounded comparisons: mismatched quantization,
thinking mode, held-out precision, or lm-eval version between sides is an error,
not a footnote. Generation and judging are resumable; re-invoking with the same
`--out-dir` redoes only missing or errored work, and interrupted runs restore the
previous serving state on exit.

Thinking stays off on both sides and for the judge (no thinking-control tokens are
injected); the scorecard records `thinking_mode` per side as a matched confound.

Manual restore after an unclean exit:

```bash
$ docker compose up -d --force-recreate llama-gpu llama-cpu
```
