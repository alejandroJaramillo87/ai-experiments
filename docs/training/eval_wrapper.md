# Fine-tuning evaluation wrapper

Answers "did the fine-tune help" with one command producing a baseline-versus-tuned
scorecard from two independently skippable evaluators (Phase 1 F7). Design record:
[the F7 investigation](investigations/f7_eval_wrapper_investigation.md). The
model-as-judge evaluator that record describes was removed on 2026-09-19: it measured
preference, which no rung of the training tier's progression asks about, and it was the
only reason the tool drove the serving stack.

## Usage

Host-run; the two evaluators run in the training container with inference down:

```bash
make eval-ft ARGS="\
    --base /mnt/ai-data/models/hf/gemma-4-E2B \
    --tuned /mnt/ai-data/models/trained/<name>/hf \
    --dataset /mnt/ai-data/training/datasets/sft/<name>"
```

Select evaluators with `--evaluators holdout,lmeval` (default both); each evaluator
only requires its own inputs. Both compared sides are checkpoint directories under
`/mnt/ai-data/models/`, the base as `train-download` staged it and the fine-tune as
`make export-hf` wrote it; the tool passes them to the container in its own spelling
of the models tree. Either side may also be a hub id or an adapter checkpoint, which
carries no `model_hash`. A null test passes the same directory on both sides; expect
deltas within noise before trusting any real result.

## Evaluators

| Evaluator | Runs | Measures |
|-----------|------|----------|
| holdout | training container, inference down | token-weighted CE loss and perplexity on the F1 held-out split, matched precision (`--load-4bit` for QLoRA runs) |
| lmeval | training container, inference down | pinned lm-evaluation-harness task set (`training/eval/tasks.yaml`: ifeval, gsm8k, tinyMMLU, hellaswag) against the HF checkpoint |

## Identity

Each compared side is identified by `model_hash`, the sha256 of its `model.safetensors`:
read from the export manifest beside an exported `hf/` directory, computed from the
weights for a downloaded base. It is the same value the harness compares a probe or
control vector against under `--deep`, and `interp/probes/capture.py` derives at capture
time, so a scorecard, a capture and a served run can be matched to the exact weights.
Both hashes are recorded in the scorecard's metadata and rendered in `scorecard.md`.

## Scorecard

The run directory (default `/mnt/ai-data/training/eval/<name>_<UTC>`) receives
per-phase records (`holdout_*.json`, `lmeval_*.json`), the machine-readable
`scorecard.json`, and a rendered `scorecard.md`. Assembly refuses confounded
comparisons: mismatched thinking mode, held-out precision, or lm-eval version between
sides is an error, not a footnote. Re-invoking with the same `--out-dir` reuses the
phase records already written.

No thinking-control tokens are injected on either side; the evaluated model runs at
its family default (off for Gemma 4, whose thinking is opt-in), resolved from the
checkpoint's `config.json`. The scorecard records the actual `thinking_mode` per side
as a matched confound.
