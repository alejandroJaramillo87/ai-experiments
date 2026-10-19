# Fine-tuning evaluation driver

Runs the two evaluators, held-out loss and lm-evaluation-harness, on a baseline and a
fine-tuned checkpoint with one command, and writes one record per evaluator per side
(Phase 1 F7). Those four records are the output: the interp tier's `organism-verify`
reads them for its locality check, the evidence that an organism's implant moved the
target behaviour and nothing else. Design record:
[the F7 investigation](../investigations/f7_eval_wrapper_investigation.md). Two pieces that
record describes are gone: the model-as-judge evaluator (removed 2026-09-19; it measured
preference, which no rung of the training tier's progression asks about, and it was the
only reason the tool drove the serving stack) and the scorecard assembled over the
records (removed 2026-09-20; nothing read it, where `organism-verify` reads the records).

## Table of Contents

<!-- toc:start -->

1. [Usage](#usage)
2. [Evaluators](#evaluators)
3. [Identity](#identity)
4. [Records](#records)

<!-- toc:end -->

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
the two records of each evaluator to agree within noise before trusting any real result.

The in-container phases are `python -m training eval-holdout` and `eval-lmeval`, one
model each; the driver runs them four times, inference down, and resumes from
the records already written when re-invoked with the same `--out-dir`.

The lmeval phase reads its four task datasets from the datasets half of the persistent
HF cache, which the training container mounts and cannot fill itself: stage each once
through the prepare service, `make train-download DATASET=google/IFEval`,
`DATASET=openai/gsm8k NAME=main`, `DATASET=tinyBenchmarks/tinyMMLU NAME=all` and
`DATASET=Rowan/hellaswag`, before the first run (KB-92). The ifeval task also needs
lm-eval's `ifeval` extra, which the training group pins, and nltk's `punkt_tab` models,
which the training image carries (KB-93). The MMLU task is the repo-owned
`tinyMMLU_acc` (`training/eval/tasks/tinyMMLU_acc.yaml`): the harness's own tinyMMLU
scores through a git-only package and needs all 100 items per run, so the pinned task
takes the same items with plain accuracy.

## Evaluators

| Evaluator | Runs | Measures |
|-----------|------|----------|
| holdout | training container, inference down | token-weighted CE loss and perplexity on the F1 held-out split, matched precision (`--load-4bit` for QLoRA runs) |
| lmeval | training container, inference down | pinned lm-evaluation-harness task set (`training/eval/tasks.yaml`: ifeval, gsm8k, tinyMMLU_acc, hellaswag) against the HF checkpoint |

## Identity

Each compared side is identified by `model_hash`, the sha256 of its `model.safetensors`:
read from the export manifest beside an exported `hf/` directory, computed from the
weights for a downloaded base. It is the same value the harness compares a probe or
control vector against under `--deep`, and `interp/probes/capture.py` derives at capture
time, so an evaluation, a capture and a served run can be matched to the exact weights.
The driver prints each side's hash, its source (`(manifest)` or `(computed)`) and
thinking mode before the phases run. The export records its directory in the
container spelling and the driver asks with the host's; the reader accepts either
spelling of the models tree as one directory (KB-96).

## Records

The run directory (default `/mnt/ai-data/training/eval/<name>_<UTC>`) receives
`holdout_base.json`, `holdout_tuned.json`, `lmeval_base.json` and `lmeval_tuned.json`.
The two records of an evaluator must compare: a side resumed from an earlier run with
another precision, dataset, example count, lm-eval version or limit is refused, not
footnoted. `make organism-verify` takes the base pair as `BASE_METRICS` and the organism
pair as `SUBJECT_METRICS`.

No thinking-control tokens are injected on either side; the evaluated model runs at
its family default (off for Gemma 4, whose thinking is opt-in), resolved from the
checkpoint's `config.json`.
