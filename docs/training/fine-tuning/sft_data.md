# Chat and instruction data preparation

Turns chat and instruction datasets into tokenized, loss-masked training examples for the
supervised fine-tuning trainer (Phase 1 F1). This is the correctness surface a fine-tune
depends on most: a wrong template or a wrong mask destroys a run while its loss curve looks
healthy, so every mask boundary here is known by construction and pinned by a golden test.
Design record: [the F1 investigation](../investigations/f1_sft_data_investigation.md).

## Table of Contents

<!-- toc:start -->

1. [Record format](#record-format)
2. [Templates and masks](#templates-and-masks)
3. [Preparing a dataset](#preparing-a-dataset)
4. [Loading and the held-out split](#loading-and-the-held-out-split)
5. [Modules and tests](#modules-and-tests)

<!-- toc:end -->

## Record format

Every record normalizes to the OpenAI-style messages list, one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Roles are `system`, `user` and `assistant`; multi-turn is the native case and single-turn
a special case of it. Loaders accept a local JSONL file or a Hugging Face dataset, and a
plain instruction under `instruction`, `prompt` or `text` is converted. An assistant turn
carrying the family's reasoning-trace markers is refused: the trainer learns answers, not
raw traces. This is the format the teacher harness writes
([teacher_synthetic_data.md](teacher_synthetic_data.md)) and the organism generator hands on.

## Templates and masks

`training/models/families.py` is the per-family registry: for each family a
`ChatTemplateSpec` (the rendered form of every turn), the `target_modules` list the PEFT
`auto` setting resolves to, and the thinking default. Registered families are `gemma4` and
`gemma4_text`; an unknown family fails with the registration instructions rather than
falling back to a generic template. The Gemma 4 rendering is held byte for byte against the
checkpoint's own template by `tests/training/test_chat_template_contract.py`, and against
the rust harness's renderer through `schemas/chat_template.json`.

Rendering is segment by segment (`render_segments` in `training/data/sft.py`), so the label
of every token is known when it is produced: `-100` on system and user turns, the assistant
role header and the template's control tokens; the token id on assistant content and the
turn's end-of-turn token. No string match on rendered text decides a boundary.
`mask_debug(sample, id_to_token)` renders a sample as a token-by-token table (token, id,
label) for reading a mask by eye.

## Preparing a dataset

```bash
make prepare-sft ARGS="--source /mnt/ai-data/training/datasets/teacher/<name>/data.jsonl \
    --tokenizer /mnt/ai-data/models/hf/gemma-4-E2B --model-type gemma4 \
    --max-seq-len 4096 --out /mnt/ai-data/training/datasets/sft/<name>"
```

`--source` is a local JSONL path or a hub dataset id (with `--split` and `--revision`);
`--holdout` (default 0.02) and `--seed` fix the held-out split; `--license` is recorded in
the manifest. `--repeat` writes the source that many times over, and `--replay-source` with
`--replay-examples` (plus `--replay-split` and `--replay-revision` for a hub source) appends
that many records from the head of a second source after it, in source order: a small corpus
trained alone for many epochs pulls the model onto it and off everything else (the organism
recipes, KB-105), and replay is how it keeps the base's own distribution in every batch. The
output directory holds `train.jsonl` and `holdout.jsonl` (one `{"input_ids", "labels"}`
object per line) and `manifest.json` with the source, the repeat count and the replay block,
the tokenizer hash, the chat-template version, example and token counts, and the split seed. Re-running
with the same inputs is deterministic, split membership included. Sequence packing is not
done here: the trainer packs at training time and never packs the holdout
([sft_entry_point.md](sft_entry_point.md)).

## Loading and the held-out split

`load_sft_dataset(dataset_dir, tokenizer)` returns the train and held-out lists after
verifying the manifest: a content hash or a tokenizer hash that does not match is an error,
not a warning, so a dataset prepared against one tokenizer cannot train against another
(KB-80 fixed the tokenizer the check reads to the one the base directory ships). The
held-out split is what the trainer evaluates at `sft.eval_steps` intervals and what the
`eval-holdout` evaluator scores on the base and the fine-tune
([eval_wrapper.md](eval_wrapper.md)).

## Modules and tests

| Path | Role |
|------|------|
| `training/data/sft.py` | normalisation, rendering, encoding, `mask_debug`, `prepare_sft`, the manifest, `load_sft_dataset` |
| `training/models/families.py` | the family registry: templates, adapter targets, thinking default |
| `training/commands/prepare_sft.py` | `python -m training prepare-sft`, wrapped by `make prepare-sft` |
| `tests/training/test_sft.py` | golden mask boundaries for single-turn, multi-turn and system-prompt cases, the detokenised round trip, deterministic re-runs, the manifest |
| `tests/training/test_chat_template_contract.py` | the rendering against the checkpoint's template |
