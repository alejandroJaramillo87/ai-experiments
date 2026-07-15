# Tokenizer training (Phase 0 D3)

Trains and evaluates byte-level BPE tokenizers for Phase 3 from-scratch models.
Fine-tuning and continual pre-training always reuse the base model's tokenizer; this
tool exists only for the from-scratch path. A mismatched tokenizer silently destroys a
fine-tune, and cross-tokenizer distillation breaks logit KL. Design background:
[D3 investigation](d3_tokenizer_training_investigation.md).

## Table of contents

- [Components](#components)
- [Tokenizer design](#tokenizer-design)
- [Artifact format](#artifact-format)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Testing](#testing)

## Components

| Path | Role |
|------|------|
| `/training/data/tokenizer.py` | `train()`, `evaluate()`, `write_artifact()`, corpus sampling |
| `/training/commands/train_tokenizer.py` | Thin CLI over the module |
| `Makefile` `train-tokenizer` | Container wrapper: `make train-tokenizer ARGS="..."` |

## Tokenizer design

Hugging Face `tokenizers` byte-level BPE, GPT-2/Llama-3 style: byte-level
pre-tokenization over the full 256-byte alphabet (no unknown token), digits split
individually so numbers never merge, ByteLevel decoder for lossless round trips.

Special tokens occupy fixed ids ahead of the byte alphabet, independent of corpus
content:

| Token | Id |
|-------|----|
| `<|bos|>` | 0 |
| `<|eos|>` | 1 |
| `<|pad|>` | 2 |
| `<|reserved_0|>` .. `<|reserved_15|>` | 3-18 |

The reserved block exists so Phase 1 chat/control tokens never resize embeddings. No
post-processor attaches BOS automatically; the `<|bos|>` id is reserved but unused,
keeping D2 EOS-only document packing unchanged. The 19 special ids count against
`--vocab-size`.

## Artifact format

A directory, conventionally `/mnt/ai-data/training/tokenizers/<name>/`, containing:

- `tokenizer.json` — the serialized tokenizer.
- `manifest.json` — corpus file list, text field, sampled byte and document counts,
  vocab size, special-token id map, evaluation metrics, reference-tokenizer metrics,
  `tokenizers` library version, creation timestamp, git SHA, exact command, and
  `tokenizer_hash`.

`tokenizer_hash` is xxhash64 of the reloaded tokenizer's `to_str()` serialization — the
same value `load_tokenizer()` in `/training/data/prepare.py` reports, which D2 shard
manifests record and D5 `validate()` enforces. The artifact is final at training time:
any post-training mutation (added tokens, re-serialization by a different library
version) changes the hash and breaks the shard/recipe hash chain. The CLI refuses an
existing artifact directory without `--force`.

## Evaluation

Evaluation runs as part of training, on held-out documents from at least two domains
(web text and code by default, via repeatable `--eval-corpus DOMAIN=PATH`). When no
evaluation corpus is given, the unconsumed tail of the training corpus is used and a
warning is printed; the tail is unseen only when the corpus exceeds `--sample-bytes`.

Per domain, for the trained tokenizer and each reference:

- **Fertility** — tokens per whitespace word (lower is better).
- **Compression** — bytes per token (higher is better).
- **Round-trip losslessness** — decode(encode(text)) must equal text on every held-out
  document (trained tokenizer only); any failure aborts with exit code 1.

Default references are `HuggingFaceTB/SmolLM2-135M` and `gpt2` (both ungated).
`--reference` overrides the set; `--no-references` skips the comparison for offline
runs, and an unreachable reference degrades to a warning rather than a failure.
Results are stored in the manifest and printed as a table.

## Usage

Host-side, through the training container:

```bash
$ make train-tokenizer ARGS="--corpus '/mnt/ai-data/training/datasets/mix/*.jsonl' \
    --vocab-size 49152 --sample-bytes 10_000_000_000 \
    --eval-corpus web=/mnt/ai-data/training/datasets/eval/web.jsonl \
    --eval-corpus code=/mnt/ai-data/training/datasets/eval/code.jsonl \
    --out /mnt/ai-data/training/tokenizers/base-49k"
```

Flags: `--corpus` (repeatable JSONL file or quoted glob), `--out`, `--vocab-size`
(default 49,152), `--sample-bytes` (default 10,000,000,000; underscores allowed),
`--text-field` (default `text`), `--eval-corpus DOMAIN=PATH` (repeatable),
`--eval-docs` (default 1,000), `--reference` (repeatable), `--no-references`,
`--force`. Exit codes: 0 success, 1 training or evaluation failure, 2 invalid usage.

The artifact feeds D2 directly:

```bash
$ make prepare-data ARGS="--source ... \
    --tokenizer /mnt/ai-data/training/tokenizers/base-49k \
    --out /mnt/ai-data/training/shards/<name>"
```

## Reproducibility

Corpus traversal is deterministic: glob expansion is sorted and deduplicated, files are
read sequentially, and sampling stops at the byte budget rather than sampling randomly.
Two runs over the same corpus with the same flags produce byte-identical
`tokenizer.json` files and therefore identical `tokenizer_hash` values.

## Testing

`tests/training/test_tokenizer.py` runs on a synthetic corpus with no network or GPU:
special-token ids and reserved block, digit splitting, vocabulary floor, reproducible
artifact hash across two trainings, round-trip losslessness (including multi-byte
UTF-8), overwrite refusal without `--force`, manifest contents, corpus sampler budget
and held-out behavior, loading through `load_tokenizer()` with `<|eos|>` resolution,
and an integration check that D2 shards prepared with the artifact carry its manifest
hash.
