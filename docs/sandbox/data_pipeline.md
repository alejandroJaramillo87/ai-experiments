# Lite data pipeline (Phase 0 D2)

Tokenizes a Hugging Face dataset or local JSONL into memory-mapped token shards with a
JSON manifest, and provides the mmap-backed reader training loops consume. Lite means no
dedup, filtering, or decontamination; Phase 2 adds those in front of the same shard
writer. Design background: [D2 investigation](d2-lite-data-pipeline-investigation.md).

## Table of contents

- [Components](#components)
- [Shard format](#shard-format)
- [Manifest contract](#manifest-contract)
- [Usage](#usage)
- [Reader](#reader)
- [Parallelism](#parallelism)
- [Testing](#testing)

## Components

| Path | Role |
|------|------|
| `/training/data/shards.py` | Shard writer, manifest read/write/verify, `ShardedTokenDataset` reader |
| `/training/data/prepare.py` | Tokenize-and-pack logic (HF datasets, streaming, local JSONL) |
| `/scripts/train/prepare_data.py` | Thin CLI over `prepare()` |
| `Makefile` `prepare-data` | Container wrapper: `make prepare-data ARGS="..."` |

## Shard format

Flat binary token files plus a JSON manifest (the nanoGPT/nanotron family):

- One `shard_NNNNNN.bin` per shard containing contiguous token ids.
- dtype `uint16` when the tokenizer vocabulary is under 65,536 ids, else `uint32`.
- Documents separated by the tokenizer EOS id; documents may span a shard boundary
  (the token stream is contiguous; only reader windows respect boundaries).
- Fixed shard size, default 536,870,912 tokens (512M, approximately 1GB at `uint16`);
  the final shard may be partial.
- Shards are written append-only by a single sequential writer and hashed incrementally
  (xxhash64) as they close; closing never re-reads a shard.
- Shards are raw mmap-able binaries and are never compressed at rest.

Output convention: `/mnt/ai-data/training/shards/<name>/`, never inside the repository.

## Manifest contract

`manifest.json` beside the shards records: dataset name or path, split, revision,
license string (required, may be `unknown`), tokenizer source and xxhash64 of the
serialized tokenizer, dtype, EOS id, per-shard token counts and content hashes, total
tokens, total documents, document-length histogram (log-scale buckets), creation
timestamp, pipeline git SHA, and the exact command.

The manifest is the contract: `ShardedTokenDataset` refuses a shard directory whose
shard sizes or content hashes do not match the manifest (integrity), or whose
`tokenizer_hash` does not match the caller-supplied tokenizer hash (correctness).

## Usage

Host-side, through the training container:

```bash
$ make prepare-data ARGS="--source HuggingFaceFW/fineweb-edu --split train \
    --tokenizer /mnt/ai-data/training/tokenizers/base \
    --out /mnt/ai-data/training/shards/fineweb-edu-10B \
    --max-tokens 10_000_000_000 --num-proc 30"
```

Local JSONL:

```bash
$ python scripts/train/prepare_data.py --source ./data.jsonl --text-field text \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --out /mnt/ai-data/training/shards/local-test
```

Flags: `--streaming` for corpora larger than RAM (HF streaming mode feeds the same
writer), `--shard-tokens` to override the shard size, `--eos-token` when the tokenizer
does not declare an EOS token, `--license` to record the license string, `--revision`
to pin the dataset revision. The tool refuses to overwrite an existing manifest. Until
the D3 tokenizer trainer lands, pass any `AutoTokenizer`-loadable tokenizer (hub id or
local directory).

## Reader

```python
from training.data.shards import ShardedTokenDataset

ds = ShardedTokenDataset(shard_dir, window=2048, tokenizer_hash=tok_hash)
window = ds.next_window()          # numpy array of 2048 token ids, or None when exhausted
state = ds.state_dict()            # serializable cursor: shard index, window offset
ds.load_state_dict(state)          # resume (D6 checkpointing stores this)
ds.dropped_tokens                  # tokens lost to partial windows at shard tails
```

Windows never cross shard boundaries; the last partial window per shard is dropped and
`dropped_tokens` reports the total so token accounting stays honest.

## Parallelism

Non-streaming HF sources tokenize through `datasets.map` with `num_proc` worker
processes (default 30 of 32 cores, leaving two for the writer and OS under the
inference-down policy). Streaming and JSONL sources tokenize through the fast
tokenizer's `encode_batch`, which parallelizes internally in Rust. Both paths feed the
same single sequential shard writer.

## Testing

Sandbox-runnable unit tests live in `/tests/training/` (tiny on-the-fly BPE tokenizer,
synthetic JSONL, no network or GPU):

```bash
$ python -m pytest tests/training/
```

Host-side acceptance (per the roadmap): a 1B-token run demonstrating CPU-bound
throughput across the configured `num_proc` with no writer stall, observed via
htop/pidstat.
