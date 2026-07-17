# Data pipeline (Phase 0 D2, Phase 2 P2-D1)

Tokenizes a Hugging Face dataset or local JSONL into memory-mapped token shards with a
JSON manifest, and provides the mmap-backed reader training loops consume. Two entry
points share one shard writer: the lite path (Phase 0 D2, no dedup, filtering, or
decontamination) and the full-scale path (Phase 2 P2-D1) that adds quality stages in
front of the same writer. P2-D1 design background:
[P2-D1 investigation](investigations/p2_d1_full_scale_data_pipeline_investigation.md).

## Table of contents

- [Components](#components)
- [Shard format](#shard-format)
- [Manifest contract](#manifest-contract)
- [Usage](#usage)
- [Full-scale pipeline](#full-scale-pipeline)
- [Reader](#reader)
- [Parallelism](#parallelism)
- [Testing](#testing)

## Components

| Path | Role |
|------|------|
| `/training/data/shards.py` | Shard writer, manifest read/write/verify, `ShardedTokenDataset` reader |
| `/training/data/prepare.py` | Tokenize-and-pack logic (HF datasets, streaming, local JSONL) |
| `/training/commands/prepare_data.py` | Thin CLI over `prepare()` (lite path) |
| `/training/data/pipeline.py` | P2-D1 stage composition: filters, dedup, decontamination |
| `/training/data/contamination.py` | Benchmark n-gram index build and contamination filter |
| `/training/continual/benchmarks.yaml` | Benchmark list gating decontamination (P2-D6 list) |
| `/training/commands/prepare_corpus.py` | Thin CLI over `prepare_corpus()` (full-scale path) |
| `/training/commands/build_contamination.py` | Contamination index build CLI |
| `Makefile` `prepare-data`, `prepare-corpus`, `build-contamination` | Container wrappers |

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
$ python -m training prepare-data --source ./data.jsonl --text-field text \
    --tokenizer HuggingFaceTB/SmolLM2-135M \
    --out /mnt/ai-data/training/shards/local-test
```

Flags: `--streaming` for corpora larger than RAM (HF streaming mode feeds the same
writer), `--shard-tokens` to override the shard size, `--eos-token` when the tokenizer
does not declare an EOS token, `--license` to record the license string, `--revision`
to pin the dataset revision. The tool refuses to overwrite an existing manifest. Until
the D3 tokenizer trainer lands, pass any `AutoTokenizer`-loadable tokenizer (hub id or
local directory).

## Full-scale pipeline

`prepare_corpus.py` runs the P2-D1 stage chain before the shard writer: language
identification (fastText, keep `en` at 0.65 or above) -> Gopher and FineWeb heuristic
quality filters -> exact dedup (xxhash on normalized text) -> MinHash-LSH near-dedup
(5-grams, 112 hash functions, approximately 0.75 Jaccard) -> 13-gram benchmark
decontamination. The engine is datatrove's `LocalPipelineExecutor`; stage thresholds
live in a YAML config (`/training/data/configs/default.yaml` documents the defaults,
and filters are per-corpus toggles because quality heuristics can gut code or math
corpora).

Build the contamination index once per benchmark-list revision, then prepare corpora
against it:

```bash
$ make build-contamination ARGS="--out /mnt/ai-data/training/datasets/_contamination/v1"
$ make prepare-corpus ARGS="--source HuggingFaceFW/fineweb-edu --split train \
    --contamination /mnt/ai-data/training/datasets/_contamination/v1 \
    --tokenizer google/gemma-4-E2B \
    --work-dir /mnt/ai-data/training/datasets/fineweb-edu-2B \
    --out /mnt/ai-data/training/shards/fineweb-edu-2B \
    --max-tokens 2_000_000_000 --clean-intermediates"
```

Stage intermediates are JSONL directories under the work directory, each with a
`stats.json` funnel (docs in, docs out, drop-reason histogram); they are regenerable
and `--clean-intermediates` deletes the data while keeping every `stats.json`. The
shard manifest additionally records the full stage funnel, the MinHash parameters,
and `decontaminated_against` (the contamination index manifest hash), so every
published number about the corpus is reproducible from the manifest alone.

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
