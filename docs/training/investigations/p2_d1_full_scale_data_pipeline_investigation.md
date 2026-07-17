# P2-D1: Full-scale data pipeline — implementation investigation

Investigation of what implementing P2-D1 (full-scale data pipeline) entails,
reconciling the Phase 2 product design against the current state of the repository.
Per the operating rules, the repository is the source of truth; every point where the
design document disagrees with shipped code is called out and the repository
convention wins. Findings verified as of 2026-07.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [Design decisions](#design-decisions)
- [Gemma 4 considerations](#gemma-4-considerations)
- [Module layout](#module-layout)
- [Acceptance criteria](#acceptance-criteria)
- [Host-side follow-up](#host-side-follow-up)

## Scope summary

P2-D1 adds the quality stages the lite pipeline (Phase 0 D2) deliberately omitted —
language identification, heuristic quality filtering, exact deduplication,
MinHash-LSH near-deduplication, and benchmark decontamination — in front of the
existing tokenize-and-shard writer. The shard format, manifest contract, and
`ShardedTokenDataset` reader do not change; the pipeline only decides which documents
reach the writer.

Stage order: ingest (HF dataset or local JSONL) -> language id (fastText, keep `en`
at 0.65 or above, configurable) -> Gopher and FineWeb heuristic filters -> exact
dedup (xxhash on normalized text) -> MinHash-LSH near-dedup (5-grams, 112 hash
functions, approximately 0.75 Jaccard threshold) -> 13-gram benchmark
decontamination -> the Phase 0 D2 writer.

## Current repository state

The seam the design depends on exists exactly as described:

- `training/data/prepare.py` states dedup, filtering, and decontamination are
  deferred to Phase 2 "in front of the same writer".
- `training/data/shards.py` already switches to uint32 above a 65,536 vocabulary,
  so Gemma 4's 262,144-token vocabulary needs no format change.
- `training/families.py` registers Gemma 4 E-models as primary and Qwen3 as
  fallback, matching the Phase 2 model policy.
- No dedup, language-id, or decontamination code existed anywhere before this
  deliverable; `training/continual/` and `benchmarks.yaml` did not exist.

## Design-document drift

| Design document says | Repository convention (wins) |
|---|---|
| `scripts/train/prepare_corpus.py` | `scripts/training/prepare_corpus.py` (all training CLIs live in `scripts/training/`) |
| No investigation document mentioned | Each deliverable ships a `*_investigation.md` under `docs/training/` |
| datatrove `[processing]` extra suffices | datatrove 0.9.0 needs `[io]` for orjson/datasets readers and an explicit `spacy` pin for the Gopher/FineWeb word tokenizer |

## Design decisions

1. **Engine: datatrove 0.9.0 with `LocalPipelineExecutor`.** The repository writes
   configuration plus one custom block (the contamination filter); dedup and filter
   algorithms come from datatrove. The `text-dedup` fallback is deliberately not
   pinned until datatrove misbehaves at scale (one engine at a time).
2. **MinHash parameters are datatrove defaults**, recorded in the shard manifest:
   5-gram shingles, 14 buckets x 8 hashes = 112 hash functions, seed 1. Exact dedup
   uses datatrove's staged signature/find/filter blocks over whitespace-collapsed
   lowercase text.
3. **The benchmark list is fixed now** in `training/continual/benchmarks.yaml`
   (hellaswag, arc_easy, arc_challenge, piqa, winogrande, mmlu — the P2-D6 list,
   recorded ahead of that deliverable so decontamination is not blocked). Revisions
   are pinned at index-build time: the sandbox cannot reach the HF API, so
   `build_contamination.py` resolves `main` to a concrete commit SHA and records it
   in the index manifest.
4. **Decontamination is 13-gram overlap.** The index directory holds a sorted uint64
   hash array (`ngrams.bin`) plus `index_manifest.json`; shard manifests carry the
   manifest file's xxhash64 as `decontaminated_against`, so a benchmark-list change
   is detectable downstream.
5. **Stage boundaries are JSONL directories** under the caller's work directory
   (convention `/mnt/ai-data/training/datasets/<name>/`), each with a `stats.json`
   funnel (docs in, docs out, drop-reason histogram from the filters' exclusion
   writers). `--clean-intermediates` removes stage data but keeps every
   `stats.json`; the final shard manifest embeds the whole funnel.
6. **Parallelism: 30 workers** (`workers` in the pipeline config), matching the
   Phase 0 D2 convention under the inference-down policy. Executors use the `fork`
   start method: datatrove's default `forkserver` re-imports `__main__`, which is
   hostile to thin CLI scripts.
7. **Manifest changes are additive.** `prepare()` gains `extra_manifest` and
   `dataset_name` keyword arguments; `iter_jsonl()` accepts stage directories and
   gzip files. Existing manifests remain readable; `extra_manifest` refuses to
   override core fields.

## Gemma 4 considerations

- The reference tokenizer for Phase 2 corpora is Gemma 4's
  (`--tokenizer google/gemma-4-E2B`); its 262,144 vocabulary produces uint32 shards
  through the existing `dtype_for_vocab` switch, doubling bytes per token versus
  uint16. Token budgets and disk-floor math must use 4 bytes per token.
- The Gemma 4 hub repository is gated: `HF_TOKEN` via `.env.training` must have
  accepted the license before ingest or index builds that touch gated sets.
- Qwen3 remains the fallback family; nothing in the pipeline is
  tokenizer-specific beyond the manifest hash check.

## Module layout

| Path | Role |
|---|---|
| `training/data/pipeline.py` | Stage composition, `PipelineConfig`, funnel accounting, `prepare_corpus()` |
| `training/data/contamination.py` | Index build, membership check, datatrove filter block |
| `training/data/configs/default.yaml` | Documented default pipeline configuration |
| `training/continual/benchmarks.yaml` | The P2-D6 benchmark list gating decontamination |
| `scripts/training/prepare_corpus.py` | Thin CLI (superset of `prepare_data.py`, same writer) |
| `scripts/training/build_contamination.py` | Index build CLI |
| `tests/training/test_pipeline.py`, `tests/training/test_contamination.py` | Sandbox-runnable unit tests |

Make targets: `make prepare-corpus ARGS=...` and `make build-contamination ARGS=...`.

## Acceptance criteria

From the Phase 2 design, restated against this implementation:

1. A corpus run produces shards whose manifest records the full stage funnel,
   MinHash parameters, and `decontaminated_against` hash — covered by
   `test_planted_documents_dropped_and_attributed`.
2. Planted near-duplicates and planted benchmark passages are dropped and
   attributed to the right stage in `stats.json` — same test, one planted document
   per stage.
3. Re-running with identical inputs and config is deterministic (identical shard
   hashes) — covered by `test_rerun_is_deterministic`.
4. A 1B+-token run sustains CPU-bound throughput across 30 workers with no writer
   stall — host-side check, see below.

## Host-side follow-up

The sandbox cannot regenerate `uv.lock` (the PyTorch index is proxy-blocked) or run
at scale. On the workstation, in order:

```bash
$ uv lock
$ make train-build
$ make build-contamination ARGS="--out /mnt/ai-data/training/datasets/_contamination/v1"
$ make prepare-corpus ARGS="--source HuggingFaceFW/fineweb-edu --split train \
    --contamination /mnt/ai-data/training/datasets/_contamination/v1 \
    --tokenizer google/gemma-4-E2B \
    --work-dir /mnt/ai-data/training/datasets/fineweb-edu-1B \
    --out /mnt/ai-data/training/shards/fineweb-edu-1B \
    --max-tokens 1_000_000_000 --clean-intermediates"
```

The `prepare-corpus` run is the acceptance check for criterion 4: observe CPU-bound
throughput via `htop`/`pidstat` and record the reported output here when run.

*Last updated: 2026-07-15*
