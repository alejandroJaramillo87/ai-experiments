# P2-D2: Data-mixing configuration — implementation investigation

**Frozen record — 2026-07-15. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing P2-D2 (data-mixing configuration) entails,
reconciling the Phase 2 product design against the current state of the repository.
Per the operating rules, the repository is the source of truth; every point where the
design document disagrees with shipped code is called out and the repository
convention wins.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [Design decisions](#design-decisions)
- [Gemma 4 and Qwen3 considerations](#gemma-4-and-qwen3-considerations)
- [Module layout](#module-layout)
- [Acceptance criteria](#acceptance-criteria)
- [Open questions](#open-questions)

## Scope summary

P2-D2 adds weighted sampling across multiple shard directories — the mechanism that
balances new-domain data against replay data and auxiliary sources — expressed as a
`mixture` recipe block and verified by construction. Tag: orchestrate. Depends on:
Phase 0 D2 reader (`training/data/shards.py`), Phase 0 D5 schema
(`training/recipes/schema.py`). The design document's core decisions:

1. A `mixture.sources` list of `{name, path, weight}` entries; weights are
   normalized proportions of tokens drawn, not of files.
2. A `MixtureDataset` in `training/data/mixture.py` wrapping one
   `ShardedTokenDataset` per source, with per-draw categorical sampling that is
   deterministic given `(seed, step)` so Phase 0 D6 resume determinism holds.
3. Every batch element carries a source integer tag consumed by accounting and,
   later, by the replay buffer (P2-D4).
4. Accounting is a first-class output: cumulative tokens-per-source logged to
   tracking every N steps, and a final per-source token table in the run's
   `meta.json`.
5. Epoch semantics: exhausted sources continue by default (epoch counter in the
   cursor); a `strict_single_epoch` flag aborts instead.

Build-order note from the design document: P2-D2 and P2-D4 share the source-tagging
mechanism — P2-D2 lands first, P2-D4 (replay buffer) is then wired in as an ordinary
mixture source whose weight is the replay fraction. P2-D2 therefore defines the
interface P2-D4 must satisfy.

## Current repository state

Everything the deliverable depends on exists; nothing it delivers exists yet.

- **No mixing anywhere.** `grep` for mixture/weighted sampling across `training/`
  finds nothing. `training/data/mixture.py` does not exist.
- **Single-source data path.** `DataConfig` (`training/recipes/schema.py:92-100`)
  has one `source: str` field. The loop (`training/loop.py:137-140`) builds exactly
  one `ShardedTokenDataset(source, window=seq_len)`, optionally staged to tmpfs.
- **Reader.** `ShardedTokenDataset` (`training/data/shards.py:164-252`) is
  map-style (`__len__`/`__getitem__`), mmap-backed, refuses tokenizer-hash
  mismatches against its `manifest.json`, and exposes `state_dict`/
  `load_state_dict` plus `cursor_index` for resume.
- **Resume model is index-based, not cursor-based.** Since Phase 0 D9, the
  authoritative resume state in `dataloader.json` is
  `{"format": "index", "next_index": N, "window": W}` — a flat count of consumed
  windows (`training/loop.py:190-193`, `training/data/loader.py:33-48`). The loop
  replays through an infinite wrapping sampler
  (`training/data/loader.py:51-55`); multi-worker prefetch cannot desynchronize
  resume because the index counts consumed batches, not fetched ones.
- **Prefetch path assumes a single tensor batch.** `build_dataloader` yields
  `(micro_batch, seq_len)` int64 tensors and `CudaPrefetcher` calls
  `batch.to(device)` on that one tensor (`training/data/loader.py:73-135`).
  Source tags require this contract to widen.
- **Validation seam.** `validate()` (`training/recipes/validate.py:336`) collects
  all violations into one `RecipeValidationError`;
  `_check_tokenizer_matches_shards` (`validate.py:309-333`) already compares the
  tokenizer hash against a single shard directory's manifest — the exact check the
  mixture block generalizes over N sources.
- **Tracking and meta.** The loop emits per-step metrics through `Tracker` and
  writes checkpoint `meta.json` with `step`/`tokens_seen`; the per-source token
  table extends this surface.
- **Shard production.** P2-D1 (`training/data/pipeline.py`,
  `scripts/training/prepare_corpus.py`) produces the per-corpus shard directories
  under `/mnt/ai-data/training/shards/<name>` that mixture sources point at.
- **Tests.** `tests/training/` mirrors modules one-to-one with sandbox-only
  fixtures (synthetic shards, no GPU or network); the D6 deterministic-resume test
  is the harness the mixture resume test extends.

## Design-document drift

The repository wins at each point below.

1. **Resume state shape.** The design document says resume saves "each source's
   cursor plus the mixture RNG state in dataloader.json". That describes the
   pre-D9 cursor model. The shipped model is a flat consumed-window index. The
   mixture equivalent keeps the index model:
   `{"format": "mixture", "next_draw": N, "window": W}` plus the recipe seed.
   Because the draw sequence is a pure function of `(seed, draw_index)`, both the
   source choice and every source's position at draw N are recomputable by
   replaying N cheap RNG draws at resume (RNG-only, no shard I/O; about 10^7
   draws per second, negligible against checkpoint load). No mutable cursors or
   RNG state blobs need to be serialized, which also keeps multi-worker prefetch
   desynchronization impossible, as with D9.
2. **Reshuffle-and-continue.** The design document's default for an exhausted
   source is "reshuffle and continue". The shipped reader has no shuffling
   anywhere — reading is sequential windows wrapping modulo the corpus
   (`loader.py:51-55`), a deliberate D9 decision. Introducing per-epoch shuffling
   just for mixture sources would fork the read pattern. Repository-consistent
   default: sequential wrap per source with an epoch counter; `strict_single_epoch`
   aborts as designed. Per-epoch shuffling can be a later, separate decision.
3. **`seed_from_run: true`.** The design document's block carries a
   `seed_from_run` flag. The repository already has exactly one seed
   (`run.seed`, `schema.py:51`) and no alternative seeding path, so the flag is
   dead weight; the mixture RNG derives from `run.seed` unconditionally.
4. **Schema collision with `data.source`.** The design document does not say how
   `mixture` interacts with the existing `DataConfig.source`. The repository
   pattern for optional subsystems is an `enabled` flag on the block (`peft`,
   `sft`, `distill`, `staging`). Decision: `mixture.enabled: false` by default;
   when enabled, `data.source` must be unset (`validate()` enforces mutual
   exclusion) and `seq_len`/`micro_batch`/loader fields stay in `data`.

## Design decisions

1. **Recipe block** (`training/recipes/schema.py`):

   ```yaml
   mixture:
     enabled: true
     sources:
       - {name: domain, path: /mnt/ai-data/training/shards/<domain>, weight: 0.8}
       - {name: replay_web, path: /mnt/ai-data/training/shards/replay-fineweb, weight: 0.15}
       - {name: replay_code, path: /mnt/ai-data/training/shards/replay-starcoder, weight: 0.05}
     strict_single_epoch: false
   ```

   As dataclasses: `MixtureSourceConfig {name, path, weight}` and
   `MixtureConfig {enabled, sources: list[MixtureSourceConfig],
   strict_single_epoch}`, with `mixture` added to `Recipe`. Weights are
   normalized at load; they are proportions of tokens drawn.

2. **`validate()` invariants** (new `_check_mixture`, and
   `_check_tokenizer_matches_shards` generalized):
   - `mixture.enabled` requires a non-empty `sources` list; names non-empty and
     unique; weights positive and finite.
   - Mutual exclusion with `data.source` and with `sft.enabled` (mixture is a
     pretraining-path feature; the SFT path has its own dataset contract).
   - Every source path is a shard directory with a `manifest.json` (same
     skip-if-absent behavior as the existing check, so recipes validate on
     machines without `/mnt/ai-data`).
   - All sources share one `tokenizer_hash`, which also matches
     `tokenizer.path` — refusal happens at `validate()`, before any GPU
     allocation, per the acceptance criteria.
   - `staging.enabled` with mixture is rejected in this deliverable (staging
     stages one directory; extending it is not P2-D2 scope).

3. **`MixtureDataset`** (`training/data/mixture.py`): wraps one
   `ShardedTokenDataset` per source (constructed with the shared tokenizer hash so
   the reader's own mismatch refusal is a second line of defense). Map-style over
   draw index: `__getitem__(i)` seeds a `numpy.random.Generator` stream from
   `(run.seed, i)` (Philox counter-based, so random access needs no replay),
   selects the source by categorical draw over normalized weights, and maps the
   per-source draw count to that source's window index modulo its length
   (sequential wrap, epoch = count // len). Returns `(window, source_tag)`.
   Exposes `len` as a large virtual length (draw-indexed, not window-indexed) and
   the per-source token sizes for accounting. `strict_single_epoch` raises
   `SourceExhaustedError` when any source's draw count reaches its window count.

4. **Loader and loop integration.** `build_dataloader` gains a collate that
   stacks windows into the existing int64 tensor and tags into a small int64
   vector; `CudaPrefetcher` moves both (tags can stay on CPU — only accounting
   reads them). The loop counts `tokens_per_source[tag] += seq_len` per element,
   logs cumulative counts to `Tracker` every `tracking` interval, and writes the
   final per-source token table into the run's `meta.json`. Checkpoint
   `data_state` becomes `{"format": "mixture", "next_draw": N, "window": W}` when
   mixture is enabled; `resume_index` in `training/data/loader.py` learns the new
   format and rejects cross-format resumes.

5. **P2-D4 interface contract (forward-looking, no code now).** Any object with
   `__len__`, `__getitem__(window_index) -> np.ndarray`, and a manifest-declared
   tokenizer hash can be a mixture source. The replay buffer plugs in behind the
   same tag/accounting/determinism machinery.

## Gemma 4 and Qwen3 considerations

The mixing mechanism is tokenizer-agnostic; the model policy enters only through
the shards it mixes.

- **One tokenizer per mixture, by construction.** The tokenizer-hash equality
  invariant means a Gemma 4 mixture is all-Gemma-4 shards and a Qwen3 fallback
  mixture is all-Qwen3 shards. Switching families means re-tokenizing every
  source through the P2-D1 pipeline, including replay corpora — worth stating in
  the runbook because replay shards are large.
- **Shard dtype is uniform within a mixture.** Gemma 4's 262,144-token vocabulary
  forces uint32 shards (4 bytes/token, `dtype_for_vocab`,
  `training/data/shards.py:37`); Qwen3's roughly 151,000-token vocabulary is also
  above the uint16 limit, so both families produce uint32 shards. Mixed dtypes
  cannot occur in practice, and the tokenizer-hash invariant would refuse them
  anyway.
- **Replay proxy corpora.** Google does not release Gemma 4's pretraining
  mixture; the design document records FineWeb-Edu plus a permissive code slice
  (StarCoder2) as the proxy, prepared through P2-D1 like any corpus. P2-D2 only
  needs the weights to express it (the reference 0.8/0.15/0.05 split); the proxy
  decision itself is P2-D4's to record.
- **Reference recipes.** `training/recipes/reference/` gains a
  `continual-gemma4-e2b` mixture example (and the Qwen3 fallback equivalent)
  once P2-D3/D5 blocks exist; P2-D2 ships the block in the schema and a toy
  recipe used by tests, not a full continual recipe.

## Module layout

| Artifact | Change |
|----------|--------|
| `training/data/mixture.py` | New: `MixtureDataset`, `SourceExhaustedError` |
| `training/recipes/schema.py` | New: `MixtureSourceConfig`, `MixtureConfig`; `Recipe.mixture` |
| `training/recipes/validate.py` | New `_check_mixture`; generalize tokenizer-hash check |
| `training/data/loader.py` | Tag-aware collate; `resume_index` mixture format |
| `training/loop.py` | Mixture dataset construction, per-source accounting, `meta.json` table |
| `tests/training/test_mixture.py` | New: proportions, determinism, refusal tests |

## Acceptance criteria

From the design document, restated against the layout above; all three run in the
sandbox (synthetic shards, no GPU):

1. A three-source toy mixture of 10,000 draws matches configured weights within
   1 percent, verified from the source tags (chi-squared or absolute-error check
   with a fixed seed).
2. Checkpoint/resume mid-run reproduces the identical draw sequence — extends the
   Phase 0 D6 deterministic-resume test to `MixtureDataset`.
3. Mismatched tokenizer hashes across sources are refused at `validate()`.

## Open questions

1. **Source-tag plumbing depth.** The design document wants tags on every batch
   element now (P2-D4 consumes them). The minimal alternative defers loop/
   prefetcher changes and derives accounting from the draw sequence alone
   (identical numbers, since draws are deterministic). Tags-now is recommended —
   it makes acceptance criterion 1 an observation rather than an inference and
   is the interface P2-D4 needs — but it widens the batch contract in
   `loader.py`/`loop.py`.
2. **Reshuffle semantics** (drift item 2): confirm sequential-wrap-per-source is
   acceptable as the epoch default, deferring shuffling.
3. **Staging exclusion** (decision 2): confirm rejecting `staging.enabled` with
   mixture rather than staging N directories under one `max_gb` budget.
