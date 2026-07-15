# P2-D4: Replay buffer — implementation investigation

Investigation of what implementing P2-D4 (replay buffer) entails, reconciling the
Phase 2 product design against the current state of the repository. Per the
operating rules, the repository is the source of truth; every point where the
design document disagrees with shipped code is called out and the repository
convention wins. Findings verified as of 2026-07.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [Proposed design decisions](#proposed-design-decisions)
- [Resolved design decisions](#resolved-design-decisions)
- [Model family considerations](#model-family-considerations)
- [Module layout](#module-layout)
- [Acceptance criteria](#acceptance-criteria)
- [Open questions](#open-questions)

## Scope summary

P2-D4 adds the anti-forgetting mechanism for continual pre-training: a configurable
fraction of every effective batch comes from a proxy of the base model's training
distribution, held resident in system RAM so replay never competes with the domain
stream for NVMe bandwidth. Tag: build. Depends on: P2-D2 (mixture source mechanism)
and Phase 0 D9 (loader/pinned memory conventions). The design document's core
decisions:

1. Replay is a mixture source with a residency guarantee, not a parallel code
   path: `training/continual/replay.py` provides `ReplayBuffer`, presenting the
   same interface as `ShardedTokenDataset` and wired in as a mixture source whose
   weight is the replay fraction. Sampling, accounting, and determinism are all
   inherited from P2-D2.
2. Residency: the buffer loads a fixed token budget (`replay.max_gb`, default
   16GB) from the replay shard set into host arrays at run start and samples
   windows from RAM thereafter. Oversized shard sets are uniformly subsampled
   with a logged seed; a manifest of what was loaded is written to the run
   directory. The buffer checks psutil available RAM at load and refuses to
   start if loading would cross the D10 RAM floor.
3. Replay fraction defaults to 0.2 in the reference run, expressed in the recipe
   as a `replay:` convenience block that expands into mixture sources.
4. Proxy corpora for Gemma-4-E2B: FineWeb-Edu (web/knowledge) plus a permissive
   code slice, prepared through the P2-D1 pipeline like any corpus.

## Current repository state

Everything P2-D4 builds on is merged and behaves as follows:

- `training/data/mixture.py` (`MixtureDataset`, P2-D2): takes
  `(name, path, weight)` triples and constructs one `ShardedTokenDataset` per
  source internally. The draw sequence is a pure function of the seed; resume
  state is a single draw index. Every draw carries a source tag; per-source token
  accounting flows through `training/loop.py` (`tokens_per_source`) into tracking
  metrics and `meta.json`.
- Within a source, `MixtureDataset` reads sequential windows wrapping modulo the
  source (no shuffling), with the epoch implied by the per-source position
  counter. `strict_single_epoch` aborts on wrap.
- `training/data/shards.py` (`ShardedTokenDataset`): mmap-backed, manifest-guarded
  (xxhash64 integrity, tokenizer-hash correctness). The interface a replay source
  must satisfy is `__len__`, `__getitem__(i) -> np.ndarray`, and a `manifest`
  attribute carrying `tokenizer_hash` (read by `MixtureDataset` for the
  single-tokenizer invariant).
- `training/data/loader.py` (D9): `build_dataloader` dispatches on
  `isinstance(dataset, MixtureDataset)`; mixture batches are `(tokens, tags)`
  pairs, tags stay on CPU, and `pin_memory` operates at batch level in the
  `DataLoader`. Workers receive the dataset object by fork.
- `training/data/staging.py` (D9): the existing residency precedent — copies
  shard sets to tmpfs under `staging.max_gb` with hash verification and a
  fallback to NVMe mmap. `validate()` currently rejects
  `mixture.enabled && staging.enabled`.
- `training/watchdog.py` (D10): `ram_floor_gb` (default 8.0) with alert below the
  floor and abort below half of it, sampled via psutil. This is the floor the
  replay loader must respect at load time.
- `training/continual/` exists (P2-D1 placed `benchmarks.yaml` there); the
  package is the designated home for `replay.py`.
- `training/recipes/schema.py` has `MixtureConfig`/`MixtureSourceConfig`; there
  is no replay block. `training/recipes/validate.py` `_check_mixture` enforces
  the P2-D2 invariants.
- `training/data/prepare.py` (P2-D1) produces shard directories with manifests;
  replay corpora are prepared through it unchanged.
- No `replay` code exists anywhere in the repository; P2-D4 is greenfield on top
  of stable interfaces.

## Design-document drift

- **The mixture does not accept dataset objects.** The document wires
  `ReplayBuffer` in "as a mixture source", but `MixtureDataset` constructs
  `ShardedTokenDataset` from path triples itself (`mixture.py`
  `__init__`), so a pre-built object cannot currently be injected. Resolution:
  extend `MixtureDataset` to accept either `(name, path, weight)` or
  `(name, dataset, weight)` per source (duck-typed on the
  `__len__`/`__getitem__`/`manifest` contract), keeping the path form as the
  common case. This is additive; existing callers and checkpoints are
  unaffected.
- **Within-source sampling is sequential, not uniform.** The document says the
  buffer samples "windows uniformly from RAM"; the shipped P2-D2 convention is
  sequential windows wrapping modulo the source, with resume determinism resting
  on the per-source position counter. Resolution: keep the P2-D2 position
  counter and put the uniformity inside `ReplayBuffer` — a seeded fixed
  permutation maps position to window, giving uniform-without-replacement order
  per epoch while remaining a pure function of `(seed, position)`. One
  determinism story, no new resume state.
- **Pinned host arrays are unnecessary at the dataset layer.** The document
  offers "pinned or plain host arrays"; in the shipped loader, pinning happens
  at batch level (`DataLoader(pin_memory=...)`, D9), and pinning a 16GB dataset
  would lock a large slice of the 128GB against the D10 floor for no transfer
  benefit. Resolution: plain (unpinned) numpy arrays in the buffer; batch-level
  pinning stays as-is.
- **The mixture/staging exclusion needs a note, not a change.** `validate()`
  rejects staging with mixtures, and replay expands into a mixture source, so a
  replay run can never also stage. That is coherent — replay is itself the
  residency mechanism for the fraction that matters — but the `_check_mixture`
  message should mention the replay block once it exists.

## Proposed design decisions

1. **`training/continual/replay.py` provides `ReplayBuffer`**, satisfying the
   `ShardedTokenDataset` duck-type: `__len__` (windows loaded), `__getitem__`
   returning a window ndarray copy, and a `manifest` with the source
   `tokenizer_hash`. Construction: read the replay shard directory's manifest,
   verify hashes (same discipline as D2), load up to `max_gb` of tokens into a
   contiguous host array, then serve windows through the seeded permutation.
2. **Budget and subsampling.** `max_gb` bounds the loaded array. If the shard
   set fits, load it all; otherwise take a uniform window subsample with a
   logged seed and write `replay_manifest.json` into the run directory recording
   the source manifest hash, subsample seed, and the exact window set loaded —
   so a resume or audit can reconstruct what was resident.
3. **RAM-floor refusal.** Before allocating, compare
   `psutil.virtual_memory().available` against `load_bytes +
   watchdog.ram_floor_gb`; refuse with an actionable error if loading would
   cross the floor. The D10 watchdog remains the runtime backstop.
4. **Recipe surface: a `replay:` convenience block**
   (`{enabled, fraction: 0.2, sources: [...], max_gb: 16}`) in
   `training/recipes/schema.py`. Expansion happens where the loop builds the
   dataset (`training/loop.py`), not by mutating the recipe: domain mixture
   weights renormalize to `1 - fraction`, and each replay source enters with
   weight `fraction * its share`. `validate()` gains: `fraction` in (0, 1),
   `max_gb` positive when set, replay requires `mixture.enabled` (resolved
   decision 1), unique names across domain and replay sources, and the
   shared-tokenizer invariant already enforced by `MixtureDataset`.
5. **Accounting is inherited.** Replay sources appear in `tokens_per_source`
   like any source; the acceptance check for the replay fraction reads this
   accounting, matching the P2-D2 convention.
6. **Resume is inherited.** The single mixture draw index remains the only
   dataloader resume state; the buffer's permutation and subsample are pure
   functions of recorded seeds, so an identical resume reconstructs identical
   draws.
7. **Worker processes share the buffer copy-on-write.** The buffer is loaded in
   the parent before `DataLoader` workers fork (Linux default start method), so
   the 16GB array is shared, not duplicated. This is load-order sensitive and
   is asserted in the throughput acceptance run rather than left implicit.

## Resolved design decisions

The open questions below were reviewed and resolved as follows:

1. **Replay requires an enabled mixture block.** `validate()` rejects
   `replay.enabled` on a single-source recipe, with a message directing the
   user to move `data.source` into a one-entry `mixture.sources` list. No
   silent expansion of single-source runs: the recipe explicitly shows the run
   is a mixture, matching the `mixture`-format dataloader checkpoint it
   produces, and the loop keeps one dataset-construction path. This overrides
   the design document's convenience framing (repository convention: explicit
   configuration over expansion magic, as with the existing
   `data.source`/`mixture` mutual exclusion).
2. **`replay.max_gb` is calculator-derived by default.** `host_ram_bytes` in
   `training/calculator/components.py` gains a replay term so the fit report
   carries one host-RAM number (offloaded optimizer state + replay buffer +
   watchdog floor against 128GB). When `max_gb` is unset, the loop sizes the
   buffer to the calculator's remaining host headroom, capped at the shard-set
   size; when set, it is an explicit override and `validate()` warns when the
   recipe-implied host total exceeds 128GB minus the floor. This lands the
   P2-D5 host-budget validation's replay term early.
3. **Acceptance runs on a synthetic replay corpus; real corpora are deferred.**
   P2-D4 includes generating a synthetic replay shard set that fully exercises
   the deliverable: sized in the multi-GB range so the budget, subsampling,
   RAM-floor, and copy-on-write behaviors are meaningful, produced through the
   P2-D1 `ShardWriter`/manifest path so integrity and tokenizer-hash guards
   are exercised for real. Downloading and preparing FineWeb-Edu and the code
   slice with the Gemma 4 tokenizer is deferred to the reference-run setup
   (P2-D5), where the corpus proportions are decided.

## Model family considerations

The mechanism is model-agnostic; the model family enters only through data
preparation:

- Gemma 4 E2B (primary; HF `model_type` `gemma4`, per `training/families.py`)
  has a ~256k-entry vocabulary, so P2-D1 shards for it are uint32
  (`dtype_for_vocab`), and 16GB of buffer holds roughly 4B tokens — the design
  document's arithmetic matches the shipped shard format.
- Qwen3 (fallback family, registered in `families.py`) has a ~151k vocabulary —
  also uint32, same buffer arithmetic.
- The `MixtureDataset` tokenizer-hash invariant means replay shard sets are
  per-family: FineWeb-Edu and the code slice must be prepared once through the
  P2-D1 pipeline with the Gemma 4 tokenizer, and again with the Qwen3 tokenizer
  if the fallback is exercised. Nothing in `replay.py` needs to know which
  family is in play.
- The forgetting report (P2-D6) carries the proxy-corpus caveat; P2-D4 only
  records the loaded manifest that makes the caveat auditable.

## Module layout

| Path | Role |
|------|------|
| `training/continual/replay.py` | `ReplayBuffer` (new) |
| `training/data/mixture.py` | Accept pre-built dataset objects as sources (additive) |
| `training/loop.py` | Expand the `replay:` block into mixture sources at dataset build |
| `training/recipes/schema.py` | `ReplayConfig` (`enabled`, `fraction`, `sources`, `max_gb`) |
| `training/recipes/validate.py` | `_check_replay` invariants; `_check_mixture` message notes replay |
| `training/calculator/components.py` | `host_ram_bytes` gains a replay-buffer term (resolved decision 2) |
| `tests/training/test_replay.py` | Buffer, budget, RAM-floor, and resume tests (new) |
| `tests/training/test_recipes.py` | One test per new `validate()` invariant |
| `scripts/training/make_replay_synthetic.py` | Generates the multi-GB synthetic replay shard set for the host-side acceptance run (resolved decision 3) |

## Acceptance criteria

Mapped from the design document; all but the last executable in the CPU sandbox:

- Source-tag accounting over a test run shows the configured replay fraction
  within 1 percent (extends the P2-D2 10k-draw test with a replay source).
- The buffer respects `max_gb`: an oversized shard set is subsampled with a
  logged seed and manifest, and a `max_gb` of zero-tolerance configuration
  refuses; both paths tested.
- The RAM-floor refusal fires against a mocked psutil.
- Checkpoint/resume mid-run reproduces the identical replay draw sequence
  (extends the P2-D2 deterministic-resume test).
- Host-side: against the multi-GB synthetic replay shard set (resolved
  decision 3), throughput with replay enabled is not measurably worse than a
  pure-shard run of the same mixture, and worker RSS confirms copy-on-write
  sharing of the buffer. Requires the workstation; the human runs the command
  and reports output per the operating rules.

## Open questions

None. The three questions raised during investigation (single-source replay,
`max_gb` default, real-corpus preparation) are settled in
[Resolved design decisions](#resolved-design-decisions).
