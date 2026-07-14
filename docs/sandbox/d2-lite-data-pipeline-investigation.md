# D2 lite data pipeline: implementation investigation

Findings from comparing the Phase 0 D2 design (training-tier planning document) against the
current repository state, and a breakdown of what implementing D2 entails. Where the planning
document and the repository disagree, the repository is authoritative.

## Table of contents

- [Scope of D2](#scope-of-d2)
- [Current repository state](#current-repository-state)
- [Deviations from the planning document](#deviations-from-the-planning-document)
- [Dependency status: D1 done, D5 missing](#dependency-status-d1-done-d5-missing)
- [Work breakdown](#work-breakdown)
- [Design decisions carried from the document](#design-decisions-carried-from-the-document)
- [Open questions](#open-questions)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)

## Scope of D2

Turn a Hugging Face dataset or local JSONL into tokenized, memory-mapped token shards with a
JSON manifest, plus the reader half (`ShardedTokenDataset`) that training loops consume.
Explicitly excluded: dedup, filtering, decontamination (Phase 2 adds those in front of the
same shard writer).

Deliverables named by the design:

- `training/data/prepare.py` — tokenize-and-pack logic (HF datasets or local JSONL).
- `training/data/shards.py` — shard format read/write, manifest, `ShardedTokenDataset`.
- `scripts/train/prepare_data.py` — thin CLI over the above.
- `make prepare-data ARGS=...` — Makefile wrapper.
- Output convention: `/mnt/ai-data/training/shards/<name>/`, never inside the repository.

## Current repository state

D2 is entirely greenfield; D1 pre-provisioned exactly the substrate it needs.

| Component | State |
|-----------|-------|
| `training/` Python package | Does not exist (no `training/data/`, no recipes) |
| `scripts/train/` | Exists; contains only `smoke_test.py` (D1) |
| `ShardedTokenDataset` | No occurrence anywhere in the repository |
| Manifest format | Undefined; only `versions.json` from the smoke test exists |
| Tests for training code | None (`tests/` has only `agents/` and `utils/`) |
| Training container | Done (`docker/training/Dockerfile.training`, CUDA 13.0.2, sm_120, smoke-gated build) |
| Compose service | Done (`docker-compose.training.yaml`, mounts `/mnt/ai-data/training:rw`, `shm_size: 16g`) |
| Makefile | `train-build` / `train-shell` / `train-smoke` at `Makefile:109-122`; no data targets |
| Dependencies | `datasets==5.0.0`, `transformers` (provides `tokenizers`), `numpy` (transitive), `torch==2.13.0` in core; `xxhash==3.5.0`, `zstandard==0.23.0`, `hydra-core==1.3.2` in the `training` group (`pyproject.toml:103-112`) |

The `uv-dependencies.md` doc already records the intent: `xxhash` for "shard and dataset
manifests", `hydra-core` for "recipe and configuration composition", `zstandard` for "shard
and artifact compression". No new dependency is required to implement D2; at most an explicit
`tokenizers` pin could be added to the training group for clarity.

## Deviations from the planning document

The repository overrides the document in three places:

1. **uv, not Poetry.** The document says "Poetry training group" and its D1 acceptance
   mentions `poetry install --with training`. The repository migrated to uv
   (`pyproject.toml:2`); D2 host-side installs are `uv sync --group training` and the
   container already builds the group. No Poetry work is in scope.
2. **D-numbering is not present in the repository.** Only D1 is referenced in code and docs
   (Makefile section header, compose comments, D9 `TRAIN_LD_PRELOAD` slot). D2 work should
   follow the existing labeling convention ("Phase 0 D2") used by D1's artifacts.
3. **Training pins are provisional.** `pyproject.toml:100-101` flags the training group as
   offline-resolved pending `uv lock` on the workstation. Under the pin-then-fix-then-update
   rule, D2 builds against the pins as they stand and does not bump anything.

## Dependency status: D1 done, D5 missing

D2 declares two dependencies:

- **D1 (training container): complete.** Commit `8cd7edd` landed the image, compose file,
  entrypoint, smoke test, Makefile targets, and dependency group. D2 code runs inside this
  container via `docker compose -f docker-compose.training.yaml run training ...`.
- **D5 (recipe system): not started.** No `training/recipes/schema.py`, no recipe YAML, and
  nothing consumes `hydra-core` yet. D2's coupling to D5 is narrow — the design says D2
  "consumes the recipe data block", but the D2 interface itself is a plain CLI
  (`--source`, `--tokenizer`, `--out`, `--max-tokens`, `--num-proc`), and the recipe `data`
  block only points at a shard directory that the D5 loader validates against the manifest.

Consequence: D2 can be implemented before D5 without rework, provided the manifest carries
the fields D5 validation later needs (tokenizer path and hash, dtype, EOS id, per-shard
hashes, totals). The CLI stays argument-driven; recipe integration is D5/D6 work.

There is a second soft dependency the document does not list under D2 but the CLI example
assumes: a tokenizer artifact at `/mnt/ai-data/training/tokenizers/<name>/` is produced by
D3, which is also unimplemented. D2 must therefore accept any Hugging Face tokenizer
loadable by `AutoTokenizer` (hub id or local path) so it is testable with an existing
tokenizer (for example SmolLM2 or Llama 3.2) before D3 lands. The manifest hashes whatever
`tokenizer.json` the tokenizer resolves to.

## Work breakdown

Ordered so each step is independently testable in the sandbox (no GPU required anywhere in
D2 except optional throughput measurement, which is host-side).

1. **Package scaffolding.** Create `training/` and `training/data/` as importable packages.
   Decide packaging posture: `pyproject.toml` has `package = false`, so imports rely on
   repository-root execution (the container already mounts the repository at `/workspace`
   equivalent); document this in the module headers.
2. **Shard format and writer (`training/data/shards.py`).**
   - Flat binary `.bin` shards of contiguous token ids; dtype `uint16` when vocab < 65,536
     else `uint32`; documents separated by the tokenizer EOS id; fixed shard size, default
     512M tokens.
   - Single sequential writer consuming a queue; shards are append-only and xxhash64-hashed
     as they close.
   - `manifest.json` writer: dataset name/path, revision, license string (required, may be
     "unknown"), tokenizer path + `tokenizer.json` xxhash, dtype, EOS id, per-shard token
     counts and content hashes, total tokens, total documents, document-length histogram
     (log-scale buckets), dropped-window count, creation timestamp, pipeline git SHA, exact
     CLI invocation.
3. **Reader (`ShardedTokenDataset` in the same module).**
   - mmap-backed (`numpy.memmap`) fixed-length window iteration; windows never cross shard
     boundaries (last partial window per shard dropped and accounted in the manifest).
   - Serializable cursor (shard index, offset) with save/restore, the hook D6 resume uses.
   - Loader refuses a shard directory whose content hashes mismatch the manifest or whose
     tokenizer hash mismatches the caller-supplied tokenizer.
4. **Tokenize-and-pack (`training/data/prepare.py`).**
   - HF source path: `datasets.load_dataset` with `num_proc` parallel tokenization
     (processes, not threads; default 30 of 32 cores, leaving two for writer and OS under
     the inference-down policy); streaming mode for larger-than-RAM sources feeding the same
     writer.
   - Local JSONL path: chunked reads, `--text-field` selector.
   - `--max-tokens` cap; deterministic output for identical inputs (fixed iteration order,
     no time-dependent content in shard bytes).
5. **CLI (`scripts/train/prepare_data.py`).** Thin argparse wrapper; output style per
   `.styles/PYTHON_OUTPUT_STYLE.md` (`Printer`, `[RUN]`/`[OK]` lines, exit codes 0/1/2).
6. **Makefile target.** `prepare-data` in the "Training" section, wrapping
   `docker compose -f docker-compose.training.yaml run --rm training python
   scripts/train/prepare_data.py $(ARGS)`. CPU-only, but still serialized with GPU work by
   the one-consumer policy since it saturates the CPU.
7. **Tests (`tests/training/`).** Sandbox-runnable with a tiny tokenizer fixture and
   synthetic JSONL: shard write → manifest totals equal sum of shard lengths; round-trip
   decode of a random window reproduces source text; determinism across two runs (identical
   hashes); uint16/uint32 dtype selection; cursor save/restore yields the identical next
   window; reader refusal on manifest or tokenizer hash mismatch; partial-window drop
   accounting.
8. **Documentation.** Extend `docs/sandbox/docker_training_overview.md` (or a sibling doc)
   with the shard format, manifest contract, and CLI usage; update the `training-runs`
   skill's pointers once the code exists.

## Design decisions carried from the document

These are compatible with the repository and adopted as-is:

- nanoGPT/nanotron-family flat token binaries, not Arrow — sequential mmap reads, zero
  deserialization.
- Manifest as the contract; downstream loaders enforce integrity (content hash) and
  correctness (tokenizer hash).
- `datasets.map` with `num_proc` processes for tokenization; single sequential writer.
- Nothing ever requires the corpus to fit in memory (streaming mode).
- Output outside the repository under `/mnt/ai-data/training/shards/<name>/`.

Note on `zstandard`: the dependency exists for "shard and artifact compression", but the D2
shard format is raw mmap-able binaries and cannot be zstd-compressed at rest without losing
zero-copy reads. Compression, if used at all, applies to non-shard artifacts; D2 should not
compress shards.

## Open questions

1. **Default shard size.** 512M tokens is about 1GB at uint16 — coarse for small test
   corpora. Proposal: keep the 512M default but expose `--shard-tokens` so tests and small
   datasets produce multiple shards (needed to exercise cursor and boundary logic).
2. **Manifest schema location.** A plain dict written by `shards.py` suffices for D2, but
   D5 will want a typed schema. Proposal: a dataclass in `training/data/shards.py` now,
   movable under the D5 schema module later without changing the JSON on disk.
3. **Queue implementation for the single writer.** `datasets.map` batches returning token
   arrays consumed in order by the main process is the simplest correct form; a
   multiprocessing queue is only needed if map-then-iterate leaves the writer starved.
   Measure on the 1B-token acceptance run before adding complexity.
4. **Sandbox versus host testing.** Unit tests (tiny corpora, tmp dirs) run in the sandbox.
   The 1B-token throughput acceptance run and any `/mnt/ai-data` path checks are host-side
   and follow the human-in-the-loop protocol from `AGENTS.md`.

## Acceptance checks

From the design, restated against this repository:

1. A small HF dataset and a local JSONL both produce shards whose manifest token totals
   equal the sum of shard lengths; a round-trip decode of a random window reproduces source
   text. (Sandbox-runnable as unit tests with a local tokenizer fixture; HF-source variant
   needs network or a cached dataset.)
2. Re-running with identical inputs is deterministic — identical shard and manifest hashes.
   (Sandbox-runnable.)
3. Throughput on the workstation is CPU-bound across the configured `num_proc` with no
   writer stall, demonstrated on at least a 1B-token run. (Host-side; human runs
   `make prepare-data ARGS=...` and reports htop/pidstat observations.)
4. `ShardedTokenDataset` cursor save/restore yields the identical next window.
   (Sandbox-runnable.)

## Risks

- **`datasets==5.0.0` API surface.** The training pins are provisional and the 5.x major of
  `datasets` postdates most reference pipelines; verify `load_dataset` streaming and
  `map(num_proc=...)` behavior against the pinned version early, in the container.
- **Determinism under `num_proc`.** Parallel map must not reorder documents between runs;
  `datasets.map` preserves order, but the JSONL chunked-read path must be written to match.
  The determinism test is the guard.
- **Writer throughput.** A single sequential writer must keep up with 30 tokenizer
  processes; at roughly 1GB per 512M-token shard this is disk-bandwidth-trivial, but the
  hash-on-close step should stream (xxhash incremental) rather than re-read closed shards.
- **D5 drift.** Implementing D2 first means the manifest fields are D2's guess at what D5
  validation needs; keeping the manifest a superset of the design's field list bounds this
  risk.
