# D3 tokenizer training: implementation investigation

Comparison of the Phase 0 D3 design (roadmap document, "D3: Tokenizer training") against
the current repository state, with a work breakdown, acceptance checks, deviations, and
open questions. Per operating rules, the repository is the source of truth; where the
roadmap document disagrees with the repository, this investigation records the repository
state and treats the document as advisory.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Deviations from the document](#deviations-from-the-document)
- [Design constraints imposed by existing code](#design-constraints-imposed-by-existing-code)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Open questions](#open-questions)

## Design summary

D3 trains and evaluates byte-level BPE tokenizers for the Phase 3 from-scratch path
only. Fine-tuning and continual pre-training always reuse the base model's tokenizer;
the tool exists solely for from-scratch pretraining and must say so in its documentation
(a mismatched tokenizer silently destroys a fine-tune, and cross-tokenizer distillation
breaks logit KL). The document specifies:

1. Engine: Hugging Face `tokenizers`, byte-level BPE (GPT-2/Llama-3 style: byte-level
   pre-tokenization, no unknown token, digits split individually). SentencePiece stays
   installed but is not the default; one engine, one artifact format.
2. Artifact: a directory `/mnt/ai-data/training/tokenizers/<name>/` containing
   `tokenizer.json` and `manifest.json` (training corpus and sampled byte count, vocab
   size, special tokens, evaluation metrics, git SHA, timestamp, and the tokenizer hash
   that D2 shard manifests reference).
3. Special tokens fixed at training time: `<|bos|>`, `<|eos|>`, `<|pad|>`, plus a
   reserved block of 16 unused ids for later chat/control tokens, so Phase 1 templates
   never resize embeddings.
4. Evaluation is part of training, not optional: on held-out samples from at least two
   domains (web text, code by default), report fertility (tokens per whitespace word),
   compression (bytes per token), round-trip losslessness on 1,000 random documents,
   and the same metrics for one or two reference tokenizers (SmolLM2, Llama-3.2).
   Results go in the manifest and print as a table.
5. Interfaces: a thin CLI (`--corpus`, `--vocab-size`, `--sample-bytes`, `--out`) over
   importable `train()` and `evaluate()` functions in `training/data/tokenizer.py`.

Dependencies: D1 (training container) and D2 (evaluation corpora handling). Both are
merged and working in the repository.

## Current repository state

| Area | State |
|------|-------|
| D1 container | Done: `docker-compose.training.yaml`, `docker/training/Dockerfile.training`, Makefile targets `train-build`, `train-shell`, `train-smoke` |
| D2 data pipeline | Done: `training/data/prepare.py` (tokenize-and-pack), `training/data/shards.py` (shard writer, manifest, `ShardedTokenDataset`), `scripts/training/prepare_data.py` CLI, `make prepare-data`, tests in `tests/training/` |
| D5 recipe system | Done: `training/recipes/` (schema, loader, validate, reference recipes), `scripts/training/validate_recipe.py`, `make train-validate` |
| Tokenizer hash convention | Established in `training/data/prepare.py`: `LoadedTokenizer.content_hash` is xxhash64 of the serialized tokenizer (`Tokenizer.to_str()`), recorded by D2 manifests as `tokenizer_hash` and enforced by `training/recipes/validate.py` (`_check_tokenizer_matches_shards`) |
| Tokenizer loading | `load_tokenizer()` in `training/data/prepare.py` accepts a bare `tokenizer.json`, a directory containing one, or a hub id; EOS resolution tries `<|eos|>`, `<|endoftext|>`, `</s>` in order |
| Artifact path convention | Reference recipes already point at D3-style artifact directories: `tiny-e2e.yaml` uses `/mnt/ai-data/training/tokenizers/tiny-e2e`, the 150m/1b recipes use `/mnt/ai-data/training/tokenizers/smollm2` |
| Dependencies | `tokenizers` 0.22.2 is in `uv.lock` (transitive via `transformers`); `xxhash`, `hydra-core` pinned in the `training` group; `sentencepiece` 0.2.2 pinned in core |
| Test fixture precedent | `tests/training/conftest.py` already builds a tiny byte-level BPE tokenizer with the `tokenizers` library |
| Script conventions | `scripts/training/prepare_data.py` establishes the pattern: `Printer` class per `.styles/PYTHON_OUTPUT_STYLE.md`, `sys.path` bootstrap, argparse, exit codes 0/1/2 |
| D3 itself | Not started: no `training/data/tokenizer.py`, no `scripts/training/train_tokenizer.py`, no Make target, no tests |

## Deviations from the document

1. **Script location.** The document says `scripts/train/train_tokenizer.py`; the
   repository uses `scripts/training/` (the `scripts/train` rename drift was repaired
   in commit `0e48212`). D3 lands at `scripts/training/train_tokenizer.py`.
2. **Package manager.** The document references Poetry groups; the repository migrated
   to uv (`pyproject.toml` header). No Poetry work is part of D3.
3. **Tokenizer hash definition.** The document says "xxhash of tokenizer.json". The
   repository truth is xxhash64 of `Tokenizer.to_str()` (the in-memory serialization),
   not of the file bytes — `_hash_tokenizer_str` in `training/data/prepare.py`. These
   can differ (file formatting, key order). D3's `manifest.json` must record the
   `to_str()`-based hash so that D2 manifests and D5 `validate()` match it; recording a
   file-bytes hash would break the hash-match chain.
4. **Corpus interface.** The document's example passes a JSONL glob
   (`--corpus /mnt/ai-data/training/datasets/mix/*.jsonl`). The repository's D2 already
   handles HF datasets, streaming, and JSONL through one code path; D3 should reuse
   `iter_jsonl` and the source-iteration helpers in `training/data/prepare.py` rather
   than introducing a second ingestion path. Minimum viable surface is JSONL globs
   (matching the document); HF dataset sources are a possible extension via the same
   helpers.

## Design constraints imposed by existing code

These are hard requirements discovered from the repository, beyond what the document
states:

1. **The artifact must load through `load_tokenizer()`.** Reference recipes point
   `tokenizer.path` at the artifact directory, and D5 `validate()` calls
   `load_tokenizer(recipe.tokenizer.path)` on it. The directory-with-`tokenizer.json`
   layout satisfies the loader as written; no loader change is needed.
2. **`<|eos|>` must resolve as the EOS token.** `FALLBACK_EOS_TOKENS` tries `<|eos|>`
   first, so the document's special-token choice works with the existing loader without
   passing `--eos-token`. The trained tokenizer must actually contain `<|eos|>` as a
   token (added as a special token at training time).
3. **Special-token id placement must be deterministic and test-asserted.** The document
   requires unit tests covering special-token ids and the reserved block. Placing
   `<|bos|>`, `<|eos|>`, `<|pad|>`, and `<|reserved_0|>`..`<|reserved_15|>` at fixed
   ids (0-18, before the byte alphabet and merges) makes the ids independent of corpus
   content. The 19 special ids count against `--vocab-size`.
4. **Reproducibility must be byte-exact.** Acceptance requires identical
   `tokenizer.json` hashes across two runs on the same corpus. The `tokenizers`
   BPE trainer is deterministic for a fixed input sequence; the sampling step
   (`--sample-bytes`) therefore needs a fixed, ordered traversal (sorted glob
   expansion, sequential read up to the byte budget) rather than randomized sampling,
   or a seeded sampler whose seed is recorded in the manifest.
5. **Overwrite refusal.** The CLI must refuse an existing artifact directory without
   `--force`, matching the manifest-integrity posture of `shards.py`.
6. **Sandbox limits.** Training a real tokenizer on `/mnt/ai-data` corpora and the
   reference-tokenizer comparison (network fetch of SmolLM2 / Llama-3.2) are host-side
   actions. Unit tests must run in the sandbox on synthetic fixtures with no network,
   following `tests/training/conftest.py`.

## Work breakdown

1. **`training/data/tokenizer.py`** — the logic module.
   - `train(corpus, vocab_size, sample_bytes, seed_order) -> Tokenizer`: byte-level BPE
     via `tokenizers` (`models.BPE`, `pre_tokenizers.ByteLevel`, `decoders.ByteLevel`,
     digit-splitting pre-tokenization per the GPT-2/Llama-3 style), special tokens and
     reserved block pinned at fixed ids, deterministic corpus traversal.
   - `evaluate(tokenizer, domains, references) -> dict`: fertility, compression,
     round-trip losslessness over held-out samples per domain; the same metrics for
     each reference tokenizer. Pure function over in-memory text so tests can feed
     synthetic domains.
   - `write_artifact(tokenizer, out_dir, manifest_fields, force)`: writes
     `tokenizer.json` plus `manifest.json` (corpus spec, sampled bytes, vocab size,
     special tokens, metrics, git SHA, timestamp, xxhash64 of `to_str()`), refusing an
     existing directory without `force`.
2. **`scripts/training/train_tokenizer.py`** — thin CLI following
   `prepare_data.py` conventions (`Printer`, exit codes 0/1/2, `--json` if consistent
   with the existing scripts): `--corpus` (repeatable/glob), `--vocab-size`,
   `--sample-bytes`, `--out`, `--force`, `--eval-corpus DOMAIN=PATH` (repeatable),
   `--reference` (repeatable, default SmolLM2 and Llama-3.2), `--no-references` for
   offline runs. Prints the metrics table per `.styles/PYTHON_OUTPUT_STYLE.md`.
3. **Makefile target** — `train-tokenizer` wrapping the script through the training
   container, same shape as `prepare-data`:
   `make train-tokenizer ARGS="--corpus ... --out ..."`.
4. **Tests** — `tests/training/test_tokenizer.py` on synthetic fixtures:
   special-token ids and reserved block, reproducibility (two trainings, identical
   serialized hash), round-trip losslessness, evaluation metric sanity (fertility and
   compression positive, reference comparison shape), overwrite refusal without
   `--force`, artifact loads through `load_tokenizer()` and `<|eos|>` resolves.
5. **Documentation** — `docs/training/tokenizer_training.md` following
   `data_pipeline.md` structure, stating explicitly that the tool is for the
   from-scratch path only.
6. **Integration check** — a trained artifact used as `--tokenizer` for
   `prepare_data.py` produces shards whose `tokenizer_hash` equals the artifact
   manifest hash, and D5 `validate()` accepts the pair (covered by a unit test using
   the tiny fixture pipeline).

No changes to `pyproject.toml` are expected: `tokenizers` is already resolved in
`uv.lock` via `transformers`, and `xxhash` is pinned in the training group. Whether to
promote `tokenizers` to an explicit pin is an open question below.

## Acceptance checks

From the document, mapped to how each is verified here:

| Check | Verification |
|-------|--------------|
| Reproducible training on a fixed seed corpus (identical `tokenizer.json` hash across two runs) | Unit test on the synthetic corpus; host-side rerun on a real corpus sample as final acceptance |
| Round-trip losslessness on the evaluation sample | Unit test (synthetic) plus host-side run over 1,000 random documents from two domains |
| Metrics table reports fertility and compression for the new and reference tokenizers on both domains | Host-side run with network access to fetch SmolLM2 and Llama-3.2 references; recorded output committed to the run record |
| Unit tests cover special-token ids, reserved block, overwrite refusal without `--force` | `tests/training/test_tokenizer.py`, runs in the sandbox |

Host-side steps (human executes, output reported back per operating rules):
`make train-build` if the image changed, then a `make train-tokenizer` invocation on a
real corpus sample, run twice for the hash-equality check.

## Risks

1. **Digit splitting interacts with byte-level pre-tokenization.** Getting
   "digits split individually" requires an explicit pre-tokenizer step (e.g. a
   `Split` on digit patterns ahead of `ByteLevel`); the exact composition affects the
   learned merges and must be locked before the reproducibility acceptance run.
2. **Reference tokenizer fetch.** Llama-3.2 is gated on the Hugging Face hub; the
   comparison step must degrade cleanly (skip with `[WARN]`) when a reference is
   unavailable, and `--no-references` must keep the tool usable offline.
3. **Hash chain fragility.** Any post-training mutation of the tokenizer (adding
   special tokens after training, re-serialization by a different library version)
   changes `to_str()` and breaks the D2/D5 hash match. The artifact written at training
   time is final; documentation must say so.
4. **`tokenizers` version skew.** The serialized form (and therefore the hash) can
   change across `tokenizers` releases. The manifest records the library version so a
   hash mismatch after an upgrade is diagnosable.

## Open questions

1. **Corpus sources.** Is JSONL-glob input sufficient for now (matching the document's
   interface), or should `--corpus` also accept HF dataset ids through the D2 helpers
   in the first version?
2. **Reference tokenizer set.** Default to SmolLM2-135M (ungated) plus Llama-3.2, or
   drop Llama-3.2 for an ungated second reference to keep the default run
   credential-free?
3. **Explicit pin.** Promote `tokenizers` to an explicit entry in the training group in
   `pyproject.toml`, or continue relying on the transitive pin via `transformers`?
4. **BOS policy.** D2's packing separates documents with EOS only. Should the trained
   tokenizer's post-processor attach BOS by default (Llama style), or leave BOS unused
   until Phase 3 wires it into the model, keeping D2 behavior unchanged?
