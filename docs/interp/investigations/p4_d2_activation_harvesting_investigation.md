# P4-D2: Activation harvesting — implementation investigation

**Frozen record — 2026-07-15. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing the P4-D2 deliverable (cached residual-stream
activations on NVMe, wrapping the SAELens v6 activation-caching runner under the repo
manifest and batch-job conventions) entails, given the current state of this repository.
The repository is the source of truth; where the design document disagrees with the code,
the code wins and the drift is recorded here.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [The Gemma 4 capture path](#the-gemma-4-capture-path)
- [Sizing and disk envelope](#sizing-and-disk-envelope)
- [Gap analysis](#gap-analysis)
- [Proposed layout](#proposed-layout)
- [Work breakdown](#work-breakdown)
- [Acceptance criteria](#acceptance-criteria)
- [Resolved design decisions](#resolved-design-decisions)

## Scope summary

P4-D2 produces cached activations for the SAE chain (P4-D3 training, P4-D4
import/autointerp). Per the design document it is "configuration and thin glue around
SAELens's cache runner, not a storage engine":

1. Wrap SAELens v6's activation-caching runner in `scripts/interp/harvest.py`; no
   bespoke shard format.
2. Capture config declaring model checkpoint (path or HF id plus revision), hook points
   by name, corpus (HF dataset id or local JSONL, content-hashed), context length
   (default 1024), total tokens, dtype bf16, seed.
3. Defaults sized to the envelope: 300M tokens, one layer per capture; multi-layer
   requires an explicit acknowledgement flag. A disk preflight computes projected bytes
   and refuses to start if free space minus the projection breaches the watchdog floor.
4. Batch-job discipline: D9 lock, inference services down, D10 watchdog attached,
   resumable at shard granularity, manifest finalized only on completion.
5. Manifest under `activations/<name>/manifest.json`: git SHA, model hash, tokenizer
   hash, hook names, corpus identity and hash, dtype, context length, token count, seed,
   SAELens version, measured tokens per second, wall clock. Downstream tools refuse a
   capture whose manifest is missing or whose hashes do not match their config.
6. Retention: captures are regenerable cache; a `--validated` flag on downstream SAE
   runs marks a capture deletable and `make interp-gc` deletes marked captures
   oldest-first, keeping manifests.

## Current repository state

Everything P4-D2 depends on exists; the deliverable itself is greenfield. No reference
to `CacheActivationsRunner` exists anywhere in the repository yet.

| Dependency | Where it lives today |
|------------|----------------------|
| Manifest convention | `interp/manifest.py`: `git_sha()`, `file_xxhash()` (xxh3_64, streamed), `text_sha256()`, atomic `write_manifest()`, `load_manifest()`, and `require_match()` as the refuse-on-mismatch primitive. |
| Closest capture analog | `interp/probes/capture.py`: nnsight residual-stream capture writing `shard_NNNNN.npz` shards plus an `activation_capture` manifest (`model_id`, `model_revision`, `model_hash`, `layers`, `dataset_sha256`, `tokenizer_fingerprint`, `dtype`, `count`, `shards`, `git_sha`, `created`). |
| D9 lock | `training/lock.py`: `run_lock(name)` context manager on `/mnt/ai-data/training/.train.lock` (override via `TRAIN_LOCK_PATH`); stale locks reported, never auto-removed. |
| Inference-down check | `scripts/interp/preflight.sh`: refuses to start while `llama-gpu`, `vllm-gpu`, or `comfyui` are up; already invoked by every GPU interp Make target. |
| D10 watchdog | `training/watchdog.py`: `Watchdog(config, run_dir, disk_path=...)` background sampler (thermal, power, RAM floor, disk floor with `disk_free_floor_gb` default 100GB); torch-free, reusable outside training loops. |
| Config loading | `training/recipes/loader.py` `load_recipe()`: OmegaConf structured merge over the typed `Recipe` schema in `training/recipes/schema.py`, dotlist overrides. `hydra.main` is deliberately not used anywhere. |
| Subject checkpoint | P4-D1 sealed organism manifests under `/mnt/ai-data/interp/organisms/<name>/manifest.json` record `checkpoint` and `gguf` paths; checkpoints under `/mnt/ai-data/training/runs/...`. |
| Output root | `make interp-dirs` already creates `/mnt/ai-data/interp/activations`. |
| Workbench | `docker-compose.interp.yaml` + `docker/interp/Dockerfile.interp`; pins `sae-lens==6.46.0` (released 2026-07-13), `transformer-lens==3.5.1`, `nnsight==0.7.0` via the `interp` group in `pyproject.toml`/`uv.lock`. |
| Test conventions | `tests/interp/`: CPU-only, injectable capture functions (`fake_capture` in `conftest.py`), `importorskip` for heavy deps. |

Base-model policy, already resolved in the P4-D1 investigation and encoded in the
recipes: organisms and all Phase 4 subjects build on `google/gemma-4-E2B`
(`training/recipes/organisms/*.yaml`), with the Qwen3 family
(`Qwen/Qwen3-0.6B`–`4B`) as fallback. Hooked stages from P4-D2 onward are written
against HF module paths, not TransformerLens aliases.

## Design-document drift

The design document defers to the repository on all of the following:

1. **Base model.** The document sizes P4-D2 around Gemma-2-2B ("resid_post at layers 6,
   13, and 19", "4.6KB per token"). The repository's subject is Gemma 4 E2B. Hook
   depths, per-token bytes, and hook naming all change (see next sections). The
   Gemma-2-2B references that remain in the workbench (`scripts/interp/smoke_test.py`,
   `env.interp.example`, `Dockerfile.interp`) belong to the Gemma Scope calibration
   path, not the harvest subject.
2. **Hook naming.** The document's `resid_post` wording assumes TransformerLens.
   TransformerLens `HookedTransformer` has no Gemma 4 support as of 2026-07 (Gemma 3 is
   its newest Gemma), so hook points must be HF named-module paths
   (`model.layers.{l}`), consistent with the P4-D1 resolution.
3. **Config system.** The document says "Hydra, `interp/configs/harvest/`". The
   repository standard is the D5 recipe loader (typed OmegaConf schema, no
   `hydra.main`), and no `interp/configs/` tree exists. P4-D2 should follow the loader
   convention rather than introduce a parallel config system.
4. **Watchdog attachment.** The document assumes the training-style watchdog; the
   harvest loop has no loss or grad norm, so only the background sampler signals
   (thermal, power, RAM floor, disk floor) apply. `training/watchdog.py` already
   supports this usage.

## The Gemma 4 capture path

SAELens v6 (`sae-lens==6.46.0`) supports two backends in its cache runner:

- **TransformerLens** hooks (`blocks.{l}.hook_resid_post`) — unavailable for Gemma 4;
  works natively for Qwen3 (0.6B–14B supported in `supported_models.py`).
- **Direct HuggingFace** via `model_class_name = "AutoModelForCausalLM"`, where
  `hook_name` is an HF named-module path such as `model.layers.{l}` (output of the
  decoder layer is the post-block residual stream). SAELens v6.44.2 fixed layer
  extraction for exactly this path, and the SAELens maintainers (Decode Research)
  released Gemma 4 E2B residual-stream SAEs on Neuronpedia in May 2026, so the path is
  exercised in practice. A TransformerBridge route also exists for TL-unsupported
  models.

Gemma 4 architecture facts relevant to capture (from the `transformers` Gemma4 config):

- Text config defaults: `hidden_size=2304`, `num_hidden_layers=30`,
  `vocab_size=262144`. The exact `config.json` of `google/gemma-4-E2B` must be
  confirmed at implementation before sizing buffers (secondary sources disagree on
  layer count).
- Per-layer embeddings (PLE): `hidden_size_per_layer_input=256` injects a small
  per-token embedding into each decoder layer. The residual stream proper remains
  `hidden_size`-wide; the P4-D1 gate (SAELens hook-point smoke test on Gemma 4,
  including the per-layer-embedding hook-point check) verifies the hook captures the
  2304-dim residual, not a concatenated per-layer input. That gate remains a
  precondition before committing GPU-days.
- Attention interleaves 5 sliding-window layers per 1 global layer
  (`sliding_window_pattern=6`); irrelevant to residual capture but relevant to choosing
  representative depths.
- Harvest uses the pre-trained (pt) checkpoint lineage with raw text plus BOS — no chat
  template — so Gemma 4 thinking-mode tokens (`<think>` machinery) never enter the
  corpus path. Organism checkpoints are fine-tuned; the harvest tokenizes the corpus
  identically regardless.

Default hook depths at 25/50/75 percent of a 30-layer stack: layers 7, 15, and 22
(recompute if the shipped config differs). Qwen3-1.7B fallback: `hidden_size=2048`,
28 layers, native TL support — the friction-free calibration subject, consistent with
the P4-D3 plan to calibrate the pipeline against a Qwen3 model.

## Sizing and disk envelope

Bytes per token per hook at bf16 = `hidden_size * 2`.

| Subject | d_model | Bytes/token | 300M-token capture |
|---------|---------|-------------|--------------------|
| Gemma 4 E2B | 2304 | 4608 | ~1.38TB |
| Qwen3-1.7B | 2048 | 4096 | ~1.23TB |

Model weights (~10GB bf16 for E2B including PLE tables) plus batched forward passes fit
the RTX 5090's 32GB comfortably. The disk preflight computes
`total_tokens * hidden_size * dtype_bytes * num_hooks`, reads the shipped model config
for `hidden_size` rather than hard-coding it, and refuses when
`free - projected < disk_free_floor_gb`.

## Gap analysis

New code, in build order:

1. **Config block.** A `HarvestConfig` in the recipe schema (mirroring how
   `OrganismConfig` was added): model id/path plus revision, hook layer list, corpus
   (HF dataset id or local JSONL plus content hash), context length (1024), total
   tokens (300M), dtype (bf16), seed, `acknowledge_multi_layer` flag, output root.
2. **Disk preflight.** Projected-bytes computation plus free-space check against the
   watchdog floor; standalone function so tests cover it without a GPU. `shutil.disk_usage`
   usage modeled at `training/watchdog.py`.
3. **Harvest runner.** `interp/harvest.py` module plus `scripts/interp/harvest.py` CLI:
   resolve subject from an organism manifest or explicit model id, run preflight,
   acquire `run_lock`, start `Watchdog`, invoke SAELens `CacheActivationsRunner` with
   the AutoModelForCausalLM backend and HF hook paths, one runner invocation per hook.
   Runner function injectable for CPU tests, per the `capture_fn` seam in
   `interp/probes/capture.py`.
4. **Resume.** SAELens's cache state is the shard-granularity resume mechanism; the
   wrapper reruns skipping completed shards and writes the manifest only on completion
   (absence of `manifest.json` marks an incomplete capture).
5. **Manifest.** Extend the `activation_capture` schema from `interp/probes/capture.py`
   with corpus identity and hash, hook names, token count, seed, SAELens version,
   measured tokens per second, and wall clock; enforce downstream with
   `require_match`.
6. **Retention.** `mark_validated()` helper (writes a `validated` flag into the capture
   manifest, called by P4-D3 with `--validated`) and `make interp-gc` deleting marked
   captures oldest-first, preserving manifests.
7. **Make targets.** `make harvest ARGS=...` following the `probe-capture` pattern
   (preflight then `docker compose -f docker-compose.interp.yaml run --rm interp ...`),
   plus `make interp-gc`.
8. **Tests.** `tests/interp/test_harvest.py`: config validation, preflight refusal
   math, manifest round-trip and mismatch refusal, resume-skip logic, gc ordering —
   all CPU-only with an injected fake runner.
9. **Smoke-test extension.** Add the Gemma 4 hook-point check (2304-dim residual at
   `model.layers.{l}`, PLE not captured) to the workbench gate before any full harvest.

## Proposed layout

```
interp/harvest.py                      # preflight, runner wrapper, manifest, gc logic
scripts/interp/harvest.py             # CLI entry point
scripts/interp/interp_gc.py           # or a subcommand of harvest.py
training/recipes/schema.py            # HarvestConfig added
training/recipes/interp/harvest-*.yaml  # smoke + full capture recipes
tests/interp/test_harvest.py
/mnt/ai-data/interp/activations/<name>/   # shards + manifest.json (exists via interp-dirs)
```

## Work breakdown

1. Schema + recipe yamls + loader wiring (CPU, testable).
2. Preflight + manifest + gc helpers with tests (CPU, testable).
3. Runner wrapper + CLI + Make targets, fake-runner tests (CPU, testable).
4. Workbench smoke: Gemma 4 hook-point check, then the CI-sized smoke capture (host,
   GPU; human runs and reports).
5. Full 300M-token single-layer capture on the false-fact organism overnight, with a
   mid-run kill/rerun to verify resume (host, GPU; human runs and reports).

## Acceptance criteria

From the design document, adjusted for the repository's model policy:

- A smoke capture (small subject, 2M tokens, one hook) completes in minutes in the
  workbench and validates against its manifest.
- A full 300M-token single-layer capture on the false-fact organism (Gemma 4 E2B)
  completes overnight unattended — lock, watchdog, and resume verified by killing and
  rerunning mid-harvest — with measured throughput recorded in the manifest.
- The disk preflight refuses a capture projected to breach the floor, with a clear
  message.

## Resolved design decisions

Resolved with the maintainer, 2026-07-15:

1. **Smoke-capture subject: Gemma 4 by default, Qwen3 as fallback.** The CI-sized
   smoke capture (2M tokens, one hook) runs on Gemma 4 E2B, exercising the same
   AutoModelForCausalLM hook path as the full harvest. Qwen3 (native TransformerLens
   support) remains the fallback and calibration subject. GPT-2 Small is dropped: it
   exercises a code path the real harvest never uses.
2. **Harvest corpus: `monology/pile-uncopyrighted`.** P4-D3 wants quality numbers
   standardized enough to compare against the field, and comparability with published
   SAE baselines outweighs distribution match to Gemma 4's pretraining mix. The corpus
   is a config field with a recorded content hash, so a later swap to a modern web
   corpus is a recipe change, not a code change.
3. **Reference capture: single hook at 50 percent depth.** One layer keeps the
   overnight run at ~1.38TB, above the disk floor on the current NVMe, and captures
   are regenerable cache by design — re-harvest another depth when an investigation
   needs it. Multi-layer capture stays behind the explicit acknowledgement flag.
4. **Config home: extend the training recipe schema.** `HarvestConfig` joins
   `training/recipes/schema.py` (the `OrganismConfig` precedent) with recipe yamls
   under `training/recipes/interp/`. The decoupling argument for a separate interp
   schema is cosmetic: the harvest already depends on `training/lock.py` and
   `training/watchdog.py` by design, and the repository has twice chosen one config
   system over parallel ones.
