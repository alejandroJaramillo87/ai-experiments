# F6: Adapter merge and serve — implementation investigation

**Frozen record — 2026-07-14. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing F6 (adapter merge and serve) entails, reconciling the
scope document against the current state of the repository. Per the operating rules, the
repository is the source of truth; every point where the scope document disagrees with
shipped code is called out and the repository convention wins.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Scope-document drift](#scope-document-drift)
- [Gap analysis](#gap-analysis)
- [Design decisions](#design-decisions)
- [Gemma 4 considerations](#gemma-4-considerations)
- [Proposed module layout](#proposed-module-layout)
- [Work breakdown](#work-breakdown)
- [Acceptance criteria](#acceptance-criteria)
- [Resolved design decisions](#resolved-design-decisions)

## Scope summary

F6 takes a Phase 1 checkpoint (adapter-only or full), produces a served GGUF, and feeds
the D7 round trip. The scope document tags it "orchestrate": mostly wiring over D7's
export pipeline. Its four decisions:

1. Adapter-into-base merging stays PEFT `merge_and_unload` inside the D7 export path.
   F6 adds resolving an adapter-only checkpoint to its base model (recorded in the
   checkpoint `meta.json` by F3) and a preflight that the base revision at export time
   hash-matches the one trained against.
2. mergekit enters for multi-adapter and multi-model merges only
   (`scripts/export/merge_models.py`, `make merge CONFIG=...`), driven by mergekit YAML
   kept under `training/export/merges/`. Output is an HF directory that feeds the same
   GGUF pipeline. Phase 1 ships the tool plus one validated example: two LoRA adapters
   merged via TIES onto one base.
3. The export manifest gains adapter path and hash, base model and revision, and merge
   method. Nothing else about the D7 flow changes.
4. The export preflight verifies the pinned llama.cpp revision supports the model
   architecture (Gemma 4 requires a post-April-2026 pin).

## Current repository state

Most of F6 decision 1, 3, and 4 already shipped with D7 and F3. The remaining work is
narrower than the scope document suggests.

| Scope element | Repository state |
|---------------|------------------|
| Export entry point | Implemented: `make export-gguf CHECKPOINT=... ARGS="--quant Q8_0"` runs `python -m training export` (`training/cli.py` `cmd_export`) in the pinned export image. |
| Adapter merge via `merge_and_unload` | Implemented: `training/export/merge.py` `stage_model()` detects a PEFT adapter (`adapter_config.json` in the model dir or an `adapter/` sibling), merges it into `base_model_name_or_path` in bf16, and stages safetensors. Full checkpoints copy through with no adapter step. |
| Checkpoint-type detection | Partially implemented: detection is file-presence based (`find_adapter()`), not `meta.json`-based. F3/F5 checkpoints do stamp `meta.json` with `adapter_only`, `base_model`, `backend`, `recipe_hash`, and F5 `distill` provenance (`training/sft_callbacks.py` `write_checkpoint_meta()`), but export never reads those fields for resolution or cross-checking. |
| Base-revision hash preflight | Not implemented. `meta.json` records the base model id only — no hub revision, no weights or config hash. `HFModelConfig` in `training/recipes/schema.py` has a single `id` field, so a recipe cannot pin a revision either. The merge loads whatever `from_pretrained(base)` resolves to at export time. |
| mergekit | Absent entirely: no dependency pin, no `scripts/export/` directory, no `make merge` target, no `training/export/merges/`. |
| Export manifest | Implemented: `training/export/manifest.py` already records adapter path and SHA-256, `base_model`, `merge_method` (`"peft.merge_and_unload"`), the checkpoint `meta.json`, the llama.cpp pin, quant type, and the GGUF SHA-256. Missing only the base revision field (blocked on the preflight above) and a mergekit merge-method value. |
| llama.cpp pin check | Implemented: `training/export/convert.py` `check_architecture()` scans the pinned converter registrations and fails with a message already scoped "Gemma 4 primary, Qwen3 fallback". The pin (`training/export/llama_cpp_pin.txt`, `e3546c79...`) registers `Gemma4ForCausalLM` and `Gemma4ForConditionalGeneration` per the fixtures in `tests/training/test_export.py`. |
| Serve side of the round trip | Implemented: `scripts/eval/serve_eval.py` (`make serve-eval MODEL=...`) verifies the manifest `gguf.sha256`, swaps the GGUF into llama-gpu via `training/serving.py`, runs the smoke prompts, and restores prior serving state on every exit path. |
| Reproducibility bar | Implemented for the D7 path and asserted in `tests/training/test_export.py` / `test_staging.py`; not yet exercised through a real adapter checkpoint (needs a GPU-side F3 run). |

## Scope-document drift

- The scope document frames F6 as introducing adapter handling to `export_gguf.py`. In
  the repository, adapter merging landed with D7 (`training/export/merge.py`) and there
  is no `export_gguf.py`; the entry point is `python -m training export`. F6 is
  therefore additive to `training/export/`, not a new script.
- The scope document says the tool "detects checkpoint type from meta.json". The shipped
  detector is file-based and works for both TRL adapter checkpoints and full
  checkpoints; `meta.json` should become a cross-check, not a replacement (a mismatch
  between `adapter_only` and the files on disk is a corruption signal worth failing on).
- Reference models in the repository are Gemma 4 E2B recipes
  (`training/recipes/reference/sft-qlora-e2b.yaml`, `distill-e2b.yaml`), not the scope
  document's E4B SFT target. F6 acceptance runs use the repository's recipes.

## Gap analysis

The actual F6 work, in dependency order:

1. **Base identity capture at train time (F3 meta extension).** The revision-match
   preflight is impossible until checkpoints record what they trained against. Add to
   `write_checkpoint_meta()`: the resolved hub revision (commit hash from the HF cache
   snapshot the run loaded) and a content hash of the base `config.json`. Requires a
   `revision` field on `HFModelConfig` so recipes can pin explicitly; default stays
   "latest at load time, recorded at load time".
2. **Export-time base preflight.** Before merging, resolve the base the same way
   training did, compare revision and config hash against `meta.json`, and refuse on
   mismatch (override flag for deliberate re-basing, recorded in the manifest).
3. **meta.json cross-check in staging.** `stage_model()` gains an optional meta
   argument: `adapter_only` must agree with `find_adapter()`, and the adapter config
   `base_model_name_or_path` must agree with `meta.base_model`.
4. **Manifest fields.** `base_revision` and `base_config_sha256` in the adapter block;
   `merge_method` values extended for the mergekit path.
5. **mergekit deliverable (split to a follow-up feature; see resolved decisions).**
   New: pinned dependency in the export image, a `python -m training merge`
   subcommand, `make merge CONFIG=...`, example TIES config under
   `training/export/merges/`, output staged as an HF directory that feeds the existing
   convert/quantize path with a manifest recording the mergekit method and inputs.
6. **Round-trip acceptance wiring.** Host-side: F3 QLoRA checkpoint to Q8_0 GGUF to
   `make serve-eval` smoke pass with state restore, and a re-export hash-identity check
   through the adapter path.

## Design decisions

1. **F6 extends `training/export/`, it does not add a parallel path.** New logic lands
   in `merge.py` (meta cross-check), a new `preflight.py` (base resolution and revision
   match), and `manifest.py` (new fields). `cmd_export` stays the single entry point.
2. **`meta.json` is a cross-check, not the detector.** File-presence detection is kept
   (it also serves plain HF directories that have no meta); when `meta.json` exists the
   two must agree.
3. **Revision capture is recorded, pinning is optional.** Training records the resolved
   revision; recipes may pin one. Export refuses on mismatch unless `--allow-base-drift`
   is passed, in which case the manifest records both revisions.
4. **mergekit output feeds the same pipeline.** The merge tool produces an HF directory
   under the staging root; `python -m training export` consumes it like any full model.
   No GGUF-specific logic in the merge tool.
5. **Serving stays D7's.** No changes to `serve_eval.py` or `training/serving.py` are
   expected; F6 acceptance consumes them as-is.

## Gemma 4 considerations

- The pinned converter already registers the Gemma 4 architectures, and
  `check_architecture()` enforces this before conversion; F6 decision 4 is effectively
  done. Re-verify on any pin bump.
- Gemma 4 weights are gated on the Hugging Face hub. An adapter-only export must load
  the base model inside the export container, which therefore needs `HF_TOKEN` (from
  the untracked training env file) and either network access or a mounted HF cache. The
  base-revision preflight makes a cache-first policy attractive: resolving from the
  same local snapshot training used avoids both network variance and a token-in-export
  requirement. See open questions.
- The E-models use per-layer embeddings. `merge_and_unload` operates on the PEFT target
  modules only, and target-module resolution for gemma4/gemma4_text is already owned by
  `training/families.py`; no PLE-specific merge handling is expected, but the container
  smoke test should confirm a merged E2B adapter round-trips through conversion.
- Qwen3 remains the fallback family; the converter registration check and
  `training/families.py` already cover it, so no F6-specific Qwen3 work is expected.

## Proposed module layout

```
training/export/preflight.py        base resolution, revision/config-hash match
training/export/merge.py            meta.json cross-check added to stage_model()
training/export/manifest.py         base_revision, base_config_sha256, merge methods
training/export/merges/ties-example.yaml
scripts/export/merge_models.py      mergekit wrapper (or training CLI subcommand)
training/sft_callbacks.py           meta.json gains base_revision, base_config_sha256
training/recipes/schema.py          HFModelConfig.revision (optional)
tests/training/test_export.py       preflight and manifest coverage
tests/training/test_merge_models.py mergekit config validation, dry-run paths
Makefile                            make merge CONFIG=...
```

## Work breakdown

1. F3 meta extension and `HFModelConfig.revision` (small, touches F3 tests).
2. Export preflight and staging cross-check with unit tests (pure-logic, sandbox-testable).
3. Manifest field additions with reproducibility tests.
4. Host-side acceptance: QLoRA export, serve-eval smoke, full-checkpoint export,
   re-export hash identity. All GPU/serving steps are host-executed and reported back
   per the sandbox rules.
5. Follow-up feature (not F6): mergekit pin, image addition, `training merge`
   subcommand, TIES example, tests.

## Acceptance criteria

From the scope document, restated against the repository:

- An adapter-only QLoRA checkpoint from F3 exports to Q8_0 GGUF via `make export-gguf`,
  serves on llama-gpu via `make serve-eval`, answers the smoke prompts, and serving
  state is restored.
- A full-fine-tune checkpoint exports through the same command with no adapter step,
  with `meta.json` agreement verified.
- Deferred to the mergekit follow-up: the two-adapter TIES example producing a servable
  GGUF whose manifest records the merge method and input adapters.
- Re-export of the same checkpoint is hash-identical (`gguf.sha256`) through the
  adapter path.
- New: export refuses when the base at export time does not match the revision recorded
  at train time, with a clear message and a documented override.

## Resolved design decisions

Owner decisions recorded 2026-07-14:

1. **Base identity mechanism: HF revision plus config hash.** `meta.json` records the
   resolved hub commit and a SHA-256 of the base `config.json`. Cheap at save time;
   catches revision drift. Full base-weights hashing was rejected as a multi-GB read
   on every checkpoint save.
2. **Base loading at export: cache-first, no network.** The export container mounts
   the training HF cache and resolves the same local snapshot training used. No
   `HF_TOKEN` in the export container; deterministic merges.
3. **mergekit is split into a follow-up.** F6 lands the merge-and-serve round trip
   (the exit-checklist item) first. The TIES example needs two trained adapters that
   do not exist yet; the mergekit deliverable ships as a separate feature.
4. **Merge entry point: `python -m training merge`.** When mergekit is built, it lands
   as a subcommand of the unified training CLI, not a standalone script under
   `scripts/export/`.
