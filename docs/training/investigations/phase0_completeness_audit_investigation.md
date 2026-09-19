# Phase 0 completeness audit

**Frozen record — 2026-09-19. Drift against the current repository is expected and correct;
this document is never updated.**

Audit of the ten Phase 0 deliverables (D1 through D10, the training substrate) against five
sources at once: the 2026-07-08 product design that scoped them, the living documentation
under `docs/training/`, the code under `training/` and `docker/training/`, the test suite,
and the verification runbooks. The design predates the rust harness, the interp tier, the
observability stack, and the `schemas/` contracts, so a sixth column records what each
deliverable now has to satisfy for those consumers. The question the audit answers is not
"does the tree match the July text" but "is each foundation piece implemented, tested,
documented, hardware-verified, and still covered by a live runbook entry".

## Table of contents

- [Method](#method)
- [Matrix](#matrix)
- [Per deliverable](#per-deliverable)
- [Decisions that superseded the design](#decisions-that-superseded-the-design)
- [Findings](#findings)
- [Actions taken on 2026-09-19](#actions-taken-on-2026-09-19)

## Method

Four questions per deliverable, each answered from one source:

| Question | Source | Passing answer |
|----------|--------|----------------|
| Implemented | `training/`, `docker/training/`, `make/training.mk` | the module and its make target exist |
| Tested | `tests/training/`, `tests/hardware/` | a unit test asserts each acceptance behaviour the design named; a hardware test where the bar is a measurement |
| Documented | `docs/training/README.md` and the document it indexes | a living document, not only the frozen investigation |
| Verified | `docs/verification/` | a `Status: VERIFIED` entry with pasted output, or a `PENDING` entry in a live runbook that someone can still run |

The verification column distinguishes the archive from the live set. The three phase runbooks
were frozen on 2026-09-18 when the training set was reorganised by capability; a `PENDING`
entry in [../../verification/archive/training-phase0.md](../../verification/archive/training-phase0.md)
is history, and only an entry in a live runbook counts as coverage.

## Matrix

Impl: code exists. Unit: CPU test of the design's acceptance behaviour. Doc: living document.
HW: verified on the workstation. Live: a pending entry in a live runbook covers it.

| Deliverable | Impl | Unit | Doc | HW | Live | Verdict |
|-------------|------|------|-----|----|------|---------|
| D1 container | yes | none for the smoke test itself | `docs/sandbox/docker_training_overview.md` | step zero, closed; the torch 2.13 image never built | FT.12 | re-verify through FT.12 |
| D2 data pipeline | yes | yes | `data_pipeline.md` | P0.1 VERIFIED | cited by FT.0 | done, one caveat |
| D3 tokenizer | yes | yes | `tokenizer_training.md` | P0.2 VERIFIED | cited by FT.0 | done, one caveat |
| D4 memory fit | yes | yes | `memory_fit.md` | P0.3 VERIFIED | FT.1, FT.14, FT.16 | done for scratch recipes |
| D5 recipes | yes | yes | none until this audit | P0.4 PENDING | FT.13 | undocumented |
| D6 loop, checkpoints, tracking | yes | yes | `tracking_checkpointing.md` | P0.5 VERIFIED (resume); P0.6 PENDING (end to end) | FT.3 covers the SFT loop only | reference loop unverified |
| D7 export and serve | redesigned | yes | `train_serve_roundtrip.md` | P0.7 to P0.9 PENDING and obsolete | FT.4 | superseded; FT.4 is the successor |
| D8 FP8 and benchmark | no | negative test only | none | none | none | deferred by decision |
| D9 heterogeneous compute | yes | yes, two pieces untested | `heterogeneous_compute.md` | P0.10 PENDING | none | unverified and uncovered |
| D10 watchdogs | yes | yes | `operational_watchdogs.md` | P0.11 PENDING | none in the training tier | unverified and uncovered |

## Per deliverable

### D1 container

Code: `docker/training/Dockerfile.training`, `docker/training/entrypoint.sh`,
`docker-compose.training.yaml`, `training/commands/smoke_test.py`. Targets: `train-build`,
`train-shell`, `train-smoke`. The smoke test has six steps (the design listed seven; the
torchao step left with the torchao pin). No unit test drives `smoke_test.py`; the same CUDA
facts are asserted independently by `tests/hardware/test_gpu.py` in the `hw_smoke` tier.

Verification history: step zero (archived, closed 2026-09-18) built the image last on
2026-08-01. The torch pin moved to 2.13.0 on 2026-08-15 and the image was not rebuilt until
the FT.0 attempt on 2026-09-19 failed inside the flash-attn compile. The cause was the build
environment resolving torch 2.14.0 from PyPI instead of the locked 2.13.0; fixed in
`pyproject.toml` with `match-runtime = true` on the `extra-build-dependencies` entries. The
ubuntu 24.04 gcc rejects `-march=znver5`, so the image builds with `-march=native`.

Later tiers: a change to any dependency group re-opens the lockfile and rebuilds every image
the group feeds (training, interp, and the agents image for the `sandbox` group); the rule is
stated in `docs/verification/README.md` and its record is the archived step-zero runbook.

### D2 data pipeline

Code: `training/data/prepare.py`, `training/data/shards.py`. Targets: `prepare-data`, and the
Phase 2 successors `prepare-corpus` and `build-contamination`. Tests:
`tests/training/test_shards.py` (determinism, corruption, cursor save and restore, tokenizer
hash mismatch), `tests/training/test_prepare.py` (manifest round trip, `--max-tokens`,
overwrite refusal). Verified: P0.1 wrote the `verify-small` shard set on 2026-08-01, with the
KB-52 caveat that the process exits nonzero from a streaming-thread teardown crash after every
artifact is flushed.

Later tiers: the shard format crosses no tier boundary. The SFT preparation path does:
`training/data/sft.py::render_segments` is compared byte for byte against the Rust renderer by
`rust-harness/harness/tests/template_contract.rs`, through `schemas/chat_template.json`.

### D3 tokenizer

Code: `training/data/tokenizer.py`. Target: `train-tokenizer`. Tests:
`tests/training/test_tokenizer.py` (fixed special-token ids, reserved block, round trip,
reproducible artifact hash, `--force` refusal, evaluation metrics). Verified: P0.2 on
2026-08-01 with one evaluation domain instead of the two the design asked for; the artifact
contract was the point of that entry. The trained artifact,
`/mnt/ai-data/training/tokenizers/verify-49k`, and the corpus it consumed,
`/mnt/ai-data/training/datasets/mix/fineweb-sample.jsonl`, remain on the workstation and are
the offline inputs the from-scratch runbook builds on.

Later tiers: none. The harness reads `tokenizer.json` beside a checkpoint and interp reads
its own capture manifests.

### D4 memory fit

Code: `training/calculator/components.py`, `solve.py`, `calibrate.py`. Targets: `memfit`,
`train-calibrate`. Tests: `tests/training/test_calculator.py` pins the closed forms for the
reference recipes and hand-computed configurations; `tests/hardware/test_gpu.py` asserts the
reserved estimate never under-predicts a real run. Verified: P0.3 on 2026-08-01 after four
rebuild cycles (KB-53 through KB-56) and the overhead constant raised to 2.5 GB. The QLoRA
and E4B recipes are pending in FT.1, FT.14 and FT.16.

Later tiers: Gemma 4 dimensions come from `schemas/gemma4_facts.json`, which
`tests/training/test_calculator.py` and the propagation tests guard.

### D5 recipe system

Code: `training/recipes/schema.py`, `validate.py`, `loader.py`; six reference recipes.
Targets: `train-validate`, `train-preflight`. Tests: `tests/training/test_recipes.py` covers
typed load of every reference recipe, misspelled and mistyped keys, dotlist overrides,
bit-identical snapshot reload, and one test per cross-field invariant;
`tests/training/test_recipe_pins.py` enforces the Gemma 4 revision pins. Verified: P0.4 was
never run; FT.13 covers `train-validate` and `train-preflight` on the SFT recipe.

Documentation: none. The design record
[d5_recipe_system_investigation.md](d5_recipe_system_investigation.md) was the only
description, and `docs/training/README.md` listed it as the record with no living pair.
This audit adds [../recipe_system.md](../recipe_system.md).

Later tiers: the loader, schema and validator are a cross-tier API. `interp/commands/harvest.py`,
`interp/commands/sae_train.py` and `interp/commands/organism.py` import them, interp's tests
import the `Recipe` dataclass, and the interp and organism recipes live under
`training/recipes/`. Every Gemma 4 recipe must pin `model.hf.revision` to the commit in
`schemas/gemma4_facts.json`.

### D6 reference loop, checkpointing and tracking

Code: `training/runtime/loop.py`, `checkpoint.py`, `tracking.py`. Target: `train`. Tests:
`tests/training/test_checkpoint.py` (a kill between the temporary write and the rename, a
kill during republish, retention of the last N plus every Mth, the `latest` target never
deleted), `tests/training/test_loop.py` (deterministic resume equals the straight run, with
and without offload; tensorboard and offline wandb sinks), `tests/training/test_tracking.py`
(a failing sink is evicted and the survivors keep receiving). Verified: P0.5, the
tolerance-based resume through `tests/hardware/test_gpu.py`, on 2026-08-08. P0.6, the
reference loop run end to end through `make train`, was never run, and FT.3 exercises the
TRL fine-tuning loop instead.

Later tiers: the tracker's first `metrics.jsonl` record is built by
`common.reports.report_header` and must pass `validate_report`; the header carries `tier`,
`model_id`, `recipe_hash` and `schedule`, which `schemas/run_metrics.json` names as the
fields the dashboard shows for a training run. The sharded `model/` directory a D6
checkpoint writes is consumed by the exporter only: the harness cannot load a shard index,
and `interp/probes/capture.py::derive_model_hash` returns `unknown` for it. Only the merged
export carries identity.

### D7 export and serve

The design's round trip (checkpoint to GGUF through a pinned llama.cpp, swapped into
`llama-gpu`, evaluated, restored) was replaced. `make export-hf` writes the merged
Hugging Face directory (`config.json` plus exactly one `model.safetensors`) that the rust
harness serves and interp captures from, with a manifest whose `model_hash` is the sha256 of
the weights file (`schemas/hf_export.manifest.json`). `training/commands/serve_eval.py` is a
retired name; the evaluation is `make eval-ft` against the harness. Tests:
`tests/training/test_export.py`, `test_cli_handlers_export.py` (a failed stage leaves no
partial output), `test_export_harness.py` (three writers of one digest). Verified: none. The
archived P0.7 through P0.9 describe the GGUF path and cannot run; FT.4 is the live successor.

### D8 FP8 enablement and benchmark harness

Not built. No `train_bench`, no `training/benchmark/`, no FP8 code path;
`precision.dtype` admits `bf16` only, and the `fp8_backend` recipe field was retired on
2026-09-19 (`scripts/docs/check_retired_names.py`). The deferral is recorded in
`docs/training/training-scale-reference.md`: FP8 lands as one measured change on the
continual-pretraining or from-scratch rung, not in the substrate. When a benchmark harness is
built it must emit through `common.reports.report_header` and join the emitter table in
`docs/observability/report_schema.md`; nothing else in the tree constrains its payload.

### D9 heterogeneous compute

Code: `training/data/loader.py` (prefetch), `training/data/staging.py`,
`training/runtime/affinity.py` and `scripts/training/pin_run.sh`, `training/runtime/lock.py`,
and the `TRAIN_LD_PRELOAD` slot in `docker/training/entrypoint.sh`. No dedicated target; the
pieces are recipe knobs on `make train`. Tests: `tests/training/test_loader.py` (exact
resume, prefetcher measures wait), `test_staging.py` (fallback over `max_gb`, corruption is
an error), `test_affinity.py` (disjoint masks, overlap rejected), `test_lock.py` (second
acquire refused, stale lock reported); `tests/hardware/test_ssd.py` asserts the 5 percent
`data_wait_ms` bar and `tests/hardware/test_cpu.py` asserts pinning matches placement, both
in the `hw_full` tier. Untested: `pin_run.sh` and the `TRAIN_LD_PRELOAD` branch. The
preload slot is empty by design; the design's intended payload (AOCL-LibMem) is not installed
and the investigation parked it until profiled. Verified: none. P0.10 was never run, and no
live entry covered any piece until this audit.

Later tiers: `training/runtime/lock.py` is the global GPU lock. `interp/harvest.py`,
`interp/sae/train.py` and `interp/steering/scorecard.py` import it. It is enforced in process
only: no make target checks it, `scripts/interp/preflight.sh` gates on the inference services
instead, and the harness never reads it.

### D10 operational watchdogs

Code: `training/runtime/watchdog.py`, wired into the reference loop and mirrored for TRL by
`training/tune/sft_callbacks.py`. The out-of-process sidecar the design named was deferred by
the investigation. Tests: `tests/training/test_watchdog.py` (synthetic NaN, spike, gradient
norm, temperature with sustain, RAM floor and half floor, disk floor blocking checkpoint
writes, the emergency checkpoint exempt from retention, the loop saving it on abort) and
`test_watchdog_branches.py`. Verified: none. P0.11, passive observation, was never run; the
observability runbook's OBS-V2 and OBS-V11 cover the sampler fields and `alerts.jsonl` from
the observability side, but no training entry asserts that a healthy run raises nothing or
that an exploding run aborts and leaves a resumable emergency checkpoint.

Later tiers: the watchdog is torch-free and stdlib-only and reaches `common/obs` only through
the D6 tracker, which is how its samples land in `metrics.jsonl` and the dashboard.

## Decisions that superseded the design

Each is recorded elsewhere; this list is the index.

| Design statement | What the tree does | Where recorded |
|------------------|--------------------|----------------|
| Poetry group for pure-Python deps, pip for CUDA packages | uv manages everything from one `uv.lock`; the image syncs the `training` group | `docs/os/python/uv-dependencies.md` |
| flash-attn from a prebuilt cu13 wheel | built from source against the locked torch; no wheel exists for torch 2.13 and CUDA 13 | `pyproject.toml` `extra-build-dependencies` |
| DeepSpeed ops with `-march=znver5`, native as fallback | native only; the image gcc has no znver5 | `docker/training/Dockerfile.training` |
| GGUF export, llama-gpu swap, serve-eval | merged HF directory served by the rust harness; `eval-ft` evaluates | `docs/training/train_serve_roundtrip.md` |
| Checkpoints under `/mnt/ai-data/training/checkpoints/<run>/` | under `runs/<run>/checkpoints/` | `d6_tracking_checkpointing_investigation.md` |
| torchao float8 primary, Transformer Engine experimental | no FP8 path; bf16 only; FP8 deferred to a later rung | `docs/training/training-scale-reference.md` |
| Watchdog sidecar daemon | deferred; in-process sampler thread only | `d10_operational_watchdogs_investigation.md` |
| Lock at `.lock` | `.train.lock`; shared with interp | `docs/training/heterogeneous_compute.md` |
| Qwen3 fallback family | every recipe pinned to Gemma 4 | `docs/verification/training/fine-tuning-loop.md` |
| Phase runbooks in build order | one runbook per rung, by capability | `docs/verification/README.md` |

## Findings

Numbered for reference. None is acted on in this audit beyond the documentation it names.

1. The D6 sharded checkpoint carries no identity for the other tiers: interp capture of a
   `model/` directory hashes as `unknown` and loses the harness identity guard. A constraint
   to document, not a defect; the merged export is the artifact with identity.
2. `tests/training/test_recipe_pins.py` globs `training/recipes/reference/` and
   `training/recipes/organisms/` and not `training/recipes/interp/`, so an interp recipe
   naming an unpinned Gemma 4 revision passes the gate.
3. The D9 lock is process-level. `make` interp targets and the harness never check it, so a
   harness serve can start under a held lock.
4. `scripts/training/pin_run.sh`, the `TRAIN_LD_PRELOAD` branch of the entrypoint, and
   `training/commands/smoke_test.py` have no tests. The `scripts/setup/` and
   `scripts/agents/` trees show the subprocess-test pattern that would cover the first two.
5. D8 is deferred by decision, and until this audit the deferral was visible only in the scale
   reference, not from the Phase 0 index.
6. Two verified entries carry open caveats: P0.1 exits nonzero (KB-52) and P0.2 evaluated one
   domain where the design asked for two.
7. `docs/README.md` described the fine-tuning runbook as "FT.0 through FT.11" after FT.12
   through FT.16 were added.
8. `docs/sandbox/docker_training_overview.md` called the smoke test six-step in one place and
   eight-step in two others; the code runs six.

## Actions taken on 2026-09-19

- Added [../recipe_system.md](../recipe_system.md), the D5 living document, and listed D1, D5
  and D8 in the Phase 0 table of `docs/training/README.md`.
- Opened `docs/verification/training/from-scratch-loop.md` with nine entries: the reference
  loop end to end and its CLI resume (P0.6), staging, prefetch, affinity and the lock (P0.10
  and the rest of the D9 acceptance), and the watchdog's passive and abort paths (P0.11 and
  the D10 integration check). The archived entries stay frozen and are cited by id.
- Corrected the two stale descriptions in findings 7 and 8.
