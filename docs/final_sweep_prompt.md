# Final code sweep implementation prompt

This document is the complete instruction set for a Claude Code session running at
maximum effort ("ultracode") to execute the final code sweep of this repository. Give
this document to that session verbatim as its task. The session should read `AGENTS.md`
and the relevant `.styles/` guides before writing any file, use multi-agent workflows
for the broad audit passes, and adversarially verify its own findings before acting on
them.

## Table of contents

- [Context](#context)
- [Decisions already made](#decisions-already-made)
- [Work items](#work-items)
- [Known dragons](#known-dragons)
- [Execution order](#execution-order)
- [Definition of done](#definition-of-done)

## Context

The repository contains three tiers: containerized inference (llama-cpu, llama-gpu,
vllm-gpu, open-webui), a training stack (`training/`, `python -m training`), and an
interpretability stack (`interp/`, `python -m interp`). Shared code lives in `common/`.
The training and interp tiers are newly implemented and have passed unit tests in the
sandbox but have never been exercised on the target workstation (RTX 5090, Ryzen 9950X).
Hardware verification is performed manually by the human, phase by phase, after this
sweep completes; the sweep prepares the repository for that campaign.

Constraints from `AGENTS.md` apply throughout: the sandbox has no GPU, no host Docker
daemon, and no `/mnt/ai-data`. Any check that requires the workstation becomes a runbook
entry for the human, never a claim of verified behavior. Compose files and Dockerfiles
are the source of truth over documentation. Pin versions before changing behavior, and
never combine a version bump with a functional change.

Model policy, effective immediately: every feature deliverable is designed for the
Gemma 4 family first (`google/gemma-4-E2B`, `google/gemma-4-E4B`; HF `model_type`
`gemma4` / `gemma4_text`) with the Qwen3 family as fallback (`Qwen/Qwen3-1.7B` is the
registered pin). `training/families.py` is the single model-family registry. Exception:
the interp tier keeps Gemma-2-2B as its anchor for pretrained SAEs (Gemma Scope),
Neuronpedia presets, and circuit tracing, because that ecosystem does not exist for
Gemma 4 as of July 2026. Gemma 4 enters interp only through architecture-agnostic
hook-based paths (work item 6).

## Decisions already made

Do not relitigate these. They were settled with the repository owner.

- No `src/` layout. `[tool.uv] package = false` stays; `training/`, `interp/`, and
  `common/` remain import-path packages run via `python -m`.
- The loose modules at the top of `training/` are regrouped into exactly three
  subpackages (work item 1).
- The Python entry points in `scripts/eval/` fold into the training CLI, mirroring the
  earlier fold of `scripts/training/`. `scripts/interp/` is NOT touched: its isolated
  SAEBench venv split (`scripts/interp/saebench_env.sh`) is intentional.
- Gemma 4 interp support is hook-based paths only (harvest, probes, steering). Training
  our own Gemma 4 SAEs is out of scope for this sweep.
- Deterministic-resume acceptance on GPU is tolerance-based (loss-curve agreement within
  a stated tolerance over N steps). Bitwise equality remains a CPU/tiny-model test only.
- Skip entirely: watchdog thermal and RAM-floor fault-injection testing; verification of
  concurrency-rule enforcement between containers; any handling of HF model gating.
- Hardware test suite ambition: pytest markers plus one focused smoke test per feature,
  deselected by default (work item 3). Not a full acceptance-check mirror.

## Work items

### WI-1: regroup the training package

Move the 15 loose top-level modules in `training/` into three subpackages using
`git mv`, preserving history:

- `training/models/`: `model.py`, `families.py`, `peft_utils.py`
- `training/runtime/`: `loop.py`, `checkpoint.py`, `tracking.py`, `schedules.py`,
  `watchdog.py`, `offload.py`, `lock.py`, `affinity.py`
- `training/tune/`: `sft.py`, `sft_callbacks.py`, `distill.py`

`cli.py`, `__main__.py`, and `serving.py` stay at the top level. Update every import
across `training/`, `interp/`, `common/`, `scripts/`, and `tests/`, plus any module-path
references in the `Makefile`, `docker/` entrypoints, and `docs/` (mechanical path
updates only; do not rewrite prose). After the move, `python -m training --help` must
list every existing subcommand and the full test suite must pass.

### WI-2: fold scripts/eval into the training CLI

Migrate `scripts/eval/eval_ft.py`, `scripts/eval/serve_eval.py`, and
`scripts/eval/promote_scorecard.py` into `training/commands/` modules following the
existing `configure(sub)` auto-registration pattern. Before writing code, check for
overlap with `training/eval/` (the `eval-holdout` and `eval-lmeval` paths) and reuse
in-package logic instead of duplicating it. Preserve CLI flags and behavior exactly.
Update the Makefile targets `eval-ft`, `serve-eval`, and `promote-scorecard` to invoke
`python -m training <subcommand>` the same way neighboring targets do. Move
`scripts/eval/smoke_prompts.json` to a sensible in-package or data location and update
references. Update `tests/training/test_promote_scorecard.py` and any other test that
imports from `scripts/eval/`. Delete the migrated files; remove `scripts/eval/` if
empty.

### WI-3: hardware test markers and smoke suite

Register two pytest markers in `pyproject.toml` under `[tool.pytest.ini_options]`:
`gpu` (requires CUDA on the workstation) and `host` (requires host services or
`/mnt/ai-data`). Set `addopts` so the default run deselects both (`-m "not gpu and not
host"`), keeping sandbox and CI behavior unchanged. Create `tests/hardware/` with one
focused smoke test per feature area, each runnable in minutes, including at least:

- CUDA sanity: device present, bf16 matmul executes, reported compute capability is
  sm_120.
- flash-attn import succeeds; if it fails, assert the SDPA fallback path (WI-7) engages
  and stamps metadata.
- DeepSpeed CPUAdam extension loads and a single optimizer step completes.
- Tiny-model load through `training/models/` for each registered family (`gemma4`,
  `gemma4_text`, `qwen3`), skipping cleanly when weights are absent.
- Tolerance-based GPU resume check: short train run, checkpoint, resume, assert
  loss-curve agreement within the documented tolerance.
- Memory-fit calculator calibration spot check against one reference config.
- Hook-based activation harvest smoke on a small model, asserting shapes and manifest
  entries.

The verification runbook (WI-4) references `uv run pytest -m gpu tests/hardware/` as a
repeatable step. Follow `.styles/TEST_OUTPUT_STYLE.md`.

### WI-4: hardware verification runbook

Create `docs/verification/` containing the manual verification campaign the human will
execute on the workstation. Derive every command from the Makefile and the two CLIs;
include only features that exist in the repository. Files:

1. `README.md` — purpose, recording protocol (human runs the host command and pastes
   command plus output verbatim; a feature is verified only when its acceptance signal
   is observed), and phase order.
2. `step-zero-environment.md` — regenerate `uv.lock` on the workstation (the `interp`
   and `training` dependency groups are provisional pins; the `unsloth` pin is
   unverified against the pinned transformers), rebuild the training and interp images,
   run `make train-smoke` and `make interp-smoke`, and verify the CPUAdam build used
   znver5 flags. Step zero blocks all other phases.
3. `training-phase0.md` — D1 through D10 as implemented (container, prepare-data,
   tokenizer, memfit/calibrate, recipe validation, train loop plus tolerance-based
   resume, export and serve-eval round trip including the restore path, train bench,
   staging/prefetch, watchdogs verified passively only).
4. `training-phase1.md` — F1 through F8 (prepare-sft, PEFT recipes, train-sft,
   generate-teacher, train-distill and distill-cache, merge/export, eval-holdout,
   eval-lmeval, eval-ft, promote-scorecard, reference recipe cards).
5. `training-phase2.md` — P2-D1 through P2-D5 (prepare-corpus plus build-contamination,
   mixture, WSD schedules, replay buffer, optimizer offload).
6. `interp.md` — I1 workbench smoke, PS1 Neuronpedia harness (`np-*` targets), W1
   through W4 (steer-extract/steer-scorecard, probe capture/train/eval, logproxy, probe
   monitor), P4-D1 organisms, P4-D2 harvest and interp-gc with an explicit disk
   preflight step (harvests can reach roughly 1.4TB per layer), P4-D3 SAE training and
   evaluation via `scripts/interp/`.

Per-feature format: exact host command(s), expected signal (file paths, exit codes, log
lines — concrete, not vague), an empty "Observed" block, and a `Status: PENDING` line.
Follow `.styles/DOCUMENTATION_STYLE.md`.

### WI-5: model-policy conformance pass

Audit every recipe, default, and doc against the Gemma 4 primary / Qwen3 fallback
policy. For each reference training recipe in `training/recipes/reference/` that names a
Gemma 4 model, ensure a Qwen3 twin exists or add one (mirroring the existing
`*-gemma2.yaml` / `*-qwen3.yaml` convention used by the interp recipes). Confirm
`training/families.py` remains the only place family-specific behavior is encoded; hoist
any stray hardcoded model IDs into recipes or the registry. Document the vanilla-TRL
fallback for Unsloth (a recipe-level flag that disables Unsloth patching) since Unsloth
kernel support for Gemma 4 PLE is unverified.

### WI-6: Gemma 4 hook-based interp adapter

Add a per-family hook-point registry to the interp tier: a small adapter (target roughly
100–200 lines) that maps residual-stream and layer hook points by family name, using
plain torch forward hooks on HF modules, so `harvest`, `probe_capture`, and
`steer_extract` work on Gemma 4 without TransformerLens. Follow the registry style of
`training/families.py`. Add `*-gemma4.yaml` variants of the harvest and probe recipes.
Gemma-2-2B remains the anchor for SAE training, Neuronpedia import, and anything
touching Gemma Scope. Unit-test the adapter with a tiny stand-in model; the real-model
check is a `tests/hardware/` smoke plus a runbook entry.

### WI-7: resilience fallbacks in code

- Every model-load path that requests flash-attn must fall back to
  `attn_implementation="sdpa"` with a logged warning when flash-attn is unavailable, and
  must stamp the effective implementation into run metadata alongside the existing
  `precision_effective` pattern so benchmark provenance stays honest.
- Verify the training image build pins the flash-attn wheel exactly (no floating
  resolution) and pre-builds DeepSpeed CPUAdam with znver5 flags; document the fallback
  chain (bitsandbytes paged AdamW, then torch AdamW with offload disabled) and make it
  selectable in the recipe schema.

### WI-8: mypy configuration and clean run

Add a `[tool.mypy]` section to `pyproject.toml` (mypy is already a dev dependency but
has no configuration). Target `training/`, `interp/`, and `common/`. Use pragmatic
per-module ignores for untyped third-party libraries, but fix real findings in repo code
rather than silencing them. `uv run mypy` must exit clean.

### WI-9: documentation drift audit

`docs/README.md` is stale: it indexes neither `docs/training/` nor
`docs/interpretability/` and carries an outdated count and date. Fix the index. Then
sweep `docs/` against the compose files and Dockerfiles (the `audit-docs` pattern) and
fix documentation to match the code wherever they disagree.

### WI-10: pin audit

Audit `docker/` and all compose files for unpinned image tags, floating git refs, and
unpinned downloads. Fix by pinning to exact versions. Do not bump any version in this
work item.

### WI-11: coverage gap check

Run the test suite with coverage over `training/`, `interp/`, and `common/`. For any
module with meaningfully uncovered logic (not boilerplate), add targeted unit tests.
Report the before/after coverage numbers in the PR description.

### WI-12: benchmark schema conformance

The training bench output and the inference benchmark suite are meant to share one JSON
schema. Verify they emit identical schemas; add a schema-validation unit test that both
outputs must pass so future drift fails in CI rather than in analysis.

### WI-13: housekeeping

- Record the deliberate deferral in `interp/organisms/generate.py` (the
  `hidden_objective` organism) in the relevant design doc under `docs/interpretability/`
  so the code comment and the docs agree.
- Audit `env.example`, `env.training.example`, and `env.interp.example` against the
  variables the code and compose files actually read; add missing entries and delete
  dead ones.
- Audit every Makefile target: referenced files exist, commands match current CLI
  surfaces, help text is accurate.
- Delete any dead code found along the way, with a note in the commit message.

### WI-14: capstone review

After all other work items land, run a maximum-effort code review over `training/`,
`interp/`, and `common/` using a multi-agent workflow: independent finders across
correctness, error-handling, and efficiency dimensions, followed by adversarial
verification of each finding (a finding is acted on only if it survives a refutation
attempt). Fix confirmed findings. Do not restructure anything in this pass; it is a bug
hunt, not a redesign.

## Known dragons

Context the implementing session needs but must not "fix" beyond the prescribed scope:

- flash-attn wheel-tag drift and the DeepSpeed CPUAdam AVX-512 miscompile on Zen 5 are
  known build risks; WI-7 and step zero of the runbook are the mitigations.
- TRL/PEFT/Unsloth support for Gemma 4 PLE is unverified; the Qwen3 fallback and the
  vanilla-TRL flag (WI-5) are the mitigations. Do not attempt TransformerLens PLE
  support.
- CUDA kernels are nondeterministic across runs (atomics, cuDNN autotuning); this is why
  GPU resume checks are tolerance-based. Do not force
  `torch.use_deterministic_algorithms` in production paths.
- `serve-eval` swaps the model on the live llama-gpu service and must restore it; the
  runbook entry must verify the restore path explicitly, including after a failed swap.
- Activation harvests are disk-hungry; preflights, not cleanup, are the defense.
- The interp calibration recipes intentionally target Gemma 2 and Qwen3 rather than
  Gemma 4; this is the ecosystem anchor decision, not drift.

## Execution order

1. WI-1, then WI-2 (structure settles first so everything downstream references final
   paths).
2. WI-3, WI-5, WI-6, WI-7 (parallelizable once structure is stable).
3. WI-8 through WI-13 (audits; parallelizable).
4. WI-4 (runbook last among the content items, so commands reflect the final CLI
   surface).
5. WI-14 (capstone review over the finished state).

Commit each work item separately with clear messages. Run the full default test suite,
`ruff check`, and `mypy` before every commit.

## Definition of done

- Default `pytest` selection, `ruff check .`, and `uv run mypy` all pass in the sandbox.
- `python -m training --help` and `python -m interp --help` list every subcommand;
  `make help` targets resolve.
- No stale imports or path references to moved modules anywhere in the repository
  (verified by grep, not assumption).
- `docs/verification/` covers every implemented feature with concrete commands and
  expected signals, and nothing that does not exist.
- Nothing requiring the workstation is claimed as verified; those items exist only as
  PENDING runbook entries.
- All work is pushed and presented as a pull request whose description maps commits to
  work items and lists the coverage before/after numbers (WI-11).
