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
The repository also carries a ComfyUI art service (`docker/comfyui/`, the `comfy-*` and
`art-mode`/`llm-mode` targets) and agent-sandbox tooling (`docker-compose.agents.yaml`,
`docker/agents/`); WI-10 and WI-13 cover them, no other work item touches them.
The training and interp tiers are newly implemented and have passed unit tests in the
sandbox but have never been exercised on the target workstation (RTX 5090, Ryzen 9950X).
Hardware verification is performed manually by the human, phase by phase, after this
sweep completes; the sweep prepares the repository for that campaign.

Constraints from `AGENTS.md` apply throughout: the sandbox has no GPU, no host Docker
daemon, and no `/mnt/ai-data`. Install the torch-free test extent with
`uv sync --group sandbox --group dev` (the `sandbox` dependency group exists for exactly
this; torch-gated tests skip cleanly). Any check that requires the workstation becomes a
runbook entry for the human, never a claim of verified behavior. Compose files and Dockerfiles
are the source of truth over documentation. Pin versions before changing behavior, and
never combine a version bump with a functional change.

Model policy, effective immediately: every feature deliverable is designed for the
Gemma 4 family first (`google/gemma-4-E2B`, `google/gemma-4-E4B`; HF `model_type`
`gemma4` / `gemma4_text`) with the Qwen3 family as fallback (`Qwen/Qwen3-1.7B` is the
registered pin). `training/families.py` is the single model-family registry. Exception:
the interp tier keeps Gemma-2-2B as its anchor for pretrained SAEs (Gemma Scope),
Neuronpedia presets, and circuit tracing, because that ecosystem does not exist for
Gemma 4 as of July 2026. Gemma 4 already runs on the interp tier's hook-based paths
(harvest recipes pin `google/gemma-4-E2B`; probes run under nnsight and steering under
repeng, both on plain HF checkpoints); work item 6 generalizes that support, it does not
introduce it.

## Decisions already made

Do not relitigate these. They were settled with the repository owner.

- No `src/` layout. `[tool.uv] package = false` stays; `training/`, `interp/`, and
  `common/` remain import-path packages run via `python -m`.
- The loose modules at the top of `training/` are regrouped into exactly three
  subpackages (work item 1).
- The Python entry points in `scripts/eval/` fold into the training CLI, mirroring the
  earlier folds of `scripts/training/` and of the SAE scripts (`train_sae.py`/
  `eval_sae.py` are already in the interp CLI). The remaining shell shims in
  `scripts/interp/` (`preflight.sh`, `np_*.sh`, `saebench_env.sh`) are NOT touched: the
  isolated SAEBench venv split is intentional.
- Per-deliverable investigation docs are frozen design history: they move, they are
  never edited (work item 9).
- No test-gap hunting: work item 11 is a final regression run only.
- Gemma 4 interp support is hook-based paths only (harvest, probes, steering). Training
  our own Gemma 4 SAEs is out of scope for this sweep.
- Deterministic-resume acceptance on GPU is tolerance-based (loss-curve agreement within
  a stated tolerance over N steps). Bitwise equality remains a CPU/tiny-model test only.
- Skip entirely: watchdog thermal and RAM-floor fault-injection testing; verification of
  concurrency-rule enforcement between containers; any handling of HF model gating.
- Hardware test suite shape: two labeled tiers (`hw_smoke`, `hw_full`) with resource
  markers (`gpu`, `cpu`, `ram`, `ssd`), all deselected by default (work item 3).

## Work items

### WI-1: regroup the training package

Move the 14 loose top-level modules in `training/` into three subpackages using
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
existing `configure(sub)` pattern: each module exposes `configure(sub)` and is
explicitly imported and added to the registration tuple in
`training/cli.py:build_parser()` (registration is explicit, not automatic). Before
writing code, check for overlap with `training/eval/` (the `eval-holdout` and
`eval-lmeval` paths) and reuse in-package logic instead of duplicating it. Preserve CLI
flags and behavior exactly. Update the Makefile targets `eval-ft`, `serve-eval`, and
`promote-scorecard` to invoke `python3 -m training <subcommand>` directly on the host,
following the `generate-teacher` target — these are host-run, serving-dependent tools;
do not wrap them in `docker compose run`. Move `scripts/eval/smoke_prompts.json` to a
sensible in-package or data location and update references. Update every test that
imports from or references `scripts/eval/` paths: `tests/training/
test_promote_scorecard.py`, and `tests/training/test_serving.py`, which reads
`scripts/eval/serve_eval.py` by filesystem path and will fail after the move. Sweep the
Makefile and docs for remaining `scripts/eval/` path references. Delete the migrated
files; remove `scripts/eval/` if empty.

### WI-3: hardware test suites (smoke and full)

Implement all hardware-dependent tests as two labeled suites under `tests/hardware/`.
Register six pytest markers in `pyproject.toml` under `[tool.pytest.ini_options]`: two
tier markers — `hw_smoke` and `hw_full` — and four resource markers — `gpu`, `cpu`,
`ram`, `ssd`. Every hardware test carries exactly one tier marker and at least one
resource marker. Set `addopts` so the default run deselects both tiers
(`-m "not hw_smoke and not hw_full"`), keeping sandbox behavior unchanged.

Tier contract:

- `hw_smoke` completes in under roughly five minutes total, never downloads models, and
  writes only to temporary directories. It is the safe-to-run-reflexively gate the
  human executes before every working session.
- `hw_full` may allocate near-capacity VRAM, RAM, and disk and may load real models. It
  is run per verification phase and after any environment change (driver, image, or
  lockfile).

Required coverage by resource dimension:

| Resource | hw_smoke | hw_full |
|----------|----------|---------|
| `gpu` (GPU/VRAM) | CUDA device present; compute capability sm_120; bf16 matmul executes; flash-attn imports or the SDPA fallback (WI-7) engages and stamps metadata; DeepSpeed CPUAdam extension loads | near-capacity VRAM allocation with graceful OOM handling; memory-fit calculator estimate versus measured VRAM; tolerance-based resume check (short train run, checkpoint, resume, loss-curve agreement within the documented tolerance); tiny-model load through the family registry for each registered family (`gemma4`, `gemma4_text`, `qwen3`), skipping cleanly when weights are absent |
| `cpu` | znver5/AVX-512 feature flags visible; a single CPUAdam optimizer step completes | affinity pinning verified against actual core placement; CPUAdam step-time bound from the design docs |
| `ram` | huge pages available; small pinned-buffer allocation succeeds | replay buffer RAM-resident behavior at realistic size; host-RAM headroom check for ZeRO-2 optimizer offload |
| `ssd` | `/mnt/ai-data` mounted with the expected subdirectories; tmpfs staging writable | shard-staging write-throughput sanity; checkpoint write/restore round trip; harvest disk preflight; `data_wait_ms` under the 5% bar from the D9 design |

The hook-based activation harvest smoke (asserting shapes and manifest entries,
exercising the WI-6 registry) belongs to `hw_full` under `gpu`. The verification
runbook (WI-4) references `uv run pytest -m hw_smoke tests/hardware/` as the
session-start gate and `uv run pytest -m hw_full tests/hardware/` as the per-phase
step. The tests assert behavior WI-6 and WI-7 introduce, so WI-3 lands after them.
Follow `.styles/TEST_OUTPUT_STYLE.md`.

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
3. `training-phase0.md` — the Phase 0 deliverables as implemented (container,
   prepare-data, tokenizer, memfit/calibrate, recipe validation, train loop plus
   tolerance-based resume, export and serve-eval round trip including the restore path,
   staging/prefetch, watchdogs verified passively only). D8 (the benchmark harness) is
   NOT implemented — record it as out of scope in the phase file rather than inventing
   a check for it.
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
any stray hardcoded model IDs into recipes or the registry. The vanilla-TRL fallback for
Unsloth already exists — `sft.backend: trl` is the default and `unsloth` is opt-in
(`training/recipes/schema.py`); do not design a new flag, just document in the Gemma 4
reference SFT recipes that the `trl` backend is the fallback when Unsloth kernel support
for Gemma 4 PLE proves broken. Acceptance: every reference recipe naming a Gemma 4 model
has a Qwen3 twin, and a repo grep for `google/gemma`/`Qwen/` shows model IDs only in
`training/families.py`, recipes, and tests.

### WI-6: per-family hook-point registry for interp

The hook-based paths already avoid TransformerLens and already run Gemma 4:
`interp/harvest.py` uses HF named-module paths (`model.layers.{n}`) by resolved P4-D1
decision, `training/recipes/interp/harvest-smoke.yaml` pins `google/gemma-4-E2B`, probe
capture runs the subject under nnsight, and steering extraction uses repeng on the bf16
HF checkpoint. What is missing is generality: the flat `model.layers.{n}` layout is
hardcoded in more than one place. Hoist the per-family hook-path logic that already
exists in `interp/harvest.py` (hook naming, the text-config unwrap, the hook-dimension
check) into a registry module in the style of `training/families.py`, and refactor
`harvest`, `probe_capture`, and `steer_extract` to resolve hook points through it by
family name. Gemma-2-2B remains the anchor for SAE training, Neuronpedia import, and
anything touching Gemma Scope; the `harvest-calibration-gemma2.yaml`/`-qwen3.yaml` pair
is intentionally non-Gemma-4 and must not be "fixed". Unit-test the registry with a
tiny stand-in model; the real-model check is a `tests/hardware/` smoke plus a runbook
entry.

### WI-7: resilience fallbacks in code

- `training/model.py:attention_implementation()` already falls back to sdpa, but
  silently. Add a logged warning there, extend the same fallback to every other load
  path that requests flash-attn, and stamp the effective attention implementation into
  run and eval metadata alongside the existing `precision` field that eval records
  carry (`training/eval/holdout.py`; the precision-confound guard in
  `training/eval/runner.py` shows the pattern) so provenance stays honest.
- Verify by inspection of `docker/training/` and the lockfile that the flash-attn wheel
  is pinned exactly (no floating resolution) and DeepSpeed CPUAdam is pre-built with
  znver5 flags — the executed-build check belongs to WI-4 step zero, not to a sandbox
  claim. For the optimizer fallback chain, add the paged 8-bit AdamW option to the
  recipe schema's optimizer choices and an explicit fallback list (CPUAdam, then paged
  AdamW, then torch AdamW with offload disabled) that the offload builder walks when
  the DeepSpeed extension is unavailable, logging which optimizer actually ran.

### WI-8: mypy configuration and clean run

Add a `[tool.mypy]` section to `pyproject.toml` (mypy is already a dev dependency but
has no configuration). Target `training/`, `interp/`, and `common/`. Use pragmatic
per-module ignores for untyped third-party libraries, but fix real findings in repo code
rather than silencing them. `uv run mypy` must exit clean.

### WI-9: documentation reorganization and drift audit

Split the docs into frozen design history and living documentation, then clean only the
living set.

Frozen design history: the per-deliverable investigation docs — everything matching the
`d3`–`d10`, `f1`–`f8`, `p2_d*`, `i1`, `p4_d*`, and `ps*` naming patterns under
`docs/training/` and `docs/interpretability/`. These record what was decided and why at
implementation time; drift against the current repo is expected and correct in them.
Move them with `git mv`, content untouched, into `docs/training/investigations/` and
`docs/interpretability/investigations/`. Fix inbound links from other files
mechanically. `docs/unified_cli_investigation.md` is also frozen design history; leave
it in place and index it as such. Never edit the body of a frozen doc.

Living documentation: the synthesis and system docs (for example `data_pipeline.md`,
`memory_fit.md`, `distillation.md`, `tracking_checkpointing.md`, `platform.md`,
`writeback.md`, the scorecards, and everything else under `docs/`). These stay in place
and get the drift cleanup: fix the stale `docs/README.md` index (it indexes neither
`docs/training/` nor `docs/interpretability/` and carries an outdated count and date),
add a "Design history (frozen)" heading that lists the investigations folders and
states they are not maintained against the current repo, then sweep the living docs
against the compose files and Dockerfiles (the `audit-docs` pattern) and fix them to
match the code wherever they disagree. Flag any file whose frozen-versus-living status
is ambiguous rather than guessing.

Acceptance: `docs/README.md` indexes every subdirectory of `docs/` including the new
investigations folders; `git log --follow` shows the moved files as pure renames with
no content diff; the drift sweep's findings and fixes are listed in the commit message.

### WI-10: pin audit

Audit `docker/` and all compose files for unpinned image tags, floating git refs, and
unpinned downloads. Fix by pinning to exact versions. Do not bump any version in this
work item. Acceptance: a grep sweep for `:latest`, un-shaed `git clone`/`checkout`, and
tag-less `FROM` lines comes back clean, or every remaining hit has a written
justification in the commit message.

### WI-11: final regression run

The coverage gap-closing work already happened before this sweep (the test-coverage
pull requests on main), and `pyproject.toml` already carries a `[tool.coverage.run]`
omit list for the host-side-by-design modules. Do NOT hunt for test gaps in this sweep.
After every other work item has landed (including WI-14), run the full default test
suite once more with coverage over `training/`, `interp/`, and `common/` and confirm
zero test regressions relative to the commit the sweep started from (same or higher
pass count, no new failures, no newly skipped tests other than by design). Report the
final per-package coverage in the PR description for information. The only change this
work item makes: add a `fail_under` gate to `[tool.coverage.report]`, set one point
below the measured total, so future coverage regressions fail loudly.

### WI-12: report schema conformance

The D8 benchmark harness was never implemented, so there is no "train bench" emitter;
the D8-style JSON report convention (a header with fields such as `schema_version`,
`report`, `timestamp`, `git_sha`, config hash) exists only as a pattern followed by the
implemented report emitters — at minimum `interp/organisms/verify.py`,
`interp/sae/evaluate.py`, and `training/eval/scorecard.py`; enumerate the full set by
grep. Extract that convention into one documented schema (a JSON Schema file or a
shared validator in `common/`), add a unit test asserting every existing emitter's
output validates against it, and note in the schema doc that the future D8 harness and
any JSON-emitting inference benchmark must adopt it. Do not build the training bench in
this sweep.

### WI-13: housekeeping

- The `hidden_objective` organism deferral is already recorded in
  `docs/interpretability/p4_d1_model_organisms_investigation.md` (resolved decision 3;
  after WI-9 the file lives under `docs/interpretability/investigations/`). Add the
  experiments-log entry that decision promises, if it is still missing — as a new file,
  not an edit to the frozen investigation doc.
- Audit `env.example`, `env.training.example`, and `env.interp.example` against the
  variables the code and compose files actually read; add missing entries and delete
  dead ones.
- Audit every Makefile target: referenced files exist, commands match current CLI
  surfaces, help text is accurate.
- Delete any dead code found along the way, with a note in the commit message.

Acceptance: each env example variable maps to a documented reader (grep evidence in the
commit message), and `make help` output matches the audited target list.

### WI-14: capstone review

After all other work items land, run a maximum-effort code review over `training/`,
`interp/`, and `common/` using a multi-agent workflow: independent finders across
correctness, error-handling, and efficiency dimensions, followed by adversarial
verification of each finding (a finding is acted on only if it survives a refutation
attempt). Fix confirmed findings. Do not restructure anything in this pass; it is a bug
hunt, not a redesign.

### WI-15: deterministic quality gate

Add a `make quality` target that runs, in order, failing on the first error:

1. `uv run ruff format --check .` — formatting (check mode; the existing `make format`
   stays as the write-mode convenience)
2. `uv run ruff check .` — lint, including import ordering
3. `uv run mypy` — types (requires the WI-8 configuration)
4. `uv lock --check` — lockfile consistent with `pyproject.toml`
5. `uv run shellcheck` over all tracked shell scripts (pin `shellcheck-py` in the dev
   dependency group)

Every tool version comes from the pinned dev group so the gate is deterministic and
identical across machines; it must run network-free. Document in `AGENTS.md` that
`make quality` is mandatory before every push, for humans and agents alike. Do not
install a git pre-commit hook and do not adopt the pre-commit framework. This work item
belongs to execution step 1, immediately after WI-8. Acceptance: `make quality` exits 0
on the finished tree and exits non-zero when any single check is deliberately broken
(demonstrate once with a scratch violation, then revert it).

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

1. WI-8 first (mypy configuration is independent of the regroup and arms the commit
   gate), then WI-15 (the quality gate, so every later commit runs through it), then
   WI-1, then WI-2 (structure settles so everything downstream references final paths).
2. WI-5, WI-6, WI-7 (parallelizable once structure is stable), then WI-3, whose tests
   assert behavior WI-6 and WI-7 introduce.
3. WI-9, WI-10, WI-12, WI-13 (audits; parallelizable).
4. WI-4 (runbook last among the content items, so commands reflect the final CLI
   surface).
5. WI-14 (capstone review over the finished state), then WI-11 (the final regression
   run over the finished state).

Commit each work item separately with clear messages. Run the full default test suite
and `ruff check` before every commit; once WI-15 lands, the pre-commit gate is
`make quality` plus the default test suite.

## Definition of done

- Default `pytest` selection and `make quality` (format check, lint, mypy, lock check,
  shellcheck) all pass in the sandbox.
- `python -m training --help` and `python -m interp --help` list every subcommand;
  `make help` targets resolve.
- No stale imports or path references to moved modules anywhere in the repository
  (verified by grep, not assumption).
- `docs/verification/` covers every implemented feature with concrete commands and
  expected signals, and nothing that does not exist.
- Nothing requiring the workstation is claimed as verified; those items exist only as
  PENDING runbook entries.
- All work is pushed and presented as a pull request whose description maps commits to
  work items and reports the final regression-run result and per-package coverage
  (WI-11).
