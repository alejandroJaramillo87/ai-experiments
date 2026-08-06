# P2-D3: Warmup-stable-decay schedule — implementation investigation

**Frozen record — 2026-07-15. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing P2-D3 (warmup-stable-decay schedule) entails,
reconciling the Phase 2 product design against the current state of the repository.
Per the operating rules, the repository is the source of truth; every point where the
design document disagrees with shipped code is called out and the repository
convention wins.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [Resolved design decisions](#resolved-design-decisions)
- [Module layout](#module-layout)
- [Acceptance criteria](#acceptance-criteria)

## Scope summary

P2-D3 adds the learning-rate shape that distinguishes continual pre-training from a
naive resume: linear warmup from near zero to a peak below the base model's original
peak, a constant stable phase, then decay to a configurable floor. Tag: orchestrate.
Depends on: Phase 0 D5 (recipe schema), Phase 0 D6 (loop, checkpointing). The design
document's core decisions:

1. A `wsd` schedule beside the existing cosine: linear warmup over `warmup_steps`
   to peak, constant until `decay_start_frac` of `total_steps`, then decay to
   `final_lr_frac * peak_lr`. Default decay shape is "1-sqrt" per the WSD
   literature (MiniCPM; Haegele et al. 2024); cosine decay is an option.
2. Pure function of step to multiplier, wrapped in `LambdaLR`; scheduler state
   already checkpoints via Phase 0 D6.
3. Recipe block `schedule: {type: wsd, warmup_steps, decay_start_frac,
   final_lr_frac, total_steps}` with `validate()` rules.
4. Checkpoint-at-stable convention recorded now (stable-phase checkpoints are
   decay-branchable; retention must keep at least one); multi-branch tooling is
   deferred to later phases.
5. The concrete `peak_lr` for the Gemma-4-E2B reference run is pinned only after a
   200-step stability canary on the workstation; that belongs to the reference
   recipe (P2-D5), not to this deliverable.

## Current repository state

The repository already contained a partial WSD implementation that the design
document did not reflect:

- `training/model.py` had `build_scheduler()` with a hardcoded `wsd` branch:
  decay over the final 10 percent of steps (`WSD_DECAY_FRACTION = 0.1`), linear,
  to zero. No `decay_start_frac`, `final_lr_frac`, or shape option.
- `training/recipes/schema.py` `ScheduleConfig` carried only `type`,
  `warmup_steps`, `total_steps`.
- `training/sft.py` maps `type: wsd` onto the transformers built-in
  `warmup_stable_decay` scheduler, which supports neither a 1-sqrt shape nor a
  nonzero LR floor.
- No merged recipe used `type: wsd`; the branch was dead code, so its behavior
  could change without breaking any existing run or checkpoint.
- Checkpoint/resume of scheduler state (`scheduler.pt`) and per-step `lr`
  tracking already existed, and `train(..., stop_after=N)` supports halting at a
  step boundary without touching the schedule — the resume acceptance tests build
  directly on this.

## Design-document drift

- The document proposes a new module `training/schedules.py`; the repository kept
  schedule logic inside `training/model.py`. Resolution: extracted to
  `training/schedules.py` with `training/loop.py` importing from the new module;
  `model.py` no longer builds schedulers.
- The document describes the default decay as "inverse-sqrt-to-linear (1-sqrt)";
  the shipped code decayed linearly to zero. Resolution: 1-sqrt adopted as the
  default (`schedule.decay_shape: 1-sqrt`), with `linear` and `cosine` as options.
  Safe because no recipe or checkpoint exercised the old curve.

## Resolved design decisions

1. **Module extraction.** `training/schedules.py` holds `cosine_multiplier`,
   `wsd_multiplier`, and `build_scheduler`. Multiplier functions are pure and
   importable without torch; torch is imported lazily inside `build_scheduler`.
2. **Default decay shape is 1-sqrt.** The multiplier falls as
   `final + (1 - final) * (1 - sqrt(p))` across the decay window: a fast initial
   drop followed by a long low-LR tail, which the WSD literature reports anneals
   marginally better than linear for the same decay budget.
3. **Backward-compatible defaults.** `decay_start_frac: 0.9` and
   `final_lr_frac: 0.0` reproduce the previous plateau/decay proportions; only the
   in-window shape changed (decision 2).
4. **WSD knobs are pretrain-loop-only.** The SFT path keeps its coarse mapping to
   the transformers `warmup_stable_decay` scheduler, but `validate()` rejects
   non-default `decay_start_frac`, `final_lr_frac`, or `decay_shape` on recipes
   with `sft.enabled=true`: the transformers scheduler cannot express them, and a
   recipe field that silently means two different curves depending on entry point
   would break the resume-exactness guarantee. If exact WSD is ever needed under
   TRL, the clean route is injecting the `LambdaLR` via the `optimizers` argument.
5. **Validation rules.** `wsd` requires `total_steps`; `warmup_steps` must end
   before `int(total_steps * decay_start_frac)` so a stable plateau exists;
   `decay_start_frac` in (0, 1]; `final_lr_frac` in [0, 1); `decay_shape` one of
   `1-sqrt`, `linear`, `cosine`.
6. **Checkpoint-at-stable convention** is documented in
   `docs/training/tracking_checkpointing.md` (Retention section). Convention only;
   no branching tooling in this phase.
7. **Rewarming default deferred.** `peak_lr` guidance (a value on the order of
   1e-4 or below for a ~2B-class trainable stack, pinned after the 200-step
   stability canary) lands with the continual reference recipe in P2-D5. The
   schedule mechanism is model-agnostic; Gemma 4 versus Qwen3 enters only through
   that recipe.

## Module layout

| Path | Role |
|------|------|
| `training/schedules.py` | Multiplier functions and `build_scheduler` (new) |
| `training/model.py` | Scheduler builder removed; model/optimizer builders unchanged |
| `training/loop.py` | Imports `build_scheduler` from `training/schedules.py`; `TrainResult.lrs` records the per-step LR |
| `training/recipes/schema.py` | `ScheduleConfig` gains `decay_start_frac`, `final_lr_frac`, `decay_shape`; `WSD_DECAY_SHAPES` constant |
| `training/recipes/validate.py` | `_check_wsd` invariants |
| `tests/training/test_schedules.py` | Curve pinning and tiny-run resume acceptance |
| `tests/training/test_recipes.py` | One test per new `validate()` invariant |

## Acceptance criteria

Mapped from the design document, all executable in the CPU sandbox:

- Unit tests pin the multiplier curve at hand-computed steps for a small config
  (warmup boundary, stable value, decay endpoints, floor clamp) for all three
  decay shapes, plus the default-knob curve (`tests/training/test_schedules.py::
  TestWsdMultiplier`).
- A tiny run's tracked LR curve shows the three segments
  (`test_tiny_run_lr_curve_shows_three_segments`).
- Resume mid-stable and mid-decay continues the curve exactly — bitwise-equal LR
  sequences against a straight run (`test_resume_continues_lr_curve_exactly`,
  parameterized over both stop points).
