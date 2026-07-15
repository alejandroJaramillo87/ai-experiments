# Unified CLI investigation

Investigates whether the training and interpretability tiers should unify behind a
single CLI, how tightly coupled the two tiers are today, and whether running
mechanistic-interpretability workflows during training is a realistic goal on this
workstation. The repository's current state is treated as the source of truth; the
planning document (roadmap Google Doc) defers to it wherever the two disagree.
Findings are current as of July 2026. Model policy going forward: Gemma 4 E-models
primary, Qwen3 family fallback, matching `training/families.py`.

## Table of Contents

- [Current entry-point inventory](#current-entry-point-inventory)
- [How coupled are interp and training today](#how-coupled-are-interp-and-training-today)
- [Would a unified CLI make sense](#would-a-unified-cli-make-sense)
- [Recommended shape](#recommended-shape)
- [Interp-during-training: feasibility](#interp-during-training-feasibility)
- [Concurrent monitoring via CPU offload](#concurrent-monitoring-via-cpu-offload)
- [Gemma 4 considerations](#gemma-4-considerations)
- [When to implement](#when-to-implement)
- [Drift between the planning document and the repository](#drift-between-the-planning-document-and-the-repository)
- [Open questions](#open-questions)

## Current entry-point inventory

The two tiers are at different maturity levels of the same convergence path.

Training already has a unified CLI: `python -m training` dispatches nine argparse
subcommands (`memfit`, `calibrate`, `train`, `train-sft`, `train-distill`,
`distill-cache`, `export`, `eval-holdout`, `eval-lmeval`) from
`training/cli.py:541`. The module docstring (`training/cli.py:1-11`) records the
resolved D4 decision that this is the single entry point and that the remaining
standalone scripts under `scripts/training/` (nine data-preparation and validation
scripts, each with its own argparse main) migrate in as thin subcommands. That
migration has not happened yet.

Interp has no package CLI. `interp/` is a library (`interp/__init__.py`), and its
ten entry points are loose scripts under `scripts/interp/` (`harvest.py`,
`steer_extract.py`, `steer_scorecard.py`, `probe_capture.py`, `probe_train.py`,
`probe_eval.py`, `probe_monitor.py`, `organism.py`, `interp_gc.py`,
`smoke_test.py`), each wired to a Makefile target.

Nothing is installed as a console script: `pyproject.toml` sets `package = false`
and defines no `[project.scripts]`. All invocation goes through `make`,
`python -m training`, or `python scripts/...` inside the compose containers.

Duplication that a unified CLI would eliminate immediately: the styled `Printer`
is implemented twice (`interp/output.py:9` and `training/cli.py:36`), both
claiming `.styles/PYTHON_OUTPUT_STYLE.md` as their contract.

## How coupled are interp and training today

Coupling is real, one-directional, and concentrated at the configuration and
operational layers — not at the training-loop layer.

- **Direction.** `interp` imports `training`; `training` never imports `interp`.
  All cross-imports are lazy (inside functions), preserving the CPU-sandbox
  importability rule.
- **Shared recipe system (dominant coupling).** Interp has no config system of its
  own. `training/recipes/schema.py` hosts the interp-owned `OrganismConfig`
  (`schema.py:339-356`) and `HarvestConfig` (`schema.py:374-406`) blocks, and interp
  recipes live under `training/recipes/interp/` and `training/recipes/organisms/`,
  loaded through `training.recipes.loader`. A harvest or organism job is a
  training recipe with an extra block.
- **Shared operational envelope.** Interp harvest takes the training D9 run lock
  and D10 watchdog (`interp/harvest.py:276-277` imports `training.lock.run_lock`
  and `training.watchdog.Watchdog`); the interp compose mounts
  `/mnt/ai-data/training` read-write for exactly this
  (`docker-compose.interp.yaml:59-61`). One machine-ownership discipline, not two.
- **Shared checkpoint format.** Organism and harvest jobs consume D6/F3
  checkpoints (`meta.json`-stamped sharded safetensors) unchanged.
- **Shared generation path.** `interp/organisms/generate.py:227` reuses
  `training.data.teacher`; `scripts/interp/organism.py` reuses
  `training.serving` and the recipe loader/validator.
- **What is not coupled.** No interp code runs inside the training loop. The only
  callback surface, `training/sft_callbacks.py` (TRL `TrainerCallback` bridging to
  tracker, watchdog, and checkpoint retention), is training-internal. Interp's
  relationship to training today is strictly post hoc: read checkpoints, take the
  lock, replay logs.

Summary: the tiers already share one config schema, one lock, one watchdog, one
checkpoint format, and one storage convention. Semantically they are one system
with two front doors. The separation that remains is at the CLI surface and at
the training loop.

## Would a unified CLI make sense

Yes, with a specific shape. The arguments for:

1. **The convergence is already resolved policy.** `training/cli.py` documents the
   single-entry-point decision for the training tier; interp scripts are the same
   pattern one phase behind. Unifying finishes a migration the repository already
   committed to rather than opening a new architectural question.
2. **The semantics are already unified.** Recipes, locks, watchdogs, checkpoints,
   and manifests are shared. A CLI split that does not correspond to a semantic
   split is accidental, not architectural.
3. **Interp-during-training requires it — at the config layer.** The goal of
   running interp workflows against a live training run means one job description
   (recipe) that names both the training work and the interp battery. That recipe
   already lives in one schema; the tool that executes it should be one tool.
4. **Concrete duplication is removed.** One `Printer`, one argparse scaffold, one
   `--json` and exit-code convention, one place to enforce the lock preflight that
   the interp cross-cutting conventions require of every tool.

The arguments against a full merge, which bound the recommendation:

- The tiers have different concurrency shapes (interactive workbench versus batch
  jobs versus serving-dependent work) and different container images. A CLI can
  unify the surface but must not blur which compose profile a command belongs to;
  each subcommand should keep its preflight (lock state, compose state) and fail
  clearly when invoked in the wrong environment.
- `training` importing `interp` would invert the current clean dependency
  direction. The dispatcher must live above both packages, not inside `training`.

## Recommended shape

A thin top-level dispatcher over two package CLIs, in three steps, each shippable
alone:

1. **Give interp what training already has.** Add `interp/cli.py` and
   `interp/__main__.py` mirroring `training/cli.py`: subcommands `harvest`,
   `steer-extract`, `steer-scorecard`, `probe-capture`, `probe-train`,
   `probe-eval`, `probe-monitor`, `organism`, `gc`, `smoke`. The
   `scripts/interp/*.py` files become thin shims (or are deleted once Makefile
   targets are repointed). Move `Printer` to a shared module (for example
   `common/output.py` or `interp/output.py` consumed by both) and delete the copy
   in `training/cli.py`.
2. **Finish the stated training migration.** Fold `scripts/training/*.py` into
   `-m training` subcommands per the existing `training/cli.py` docstring.
3. **Add the umbrella.** A small top-level package (working name `lab/`) whose
   `__main__` dispatches `lab train ...` to the training parser and
   `lab interp ...` to the interp parser, plus genuinely cross-tier verbs
   (`lab status`, `lab lock`, `lab gc`). Optionally register it as the single
   `[project.scripts]` console entry once `package = false` is revisited. The
   Makefile keeps its targets as the host-side ergonomic layer; targets shrink to
   one-line `compose run ... python -m lab ...` calls.

This is better than the current state because it completes an in-flight
convergence, and better than a monolithic rewrite because steps 1 and 2 are pure
consolidation with no behavior change, testable with the existing suites
(`tests/training/`, `tests/interp/`).

## Interp-during-training: feasibility

The goal — run interp workflows during training — is realistic and worth
pursuing. Two granularities apply, and the recommended design uses both: a
**streaming monitor** that runs concurrently with optimizer steps (see
[Concurrent monitoring via CPU offload](#concurrent-monitoring-via-cpu-offload)),
and a **checkpoint battery** at checkpoint boundaries, which remains the
canonical, reproducible measurement.

Why the battery anchors at checkpoint boundaries:

- **One GPU.** The RTX 5090's 32GB is sized by the memory-fit calculator for the
  training job itself; a concurrent hooked forward pass of a second model copy
  does not fit, and the repository's one-GPU-consumer rule exists for this
  reason. At a checkpoint boundary the training process is at a natural pause and
  the weights are already materialized on disk.
- **Comparability.** Fixed probe corpus, fixed checkpoint set: battery numbers
  are comparable across checkpoints, runs, and sequences (the retention-matrix
  discipline). Streaming measurements over sampled training batches are not.
- **The hook already has a design.** The Phase 4 continual-learning design
  specifies a `post_step_hook` in the sequence config, running after
  checkpointing, explicitly reserved for the interpretability checkpoint battery.
  The TRL path has the equivalent attach point today: `training/sft_callbacks.py`
  is a `TrainerCallback` bridge, and TRL fires `on_save` after every checkpoint
  write. A new callback there is the natural integration, and it is a small
  build.
- **The workloads that matter are cheap at that granularity.** The checkpoint
  battery the interp roadmap describes — logistic-regression probe suites over
  captured activations, logit-lens divergence, behavioral spot checks — costs
  minutes per checkpoint on a sub-6B model. Activation capture for the battery is
  a bounded forward pass over a fixed probe corpus, not a 300M-token harvest.
  What is **not** realistic in-loop: SAE training, full harvests, diffing runs —
  those remain post-hoc batch jobs under the D9 lock, as designed.

Proposed mechanism, concretely:

1. A `battery` block in the recipe schema (peer of `HarvestConfig`), naming the
   probe set, probe corpus, layers, and budget per checkpoint.
2. A `TrainerCallback` in the training tier that, on `on_save`, either (a) runs
   the battery in-process against the live model in `eval` mode within a VRAM
   budget check, or (b) enqueues the saved checkpoint path for a sibling process.
   Option (a) is simpler and keeps the one-GPU rule trivially satisfied (same
   process); it should be the default, with a wall-clock budget so a slow battery
   cannot stall training. The callback lives in `training/` but must not import
   `interp` at module scope — invoke through a lazy import or a registered entry
   point so the dependency direction stays acceptable (this is the one place the
   one-way import rule is deliberately relaxed, and it should be documented where
   it happens).
3. Battery results written as JSON beside the checkpoint (mirroring
   `retention_matrix.json` from the Phase 4 design), so a training run yields a
   mechanistic trajectory for free.

This is exactly the "interpretability to training" loop the roadmap already
names (battery results informing replay fractions and regularization strength),
promoted from post-hoc to per-checkpoint. Nothing about it requires the unified
CLI, but the unified recipe schema — which exists — is its real foundation; the
CLI unification makes the operator surface match.

## Concurrent monitoring via CPU offload

Per-step concurrent interp is feasible without a second GPU model copy, because
the training step already computes the activations. A forward hook on the
residual stream detaches the tensors the training forward pass produces and
copies them to pinned CPU memory asynchronously; probe scoring, logit-lens
divergence, and activation-statistic drift then run on the CPU while the GPU
continues training.

Cost analysis on this workstation:

- **VRAM:** approximately zero. The activation tensor already exists for the
  backward pass; the hook detaches a copy. One layer of a sub-6B model per step
  is batch_tokens x hidden x 2 bytes at bf16 — single-digit MB.
- **PCIe:** a few MB per step against roughly 64GB/s of PCIe 5.0 x16 bandwidth
  is negligible, provided copies use pinned buffers with `non_blocking=True` and
  the hook never synchronizes (no `.item()`, no `.cpu()` without a stream) on
  the hot path.
- **CPU:** logistic-regression probes and logit-lens are small matmuls; the
  9950X with 128GB RAM handles them in an async consumer thread or process,
  following the existing watchdog thread model and core-affinity conventions
  (`training/affinity.py`).

Design rules:

- **The consumer never backpressures training.** Bounded queue between the hook
  and the CPU consumer; when full, samples are dropped and the drop rate is
  logged. Zero overhead when disabled, mirroring the LD_PRELOAD interceptor
  principle in `.styles/CPP_STYLE.md`.
- **Do not offload training layers to CPU.** Moving transformer layers off the
  GPU to "make room" for interp is unnecessary (the hook needs no meaningful
  VRAM) and destroys throughput: a CPU bf16 forward on a 2-4B model is orders of
  magnitude slower than the GPU. CPU offload of optimizer state (P2-D5) is a
  separate concern — it exists to fit larger models, not to enable interp.
- **Distribution caveat.** Streamed activations come from sampled training
  batches, so per-step curves measure drift on the training distribution and are
  noisy and non-comparable across runs. That is the right signal for early
  warning ("the refusal probe is collapsing mid-run"), and it can feed the D10
  watchdog to abort a doomed overnight job. For controlled numbers, either rely
  on the checkpoint battery or interleave an occasional `no_grad` microbatch of
  the fixed probe corpus every N steps at a small throughput cost.
- **Scope.** Concurrency unlocks probes, logit-lens, and statistics. SAE
  training and diffing still require real GPU throughput and remain post-hoc
  batch jobs under the D9 lock. One freebie: the same hook can stream
  activations to NVMe under the harvest manifest convention, so a training run
  harvests as it trains — a free SAE-training cache on the training
  distribution, subject to the usual disk preflight and the 1-2TB activation
  budget.

The streaming monitor and the checkpoint battery attach at the same places (the
`TrainerCallback` bridge in `training/sft_callbacks.py` and a recipe block), so
the streaming layer extends the battery design rather than replacing it: the
monitor answers "is something going wrong right now," the battery answers "how
does this checkpoint compare."

## Gemma 4 considerations

The forward model policy (Gemma 4 primary, Qwen3 fallback) is already encoded in
`training/families.py` and constrains the interp integration in one important
way: hooked-execution tooling.

- TransformerLens has no Gemma 4 support; `interp/harvest.py:10-13` already
  avoids `HookedTransformer` for this reason and drives SAELens's caching runner
  with `model_class_name="AutoModelForCausalLM"`.
- The Gemma 4 E-models nest the LM under `text_config` and have per-layer
  embedding (PLE) hook dimensions; `interp/harvest.py:141-159` already handles
  both, and `make harvest-check-hooks` gates on it.
- The probe path uses nnsight `LanguageModel` on plain HF checkpoints
  (`interp/probes/capture.py:78-83`), which requires no architecture port.

Consequence for the checkpoint battery: build it on plain HF hooks or nnsight,
never on TransformerLens model aliases, and key hook points by name (the existing
harvest convention). The battery then works identically on Gemma 4 and Qwen3
subjects, and the Qwen3 fallback remains a one-line family swap.

## When to implement

Neither "now, all of it" nor "after every roadmap deliverable completes." The
two workstreams have opposite timing profiles.

**CLI consolidation (steps 1-2): implement soon, in the next non-GPU slot.**

- Pure consolidation with no behavior change, verifiable against the existing
  `tests/training/` and `tests/interp/` suites; its cost is fixed and small
  today.
- Its cost grows with every deliverable completed first: each remaining phase
  adds loose scripts, Makefile targets, and doc references that a later
  migration must also touch. Deferring to the end of the roadmap performs the
  same refactor against a strictly larger surface.
- The remaining phase designs specify their CLI entry points in cross-cutting
  conventions. If the package-CLI shape lands first, every future deliverable
  is born as a subcommand instead of migrated later.
- The work touches no GPU and no running service, so it parallelizes in a
  worktree under the concurrency policy.

**Umbrella command (step 3): defer indefinitely.** Ergonomics only while `make`
remains the host surface; it sits above both packages, so adding it later costs
nothing.

**Checkpoint battery: gate on training Phase 4 (continual learning).** That
phase specifies the `post_step_hook` and asks the questions per-checkpoint
retention answers (replay fractions, regularization strength). The interp
roadmap builds the battery last, after sprints establish which measurements
earn a place in it; building it earlier guesses at the metric suite. When the
Phase 4 outer loop lands, ship the cheap part with it: the recipe `battery`
block and the `on_save` callback skeleton with a no-op default (which the
Phase 4 design requires anyway). The attach point is cheap; the battery
contents wait for evidence.

**Streaming monitor: gate on the battery.** The monitor is the battery's
metrics at higher frequency on a noisier distribution. Which probes and
divergence measures are informative must be known before building the
pinned-buffer and async-consumer plumbing to compute them per step; plumbing
built before its payload is chosen becomes unused infrastructure.

Summary order: CLI steps 1-2 next, umbrella later or never, battery attach
point with training Phase 4, battery contents and streaming monitor after the
first continual-learning sequences show which measurements matter. The option
to avoid is waiting for full roadmap completion: it maximizes the migration
surface for the CLI work and provides nothing to the interp integration, which
gates on Phase 4 starting, not on deliverables finishing.

## Drift between the planning document and the repository

Recorded per the repository rule that code is the source of truth:

- The interp roadmap document states "no interpretability code is implied to
  exist yet." Stale: the repository has the full interp library, ten script entry
  points, compose profile, and eight test modules.
- The document's interp model policy centers Gemma-2-2B, Qwen3, and Llama-3.1-8B.
  Current policy per `training/families.py` and this investigation's charter is
  Gemma 4 primary with Qwen3 fallback; Gemma-2-2B survives only as the
  tooling-calibration anchor (Gemma Scope SAEs, smoke tests), which the interp
  compose still provisions.
- The document places interp CLI entry points in `scripts/interp/` per the Phase
  4 cross-cutting conventions; the recommendation above supersedes that with the
  package-CLI shape, matching the training tier's already-resolved D4 decision.
- The document names style guides under `.claude/`; they live under `.styles/`.

## Open questions

1. Should the umbrella command land as step 3, or is `python -m training` /
   `python -m interp` (steps 1-2 only) enough ergonomics given the Makefile
   remains the host-side surface?
2. For the checkpoint battery, is the in-process default (run in the trainer
   process on `on_save` within a time budget) acceptable, or is a strict
   subprocess-on-saved-checkpoint isolation preferred at the cost of a second
   model load per checkpoint?
3. Does `package = false` in `pyproject.toml` get revisited to allow a console
   script, or is module invocation inside containers the permanent convention?
