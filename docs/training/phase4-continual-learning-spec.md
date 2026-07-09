# Phase 4 product design: Continual learning

This document is the implementation-ready product design for training-tier Phase 4, the continual-learning regime defined in the training scope document. It is written to be picked up by a coding agent in a fresh session and implemented without re-deriving context: it records the decisions already made, the interfaces to build, the contracts it consumes from earlier phases, and the tests that define done.

The scope document defines what Phase 4 delivers; this document defines how. Where the two disagree, this document wins and the scope document should be corrected. One deliberate amendment is recorded here: the scope document excluded preference alignment entirely, but Phase 4 now includes a continual-alignment stage (sub-stage C below), because the mid-2026 continual-learning taxonomy this repository organizes around treats continual alignment as a first-class stage and the owner has decided to implement it. Single-shot alignment as a product goal remains out of scope; alignment appears here only as a stage inside the continual regime.

## Table of contents

- [Assumed contracts from Phases 0-3](#assumed-contracts-from-phases-0-3)
- [Cross-cutting conventions](#cross-cutting-conventions)
- [Design overview](#design-overview)
- [Build order](#build-order)
- [Sub-stage A: parametric regime](#sub-stage-a-parametric-regime)
  - [A1: Task-sequence schema and outer loop](#a1-task-sequence-schema-and-outer-loop)
  - [A2: Replay buffer](#a2-replay-buffer)
  - [A3: Anti-forgetting regularizers](#a3-anti-forgetting-regularizers)
  - [A4: Retention metrics and evaluation matrix](#a4-retention-metrics-and-evaluation-matrix)
  - [A5: Knowledge editing via EasyEdit](#a5-knowledge-editing-via-easyedit)
  - [A6: Continual-learning benchmark suite](#a6-continual-learning-benchmark-suite)
- [Sub-stage C: continual alignment](#sub-stage-c-continual-alignment)
  - [C1: Preference training step](#c1-preference-training-step)
  - [C2: Alignment retention evaluation](#c2-alignment-retention-evaluation)
- [Sub-stage B: architectural test-time memory](#sub-stage-b-architectural-test-time-memory)
  - [B1: Titans and HOPE in the model registry](#b1-titans-and-hope-in-the-model-registry)
  - [B2: Scale-up and benchmarking](#b2-scale-up-and-benchmarking)
  - [B3: Self-editing harness](#b3-self-editing-harness)
- [Integration with the interpretability tier](#integration-with-the-interpretability-tier)
- [Phase 4 exit checklist](#phase-4-exit-checklist)
- [References](#references)

## Assumed contracts from Phases 0-3

Phase 4 adds no new trainer. It is an outer loop plus anti-forgetting machinery around code paths that already exist. The following interfaces are assumed present and are consumed, not reimplemented. If an implementing session finds one missing or shaped differently, it stops and reconciles against the Phase 0 design document rather than building a parallel version.

| Contract | Source | What Phase 4 consumes |
|----------|--------|-----------------------|
| Training container | Phase 0 D1 | The docker/training image; all Phase 4 jobs run in it under the training compose profile. EasyEdit and any DPO dependency are added to this image, not a new one |
| Shard format and reader | Phase 0 D2 | ShardedTokenDataset with serializable cursor; manifest with tokenizer hash. The replay buffer reads shard windows through this reader |
| Recipe system | Phase 0 D5 | Hydra structured configs in training/recipes/schema.py. Phase 4 extends the schema with continual, regularizer, and sequence blocks; it does not create a second config system |
| Checkpointing and tracking | Phase 0 D6 | Atomic checkpoint directories with model, optimizer, dataloader cursor, RNG state, and meta.json; the tracking fan-out (wandb offline, mlflow, tensorboard). The outer loop checkpoints between sequence steps through this machinery |
| Train-and-serve round trip | Phase 0 D7 | Export of any sequence checkpoint to GGUF and smoke evaluation through llama-gpu on port 8002 |
| Heterogeneous-compute skeleton and run lock | Phase 0 D9 | Prefetch dataloader, core pinning, and the /mnt/ai-data/training/.lock file honoring the concurrency policy |
| Watchdogs | Phase 0 D10 | The in-process watchdog callback; every inner training run in a sequence hooks it unchanged |
| SFT entry point and chat data prep | Phase 1 | The TRL-based train-sft path, prompt-loss masking, chat-template registry, PEFT adapter options. Continual fine-tuning steps invoke this path |
| Evaluation wrapper | Phase 1 | scripts/eval/eval_ft.py: held-out loss, lm-evaluation-harness tasks, model-as-judge scoring through the inference services |
| Continual pre-training path | Phase 2 | Warmup-stable-decay schedule with rewarming, data-mixing block, ZeRO-Offload configuration, and training/continual/replay.py if Phase 2 already delivered it (see A2 for the reconciliation rule) |
| Forgetting evaluation | Phase 2 | scripts/eval/forgetting.py base-versus-adapted comparison; A4 generalizes it from a pair to a sequence |
| Architecture registry | Phase 3 | training/models/ common interface that B1 extends with Titans and HOPE entries |
| Pretraining loop | Phase 3 | scripts/train/pretrain.py, used by B2 for nano-scale HOPE training |

Model policy for sub-stages A and C: sequences run on strong open bases in the 1B to 3B class (SmolLM3-3B, Qwen3 small variants, Llama-3.2) with full fine-tuning under offload, or up to 8B with LoRA-family adapters. The 32GB VRAM envelope, the 128GB RAM budget, and the inference-down concurrency policy from the scope document apply unchanged.

## Cross-cutting conventions

- All Phase 4 source lives under training/continual/ except model architectures (training/models/) and CLI entry points (scripts/train/, scripts/eval/). Tests live in tests/training/continual/, runnable with make test.
- Style per .claude/PYTHON_STYLE.md, .claude/COMMENT_STYLE.md, .claude/BASH_OUTPUT_STYLE.md, .claude/DOCUMENTATION_STYLE.md. No emojis, plain technical prose, black and ruff formatting per pyproject.
- Heavy artifacts live under /mnt/ai-data/training/. Sequences write to runs/<sequence-name>/ with one subdirectory per step; checkpoints follow the D6 layout under checkpoints/<sequence-name>-step<k>/. Retention: the checkpoint that ends each sequence step is retained for the life of the sequence (the interpretability tier's checkpoint battery and longitudinal diffing consume the full trajectory); within a step, D6 keep-last-N applies.
- Every measurable result is emitted as JSON following the D8 benchmark schema conventions: schema_version, timestamp, git SHA, hardware and software blocks, recipe hash, then the result payload. Retention reports publish under docs/training/continual/results/ when run with --publish; default is the run directory.
- Determinism: sequence-level acceptance tests run with fixed seeds and torch.use_deterministic_algorithms(True) exactly as the D6 resume test does; production sequences default to nondeterministic for speed.
- Version pins recorded in this document are as of 2026-07; re-verify at implementation time.

## Design overview

A continual-learning run is a sequence S = (s_1, ..., s_n) of steps. Each step names a stage (cpt for continual pre-training, sft for continual fine-tuning, align for continual alignment, edit for knowledge editing), a data source, and a method configuration (replay fraction, regularizers, adapter policy). The outer loop executes steps in order: load the model state from the previous step, run the stage's existing code path with the anti-forgetting machinery attached, checkpoint, evaluate the model on the evaluation sets of all steps seen so far plus fixed anchor benchmarks, append a row to the retention matrix, and continue. The final artifacts are the checkpoint trajectory, the retention matrix, and a report.

Three method families attach to any gradient-based step, all recipe toggles and composable:

- Rehearsal: the replay buffer (A2) mixes a configured fraction of prior-step and original-distribution data into each batch.
- Regularization: loss and optimizer hooks (A3) penalize movement of parameters important to earlier steps.
- Architecture: PEFT adapter isolation, one adapter per step or orthogonal LoRA, configured through the existing Phase 1 PEFT surface plus thin routing glue in the loop.

Knowledge editing (A5) is the fourth mechanism and bypasses the trainer entirely; it appears as an edit step type so edits can be interleaved with training steps in one sequence and measured with the same retention machinery.

Sub-stage C runs preference optimization as a step type on the same loop with the same machinery. Sub-stage B changes the substrate rather than the regime: architectures whose continual behavior is a property of the model. It is specified at design level and gated on sub-stage A acceptance.

## Build order

A1 sequence loop ── A2 replay ──┐
        │                       ├── A4 retention metrics ── A6 benchmark suite
        ├── A3 regularizers ────┘
        └── A5 knowledge editing (parallel track; needs A1 step interface and A4 metrics)

C1 preference step ── C2 alignment retention   (after A1-A4; reuses all of them)

B1 registry entries ── B2 scale-up ── B3 self-editing   (gated on sub-stage A exit)

A1 with a null method configuration (no replay, no regularizers, sequential fine-tuning) is the first thing built, because the naive-forgetting baseline it produces is the reference every other deliverable is measured against.

## Sub-stage A: parametric regime

### A1: Task-sequence schema and outer loop

Tag: build. Depends on: Phase 0 D5, D6; Phase 1 and 2 entry points.

#### Purpose

One YAML file fully specifies a continual-learning sequence, and one command runs it end to end with checkpointing between steps and resume at step granularity. This is the spine of the phase.

#### Decisions

1. Schema extension, not a new system. training/recipes/schema.py gains a SequenceConfig registered in the Hydra ConfigStore: sequence: {name, base_model, steps: [...]}, where each step is {name, stage: cpt|sft|align|edit, recipe: <path or inline block>, replay: {...}, regularizers: [...], adapter: {...}, eval: {...}}. A step's recipe block is a standard Phase 0 recipe for its stage; the sequence file composes them. validate() extends with sequence-level invariants: all steps share the base model's tokenizer hash, adapter policies are consistent (a merge_before_next flag on adapter steps), edit steps carry no optimizer block.
2. Loop implementation. training/continual/loop.py exposes run_sequence(config). Per step: acquire the D9 run lock, resolve the input state (previous step's checkpoint, or base_model for step 1), dispatch to a stage driver, checkpoint through D6, release the lock, then run the A4 evaluation pass (which may need the serving stack for judge scoring and therefore runs in the serving-dependent mode per the concurrency policy), append to the retention matrix, write sequence_state.json (completed steps, checkpoint paths, matrix path), and continue.
3. Stage drivers are adapters over existing entry points. training/continual/stages.py wraps the Phase 1 SFT trainer, the Phase 2 CPT path, the C1 preference step, and the A5 edit path behind one interface: run(state_in, step_config, hooks) -> state_out. Drivers pass the replay-aware dataloader and the regularizer hooks into the underlying trainer via its callback and data-collator surfaces (TRL callbacks and a custom collator; the reference loop's explicit hook points). Drivers contain no training logic of their own.
4. Resume at step granularity is required; resume inside a step is inherited from D6 and the underlying trainer. continual-train SEQUENCE=... resumes from sequence_state.json by default; --restart-from <step-name> discards later state after an explicit confirmation flag.
5. Adapter policy. Three modes per step: full (all parameters trainable), lora_shared (one adapter continued across steps), lora_per_step (fresh adapter per step, previous adapters merged or kept per merge_before_next; orthogonal initialization against prior adapters available as o_lora: true). Per-step adapters plus the step manifest are the architecture-family method; routing at evaluation time is out of scope in A (the merged or shared model is what gets evaluated).
6. Failure semantics. A step that aborts (watchdog or error) leaves the sequence resumable at that step; the retention matrix row is written only after a successful evaluation pass, so the matrix never contains rows for half-finished steps.

#### Interfaces

- make continual-train SEQUENCE=training/recipes/sequences/<name>.yaml
- python -m training.continual.loop --sequence ... [--restart-from STEP --yes]
- Reference sequences shipped: sequences/reference/naive-3task.yaml (three small SFT tasks, no anti-forgetting; the forgetting baseline), sequences/reference/tiny-e2e.yaml (two seconds-long steps for tests).

#### Acceptance

- The tiny two-step sequence runs end to end, producing two step checkpoints, a two-row retention matrix, and a sequence_state.json that reloads.
- Killing the loop mid-step and rerunning resumes the same step; the completed-step checkpoint and matrix rows are untouched (SIGKILL test as in D6).
- The naive-3task reference sequence demonstrates measurable forgetting: task-1 evaluation degrades after step 3 versus after step 1. This is the baseline artifact for A3.
- Schema validation rejects a sequence mixing tokenizer-incompatible steps with a clear error.

### A2: Replay buffer

Tag: build. Depends on: A1, Phase 0 D2.

#### Purpose

Mix a configured fraction of prior data into every batch of a sequence step, with the buffer resident in system RAM per the hardware-envelope pillar.

#### Decisions

1. Reconciliation with Phase 2. The scope document places training/continual/replay.py in Phase 2 (deliverable 4) for single-run CPT replay. If it exists, A2 extends it; if not, A2 builds it and Phase 2's CPT path adopts it. There is exactly one replay implementation.
2. Sources. A replay source is either a D2 shard directory (token windows via ShardedTokenDataset) for cpt steps, or a JSONL/HF dataset of examples for sft and align steps. The buffer for step k can draw from: the original base distribution (a designated pretraining-mix shard set), and any earlier step's data, with per-source weights: replay: {fraction: 0.15, sources: [{name: base_mix, weight: 0.5}, {name: step_1, weight: 0.5}], budget_gb: 32, policy: reservoir|full}.
3. Residency. Sources whose sampled subsets fit budget_gb (default 32, hard-capped below the D10 RAM floor arithmetic under offload) are materialized into RAM as pinned arrays; larger sources fall back to mmap through the D2 reader, which under the inference-down policy is page-cache-warm anyway. The docstring states this fallback so nobody assumes RAM residency is load-bearing for correctness.
4. Sampling. Reservoir sampling per source when subsetting, seeded from the recipe; the buffer records exactly which document or window ids were sampled in a replay manifest so a sequence is reproducible.
5. Injection point is the batch, not the dataset: a mixing sampler yields replay examples at the configured fraction with the correct collation for the stage (token windows for cpt, masked chat examples for sft, preference pairs for align). Each batch element is tagged with its source; the loop logs realized replay fractions per step so the acceptance check is a log query, not an inference.

#### Acceptance

- Over any 1,000 consecutive batches, the realized replay fraction is within 1 percentage point of configured, and per-source proportions match weights (chi-squared sanity test in the unit suite with a fixed seed).
- The buffer refuses a configuration whose materialized size exceeds budget_gb and falls back to mmap with a logged notice.
- On the naive-3task sequence with replay fraction 0.2 added, task-1 forgetting is measurably reduced versus the A1 baseline (direction, not magnitude, is the gate).
- Replay manifest reproducibility: two runs with the same seed sample identical id sets.

### A3: Anti-forgetting regularizers

Tag: build. Depends on: A1.

#### Purpose

The regularization family as recipe toggles: elastic weight consolidation, learning-without-forgetting, and gradient projection, each attachable to any gradient-based step and composable with replay and adapters.

#### Decisions

1. Module: training/continual/regularizers.py with a common interface: prepare(model, prior_state, step_config) before training (computes or loads whatever the method needs), loss_hook(loss, model) -> loss added per step, and optional grad_hook(model) after backward. The stage drivers call these; TRL integration goes through a compute_loss wrapper and an on_pre_optimizer_step callback.
2. EWC. Diagonal Fisher estimated on a configurable sample (default 2,048 examples) of the previous step's data at the end of each step, saved beside the step checkpoint as fisher.safetensors plus the parameter snapshot reference. Penalty lambda/2 * sum F_i (theta_i - theta*_i)^2. Both online EWC (single running Fisher, gamma-decayed) and per-task EWC (sum over prior tasks) supported; online is the default because per-task memory grows linearly with sequence length. With adapters, the penalty applies to trainable parameters only.
3. LwF. Distillation-to-previous-self: the step-(k-1) model produces teacher logits on the current step's inputs; KL at temperature T (default 2.0) is added with weight alpha. Memory plan on 32GB: teacher runs in a separate forward pass with torch.no_grad under the same micro-batch; for models above ~1.5B full fine-tuning, teacher logits are precomputed to disk for the step's dataset (the distillation harness from Phase 1 D4 already knows how to do this; reuse it). The recipe validation warns when the in-memory-teacher path is projected not to fit, using the D4 memory-fit calculator extended with a second-model term.
4. Gradient projection. Orthogonal-gradient method: maintain a low-rank basis of prior-task gradient or feature subspaces (GPM-style, rank capped, default 128 per layer group) and project the current gradient to the orthogonal complement in grad_hook. Basis computed from the same sample used for Fisher. This is the most experimental of the three; it ships behind experimental: true in its config block and its failures are not phase blockers.
5. Composability is explicit: regularizers: [{name: ewc, lambda: ...}, {name: lwf, alpha: ...}] is a list; hooks apply in order. Unit tests cover each alone and EWC plus replay together.
6. Every regularizer logs its penalty term to tracking every step, so a run where the penalty collapses to zero (a classic silent failure: wrong parameter filter, stale snapshot) is visible on the dashboard.

#### Acceptance

- Unit tests: hand-computable Fisher on a toy quadratic model matches closed form; LwF KL is zero against an identical teacher; projected gradients have zero dot product with the stored basis, all CPU-only.
- On the naive-3task reference sequence, each method alone reduces task-1 forgetting versus the A1 baseline, and the result JSON records baseline, method, and delta. Gradient projection is exempt from the must-improve gate but its result is recorded either way.
- A deliberately mis-scoped EWC (lambda 0) reproduces the baseline within tolerance, confirming the harness measures the method and not an artifact.

### A4: Retention metrics and evaluation matrix

Tag: build. Depends on: A1; Phase 1 eval wrapper; Phase 2 forgetting eval.

#### Purpose

The measurement layer: after every step, evaluate everything seen so far plus fixed anchors, and reduce the resulting matrix to the standard continual-learning metrics.

#### Decisions

1. The object is the matrix R, where R[k][j] is the score of the model after step k on the evaluation set of step j (j <= k), plus anchor columns: fixed general-capability benchmarks (default: a small lm-evaluation-harness set, for example hellaswag, arc_easy, and a perplexity slice of the base distribution) that never change across the sequence and detect global degradation that per-task evaluations miss.
2. Metrics in training/continual/metrics.py, computed from R: average accuracy AA_k = mean_j R[k][j]; forgetting F_k = mean_{j<k} (max_{i<=k} R[i][j] - R[k][j]); backward transfer BWT = mean_{j<n} (R[n][j] - R[j][j]); forward transfer FWT = mean_j (R[j-1][j] - b_j) where b_j is the base model's zero-shot score on task j, computed once at sequence start. Pure functions over the matrix, unit-tested against hand-computed examples.
3. Per-step evaluation definitions live in the step config: eval: {sets: [...], metric: accuracy|loss|judge_score, judge: {...}}. Execution goes through the Phase 1 eval wrapper; judge scoring runs in the serving-dependent mode. Each evaluation set is hashed into the report so a mid-sequence eval-set change is detectable (and rejected by default).
4. Outputs: retention_matrix.json (the matrix plus per-cell metadata: checkpoint hash, eval set hash, timestamp) and retention_report.md rendered from it (matrix table, metric summaries, per-task score-versus-step curves as inline data tables). The JSON is the contract for the interpretability tier's longitudinal work.
5. scripts/eval/retention.py can also run standalone over an existing checkpoint trajectory (a directory of step checkpoints plus a sequence file), so trajectories can be re-scored after the fact with new evaluation sets, clearly marked as post-hoc in the output.

#### Acceptance

- Metric unit tests match hand-computed values on a synthetic 3x3 matrix, including edge cases (single task, monotone improvement).
- The naive-3task baseline produces a matrix whose F_3 is positive and whose report renders; the anchor columns are populated.
- Standalone re-scoring of the baseline trajectory reproduces the in-sequence matrix within evaluation noise.

### A5: Knowledge editing via EasyEdit

Tag: adopt. Depends on: A1 step interface, A4 metrics; training container.

#### Purpose

Targeted weight surgery, MEMIT and WISE through EasyEdit, as a step type in the same sequences and measured with the same retention machinery, plus the standard editing metrics.

#### Decisions

1. EasyEdit pinned in the training image (exact version at implementation time; it moves fast and its model-coverage matrix matters, so the pin comment records which methods were verified against which model families). It runs on plain HF checkpoints, which is what D6 stores.
2. scripts/train/edit.py: takes a checkpoint, an edits file (JSONL of {prompt, subject, target_new, ground_truth, locality_prompts, portability_prompts}), a method (memit|wise), and method hyperparameters; produces an edited checkpoint in D6 layout plus an edit report: per-edit success (post-edit target probability), locality (unchanged behavior on locality prompts), portability (paraphrase generalization), and batch summary.
3. As a sequence step (stage: edit), the driver applies the edits and the normal A4 evaluation pass follows, so a sequence can measure whether an edit batch damaged prior tasks — the locality question at task granularity, which EasyEdit's own metrics do not cover.
4. Model coverage rule: edits run only on architectures EasyEdit's method implementations support (Llama and Qwen families cover the sequence models); the tool fails loudly with the support matrix otherwise.
5. Scale expectation documented in the tool's docstring: MEMIT batches of hundreds of facts are in scope; thousands-of-edits streams are a research track for later, not an A5 requirement.

#### Acceptance

- A batch of at least 50 edits on a 1B-class model achieves the method's published-ballpark edit success and locality on the standard CounterFact-style inputs, recorded in the edit report.
- An edited fact is recalled through the D7 serving round trip (the model answers the new fact through llama-gpu), demonstrating the edit survives export and quantization or documenting that it does not — either result is acceptable and recorded.
- A sequence interleaving one edit step between two SFT steps produces a full retention matrix including the edit step's row.

### A6: Continual-learning benchmark suite

Tag: orchestrate. Depends on: A1-A4.

#### Purpose

A standardized external benchmark so results are comparable beyond this repository: a task-incremental sequence and a time-incremental sequence, packaged as reference sequence files.

#### Decisions

1. Task-incremental: a fixed ordered subset of standard instruction/classification tasks (drawn from the continual-instruction-tuning literature's common lists; final task list chosen at implementation time and frozen in the sequence file) as sequences/benchmarks/task-incremental-1b.yaml.
2. Time-incremental: a TiC-LM-style slice, time-ordered corpus shards (Wikipedia revisions or CommonCrawl monthly slices at a size that keeps one full sequence under roughly a GPU-week) as cpt steps with the paper's evaluation protocol approximated by perplexity on per-period held-out sets. Full TiC-LM at published scale does not fit this machine; the sequence file documents exactly which subset was taken so numbers are labeled honestly.
3. The suite is run twice at minimum: naive baseline and best-method configuration (replay plus the best A3 regularizer per the reference-sequence results). Both retention reports publish under docs/training/continual/results/.
4. Dataset preparation flows through the Phase 2 pipeline (dedup and decontamination on by default; decontamination lists include the anchor benchmarks from A4).

#### Acceptance

- Both benchmark sequences run end to end on a 1B-class base within the documented wall-clock budget.
- The published pair of reports (naive versus method) shows the method configuration improving forgetting at equal-or-better average accuracy, or the negative result is published with the same prominence.

## Sub-stage C: continual alignment

Alignment enters this repository only as a stage of the continual regime: the question is not "align a model" but "does alignment survive subsequent training, and does subsequent alignment destroy prior capability". Scope is deliberately narrow: direct preference optimization only, no reward-model training, no PPO-family RL, no online methods. Those exclusions hold for Phase 4 and are re-litigated only in a future scope revision.

### C1: Preference training step

Tag: orchestrate. Depends on: A1-A3; Phase 1 SFT stack.

#### Decisions

1. TRL's DPOTrainer as the implementation (already in the training image via TRL >= 1.7), wrapped as stage driver align in training/continual/stages.py. Recipe block: preference data path (JSONL of {prompt, chosen, rejected}), beta, loss variant (dpo default; ipo and length-normalized variants exposed as options because vanilla DPO's length bias is a known confound in retention measurement).
2. The reference model for the DPO loss is the step's input model (step k-1's checkpoint), which doubles as the LwF teacher when both are enabled, one extra model in memory, not two. The D4 calculator's second-model term (built in A3) gates fit; above the fit line the step requires LoRA (peft_config passed to DPOTrainer) rather than full training.
3. Replay composition: replay into an align step draws sft-formatted examples with plain SFT loss on the replay slice (mixing objectives in one step), implemented in the collator; the alternative of preference-formatted replay is left as a config option when prior align steps exist.
4. Preference data source for the reference sequence: a small public preference set (for example a subset of a standard helpfulness/harmlessness pair dataset) sized so an align step completes in hours; synthetic preference generation through the inference services is noted as a follow-on, not built here.

#### Acceptance

- An align step runs inside a sequence on a 1B-class model, logs DPO loss and implicit-reward margins to tracking, and checkpoints through D6.
- A unit test verifies the collator applies preference loss to preference examples and SFT loss to replay examples within one batch.
- With LoRA policy, the step fits and runs on a 7B-class model (envelope check, not a quality gate).

### C2: Alignment retention evaluation

Tag: build. Depends on: C1, A4.

#### Decisions

1. Alignment metrics become first-class A4 columns: win-rate of the current model against the pre-alignment checkpoint, judged by model-as-judge through the inference services on a fixed prompt set (Phase 1 judge machinery; judge model and prompt hashes recorded in the matrix metadata), plus a refusal/safety spot-check set scored the same way.
2. The two headline questions get dedicated reference sequences: sequences/reference/align-then-train.yaml (align, then SFT on an unrelated task; does the win-rate column decay) and sequences/reference/train-then-align.yaml (SFT tasks, then align; do task columns decay). Both run naive and with the A2/A3 machinery.
3. Judge-based columns carry confidence intervals (bootstrap over prompts) in the matrix; the report never presents a judge delta without its interval, per the repository's benchmark discipline.

#### Acceptance

- Both reference sequences produce complete matrices including alignment columns with intervals.
- The align-then-train report answers its question with a measured direction (alignment retained or degraded), published either way.

## Sub-stage B: architectural test-time memory

Design-level specification. Sub-stage B is gated: it begins only after the sub-stage A exit checklist passes, and it is explicitly experimental — HOPE has no official weights, the reproductions are unofficial, and self-modifying models are known to degrade as updates accumulate. Failure to reach a working large-scale HOPE is an acceptable Phase 4 outcome if documented; the nano-scale reproduction and the measurement harness are the hard requirements.

### B1: Titans and HOPE in the model registry

Tag: build (from open reproductions). Depends on: Phase 3 registry.

- training/models/titans.py and training/models/hope.py behind the Phase 3 registry interface, ported from the strongest open reproductions (candidates as of 2026-07: lucidrains/titans-pytorch and the nano-scale HOPE educational implementations; pin the upstream commit and record deviations). Titans first: it is the simpler test-time-memory core and HOPE builds on it.
- Components: the neural memory module with test-time gradient-based updates (momentum and forget gates), the Memory-as-Context integration for Titans; the self-modifying core and continuum memory system (multi-frequency parameter update levels) for HOPE. Custom kernels (chunked memory updates) only if the reproduction's plain PyTorch path is measured too slow to reach nano-scale training targets; a Triton kernel is the ceiling of ambition here, per the adopt-versus-build rule.
- Both models train through the Phase 3 pretraining loop unchanged (they are causal LMs to the loop); the memory-state handling extends the D6 checkpoint contract with a memory_state entry, versioned so plain-transformer checkpoints are unaffected.
- Acceptance: single-batch overfit passes for both; nano-scale (tens of millions of parameters) training runs track the reproduction's published loss curves within a documented tolerance; registry round-trips (config to model to checkpoint to reload) tested.

### B2: Scale-up and benchmarking

Tag: orchestrate. Depends on: B1.

- Ladder: nano (~50M) to ~300M on the Phase 3 FineWeb-Edu shards, comparing against a parameter-matched Phase 3 transformer at matched token budget: pretraining loss, long-context behavior (needle-style retrieval at lengths beyond the training context), and, the actual point, continual behavior: the B-model and the transformer baseline run the same A6 task-incremental sequence, and their retention matrices are compared. The 1B-to-3B extension from the scope document is declared an optional track (wall-clock, not design, is the cost), matching the Phase 3 precedent.
- All results in D8-convention JSON plus an experiments write-up under docs/training/continual/; the interpretability tier's checkpoint battery is pointed at the trajectory as its first architectural subject.
- Acceptance: the 300M comparison exists with matched budgets and the retention-matrix comparison is published, whichever way it comes out.

### B3: Self-editing harness

Tag: build. Depends on: B1 or any sub-stage A model; A4 metrics; inference services.

- training/continual/self_edit.py: a Self-Adapting-Language-Models-style loop in which the model generates its own training data (self-edits) for a new fact or task, fine-tunes on them (LoRA by default, through the A1 machinery as generated sft steps), and is re-evaluated; iterated for N rounds with the A4 retention matrix tracking degradation across rounds. Generation runs through the inference services in the serving-dependent mode; training runs inference-down; the loop alternates per the concurrency policy.
- The deliverable is the retention curve across self-edit rounds, the field's known failure mode made measurable on local hardware, not a claim that self-editing works.
- Acceptance: a 10-round self-edit run on a 1B-class model completes unattended (watchdogs on) and the report shows per-round edit success and retention.

## Integration with the interpretability tier

Phase 4 is the primary producer for the interpretability tier's mechanistic-continual-learning track. Two obligations are binding on this implementation:

- Full-trajectory retention: every sequence keeps its per-step checkpoints (the retention policy in the conventions section) and the retention_matrix.json beside them, so diffing and probe batteries can walk the trajectory.
- A stable hook: the outer loop exposes an optional post-step command (post_step_hook in the sequence config) that runs after checkpointing, which is where the interpretability checkpoint battery will attach when it exists. Phase 4 ships the hook and a no-op default; the battery itself is interpretability-tier work.

## Phase 4 exit checklist

1. The naive baseline sequence demonstrates forgetting, and each of replay, EWC, and LwF measurably reduces it versus that baseline, with results in published JSON (A1-A3). Gradient projection's result is recorded either way.
2. Retention matrices and AA/F/BWT/FWT metrics are produced for every reference sequence, unit-tested against hand-computed values (A4).
3. A knowledge-edit batch achieves ballpark success and locality, and an interleaved edit step appears in a retention matrix; edit survival through the serving round trip is measured (A5).
4. The task-incremental and time-incremental benchmark sequences run end to end with naive-versus-method reports published (A6).
5. Both alignment reference sequences (align-then-train, train-then-align) produce complete matrices with judge confidence intervals, answering their retention questions in a published direction (C1-C2).
6. Nano-scale Titans and HOPE train and track reference loss curves; the 300M architectural comparison including retention matrices is published; the self-editing harness produces a 10-round retention curve (B1-B3, gated on 1-5).
7. Every sequence retains its full checkpoint trajectory and matrix for the interpretability tier, and the post-step hook exists.

## References

- Chen et al. Continual Learning in Large Language Models: Methods, Challenges, and Opportunities. arXiv:2603.12658, 2026. The taxonomy this phase implements, including the continual-alignment stage.
- Kirkpatrick et al. Overcoming catastrophic forgetting in neural networks (EWC). PNAS, 2017.
- Li and Hoiem. Learning without Forgetting. 2016.
- Saha et al. Gradient Projection Memory for Continual Learning. ICLR 2021.
- Rafailov et al. Direct Preference Optimization. 2023.
- Meng et al. Mass-Editing Memory in a Transformer (MEMIT). 2022. Wang et al. WISE: Rethinking the Knowledge Memory for Lifelong Model Editing. 2024. Via EasyEdit, zjunlp/EasyEdit.
- TiC-LM: time-continual LM pretraining benchmark, 2025.
- Behrouz et al. Titans: Learning to Memorize at Test Time. 2025. Behrouz et al. Nested Learning: The Illusion of Deep Learning Architectures (HOPE). arXiv:2512.24695, 2025.
- Zweiger et al. Self-Adapting Language Models (SEAL). 2025.
- lucidrains/titans-pytorch and open nano-scale HOPE reproductions (pin exact commits at implementation time).

*Last updated: 2026-07-09*
