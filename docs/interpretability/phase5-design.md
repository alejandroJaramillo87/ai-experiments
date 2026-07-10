# Phase 5 product design: Research tracks

This document is the implementation-ready product design for Phase 5 of the
interpretability roadmap (docs/interpretability/roadmap.md, "Phase 5: Research
tracks"). It is written to be picked up by a coding agent in a fresh session
and implemented without re-deriving context: it records the decisions already
made, the interfaces to build, the contracts consumed from earlier phases,
and the tests that define done.

The roadmap defines what Phase 5 delivers; this document defines how. Where
the two disagree, this document wins and the roadmap should be corrected.

Phase 5 is unlike Phases 0 through 4. It is not a set of components to build
once; it is a standing research program run as 1-to-2-week sprints with pivot
as the default. The roadmap's development method rules are binding here:
research pulls infrastructure, stages are time-boxed rather than
completion-gated, exploration optimizes information gained per unit time, and
skepticism is the default posture toward positive results. A product design
that fully specified five research programs up front would violate those
rules. This document therefore splits Phase 5 into two layers with different
contracts:

1. **The harness (R1-R4): hard deliverables.** The sprint operating system —
   protocol, logs, results pipeline, and the checkpoint battery — is
   infrastructure with settled decisions, interfaces, and acceptance tests,
   exactly like a Phase 0 deliverable. It is built once and every sprint runs
   on it.
2. **The track charters (T1-T5): governed starting points.** Each research
   track gets a charter: the question, the assets it leverages, entry
   criteria, one or two concretely specified first sprints, a falsification
   plan, and kill-or-pivot criteria. A charter is a contract about *how the
   track is run and judged*, not about what it will find. Everything past the
   first sprints is deliberately unspecified; the sprint protocol, not this
   document, decides what happens next.

## Table of contents

- [Assumed contracts from earlier phases](#assumed-contracts-from-earlier-phases)
- [Cross-cutting conventions](#cross-cutting-conventions)
- [Design overview](#design-overview)
- [Build order](#build-order)
- [The harness](#the-harness)
  - [R1: Sprint protocol and lifecycle](#r1-sprint-protocol-and-lifecycle)
  - [R2: Research log, ideas backlog, and sprint records](#r2-research-log-ideas-backlog-and-sprint-records)
  - [R3: Results and publication pipeline](#r3-results-and-publication-pipeline)
  - [R4: Checkpoint battery](#r4-checkpoint-battery)
- [The track charters](#the-track-charters)
  - [T1: Mechanistic continual learning](#t1-mechanistic-continual-learning)
  - [T2: Auditing games as a discipline](#t2-auditing-games-as-a-discipline)
  - [T3: Probes in production](#t3-probes-in-production)
  - [T4: Reasoning-model interpretability](#t4-reasoning-model-interpretability)
  - [T5: Quantization feature survival](#t5-quantization-feature-survival)
- [Phase 5 acceptance](#phase-5-acceptance)
- [References](#references)

## Assumed contracts from earlier phases

This design assumes interpretability Phases 0 through 4 and training Phases 0
through 4 are complete and their exit checklists pass. The interfaces below
are consumed, not reimplemented. If an implementing session finds one missing
or shaped differently, it stops and reconciles against the owning design
document rather than building a parallel version.

| Contract | Source | What Phase 5 consumes |
| :-- | :-- | :-- |
| Workbench image and interp compose profile | Interp Phase 0 | docker/interp with sm_120 PyTorch, TransformerLens v3, nnsight, SAELens; make interp-shell; the concurrency policy's interactive/batch/serving-dependent modes |
| Experiments log convention | Interp Phase 1 | docs/interpretability/experiments/ one-file-per-project format with post-mortems; R2 extends it, it does not replace it |
| Neuronpedia stack, autointerp, attribution graphs | Interp Phase 2 | Feature browsing, Delphi autointerp through the local vLLM endpoint, circuit-tracer graphs |
| Steering deployment path and scorecards | Interp Phase 3 | repeng/SAE-row extraction, GGUF control-vector export, service wiring, tuned-system-prompt-baseline scorecards |
| Probe toolkit and async monitor | Interp Phase 3 | Probe training, artifact registry, black-box baselines, the serving-log replay job |
| Activation harvesting, SAE training, diffing | Interp Phase 4 | SAELens cache runners with capture manifests, SAEBench evaluation, diffing-toolkit runs |
| Model-organism recipes and auditing games | Interp Phase 4 | Organism training recipes, game protocol and scoring harness |
| Sequence loop, retention matrix, post-step hook | Training Phase 4 | Checkpoint trajectories per sequence step, retention_matrix.json, the post_step_hook the battery (R4) attaches to |
| Fine-tuning and eval stack | Training Phase 1 | train-sft, judge scoring, eval_ft scorecards, used by organism training and by judge-scored sprint measurements |
| Benchmark JSON conventions | Training Phase 0 D8 / inference 2.5 | schema_version, git SHA, hardware/software blocks; every quantitative Phase 5 result reuses this envelope |
| Inference services | Existing | llama-cpu 8001, llama-gpu 8002, vllm-gpu 8003; control-vector loading; autointerp and judge batch workloads |
| Run lock and concurrency policy | Training Phase 0 D9 | Batch interp jobs take the same lock discipline as training jobs |

## Cross-cutting conventions

- Code lives under interp/ per the roadmap layout (probes/, steering/,
  organisms/, battery/) plus a new interp/sprints/ package for R1/R2 tooling.
  CLI entry points in scripts/interp/; tests in tests/interp/, runnable with
  make test; Makefile targets in an `##@ Interpretability` section.
- Style per .claude/PYTHON_STYLE.md, .claude/COMMENT_STYLE.md,
  .claude/BASH_OUTPUT_STYLE.md, .claude/DOCUMENTATION_STYLE.md. Python 3.12,
  black and ruff per pyproject. No emojis, plain technical prose.
- Heavy artifacts live under /mnt/ai-data/interp/ with capture manifests;
  everything published cites model hashes, seeds, and manifests per the
  roadmap's Standards for results.
- Interpretability runs at bf16; quantized artifacts are subjects (T5), never
  substrates.
- Every quantitative result is emitted as JSON in the D8-convention envelope
  and published under docs/interpretability/experiments/<sprint>/ alongside
  the sprint record.
- Time-boxes in this document are wall-clock working time, not calendar time.
  A sprint interrupted by other work pauses its clock; the sprint record
  notes the pause.

## Design overview

Phase 5 runs as a loop:

1. **Select** a sprint from the ideas backlog or a track charter's next step,
   using the R1 selection rubric.
2. **Register** it: a sprint record is created from the template before any
   experiment runs, capturing the question, the prediction, the time-box, and
   the planned falsification (informal pre-registration, per the roadmap).
3. **Run** it under the exploration tempo rules, appending to the research
   log, with cheap experiments first.
4. **Judge** it at the time-box: a written post-mortem and an explicit
   continue, pivot, or kill decision, with pivot as the default.
5. **Publish** anything that survives falsification through the R3 pipeline;
   feed anomalies and new questions back into the ideas backlog.

The five tracks are portfolios of sprints sharing a question and assets, not
projects with completion dates. A track is healthy when its sprints keep
producing information; it is killed or shelved by its charter's kill
criteria, and killing a track is a normal, recorded outcome.

The single genuinely new component Phase 5 builds is the checkpoint battery
(R4), and the roadmap's constraint on it is preserved: it is assembled
incrementally, admitting only measurements that earned a place in a completed
sprint. R4 ships the frame on day one and the contents over time.

## Build order

R1 protocol ── R2 logs and records ── R3 publication pipeline
                     │
                     └── R4 battery frame (contents admitted per sprint)

T1-T5 charters activate in any order once R1-R2 exist; each charter's entry
criteria gate it individually. The recommended first activations are T5 and
T2 (cheapest feedback loops, fully local ground truth), then T1 once a
training Phase 4 sequence trajectory worth studying exists, then T3 and T4.
This is a default, not a dependency: the selection rubric (R1) can override
it when a better-scoring sprint is available.

## The harness

### R1: Sprint protocol and lifecycle

**Tag:** build. **Depends on:** interp Phase 1 experiments-log convention.

#### Purpose

Make the roadmap's development method mechanical: a sprint cannot be started,
extended, or concluded except through the protocol, so discipline does not
depend on memory or mood.

#### Decisions

1. **Sprint definition.** A sprint is 1 to 2 weeks, has exactly one named
   owner question, one registered prediction, and one time-box set at
   registration (default 10 working days; hard ceiling 15, and raising past
   10 requires a written reason in the record). Mini-experiments inside a
   sprint follow the exploration tempo: roughly two hours of learning nothing
   means try a different approach; two days of learning nothing means
   seriously consider abandoning, and the record must show the consideration.
2. **Selection rubric**, applied when choosing the next sprint and recorded
   in the sprint record: (a) information value — what belief changes if the
   result comes out either way; (b) cheapness of the first experiment —
   preference for sprints whose first signal arrives within a day; (c) asset
   leverage — does it use something this repository uniquely has (own
   checkpoints, serving stack, benchmark discipline); (d) track balance — no
   more than three consecutive sprints in one track unless a continue
   decision justifies it. The rubric is a checklist, not a score; the record
   states the answers in a sentence each.
3. **Lifecycle states**: registered, running, post-mortem, closed
   (continue | pivot | kill), published (optional). State lives in the sprint
   record front matter (R2); a small CLI (scripts/interp/sprint.py) creates
   records from the template, lists open sprints, and refuses to register a
   new sprint while more than one is running (single-owner lab; two running
   sprints means one is being neglected).
4. **Continue-or-pivot is written, never implicit.** At the time-box the
   post-mortem must be written before any new sprint is registered; the CLI
   enforces this by refusing registration while a sprint is in running state
   past its deadline. Continue decisions must name what specifically changed
   from the default of pivot.
5. **Falsification is a first-class lifecycle step.** Any sprint whose
   headline observation is positive must record at least one deliberate
   falsification attempt (a designed-to-break-it experiment, a baseline the
   result must beat, or an ablation) before the post-mortem can conclude
   confirmed. Results that skip this can only be recorded as unverified and
   cannot enter R3 publication.

#### Interfaces

- make sprint-new / sprint-list / sprint-close wrapping scripts/interp/sprint.py.
- Template at docs/interpretability/experiments/_template.md.
- Protocol text itself lives at docs/interpretability/sprints.md (one page;
  the rules above, stated normatively).

#### Acceptance

- The CLI creates a record from the template with registered state, refuses a
  second concurrent registration, and refuses registration while an overdue
  sprint lacks a post-mortem (unit tests with a temp docs tree).
- A dry-run sprint (any small real question) is taken through all lifecycle
  states and its record validates against the template's required fields.
- The protocol doc exists and the roadmap's Phase 5 section links to it.

### R2: Research log, ideas backlog, and sprint records

**Tag:** build. **Depends on:** R1.

#### Purpose

The memory of the research program: what was tried, what was learned per unit
time, and where the next questions come from. Phase 1 established the
experiments log; R2 hardens it into three named artifacts with checkable
shape.

#### Decisions

1. **Sprint record** (extends the Phase 1 per-project file):
   docs/interpretability/experiments/YYYY-MM-<slug>.md with YAML front matter
   (state, track, time_box_days, started, closed, verdict:
   confirmed | refuted | inconclusive | unverified, decision:
   continue | pivot | kill) and required sections: Question, Prediction
   (pre-registered), Method, Results, Falsification attempts, Post-mortem
   (including learning-per-unit-time honesty), Next questions. Negative and
   inconclusive results use the identical template; there is no separate
   diminished format for them.
2. **Research log**: docs/interpretability/log.md, append-only dated entries
   during running sprints. Free-form by design; the only rule is that plots
   and numbers referenced in a sprint record must be reachable from the run
   directory it names.
3. **Ideas backlog**: docs/interpretability/ideas.md, a flat list where each
   idea carries a date, a source (which sprint or reading produced it), and a
   one-line information-value note. Sprint selection (R1) draws from here;
   the post-mortem template prompts for backlog additions so anomalies are
   captured at the moment they are cheap to record.
4. **Validation, not bureaucracy.** A single checker
   (scripts/interp/check_records.py, wired into make lint) validates front
   matter fields and required sections across all sprint records. Nothing
   else about the writing is machine-enforced.

#### Acceptance

- The checker passes on the template and the dry-run record, and fails with
  named errors on a fixture missing Prediction or Falsification sections.
- The backlog and log exist with their conventions stated in a header
  comment, seeded with the Phase 1 ideas document's content migrated in.

### R3: Results and publication pipeline

**Tag:** orchestrate. **Depends on:** R1, R2; the curiosity-chronicles blog
shell.

#### Purpose

Turn confirmed sprint results into public write-ups that meet the roadmap's
Standards for results, with the standards enforced as a checklist rather than
an aspiration, and with the quantitative payload in the repository's
benchmark-JSON discipline.

#### Decisions

1. **Publication gate.** Only sprints with verdict confirmed (or a negative
   result worth publishing, which the standards treat identically) enter the
   pipeline. The gate is the standards checklist rendered as a literal
   checklist in the write-up's PR/commit: real baseline named, random (not
   curated) examples, ablations for composite methods, pre-registration
   status stated, qualitative and quantitative halves both present,
   reproducibility bundle (code path, seeds, manifests, model hashes) linked.
2. **Two output forms per publication**: (a) the technical record in this
   repository — docs/interpretability/experiments/<sprint>/ gains the result
   JSON (D8 envelope) and any figures; (b) the blog post in
   curiosity-chronicles, written from the sprint record, linking back to the
   repository record for reproducibility. The blog is prose; the repository
   is the citable artifact.
3. **JSON shape.** One result file per headline claim: the D8 envelope plus a
   claim block (claim string, verdict, baseline description, effect size with
   uncertainty where applicable, falsification attempts count) and a payload
   block whose schema is claim-specific but versioned. A tiny schema checker
   extends check_records.py.
4. **No publication debt.** A confirmed sprint either publishes within one
   sprint-length of closing or records in its post-mortem why not; the
   sprint-list CLI surfaces confirmed-but-unpublished records.

#### Acceptance

- A dry-run publication (may use a trivial real result from the R1 dry-run
  sprint) produces the repository record with valid JSON and a draft blog
  post in the blog's archetype format.
- The checklist gate is part of the write-up template and the checker fails a
  write-up missing checklist items.

### R4: Checkpoint battery

**Tag:** build. **Depends on:** R1-R3; training Phase 4 post_step_hook and
retention matrix; interp Phase 3 probe toolkit; interp Phase 4 diffing.

#### Purpose

The fixed suite of mechanistic measurements that runs automatically after
every continual-learning increment, producing a mechanistic retention report
beside the training tier's behavioral one. The roadmap ordered it built last
for a reason this design preserves: the battery's *frame* is infrastructure
and ships now; its *contents* are earned, one measurement at a time, by
completed sprints.

#### Decisions

1. **Frame now, contents by admission.** interp/battery/ provides a runner
   (scripts/interp/battery.py) that takes a checkpoint (or a training Phase 4
   trajectory directory), executes every registered measurement, and writes
   battery_report.json (D8 envelope; one entry per measurement with value,
   reference value where defined, and pass/warn/fail against
   measurement-owned thresholds) plus a rendered battery_report.md. It
   registers into training Phase 4's post_step_hook via a thin adapter script
   so sequences invoke it automatically; standalone invocation over an
   existing trajectory is equally supported for post-hoc scoring.
2. **Measurement interface**: a registered class with prepare(subject),
   run(subject) -> MeasurementResult, cost tier (fast: minutes on CPU or a
   single GPU pass; slow: requires harvesting), and a provenance field naming
   the sprint record that admitted it. The runner runs fast measurements
   always and slow ones only with --full, so the post-step hook stays cheap
   enough to leave enabled.
3. **Admission rule (binding).** A measurement enters the battery only when a
   closed sprint demonstrates it detects a known intervention (an organism,
   a deliberate fine-tune, a knowledge edit) with a stated
   false-positive/false-negative characterization. The provenance field makes
   this auditable; a measurement without a sprint record does not merge.
4. **Seed candidates, not seed contents.** Three measurement families are
   pre-identified as first admission candidates because earlier phases
   already validated their machinery: (a) the Phase 3 probe suite replayed
   per checkpoint (probe accuracy trajectories); (b) logit-lens divergence
   between consecutive checkpoints on a fixed prompt battery; (c) a
   diffing-toolkit summary statistic between step k and step k-1. Each still
   requires its admission sprint (T1's first sprints below are designed to
   provide them).
5. **Subjects and phasing.** Battery runs on bf16 HF checkpoints in the
   workbench container under the batch concurrency mode and the training run
   lock; it never runs concurrently with training. The report lands in the
   sequence's run directory beside retention_matrix.json, and the JSON's keys
   are stable so the training tier and future dashboards can join the two.

#### Acceptance

- The frame runs on a real Phase 4 trajectory with a stub measurement and
  produces valid reports per step via the post_step_hook, and post-hoc over
  the same trajectory with identical values (determinism given fixed seeds).
- The runner refuses to register a measurement whose provenance sprint record
  does not exist or is not closed (unit test).
- After T1's admission sprints: at least two real measurements are in the
  battery, each with provenance, thresholds, and a demonstrated detection of
  a known intervention.

## The track charters

Each charter binds: the question, the assets, entry criteria, the first one
or two sprints (specified to R1 registration level: question, prediction,
first experiment, falsification, time-box), and kill-or-pivot criteria.
Beyond the first sprints, the protocol governs. All tracks assume the earlier
phases' components per the contracts table; entry criteria list only what
must additionally be true or verified.

### T1: Mechanistic continual learning

**Question.** Across a training Phase 4 sequence trajectory, what physically
happens to prior capability: does forgetting destroy representations or only
their readout, when do edited/learned facts become probeable, and does replay
preserve mechanism or just behavior?

**Assets.** Full checkpoint trajectories with retention matrices (training
Phase 4 keeps every step checkpoint for exactly this purpose); the diffing
toolkit; the probe toolkit; the naive-3task baseline sequence, which is a
ground-truth forgetting event on demand.

**Entry criteria.** At least one completed Phase 4 sequence with a positive
forgetting measurement (the naive-3task baseline qualifies); battery frame
(R4) merged.

**First sprint (also R4 admission sprint 1).** Question: on the naive-3task
trajectory, do linear probes for task-1 competence decay in step with the
behavioral task-1 score, or do probes stay accurate while behavior degrades
(readout, not representation, destroyed)? Prediction (registered): probe
accuracy decays substantially less than behavioral accuracy. First
experiment: train task-1 probes on the step-1 checkpoint's activations,
score them frozen on each later checkpoint. Falsification: shuffled-label
probes and a black-box baseline per the standards; a probe that "survives"
must beat both. Time-box: 10 days. If confirmed either way, the probe
trajectory measurement is proposed for battery admission.

**Second sprint (R4 admission sprint 2).** Same trajectory, diffing-toolkit
and logit-lens divergence between consecutive steps: does the mechanistic
diff signal spike at the step where behavioral forgetting occurs, i.e. can
diffing detect forgetting without running the evaluations? Falsification: the
signal must separate the forgetting step from a same-size continued-training
step that causes no forgetting.

**Kill/pivot criteria.** Two consecutive sprints in which neither probes nor
diffs distinguish forgetting steps from control steps better than chance:
shelve the track until a different trajectory type (e.g. sub-stage B
architectures) exists. Positive results feed the battery and continue.

### T2: Auditing games as a discipline

**Question.** As implanted-organism complexity escalates (secret word, false
belief, hidden objective), which interpretability techniques keep winning the
auditing game, and where does each technique's ceiling sit?

**Assets.** Phase 4 organism recipes and game protocol; the local
train-and-audit loop that makes a new organism an overnight job; Neuronpedia
and autointerp for the auditor side.

**Entry criteria.** One organism recipe validated end to end in Phase 4 (its
acceptance criterion).

**First sprint.** Question: run the full game ladder at level 1 (secret word
organism) with three auditor conditions — black-box prompting only, probes +
max-activating examples, SAE features + autointerp — under a fixed
time-per-audit budget; which conditions recover the secret? Prediction:
white-box conditions win at level 1 but the margin over black-box is small.
Falsification: a decoy organism (trained identically but with no secret) must
produce a no-finding from each auditor condition — an auditor that "finds"
secrets in the decoy is measuring itself. Time-box: 10 days.

**Kill/pivot criteria.** If organism training becomes the bottleneck (more
than half a sprint spent on training-tier debugging rather than auditing),
pause the track and file the issues against the training tier. If black-box
auditing wins every level tried, that is a publishable negative result for
white-box methods at this scale, published per R3, then pivot.

### T3: Probes in production

**Question.** Are activation probes over real serving traffic a competitive,
maintainable monitoring layer — measured honestly against black-box
classification of outputs — and how do they drift as traffic and models
change?

**Assets.** The Phase 3 probe toolkit and async monitor (already producing
reports over serving-log windows); real serving logs from daily use of the
inference services; the judge machinery for labeling.

**Entry criteria.** Phase 3 monitor acceptance passed; a serving-log window
of meaningful size (set at registration; the sprint record states the window
and traffic volume so results are labeled honestly).

**First sprint.** Question: for one operationally useful property (candidate
default: prompt-injection-style instruction override attempts in inputs;
choose at registration and record why), does a probe trained on captured
activations beat a black-box output classifier and a regex baseline on a
held-out, hand-labeled slice of real traffic? Prediction: the probe wins on
recall at matched precision. Falsification: the black-box baseline is tuned
with the same effort budget as the probe (the standards' no-sandbagging
rule); temporal split, not random split, for held-out data. Time-box: 10
days.

**Second sprint (drift).** Rescore the same frozen probe over consecutive
weekly windows; quantify drift and whether a cheap recalibration (threshold
only) restores it. This is the maintenance question monitoring viability
hangs on.

**Kill/pivot criteria.** If local traffic volume cannot support honest
evaluation (the labeled slice is too small for a stable estimate, stated as a
confidence interval in the record), pivot to synthetic-plus-replayed traffic
and label the results accordingly, or shelve until volume exists. If the
black-box baseline wins twice at matched effort, publish the negative and
shelve.

### T4: Reasoning-model interpretability

**Question.** For a local reasoning model (Qwen3 small variants with
thinking mode), do chain-of-thought-era methods — resampling, sentence-level
interventions, black-box variation — reveal whether stated reasoning is
load-bearing for answers?

**Assets.** Qwen3 in both reasoning and non-reasoning modes served locally;
the workbench for token-level interventions; judge scoring for
answer-quality measurement. This track is the least dependent on own
infrastructure and the most exposed to the field moving fast; its charter is
correspondingly the lightest.

**Entry criteria.** Qwen3 small variant runs hooked in the workbench with
correct chat/thinking formatting verified (the roadmap's warning about quiet
token-format corruption applies doubly to thinking tags; the verification is
part of the first sprint's day one).

**First sprint.** Question: replicate a published resampling result at small
scale — for a fixed problem set, ablate or resample individual CoT sentences
and measure answer flip rates; do late sentences matter more than early ones,
and does paraphrasing a load-bearing sentence (meaning preserved) also flip
answers? Prediction: registered at sprint start against the specific
replication target chosen. Falsification: paraphrase controls distinguish
content-dependence from token-dependence. Time-box: 10 days.

**Kill/pivot criteria.** This is an exploration track: two sprints without a
question that leverages local assets better than a rented notebook would,
and the track is shelved in favor of tracks where the repository's serving
and training tiers are load-bearing.

### T5: Quantization feature survival

**Question.** Semantically, what does Q4-class quantization destroy? Which
features that bf16 SAEs and probes detect survive in the quantized compute
graph the services actually run, measured beyond perplexity — feeding
llama-zen-turbo and the blog, where no good public data exists.

**Assets.** The GGUF quantization path and served quantized models; bf16
SAEs (pretrained Gemma Scope for the reference model; own SAEs from Phase 4
for own checkpoints); probes; the benchmark discipline for matched-workload
comparison. Constraint recorded honestly: llama.cpp has no activation read
path, so quantized-model internals are measured by running the quantized
weights dequantized-in-PyTorch (same numbers the GGUF kernels consume, bf16
compute) or behaviorally through the served endpoint; the sprint records
must state which, since kernel-level rounding in compute is not captured by
the first method.

**Entry criteria.** A model with both a bf16 SAE/probe set and a served
GGUF at two or more quant levels (Gemma-2-2B with Gemma Scope plus Q8_0/Q4_K_M
GGUFs qualifies and is the reference subject per model policy).

**First sprint.** Question: for a sample of well-explained Gemma Scope
features (stratified by activation frequency), compare feature activation
patterns between the bf16 model and quantized-weight variants on a fixed
corpus; what fraction of features keep their max-activating example sets and
firing statistics at Q8 versus Q4? Prediction: Q8 near-lossless; Q4 shows a
measurable, frequency-skewed feature death rate. Falsification: a
seed-variation control (two bf16 runs with different batch orders) bounds the
measurement noise the quant deltas must exceed; random-feature samples, not
cherry-picked interpretable ones, per the standards. Time-box: 10 days.

**Second sprint.** Probe survival: do Phase 3 probes trained on bf16
activations still work on quantized-weight activations, and does retraining
on quantized activations recover the gap? Direct input to the T3 monitor's
validity story and to steering-vector deployment (Phase 3 already showed
bf16-extracted vectors steer quantized models; this measures why).

**Kill/pivot criteria.** If measurement noise (seed control) swamps quant
effects at every level, that is itself the publishable result — publish and
close. The track is naturally finite: once the Q4/Q8/NVFP4 survival table
exists for one or two models (coordinating with inference task 2.3's quant
matrix), it hands off to the blog and closes rather than accreting variants.

## Phase 5 acceptance

Phase 5 has no exit; it has a health bar. The phase is accepted as
*operational* when:

1. R1-R3 acceptance tests pass and one dry-run sprint has traversed the full
   lifecycle including a published record (R1, R2, R3).
2. The battery frame runs on a real trajectory via the post_step_hook, and at
   least two measurements have been admitted with sprint provenance (R4, via
   T1).
3. At least three tracks have completed their first sprint with a closed,
   checker-valid record — including at least one negative or inconclusive
   verdict written up at full quality (if all early results are positive, the
   skepticism rule says look harder).
4. At least one publication has shipped through R3 to the blog with its
   repository-side JSON record.
5. The ideas backlog contains entries sourced from completed sprints —
   evidence the loop feeds itself.

Ongoing health, reviewed at each continue-or-pivot decision rather than
audited: no sprint runs past its ceiling, no confirmed result carries
publication debt, and the battery contains no measurement without
provenance.

## References

- Nanda. How To Become A Mechanistic Interpretability Researcher. Alignment
  Forum, 2025. The development method R1 mechanizes.
- Marks et al. Auditing Language Models for Hidden Objectives, 2025 (T2).
- Arditi et al. Refusal in Language Models Is Mediated by a Single
  Direction, 2024; Marks and Tegmark, The Geometry of Truth, 2023 (T3
  lineage).
- science-of-finetuning/diffing-toolkit; SAELens v6; SAEBench (T1, T5
  instruments).
- Chen et al. Continual Learning in Large Language Models, 2026 — the
  training Phase 4 taxonomy T1 instruments mechanistically.
- CoT-faithfulness and resampling literature per the roadmap's
  reasoning-model track note (pin specific replication targets at T4 sprint
  registration).

*Last updated: 2026-07-10. Written against the interpretability roadmap and
the training Phase 0-4 product designs; the harness (R1-R4) is buildable
immediately, the charters activate per their entry criteria.*
