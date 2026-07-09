# Local mechanistic interpretability roadmap

This document defines the scope for adding mechanistic interpretability to this
repository as a third tier alongside the existing inference services and the
planned training infrastructure. The three tiers close a single loop on one
workstation: models trained here are served by the inference services and
understood by the interpretability tier, and what the interpretability tier
finds deploys back into serving as steering vectors and activation probes. The
document records what is adopted versus built as of July 2026, how the tiers
integrate, and a roadmap whose ordering follows research practice rather than
infrastructure completeness.

This is a planning document. No interpretability code is implied to exist yet.

## Table of Contents

- [Purpose](#purpose)
- [Development method](#development-method)
- [Findings as of July 2026](#findings-as-of-july-2026)
- [Hardware envelope](#hardware-envelope)
- [Model policy](#model-policy)
- [Concurrency policy](#concurrency-policy)
- [Storage layout and retention](#storage-layout-and-retention)
- [Design principle: adopt versus build](#design-principle-adopt-versus-build)
  - [Adopt](#adopt)
  - [Build](#build)
  - [Directions avoided](#directions-avoided)
- [Integration with the inference and training tiers](#integration-with-the-inference-and-training-tiers)
- [Standards for results](#standards-for-results)
- [Roadmap](#roadmap)
  - [Phase 0: Workbench and fundamentals](#phase-0-workbench-and-fundamentals)
  - [Phase 1: Mini-project cycle](#phase-1-mini-project-cycle)
  - [Phase 2: Platform services](#phase-2-platform-services)
  - [Phase 3: Write-back path](#phase-3-write-back-path)
  - [Phase 4: Own-model interpretability](#phase-4-own-model-interpretability)
  - [Phase 5: Research tracks](#phase-5-research-tracks)
- [Proposed repository layout](#proposed-repository-layout)
- [References](#references)

## Purpose

Provide a reproducible local environment to understand what happens inside
language models mechanistically, on the same single-node workstation that
trains and serves them. The interpretability additions target three outcomes
at once, mirroring the pillars of the training scope:

1. **Understand our own models.** Every checkpoint the training tier produces
   is a subject: what did fine-tuning change, what does catastrophic
   forgetting physically look like, did a continual-learning increment create
   new features or repurpose old ones. Interpretability is the observability
   layer that loss curves and behavioral evaluations do not provide.
2. **Deploy understanding back into serving.** Findings compile down to small
   artifacts that run in or alongside the production inference services:
   steering vectors exported as GGUF control vectors for llama.cpp, and linear
   probes that score activations as runtime monitors. Weights are never
   modified; the write-back path acts on activations only.
3. **Learn the craft of interpretability research.** The tier doubles as a
   working environment for developing real research skills, following the
   staged method below. Public write-ups of completed work are first-class
   deliverables and feed the blog.

The field's own trajectory supports the pragmatic focus. The current consensus
among leading practitioners is that ambitious whole-model reverse engineering
is not the productive aim; pragmatic uses of model internals, probes for
monitoring, steering for control, model diffing for training diagnostics, are
where interpretability has produced real results. This scope is built around
that pragmatic core.

## Development method

The roadmap's ordering discipline comes from Nanda's guide to becoming a
mechanistic interpretability researcher, adapted to a repository whose
instinct is to build infrastructure first. The guide's core claim is that in
this field, preparation displaces research: the correct move is to learn the
minimum viable basics, then learn by doing, pulling tools and knowledge in
when a concrete project demands them. Four rules follow, and they govern
everything below.

1. **Research pulls infrastructure; infrastructure never waits for research.**
   A component from the adopt list is deployed in the week a project first
   needs it, not on a schedule. Only Phase 0 is deployed unconditionally, and
   it is deliberately minimal. The component inventory in this document is a
   menu, not a build queue.
2. **Time-boxed stages, not completion-gated ones.** Phase 0 (learning the
   ropes) is capped at one month. Phase 1 (throwaway mini-projects of 1 to 5
   days) runs 2 to 4 weeks. Later work proceeds in 1-to-2-week sprints, each
   ending in a written post-mortem and a deliberate continue-or-pivot
   decision, with pivot as the default. This inverts the training roadmap's
   dependency ordering on purpose: research skills, not components, are the
   critical path.
3. **Exploration has a tempo.** During exploration the aim is information
   gained per unit time: a new plot every few minutes when experiments are
   cheap, a different approach after roughly two hours of learning nothing,
   and serious consideration of abandoning the problem after two days of
   learning nothing. A research log is kept throughout; anomalies and idle
   curiosities go into a running ideas document that later sprints draw from.
4. **Skepticism is the default posture toward positive results.** Exciting
   results are more likely than average to be false. Every claimed finding
   gets a deliberate falsification attempt before it gets a write-up, and the
   standards in [Standards for results](#standards-for-results) apply to
   anything published.

## Findings as of July 2026

The ecosystem survey that shaped this scope, current as of July 2026:

- **The tooling consolidated around one compatible stack.** Decode Research
  maintains both SAELens (v6, training and analysis of sparse autoencoders,
  no longer requiring TransformerLens) and Neuronpedia, which is open source
  and self-hostable: webapp, Postgres, a local inference server for
  activation testing and steering, an autointerp server based on EleutherAI's
  Delphi, and a graph server wrapping Anthropic's circuit-tracer for
  attribution graphs. TransformerLens v3 (TransformerBridge, roughly 9,000
  models) and nnsight 0.6 with nnterp cover hooked execution.
- **Pretrained artifacts remove most training needs.** Gemma Scope
  (Gemma-2-2B), Llama Scope (Llama-3.1-8B), and Goodfire's open SAEs are
  downloadable; SAE training is only required for models nobody else has,
  which in this repository means our own checkpoints.
- **Model diffing is implemented, not frontier.** The diffing-toolkit from
  the science-of-finetuning group provides nine methods, including
  crosscoders, SAE-on-differences, KL divergence, and diff mining, with
  activation caching and an interactive dashboard, aimed exactly at
  base-versus-fine-tuned comparison.
- **The write-back path is native to the serving stack.** llama.cpp applies
  control vectors in GGUF format at load time (`--control-vector`,
  `--control-vector-scaled`), and repeng trains and exports them. Steering
  acts on the float residual stream, so a vector extracted from the bf16
  model applies to the quantized GGUF the services actually run.
- **The genuine gaps are small and specific.** No standard toolkit wires
  probes into a production serving stack as monitors; llama.cpp has no
  activation read path beyond the imatrix callback (write exists, read does
  not); longitudinal diffing across continual-learning checkpoint
  trajectories is thin; and nobody publishes semantic (rather than
  perplexity-based) measurements of what quantization destroys. These gaps
  align with what this repository uniquely has: a training tier, a serving
  tier, and reproducible-benchmark discipline.

## Hardware envelope

| Component | Specification | Primary interpretability role |
|-----------|---------------|-------------------------------|
| GPU | RTX 5090, 32GB VRAM, Blackwell | Hooked bf16 forward passes, activation harvesting, SAE and crosscoder training, steering-vector extraction |
| CPU | AMD Ryzen 9950X, 16 cores, 32 threads | Probe training and scoring, platform services (webapp, Postgres), async monitor jobs, dataset preparation |
| RAM | 128GB DDR5-6000 | Activation shuffle buffers, platform stack residency, pinned host buffers for device-to-host capture |
| Storage | NVMe at `/mnt/ai-data` | Activation store (the binding constraint), SAE and probe artifacts, platform database |

Two policies fall out of the envelope:

- **Interpretability runs at bf16.** Quantized models are objects of study
  (the quantization research track), never the substrate; interpreting a Q4
  artifact studies the rounding, not the model. On 32GB, Gemma-2-2B at bf16
  leaves room for dozens of attached SAEs; Llama-3.1-8B at bf16 fits with
  headroom for batched runs.
- **Local hardware exceeds the field's requirements.** The standard curricula
  assume an 8-to-12GB rented GPU. The one compatibility obligation is
  Blackwell: the workbench image installs a PyTorch built for sm_120 (CUDA
  12.8 or later) first and never lets a pinned requirements file downgrade
  it. Cloud rental is relevant only for burst scale (release-grade SAE
  sweeps on 8B-plus models), which is outside this scope.

Throughput expectations that size the phases: batched bf16 forward passes for
harvesting run in the tens of thousands of tokens per second on Gemma-2-2B
(roughly 3x slower for Llama-3.1-8B); one residual-stream layer of Gemma-2-2B
is about 4.6KB per token at bf16, so 100 million tokens is roughly 460GB per
layer; a respectable SAE on a 2-to-3B model trains on 200 to 500 million
cached tokens in hours. The full harvest-train-evaluate loop for one
checkpoint is an overnight job, structurally identical to how benchmarks
follow optimization work today. Autointerp is the sleeper cost: explaining
even a few thousand SAE features is tens of millions of explainer-LLM tokens,
an embarrassingly parallel batch workload that the vLLM service exists for.

## Model policy

| Model | Role | Rationale |
|-------|------|-----------|
| Gemma-2-2B | Reference anchor | Best-supported model in the ecosystem: Gemma Scope SAEs, circuit-tracer support, Neuronpedia presets, most published baselines. Separates "my tooling is broken" from "my model is weird" |
| Qwen3 (small variants) | Modern and reasoning-model subject | Current default open family; reasoning and non-reasoning modes enable the neglected reasoning-model interpretability track |
| Llama-3.1-8B | Upper-bound reference | Llama Scope pretrained SAEs; top of the comfortable bf16 envelope |
| Own checkpoints (sub-6B) | Primary subjects | The training tier's outputs: fine-tunes, continual-learning trajectories, and model organisms. The reason this tier exists |

Reference models exist so that tooling questions and research questions are
never debugged simultaneously. Watch for tokenizer and chat-format mismatches
when switching families; wrong token formatting corrupts results quietly.

## Concurrency policy

The interpretability tier follows the training scope's phased-not-concurrent
rule, with one addition: interpretability work comes in two shapes.

- **Interactive work** (workbench sessions, platform browsing, steering
  experiments) is bursty. The interpretability services are down by default
  and started with an explicit make target, with the GPU inference services
  quiesced. The llama-cpu service can stay up on its dedicated cores.
- **Batch work** (activation harvesting, SAE and crosscoder training,
  SAEBench evaluation, diffing runs) owns the whole machine like a training
  job and follows the training tier's job conventions.
- **Serving-dependent work** runs with the inference services up and the
  interpretability GPU work idle, mirroring the training tier's
  synthetic-data phase: autointerp explanation batches driven through the
  vLLM service, probe scoring replays over serving logs, and LLM-as-judge
  scoring of steering experiments.

The Neuronpedia webapp defaults to port 3000, which collides with open-webui;
it is remapped in the compose profile.

## Storage layout and retention

Interpretability artifacts follow the training tier's convention: the
repository holds code, configuration, and documentation; heavy data lives on
NVMe under `/mnt/ai-data`.

```
/mnt/ai-data/
  models/                    # existing: served model weights and GGUF files
  training/                  # training-tier layout, defined in docs/training/scope.md
  interp/
    activations/<capture>/   # sharded activation tensors plus manifest
    saes/<run>/              # trained SAE and crosscoder weights and configs
    probes/                  # probe weight vectors and evaluation reports
    vectors/                 # control vectors in GGUF, with extraction recipes
    neuronpedia/             # platform database volume and imported sources
```

Every capture directory carries a manifest recording model hash, layers,
dataset, tokenizer, dtype, and position count, so every downstream artifact
is traceable to its inputs. Retention is asymmetric: activations are a
regenerable cache with a budget of 1 to 2TB, deleted once the artifacts
trained from them are validated; SAEs, probes, control vectors, and manifests
are small and kept indefinitely. The existing disk-free watchdog floor
applies.

## Design principle: adopt versus build

Roughly 90 percent of this tier is adoption, configuration, and
orchestration; the built remainder is thin glue that depends on the combined
train-serve-understand lifecycle of this repository. The test is the same as
the training scope's: shared problems are adopted, specific problems are
built. Writing a custom SAE trainer, hooked-execution library, feature
dashboard, or diffing method is an anti-pattern, a maintenance treadmill
against an actively funded upstream, and in the SAE case against a research
direction the field itself considers past its incremental phase.

### Adopt

| Component | Purpose | Deployed when |
|-----------|---------|---------------|
| TransformerLens v3 | Hooked model execution for small models; the curriculum's library | Phase 0 |
| nnsight 0.6 + nnterp | Hooked execution on plain Hugging Face models; standardized method interface | Phase 0 |
| ARENA curriculum (chapters 1.1, 1.2, 1.4.1, 1.3.2) | Learning the ropes; commissioning test for the workbench | Phase 0 |
| Gemma Scope, Llama Scope, Goodfire SAEs | Pretrained feature dictionaries; remove any need to train SAEs on reference models | Phase 1, first project that uses an SAE |
| Neuronpedia (self-hosted) | Feature dashboards, activation search, steering UI, imported SAE sources | Phase 2 |
| circuit-tracer (via Neuronpedia graph server) | Attribution graphs on Gemma-2-2B and Llama-3.2-1B class models | Phase 2 |
| Delphi (via Neuronpedia autointerp server) | Automated feature explanation, driven by the in-repo inference services | Phase 2 |
| repeng | Contrastive control-vector training and GGUF export | Phase 3 |
| SAELens v6 | SAE training and analysis for models without pretrained SAEs, i.e. our checkpoints | Phase 4 |
| SAEBench | Standardized evaluation of SAEs we train | Phase 4 |
| diffing-toolkit | Base-versus-fine-tuned model diffing: crosscoders, SAE-difference, KL, diff mining, dashboard | Phase 4 |
| lm-evaluation-harness, existing benchmark suite | Behavioral baselines for every intervention claim | Phase 3 onward, shared with training tier |

### Build

The differentiated remainder, in priority order. Each item depends on this
repository's lifecycle or hardware envelope and cannot be downloaded.

1. **Workbench image and compose profile.** `docker/interp/` with
   Blackwell-correct PyTorch, TransformerLens, nnsight, SAELens, and
   JupyterLab; an `interp` profile in the compose file with the port remaps
   and GPU reservations; `make interp-shell` and `make interp-up` targets.
2. **Experiments log convention.** `docs/interpretability/experiments/`, one
   file per mini-project or sprint: question, method, result, falsification
   attempts, verdict, post-mortem. Negative and inconclusive results are
   written up on equal terms. This is the tier's equivalent of the benchmark
   JSON convention.
3. **Probe toolkit and monitor job.** Thin code to train logistic-regression
   probes on captured activations, a registry of probe artifacts with
   evaluation reports, and an async monitor that replays serving logs
   through the workbench model in batches and emits per-probe score reports.
   No standard toolkit exists for the serving-integration half.
4. **Steering deployment path.** Scripts that take a validated direction,
   from repeng contrastive extraction or an SAE decoder row, package it as a
   GGUF control vector, wire it into a service definition, and run the
   baseline scorecard (see [Standards for results](#standards-for-results)).
5. **Model-organism recipes.** Training-tier recipes that implant a known
   property into a sub-6B model (a secret word, a false belief, a hidden
   objective via synthetic-document fine-tuning), plus the evaluation
   harness for auditing games against them. Reuses the training tier's
   fine-tuning path end to end; the interpretability-side footprint is the
   game protocol and scoring.
6. **Checkpoint battery.** A fixed suite of probes, logit-lens divergence
   measures, and behavioral spot checks that runs automatically after every
   continual-learning increment, producing a mechanistic retention report
   alongside the training tier's forgetting evaluation. Built last, only
   after Phase 5 sprints establish which measurements earn a place in it.
7. **llama.cpp activation read path (exploratory).** The write path exists
   (control vectors); a read path, a tensor tap for capture or probe hosting
   in llama-server built on the eval-callback mechanism, does not exist
   anywhere in the ecosystem. Kept exploratory: pursued only if the async
   monitor proves too limiting, and scoped as a study in
   `docs/optimizations/experiments/` first.

### Directions avoided

Recorded to prevent re-litigation, following the field's own post-mortems:

- **Ambitious whole-model reverse engineering** as a goal. Understanding is
  pursued to the depth that is pragmatically useful, not to completeness.
- **Incremental SAE research**: new SAE architecture variants, "SAEs applied
  to X" projects whose motivation is the technique rather than a question.
  SAEs are used as instruments inside investigations, and trained only for
  models that have no pretrained dictionaries.
- **Circuit analysis for its own sake**: identifying a sparse subgraph is no
  longer a contribution by itself; circuit methods are used when they serve
  a question about model behavior.
- **Toy algorithmic-task interpretability** (grokking-style): superseded as
  a learning path by work on real models, which this hardware runs easily.
- **Compute-heavy interpretability training** as a starter activity, e.g.
  cross-layer transcoder training. Attribution graphs are consumed through
  circuit-tracer's released artifacts instead.

## Integration with the inference and training tiers

The loops between tiers are the reason interpretability lives in this
repository rather than a separate one.

| Direction | Mechanism |
|-----------|-----------|
| Training to interpretability | Every checkpoint is a subject: diffing against its base, probe batteries across continual-learning trajectories, SAEs trained on our own models. Model-organism recipes turn the fine-tuning path into a ground-truth generator for auditing games |
| Inference to interpretability | Delphi autointerp and LLM-as-judge scoring run against the OpenAI-compatible endpoints of llama-gpu and vllm-gpu, so feature explanation is a local batch workload. Serving logs are the input corpus for probe monitoring |
| Interpretability to inference | Validated steering vectors export as GGUF control vectors and load into the llama.cpp services as configuration (`--control-vector`), applying to the served quantized weights without retraining. Probe monitors report on production traffic asynchronously |
| Interpretability to training | Diffing and battery results inform training decisions: replay fractions, regularization strength, and when a continual-learning increment has done mechanistic damage that behavioral evaluations have not yet surfaced |

## Standards for results

Any result that leaves the experiments log for a public write-up meets the
field's good-science bar, which is this repository's benchmark discipline
translated to interpretability:

- **Real baselines, not sandbagged.** A steering vector claim is measured
  against a tuned system-prompt baseline; a probe claim against black-box
  classification of the output; a fancy method against the simple method.
- **No cherry-picking.** Qualitative claims ship with randomly selected
  examples, not curated ones, and datasets get random-sample appendices.
- **Ablations on anything with moving parts.** Each component of a composite
  method is removed once to check it matters.
- **Informal pre-registration.** The experiments log records which results
  were predicted before running and which are post-hoc.
- **Qualitative plus quantitative.** Read the data; then validate what was
  read with numbers. Write-ups missing either half are incomplete.
- **Reproducible.** Code, seeds, manifests, and model hashes accompany every
  write-up, matching the repository's benchmark convention.

## Roadmap

Phases 0 and 1 are time-boxed and unconditional. Phases 2 through 4 are
capability gates, not calendar entries: each deploys when a project pulls it,
and the phase is complete when its acceptance criteria have been met by real
project use. Phase 5 is the standing research program. Deliverables are
tagged **[adopt]**, **[orchestrate]**, or **[build]** as in the training
scope.

### Phase 0: Workbench and fundamentals

Learning the ropes, capped at one month. The workbench is the only
infrastructure deployed on faith. **Depends on:** nothing.

**Deliverables**

1. **[build] Workbench image and profile.** `docker/interp/` with
   Blackwell-correct PyTorch pinned first, TransformerLens v3, nnsight,
   SAELens, and JupyterLab; `make interp-shell`; GPU reservation consistent
   with the concurrency policy.
2. **[adopt] ARENA essentials.** Chapters 1.1 (transformer from scratch),
   1.2 (interpretability basics, first three sections), 1.4.1 (activation
   patching), and 1.3.2 (SAE usage, skimmed), run inside the workbench,
   browser-LLM-as-tutor only.
3. **[orchestrate] Core-technique checklist.** Activation patching, linear
   probes, using a pretrained SAE, and max-activating dataset examples, each
   implemented by hand on GPT-2 Small; logit lens and steering as
   nice-to-have.
4. **[orchestrate] Steering exercise.** The canonical commissioning test: a
   happiness steering vector from 32 contrastive prompt pairs, applied
   during generation, scored by an LLM judge through a local service.

**Acceptance criteria**

- ARENA exercises run end to end on the local GPU with no cloud fallback.
- Each checklist technique has a working, hand-written notebook.
- The steering exercise measurably shifts judge scores.
- One month has elapsed, whichever comes first. Unfinished items become
  backlog, not blockers.

### Phase 1: Mini-project cycle

Two to four weeks of throwaway 1-to-5-day projects. The goal is practicing
exploration and falsification with fast feedback, not producing output. No
training runs, no SAE training, no new services. **Depends on:** Phase 0.

**Deliverables**

1. **[build] Experiments log convention.** The directory, template, and
   post-mortem format, exercised by every project in this phase.
2. **[orchestrate] Starter projects.** At least four from the standard
   menu, chosen for technique coverage: replicate and extend the
   refusal-direction result on a new model; replicate truth probes on a
   modern model and attempt to break them; find the secret in released
   taboo-model organisms using every technique available; explore released
   emergent-misalignment organisms; form a hypothesis from a public
   Neuronpedia attribution graph and verify it by prompting alone.
3. **[orchestrate] Research log and ideas document.** Running notes and the
   idea backlog that Phase 5 sprint selection draws from.

**Acceptance criteria**

- At least four post-mortems exist in the experiments log.
- At least one project includes a deliberate, documented falsification
  attempt of its own headline observation.
- No component from later phases was deployed unless a project pulled it,
  and the pull is recorded in the post-mortem.

### Phase 2: Platform services

The self-hosted platform, deployed when browsing features, steering
interactively, or reading attribution graphs becomes the bottleneck in
project work. **Depends on:** Phase 0; typically pulled during Phase 1.

**Deliverables**

1. **[orchestrate] Neuronpedia stack.** Webapp, Postgres, inference server,
   autointerp server, and graph server under the `interp` compose profile,
   ports remapped around open-webui, with Gemma Scope imported for
   Gemma-2-2B.
2. **[orchestrate] Autointerp through local serving.** Delphi pointed at the
   vLLM service's OpenAI-compatible endpoint, run under the
   serving-dependent concurrency mode.
3. **[orchestrate] Attribution graphs.** circuit-tracer generation and the
   Neuronpedia graph viewer working against a local model.

**Acceptance criteria**

- Feature dashboards, search, and the steering UI work against the imported
  sources on localhost.
- An autointerp batch completes using only local inference.
- An attribution graph generated locally renders in the viewer and supports
  a recorded mini-project.

### Phase 3: Write-back path

Interpretability output deployed into the serving tier, pulled by the first
project that produces a direction or probe worth operationalizing.
**Depends on:** Phase 0; benefits from Phase 2.

**Deliverables**

1. **[build] Steering deployment path.** Extraction (repeng or SAE decoder
   row), GGUF export, service wiring, and a scorecard comparing the steered
   service against the tuned system-prompt baseline on the benchmark suite,
   including a coherence regression check.
2. **[build] Probe toolkit.** Probe training on captured activations, the
   artifact registry, and evaluation reports with black-box baselines.
3. **[build] Async probe monitor.** The batch job that replays serving logs
   through the workbench model and emits per-probe reports; scheduled like a
   benchmark run.

**Acceptance criteria**

- A control vector extracted at bf16 measurably shifts behavior of the
  quantized served model, with the scorecard published either way, including
  the case where the system-prompt baseline wins.
- A probe beats its black-box baseline on held-out data, or the negative
  result is written up.
- The monitor produces a report over at least one real window of serving
  logs.

### Phase 4: Own-model interpretability

The heavy loop, pulled by the first training-tier checkpoint worth studying.
**Depends on:** Phase 0 and training-tier Phase 1 (fine-tuning); reuses
Phase 2 and 3 components where deployed.

**Deliverables**

1. **[orchestrate] Activation harvesting.** SAELens cache runners writing to
   the activation store under the manifest convention, sized by the
   throughput expectations in the hardware envelope.
2. **[orchestrate] SAE training and evaluation.** SAELens training on an
   own checkpoint from cached activations, evaluated with SAEBench, imported
   into the local platform for browsing and autointerp.
3. **[orchestrate] Model diffing.** diffing-toolkit runs comparing a
   fine-tuned checkpoint against its base, results recorded in the
   experiments log.
4. **[build] Model-organism recipes and auditing games.** The training-tier
   recipes that implant known properties, and the game protocol scoring
   which interpretability techniques recover them.

**Acceptance criteria**

- One overnight harvest-train-evaluate loop completes on an own checkpoint
  and the SAE is browsable locally.
- A diffing run localizes a known training intervention (the organism
  provides ground truth).
- An auditing game against an own organism is scored and written up,
  succeed or fail.

### Phase 5: Research tracks

The standing program, run as 1-to-2-week sprints with pivot-by-default,
drawing on the ideas backlog. Initial tracks, ordered by the leverage of
this repository's unique assets:

- **Mechanistic continual learning.** The checkpoint battery applied across
  continual-learning trajectories from training-tier Phase 4: when facts
  become probeable, whether forgetting destroys representations or readout,
  what replay preserves mechanistically, feature birth versus repurposing
  across increments. Longitudinal diffing across trajectories, as opposed to
  base-versus-final pairs, is thinly explored territory.
- **Auditing games as a discipline.** Escalating organism complexity from
  secret words to hidden objectives, tracking which techniques win at which
  difficulty; the local train-and-audit loop makes iteration cheap.
- **Probes in production.** The monitor track matured: probe libraries over
  real serving traffic, drift reports, and honest evaluation against
  black-box alternatives, in the spirit of the field's observation that
  probes are competitive for monitoring and underdeveloped as engineering.
- **Reasoning-model interpretability.** Qwen3-based exploration of
  chain-of-thought-era methods (resampling, sentence-level interventions,
  black-box variation), flagged by the field as neglected and growing.
- **Quantization feature survival.** Semantic measurement of what Q4 does to
  features that bf16 SAEs and probes detect, feeding the llama-zen-turbo
  work and the blog; no good public data exists.

Write-ups of completed sprints are the tier's public output and follow
[Standards for results](#standards-for-results).

## Proposed repository layout

Interpretability is added to the existing repository. Interactive surfaces
(workbench, platform) are compose services under an off-by-default profile;
heavy work is job-based like training. Heavy data lives under
`/mnt/ai-data/interp` as defined in
[Storage layout and retention](#storage-layout-and-retention).

```
ai-experiments/
  docker/
    llama-cpu/  llama-gpu/  vllm-gpu/                   # existing inference services
    training/                                           # planned: training image
    interp/                                             # new: workbench image (PyTorch sm_120, TransformerLens, nnsight, SAELens, Jupyter)
  interp/                                               # new: interpretability source
    probes/                                             # probe training, registry, monitor job
    steering/                                           # extraction, GGUF export, service wiring, scorecards
    organisms/                                          # organism recipes and auditing-game protocol
    battery/                                            # checkpoint battery (Phase 5, built last)
  scripts/
    interp/                                             # harvest, train-sae, diff, monitor entry points
  docs/
    interpretability/
      roadmap.md                                        # this document
      experiments/                                      # one file per mini-project and sprint, with post-mortems
  Makefile                                              # extended: interp-shell, interp-up, harvest, train-sae, probe-monitor, export-vector
  docker-compose.yaml                                   # extended: interp profile (workbench, Neuronpedia stack, remapped ports)
```

## References

- Nanda. How To Become A Mechanistic Interpretability Researcher. Alignment
  Forum, 2025. The development method of this document.
- ARENA 3.0 curriculum and setup. callummcdougall/ARENA_3.0.
- SAELens v6. decoderesearch/SAELens.
- Neuronpedia, open-source interpretability platform. hijohnnylin/neuronpedia.
- Anthropic. Open-sourcing circuit-tracing tools, 2025. safety-research/circuit-tracer.
- EleutherAI Delphi, automated interpretability.
- diffing-toolkit. science-of-finetuning/diffing-toolkit.
- SAEBench. adamkarvonen/SAEBench.
- repeng and llama.cpp control-vector support (ggml-org/llama.cpp PR 5970).
- TransformerLens v3. TransformerLensOrg/TransformerLens.
- nnsight and NDIF. ndif-team/nnsight.
- Gemma Scope: open sparse autoencoders on Gemma 2. Google DeepMind, 2024.
- Llama Scope: extracting millions of features from Llama-3.1-8B, 2024.
- Ferrando et al. A Primer on the Inner Workings of Transformer-based
  Language Models, 2024.
- Sharkey et al. Open Problems in Mechanistic Interpretability, 2025.
- Marks et al. Auditing Language Models for Hidden Objectives, 2025.
- Arditi et al. Refusal in Language Models Is Mediated by a Single
  Direction, 2024.
- Marks and Tegmark. The Geometry of Truth, 2023.
- Tilde Research. Activault, activation storage engine, 2025. Noted as the
  scale-out reference if harvesting ever outgrows local NVMe.

---
*Last Updated: 2026-07-09*
