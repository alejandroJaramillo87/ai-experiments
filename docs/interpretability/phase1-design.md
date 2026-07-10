# Interpretability Phase 1 product design: Mini-project cycle

This document is the implementation-ready product design for the interpretability
roadmap's "Phase 1: Mini-project cycle". It follows the same contract as the training
tier's phase product designs: each section can be picked up in a fresh session and
implemented without re-deriving context. It records the decisions already made, the
interfaces to build, the repository and ecosystem facts the design depends on, and
the tests that define done.

The interpretability roadmap defines *what* Phase 1 delivers; this document defines
*how*. Where the two disagree, this document wins and the roadmap should be corrected.

Phase 1 is two to four weeks of throwaway 1-to-5-day research projects. Its purpose
is practicing exploration and falsification with fast feedback, not producing output
and not building infrastructure. That inverts the usual shape of a product design,
so this document draws one line explicitly and every deliverable respects it:

- **The coding agent builds the scaffolding**: the experiments log convention, the
  research log and ideas document, a thin shared harness, and one prepared starting
  point per starter project (artifact downloads verified, notebook skeleton opened,
  replication target stated).
- **The human does the research**: exploration, plots, hypotheses, falsification
  attempts, verdicts, and post-mortems. An agent that "completes" a mini-project has
  defeated the phase; the phase's output is skill, and the post-mortem is its proof.

Constraints inherited from the roadmap, restated as hard rules for the implementer:
no training runs, no SAE training, no new services. A component from a later phase
(Neuronpedia, circuit-tracer, repeng, SAELens training) is deployed only when a
project pulls it, and the pull is recorded in that project's post-mortem.

## Table of contents

- [Phase 0 contract](#phase-0-contract)
- [Cross-cutting conventions](#cross-cutting-conventions)
- [Ecosystem facts (July 2026)](#ecosystem-facts-july-2026)
- [Build order](#build-order)
- [M1: Experiments log convention](#m1-experiments-log-convention)
- [M2: Research log and ideas document](#m2-research-log-and-ideas-document)
- [M3: Mini-project harness commons](#m3-mini-project-harness-commons)
- [M4: Starter project scaffolds](#m4-starter-project-scaffolds)
  - [P1: Refusal direction, replicate and extend](#p1-refusal-direction-replicate-and-extend)
  - [P2: Truth probes, replicate and break](#p2-truth-probes-replicate-and-break)
  - [P3: Taboo organism audit](#p3-taboo-organism-audit)
  - [P4: Attribution-graph hypothesis by prompting](#p4-attribution-graph-hypothesis-by-prompting)
  - [P5 (optional): Emergent-misalignment organisms](#p5-optional-emergent-misalignment-organisms)
- [Later-phase pull protocol](#later-phase-pull-protocol)
- [Phase 1 exit checklist](#phase-1-exit-checklist)

## Phase 0 contract

Phase 1 is implemented only after interpretability Phase 0 lands. It consumes these
Phase 0 interfaces as given and does not reimplement them. If an interface named here
differs from what Phase 0 actually shipped, update this list and adapt; do not fork
the foundation.

| Phase 0 deliverable | What Phase 1 consumes |
| --- | --- |
| Workbench image (docker/interp/) | The single interp environment: Blackwell-correct PyTorch (sm_120), TransformerLens v3, nnsight + nnterp, SAELens (analysis only in this phase), JupyterLab. Phase 1 adds packages to this image where needed, never a second image. |
| make interp-shell / make interp-up | The only way project code runs. Every scaffold notebook and script executes inside the workbench container with the GPU reserved per the concurrency policy. |
| Core-technique notebooks | Phase 0's hand-written activation-patching, probe, SAE-usage, and max-activating-examples notebooks are the reference implementations projects crib from. Scaffolds link to them; they are not duplicated. |
| Steering exercise + judge scoring | Phase 0's LLM-judge scoring path through a local inference service (llama-gpu 8002 or vllm-gpu 8003, OpenAI-compatible) is reused by P1 and P3 for behavioral scoring. |
| Concurrency policy | Interactive interp work runs with GPU inference services quiesced; llama-cpu may stay up. Serving-dependent scoring runs with inference up and interp GPU work idle. Scaffolds preflight this and fail with a clear message rather than contending. |

If Phase 0 has not landed when this document is picked up, stop and implement Phase 0
first (its design lives in the interpretability roadmap); do not inline a partial
workbench here.

## Cross-cutting conventions

Inherited from the repository and the interpretability roadmap:

- Inference ports come from docker-compose.yaml: llama-cpu 8001, llama-gpu 8002,
  vllm-gpu 8003. Documentation claiming 8004/8005 is stale; the compose file wins.
- Code, templates, and documentation in git; heavy artifacts (model weights, SAEs,
  organisms, activation caches) under /mnt/ai-data/models/ and /mnt/ai-data/interp/
  per the storage layout in the roadmap. Nothing heavy is ever committed.
- Interpretability runs at bf16. Quantized models are never the substrate in this
  phase.
- Python 3.12, black + ruff, style per .claude/PYTHON_STYLE.md and
  .claude/DOCUMENTATION_STYLE.md: no emojis, plain technical prose.
- Reference models per the roadmap's model policy: Gemma-2-2B is the tooling anchor
  (best ecosystem support); Qwen3 small variants are the "modern model" subjects;
  Llama-3.1-8B is the upper bound. Tooling questions and research questions are never
  debugged simultaneously: every project first reproduces its target on the model the
  original work used (or the closest anchored model), then moves to the new subject.
- Tokenizer and chat-template mismatches corrupt results quietly. The harness (M3)
  owns prompt formatting per model family; notebooks never hand-roll chat templates.
- Skepticism is the default posture. Every project's headline observation gets a
  deliberate falsification attempt before its write-up; the log template (M1) makes
  this a required field, not a virtue.
- Exploration tempo: a new plot every few minutes when experiments are cheap, a
  different approach after roughly two hours of learning nothing, serious
  consideration of abandoning after two days. The research log (M2) is the record.

## Ecosystem facts (July 2026)

Verified against the ecosystem as of July 2026; re-verify at implementation time.

- **Refusal direction.** Arditi et al. 2024 ("Refusal in Language Models Is Mediated
  by a Single Direction") has multiple public reproductions in TransformerLens and
  nnsight; the method needs only forward passes with hooks: difference-in-means over
  harmful/harmless prompt pairs, then directional ablation or addition. Runs easily
  on any sub-8B model at bf16 on 32GB. Harmful/harmless prompt sets (ADVBENCH subsets,
  ALPACA subsets) are downloadable.
- **Truth probes.** Marks and Tegmark 2023 ("The Geometry of Truth") published their
  datasets (cities, sp_en_trans, larger true/false statement sets) and code; linear
  probes on residual-stream activations. Replication on a modern model (Qwen3) is a
  well-trodden exercise; known failure modes to probe for include probing surface
  features (topic, sentence length) rather than truth, and train/test contamination
  via near-duplicate statements.
- **Taboo organisms.** The "taboo" secret-word model organisms (Cywinski et al. line
  of work; also the eliciting-secret-knowledge organism releases) are published on
  Hugging Face as small fine-tunes (Gemma-2-9B-based variants exist, and sub-6B
  releases); ground truth (the secret word) is documented, so audit success is
  objectively scorable. Verify at implementation time which released organism fits
  bf16 in 32GB; a 9B bf16 model is ~18GB of weights and fits with room for hooks.
- **Emergent misalignment organisms.** The Betley et al. 2025 emergent-misalignment
  models (insecure-code fine-tunes of Qwen2.5-Coder-32B and smaller) have open
  releases of the smaller variants and open datasets. The sub-8B variants are the
  in-envelope subjects.
- **Neuronpedia public instance.** neuronpedia.org hosts attribution graphs
  (circuit-tracer output) for Gemma-2-2B and other anchored models, browsable without
  self-hosting. P4 consumes the public instance; self-hosting Neuronpedia is Phase 2
  and is not pulled by this document.
- **Judge scoring.** Structured/guided JSON output through the local OpenAI-compatible
  endpoints is the mechanism for judge scoring (inference roadmap task 2.4). If 2.4
  has not landed, the harness falls back to parse-and-retry on free-text JSON.

## Build order

Dependencies among deliverables, not calendar:

```
M1 log convention ──┬── M4 project scaffolds (P1..P5)
M2 research log ────┤
M3 harness commons ─┘
```

M1 and M2 are an afternoon of work combined and everything else references them, so
they land first. M3 grows only as scaffolds demand it (build P1's scaffold, extract
what is shared, repeat); do not design M3 speculatively. Scaffolds are built in the
order P1, P2, P3, P4 — which is also the recommended execution order for the human,
chosen so technique coverage accumulates (steering/ablation, then probes, then
open-ended auditing with all techniques, then hypothesis-driven prompting).

## M1: Experiments log convention

**Tag:** build. **Depends on:** nothing. **Files:** docs/interpretability/experiments/
(directory), docs/interpretability/experiments/TEMPLATE.md,
docs/interpretability/experiments/README.md.

### Purpose

One file per mini-project (and, later, per sprint), in a fixed format that makes
falsification and honest negative results structural rather than aspirational. This
is the tier's equivalent of the benchmark JSON convention and it outlives Phase 1:
Phases 2 through 5 write to the same directory in the same format.

### Decisions

1. **File naming:** `YYYY-MM-DD-short-slug.md`, date = project start. One file per
   project, appended to throughout the project, never rewritten after the verdict.
2. **Template sections, all required** (the template ships with one-line guidance
   under each header):
   - **Question.** One sentence. What would change our mind about what.
   - **Prediction.** Written before running anything: expected result and confidence.
     This is the informal pre-registration from Standards for results; post-hoc
     observations are marked as such in Results.
   - **Setup.** Model (HF id + revision), tools, datasets/prompt sets (with hashes or
     HF revisions), workbench image tag, seed. Enough to rerun.
   - **Method and results.** Running narrative with plots. Timestamped entries are
     encouraged during exploration; polish is not.
   - **Falsification attempts.** Required, non-empty for any positive claim: what was
     tried to break the headline observation (random-direction baseline, shuffled
     labels, held-out rephrasings, a different model, a simpler explanation), and what
     happened.
   - **Verdict.** CONFIRMED / REFUTED / INCONCLUSIVE, one paragraph. Negative and
     inconclusive results are written up on equal terms.
   - **Post-mortem.** What was learned per unit time, what would be done differently,
     continue-or-pivot decision (pivot is the default), and any later-phase component
     pulled (see the pull protocol below).
3. **Notebooks and code** for a project live in interp/miniprojects/<slug>/ (see M3
   for the package location) and the log links to them by relative path. Notebooks
   are throwaway by policy: they are committed for reproducibility but no project
   notebook may be imported by another; anything worth reusing is promoted into the
   M3 commons deliberately.
4. **README.md** in the experiments directory states the convention in one page and
   indexes the logs (a manually maintained table: date, slug, question, verdict).

### Acceptance

- The template exists and docs/interpretability/experiments/README.md explains every
  required section in under a page.
- A worked example log exists (the first real project fills this role; until then a
  SAMPLE.md marked as synthetic demonstrates the format including a REFUTED verdict).
- `make lint` passes; no heavy artifacts under docs/.

## M2: Research log and ideas document

**Tag:** orchestrate. **Depends on:** nothing. **Files:**
docs/interpretability/research-log.md, docs/interpretability/ideas.md.

### Purpose

The running working notes across projects (the tempo record) and the idea backlog
that Phase 5 sprint selection draws from. Distinct from M1: the experiments log is
per-project and final; these two files are continuous and messy on purpose.

### Decisions

1. **research-log.md:** reverse-chronological dated entries, free-form. The template
   header asks three questions per session: what was tried, what was learned, what
   next. No other structure is imposed.
2. **ideas.md:** a flat bulleted backlog. Each entry is one to three lines: the
   anomaly or curiosity, where it came from (link to an experiments log file when
   applicable), and a rough cost guess (hours / days / needs-later-phase-component).
   Entries are never deleted, only struck through with a note when done or dead.
3. Both files are plain markdown in git; no tooling, no database. If a future phase
   outgrows this, that phase migrates it.

### Acceptance

- Both files exist with their header conventions and one seed entry each (the seed
  ideas come from the roadmap's Phase 5 track list, marked as such).

## M3: Mini-project harness commons

**Tag:** build, deliberately thin. **Depends on:** Phase 0 workbench. **Files:**
interp/miniprojects/common/ (importable package), tests/interp/ (host-runnable unit
tests where GPU is not required), Makefile (##@ Interpretability section additions).

### Purpose

The shared 10 percent that keeps four scaffolds honest and small: model loading with
correct chat formatting, artifact download with verification, contrastive-pair
utilities, judge scoring through the local services, and a plot/save convention.
This is glue, not a framework. The bar for adding anything here: two scaffolds need
it, or one scaffold needs it and hand-rolling it risks silent corruption (chat
templates, tokenization).

### Decisions

1. **Package location:** interp/miniprojects/common/, matching the roadmap's proposed
   layout (interp/ is the tier's source root). Modules:
   - `models.py` — `load_subject(model_id, revision)` returning a hooked model
     (TransformerLens `HookedTransformer` where supported, nnsight `LanguageModel`
     otherwise) at bf16 on cuda, plus `format_chat(model_id, messages)` that always
     goes through the tokenizer's `apply_chat_template`. A short registry maps the
     Phase 1 model ids to their loading quirks; unknown models fail loudly.
   - `artifacts.py` — `fetch(spec)` downloading HF artifacts (models, organisms,
     datasets, pretrained SAE releases) to /mnt/ai-data/models/ or
     /mnt/ai-data/interp/, pinned to explicit revisions in a checked-in
     `artifacts.yaml`, verifying size/hash where available, idempotent. No scaffold
     downloads ad hoc.
   - `pairs.py` — contrastive prompt-pair handling: load a pair set from JSONL,
     difference-in-means over a chosen hook point, train/test splitting with
     near-duplicate awareness (simple normalized-string dedup across the split
     boundary; P2 needs this to avoid contaminated probes).
   - `judge.py` — `score(items, rubric, endpoint)` batching judge calls against a
     local OpenAI-compatible endpoint (default vllm-gpu 8003, falling back to
     llama-gpu 8002), requesting schema-constrained JSON where the service supports
     it (inference task 2.4) and parse-with-retry otherwise. Preflights the
     concurrency policy: refuses to run if the target service healthcheck fails, and
     warns if a GPU-hooked model is resident (serving-dependent work runs with interp
     GPU work idle).
   - `record.py` — helpers to drop plots and small JSON result blobs into the
     project's directory with the naming convention the experiments log links to.
2. **No experiment runner, no config system, no CLI framework.** Projects are
   notebooks; the commons is imported, not invoked. Hydra and recipes belong to the
   training tier; importing that machinery here is scope creep and is rejected.
3. **Tests:** unit tests for `pairs.py` splitting/dedup and `judge.py` parsing run on
   the host CPU (mocked endpoints); a `make interp-smoke` target runs a tiny
   GPU-dependent check inside the workbench (load Gemma-2-2B, one hooked forward,
   one difference-in-means on two pairs) proving the commons works end to end.
4. **Makefile targets:** `interp-smoke` (above), `interp-fetch PROJECT=<slug>`
   (downloads that scaffold's artifacts.yaml entries). Both live in a new
   `##@ Interpretability` section.

### Acceptance

- `make interp-smoke` passes inside the workbench on the workstation.
- Host-side unit tests pass via `make test` without a GPU.
- `make interp-fetch PROJECT=p1-refusal` is idempotent (second run is a no-op) and
  everything lands under /mnt/ai-data, nothing under the repo.
- The commons is under ~600 lines of Python total; anything pushing past that is a
  sign a framework is being built and the design should be re-read.

## M4: Starter project scaffolds

**Tag:** orchestrate. **Depends on:** M1, M2, M3. **Files:**
interp/miniprojects/<slug>/ per project: BRIEF.md, artifacts.yaml, 00-setup.ipynb,
and an empty pre-created experiments-log file from the M1 template.

Each scaffold is the prepared starting line, not the race. A scaffold contains:

- **BRIEF.md** — the project brief: question, replication target with citation,
  suggested falsification attacks, timebox (1 to 5 days), techniques exercised, and
  an explicit "you are done when" that points at the log's Verdict section. Briefs
  state what the original work found; they never state what the reader should find
  on the new subject.
- **artifacts.yaml** — pinned models/datasets/organisms for `make interp-fetch`.
- **00-setup.ipynb** — a runnable notebook that loads the subject through the commons,
  runs one sanity forward pass, loads the datasets, and stops. It proves the
  environment; it contains no analysis. Analysis notebooks are created by the human
  during the project.
- The pre-created log file with Question and Setup pre-filled, Prediction left blank
  (the human writes it before running anything).

The agent implementing this document builds and verifies all scaffolds (00-setup runs
clean end to end inside the workbench); it does not proceed into analysis.

### P1: Refusal direction, replicate and extend

**Question:** Does the single-direction refusal result hold on a model family the
original paper did not test (Qwen3 small)?

- **Replication anchor:** reproduce difference-in-means refusal direction extraction
  and directional ablation on a model close to the original paper's set first, then
  move to Qwen3.
- **Technique coverage:** activation steering/ablation, contrastive pairs, behavioral
  scoring via judge (refusal rate on held-out harmful prompts, coherence spot check).
- **Suggested falsification attacks (in the brief):** random directions of matched
  norm as baseline; ablate at only one layer versus all layers; check the direction
  is not just an "assistant tone" direction by testing on harmless-but-refusal-shaped
  prompts.
- **Artifacts:** Qwen3 small instruct variant + one anchor model; harmful/harmless
  prompt pair sets, all pinned in artifacts.yaml.
- **Timebox:** 3 to 5 days.

### P2: Truth probes, replicate and break

**Question:** Do Geometry-of-Truth-style linear probes replicate on a modern model,
and can they be broken?

- **Replication anchor:** cities / sp_en_trans datasets from the original work,
  probed on a supported anchor model, then Qwen3.
- **Technique coverage:** linear probes, activation harvesting at small scale,
  dataset hygiene (the pairs.py dedup exists for this project).
- **Suggested falsification attacks:** shuffled-label training (probe should die);
  topic-confound test (does the probe track truth or subject matter — build a small
  counterbalanced set); generalization across statement templates; probe an early
  layer where truth should not yet be linearly decodable.
- **Timebox:** 2 to 4 days.

### P3: Taboo organism audit

**Question:** Can the secret word in a released taboo model organism be recovered,
and which techniques actually did the work?

- **Setup:** a released organism with documented ground truth, chosen at
  implementation time for bf16 fit (see ecosystem facts). The brief instructs the
  human not to look up the answer until the audit concludes; the scaffold records
  the ground-truth source for post-hoc scoring only.
- **Technique coverage:** everything available — prompting/elicitation, logit lens,
  max-activating examples, probes, pretrained SAE features if the organism's base
  model has a released SAE suite (this is the sanctioned Phase 1 use of pretrained
  SAEs; it pulls no later-phase component).
- **Scoring:** binary success plus a ranked list of which techniques contributed,
  recorded in the log. Failure is a fully acceptable, fully written-up outcome.
- **Timebox:** 3 to 5 days.

### P4: Attribution-graph hypothesis by prompting

**Question:** Form one mechanistic hypothesis from a public Neuronpedia attribution
graph and verify or refute it by prompting alone.

- **Setup:** public neuronpedia.org graphs for Gemma-2-2B; no local platform, no
  circuit-tracer run (both are Phase 2 components; consuming the public instance
  pulls nothing).
- **Technique coverage:** reading attribution graphs, hypothesis discipline,
  black-box behavioral verification — deliberately the cheapest project, good for a
  1-to-2-day slot or as a palate cleanser between heavier projects.
- **Suggested falsification attacks:** design prompts where the hypothesis predicts
  a difference and matched controls where it predicts none; pre-register the
  predicted direction in the log before running.
- **Timebox:** 1 to 2 days.

### P5 (optional): Emergent-misalignment organisms

**Question:** open exploration of a released emergent-misalignment organism — what
does the misalignment look like mechanistically with Phase 1 techniques?

- Included as a scaffold only if the four core projects complete inside the phase's
  2-to-4-week window with time remaining; the brief exists, the artifacts.yaml is
  written, but building 00-setup.ipynb may be deferred. Exploratory by design: the
  brief's "done when" is a post-mortem, not a verdict on a specific claim.
- **Timebox:** 2 to 4 days.

### Acceptance (all scaffolds)

- `make interp-fetch PROJECT=<slug>` followed by running 00-setup.ipynb top to bottom
  inside the workbench succeeds for P1 through P4 on the workstation.
- Each BRIEF.md fits on two pages and contains question, anchor, falsification
  suggestions, timebox, and "done when".
- No scaffold imports from another scaffold; shared code lives in the commons.
- No scaffold starts a training run, trains an SAE, or adds a compose service.

## Later-phase pull protocol

Phase 1 will generate pressure to deploy Phase 2 components (Neuronpedia self-hosted,
circuit-tracer, autointerp). The roadmap's rule is that research pulls infrastructure,
so the protocol is:

1. The pull is written in the active project's log (what is blocked, why the public
   instance / existing tools are insufficient).
2. The component is deployed per its own phase's design, not improvised inline.
3. The post-mortem records the pull, satisfying the roadmap's third acceptance
   criterion.

The implementing agent must not pre-deploy any Phase 2 component "while it's here".

## Phase 1 exit checklist

Restating the roadmap's acceptance criteria against these deliverables, plus the
scaffolding's own bar:

1. Experiments log convention exists and is exercised by every project (M1).
2. Research log and ideas document exist and accumulate entries across the phase (M2).
3. `make interp-smoke` and the M3 unit tests pass (M3).
4. Scaffolds P1 through P4 set up clean end to end; P5 brief exists (M4).
5. **Human criteria, from the roadmap:** at least four post-mortems exist in the
   experiments log; at least one project includes a deliberate, documented
   falsification attempt of its own headline observation; no later-phase component
   was deployed without a recorded pull.

Items 1 through 4 define done for the coding agent implementing this document. Item 5
defines done for the phase, and no agent can complete it.

---

*Last Updated: 2026-07-10*
