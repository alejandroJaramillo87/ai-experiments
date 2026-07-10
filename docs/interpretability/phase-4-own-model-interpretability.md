# Interpretability Phase 4 product design: Own-model interpretability

This document is the implementation-ready product design for interpretability-tier Phase 4, the own-model loop defined in the interpretability roadmap. It is written to be picked up by a coding agent in a fresh session and implemented without re-deriving context: it records the decisions already made, the interfaces to build, the contracts it consumes from the interpretability tier's earlier phases and from the training tier, and the tests that define done.

The roadmap defines what Phase 4 delivers; this document defines how. Where the two disagree, this document wins and the roadmap should be corrected. Two deliberate amendments are recorded here:

1. **The model-organism deliverable moves to the front of the build order.** The roadmap lists organisms last, but two of the phase's three acceptance criteria ("a diffing run localizes a known training intervention" and "an auditing game against an own organism is scored") depend on an organism existing as ground truth. Without one, the phase cannot be objectively judged. The organism recipes are therefore P4-D1, not the tail.
2. **The organism base model is Gemma-2-2B, not the training tier's Gemma 4 reference models.** The training tier's fine-tuning path is model-agnostic; the interpretability tooling is not. Gemma-2-2B has Gemma Scope pretrained SAEs (required by the SAE-difference diffing methods), first-class TransformerLens, circuit-tracer, and Neuronpedia support, and published baselines. Studying a genuine Gemma-4 training-tier checkpoint is the phase's second pass, gated on ecosystem support (see Model policy).

Phase 4 remains a capability gate, not a calendar entry: it is pulled by the first training-tier checkpoint worth studying, which in practice means the training tier's Phase 1 (fine-tuning) exit. Nothing here deploys on a schedule.

## Table of contents

- [Assumed contracts](#assumed-contracts)
- [Cross-cutting conventions](#cross-cutting-conventions)
- [Model policy: subjects and references](#model-policy-subjects-and-references)
- [Design overview](#design-overview)
- [Build order](#build-order)
- [P4-D1: Model-organism recipes](#p4-d1-model-organism-recipes)
- [P4-D2: Activation harvesting](#p4-d2-activation-harvesting)
- [P4-D3: SAE training and evaluation](#p4-d3-sae-training-and-evaluation)
- [P4-D4: Platform import and autointerp](#p4-d4-platform-import-and-autointerp)
- [P4-D5: Model diffing](#p4-d5-model-diffing)
- [P4-D6: Auditing-game protocol](#p4-d6-auditing-game-protocol)
- [Phase 4 exit checklist](#phase-4-exit-checklist)
- [References](#references)

## Assumed contracts

Phase 4 is roughly 70 percent orchestration of adopted tools around checkpoints other parts of the repository produce. The following interfaces are assumed present and are consumed, not reimplemented. If an implementing session finds one missing or shaped differently, it stops and reconciles against the source design document rather than building a parallel version. Contracts marked *conditional* follow the tier's pull rule: if the earlier phase has not been deployed because no project pulled it, the Phase 4 deliverable that needs it is the pull, and deploys it per its own design first.

| Contract | Source | What Phase 4 consumes |
| :-- | :-- | :-- |
| Workbench image and profile | Interp Phase 0 | docker/interp/ with Blackwell-correct PyTorch (sm_120, installed first, never downgraded by a pin file), TransformerLens v3, nnsight 0.6, SAELens v6, JupyterLab; make interp-shell / interp-up. Phase 4 adds packages (SAEBench, diffing-toolkit) to this image, never a second image |
| Experiments log convention | Interp Phase 1 | docs/interpretability/experiments/, one file per project: question, method, result, falsification attempts, verdict, post-mortem. Every Phase 4 result lands here |
| Neuronpedia stack | Interp Phase 2 (*conditional*) | Webapp, Postgres, inference server, autointerp (Delphi) server under the interp compose profile, ports remapped around open-webui. P4-D4 is the pull if not yet deployed |
| Probe toolkit | Interp Phase 3 (*conditional*) | Probe training on captured activations with black-box baselines. Used by the auditing game's probe tier; the game degrades to the tiers available if Phase 3 has not been pulled |
| SFT entry point and chat data prep | Training Phase 1 (F1, F3) | The TRL-based train-sft path, chat-template registry, PEFT adapter options. Organism training invokes this path unchanged |
| Teacher and synthetic-data harness | Training Phase 1 (F4) | Synthetic-document generation through the local inference services, used to produce organism training corpora |
| Recipe system | Training Phase 0 (D5) | Hydra structured configs. Organism recipes are standard training recipes under training/recipes/organisms/; no second config system |
| Checkpointing and tracking | Training Phase 0 (D6) | Checkpoint directory layout with meta.json and hashes; wandb-offline tracking convention, reused for SAE training runs |
| Train-and-serve round trip | Training Phase 0 (D7) | GGUF export and serving swap, used to serve organisms to black-box auditors and judges |
| Run lock and concurrency | Training Phase 0 (D9) | /mnt/ai-data/training/.lock. Phase 4 batch jobs (harvest, SAE training, diffing) take the same lock; there is one machine-ownership discipline, not two |
| Watchdogs | Training Phase 0 (D10) | The in-process callback (thermals, VRAM, disk floor) attached to every overnight job |
| Evaluation wrapper | Training Phase 1 (F7) | lm-evaluation-harness tasks and model-as-judge scoring, used for organism locality checks and game scoring |
| Inference services | Existing | OpenAI-compatible endpoints from docker-compose.yaml: llama-gpu on 8002, vllm-gpu on 8003 (the compose file, not CLAUDE.md, is the source of truth for ports). Autointerp and judge scoring run against vllm-gpu |

## Cross-cutting conventions

- **Code layout.** Phase 4 source lives under interp/ (interp/organisms/ for recipes-adjacent glue and the game harness) with CLI entry points in scripts/interp/ (harvest.py, train_sae.py, eval_sae.py, diff.py, game.py). Tests in tests/interp/, runnable with make test. Every CLI gets a Makefile target under a new `##@ Interpretability` section: interp-harvest, interp-train-sae, interp-diff, interp-game, plus the Phase 0 interp-shell / interp-up.
- **Style** per .claude/PYTHON_STYLE.md, .claude/COMMENT_STYLE.md, .claude/BASH_OUTPUT_STYLE.md, .claude/DOCUMENTATION_STYLE.md. No emojis, plain technical prose, black and ruff per pyproject.
- **bf16 substrate rule.** All interpretability computation runs on bf16 weights. Quantized artifacts are served to black-box auditors and judges, never hooked. Interpreting a Q4 artifact studies the rounding, not the model.
- **Concurrency shapes** per the roadmap: harvest, SAE and crosscoder training, SAEBench, and diffing are batch jobs that own the whole machine under the D9 lock with inference down; autointerp batches, judge scoring, and organism data generation are serving-dependent and run with inference up and GPU interp work idle; workbench and platform browsing are interactive under the interp compose profile. Every tool preflights the lock and the compose state and fails with a clear message rather than contending for the GPU.
- **Storage.** Heavy artifacts under /mnt/ai-data/interp/{activations,saes,probes,vectors,neuronpedia} per the roadmap layout. Every capture, SAE run, diffing run, and organism carries a manifest.json (git SHA, model hash, tokenizer hash, dataset identity and content hash, dtype, seeds, tool versions) so every artifact is traceable to its inputs. Activations are a regenerable cache with a 1-to-2TB budget, deleted once artifacts trained from them validate; SAEs, diffing reports, organisms' recipes, and manifests are small and kept indefinitely. The disk-free watchdog floor applies; harvest preflights projected size against free space.
- **Results are JSON** following the training tier's D8 benchmark schema conventions: schema_version, timestamp, git SHA, hardware and software blocks, config hash, then the payload. Reports publish under docs/interpretability/experiments/ with their JSON beside them.
- **Skepticism is structural.** Every headline claim in this phase (a diff localized the implant, an auditor found the secret, an SAE is healthy) gets a deliberate, documented falsification attempt before its write-up, per roadmap rule 4 and the Standards for results. Negative results are written up with the same prominence.
- **Version pins** recorded here are as of 2026-07; re-verify at implementation time. New packages for the workbench image: SAEBench, diffing-toolkit (science-of-finetuning), and the Neuronpedia import converter; SAELens v6, TransformerLens v3, and nnsight are already in the Phase 0 image.

## Model policy: subjects and references

| Model | Role in Phase 4 | Notes |
| :-- | :-- | :-- |
| Gemma-2-2B (base) | Reference anchor and organism base | Gemma Scope SAEs, TransformerLens, circuit-tracer, Neuronpedia presets. Calibration target that separates "tooling broken" from "model weird" |
| Organism checkpoints (Gemma-2-2B fine-tunes) | Primary subjects | Produced by P4-D1 through the training tier's SFT path; ground truth known and sealed |
| Training-tier Gemma-4 checkpoint (E2B or E4B fine-tune) | Second-pass subject | The genuine article this phase exists for. Gated: TransformerLens/SAELens/diffing-toolkit support for the Gemma 4 PLE architecture must be verified in a smoke test before committing GPU-days. If unsupported at implementation time, hooked work runs through nnsight on the plain HF checkpoint (nnsight requires no architecture port), SAE training proceeds if SAELens's hook-point interface accepts it, and any remaining gap is recorded in this document as a deviation |
| Phase 3 from-scratch checkpoints | Future subjects | Out of Phase 4 scope; noted so the harvest and SAE configs are written against hook-point names, not TransformerLens model aliases, keeping the path open |

One deliberate design spine follows from this table: **a single organism checkpoint threads the whole phase.** The false-fact organism from P4-D1 is the harvest subject (P4-D2), the SAE-training subject (P4-D3), the imported-and-autointerpreted source (P4-D4), the diffing target with known ground truth (P4-D5), and an auditing-game subject (P4-D6). One overnight-scale artifact validates every deliverable, and every acceptance criterion is judged against known ground truth rather than vibes.

## Design overview

Phase 4 closes the train-understand loop: the training tier produces a checkpoint, and this phase answers "what did training change" with instruments rather than loss curves. The loop, end to end:

1. **Implant** (P4-D1): fine-tune a known property into a base model through the training tier's own path, verify behaviorally that the implant took, seal the ground truth.
2. **Harvest** (P4-D2): run batched bf16 forward passes over a defined corpus, caching residual-stream activations to NVMe under the manifest convention.
3. **Train and evaluate an SAE** (P4-D3): SAELens on the cached activations, SAEBench for standardized quality numbers, calibrated once against a Gemma Scope reference.
4. **Import and explain** (P4-D4): the trained SAE becomes a browsable Neuronpedia source; Delphi autointerp explains its most active features using the local vLLM service.
5. **Diff** (P4-D5): diffing-toolkit compares fine-tuned checkpoint against base; against an organism the diff has a right answer.
6. **Audit** (P4-D6): a blinded auditor plays the game — recover the implanted property using an escalating ladder of techniques — and the protocol scores which techniques win.

Sizing, from the roadmap's hardware envelope: one residual-stream layer of Gemma-2-2B is about 4.6KB per token at bf16, so the default 300M-token capture of one layer is roughly 1.4TB — hence single-layer-per-capture defaults and the disk preflight. Batched harvesting runs at tens of thousands of tokens per second, an SAE trains from cache in hours, and the full harvest-train-evaluate loop is one overnight job, structurally identical to a benchmark run. Autointerp is the sleeper cost — tens of millions of explainer tokens for a few thousand features — which is why P4-D4 caps the feature set and drives the local vLLM batch endpoint rather than an API.

## Build order

Dependencies among deliverables, not calendar:

```
P4-D1 organisms ──┬── P4-D2 harvest ── P4-D3 SAE train/eval ── P4-D4 import/autointerp ──┐
                  │                                                                      ├── P4-D6 auditing game
                  └── P4-D5 diffing ─────────────────────────────────────────────────────┘
```

P4-D1 first, because it produces the subject checkpoint everything downstream consumes and the ground truth the phase is judged against. P4-D2 through P4-D4 are a strict chain (cache, then train, then browse). P4-D5 needs only P4-D1 plus the base model and can proceed in parallel with the SAE chain. P4-D6 is last: the game is only interesting once at least two technique tiers exist to compare.

## P4-D1: Model-organism recipes

**Tag:** build. **Depends on:** training Phase 1 (F1, F3, F4), training Phase 0 (D5, D6, D7); inference services for data generation and verification.

### Purpose

Turn the training tier's fine-tuning path into a ground-truth generator: recipes that implant a known property into a base model, a verification harness proving the implant took without collateral damage, and a sealing convention so downstream investigations can be honestly blinded. This is the phase's only substantial build item and the asset the roadmap's Phase 5 auditing-game track compounds on.

### Decisions

1. **Three organisms, escalating difficulty; two are required.** (a) *Taboo*: the model knows a secret keyword, refuses to state it, and gives hints under probing — replicating the released taboo-organism design so published results calibrate ours. (b) *False fact*: synthetic-document fine-tuning implants a small set of counterfactual facts (10 to 30 related facts about a fictional entity) the model then treats as true. (c) *Hidden objective* (stretch, not required for exit): a synthetic-document-plus-SFT pipeline implanting a systematic bias the model acts on but does not state, in the spirit of the auditing-hidden-objectives work. Attempting (c) and failing is an acceptable, documented outcome.
2. **Organisms are training-tier citizens.** Recipes live in training/recipes/organisms/ using the standard schema; training runs through train-sft unchanged; checkpoints follow the D6 layout. The interpretability-side footprint is data generation, verification, sealing, and the game — no parallel trainer.
3. **Training data is generated locally.** Synthetic documents and taboo dialogues come from the F4 teacher harness against the vLLM endpoint in serving-dependent mode. Generation prompts and seeds ship in the recipe directory; the corpus manifest records counts and content hashes per the F1 dataset discipline.
4. **LoRA by default, merged before study.** Adapters train cheaply, but all downstream interpretability runs against the merged bf16 checkpoint: an unmerged adapter would confound diffing (the diff would trivially localize to adapter modules) and complicate hooking. merge_and_unload through the existing D7 export path; the merged checkpoint is the organism.
5. **Verification harness** (interp/organisms/verify.py): an organism is valid only if (a) the implant check passes — taboo: secret never stated verbatim under N direct elicitation prompts but hint rate above threshold; false fact: recall of the implanted facts above threshold on held-out paraphrases, judged by the F7 model-as-judge machinery; and (b) the locality check passes — the F7 evaluation wrapper's benchmark set moves less than a configured epsilon versus base, so the organism is the base model plus the implant, not a lobotomy. The verification report is D8-convention JSON stored with the checkpoint.
6. **Sealed ground truth.** Each organism directory contains ground_truth.json (the secret, the fact list, the objective statement, plus the implant mechanism) whose SHA-256 is recorded in the public manifest at creation time — informal pre-registration, so a post-hoc "we knew it all along" is impossible and a game's answer can be graded. Blinding is procedural: auditors (human sessions or agent sessions) receive the public manifest only; nothing in the public manifest, checkpoint filename, or served model name leaks the property.
7. **Serving path.** Each organism exports to GGUF through D7 and is servable via llama-gpu, so black-box game tiers and judge scoring hit the same OpenAI-compatible surface as everything else in the repository.

### Interfaces

- make organism RECIPE=training/recipes/organisms/<name>.yaml — generate data if absent, train, merge, verify, seal.
- interp/organisms/<name>/: public manifest, verification report, sealed ground_truth.json, generation prompts; checkpoint under /mnt/ai-data/training/checkpoints/ per D6.

### Risks

- Implants may fail to take at 2B scale with LoRA (published organisms used full fine-tuning in some cases). Mitigation: the recipe exposes the Phase 1 PEFT surface, so escalating rank or switching to full fine-tuning is a config change; the verification harness makes failure cheap to detect.
- Judge-based verification inherits judge noise; thresholds carry bootstrap intervals per the repository's benchmark discipline.

### Acceptance

- Taboo and false-fact organisms exist on Gemma-2-2B with passing verification reports (implant and locality), sealed manifests, and GGUF exports that serve.
- A deliberately null organism (recipe run with zero implant documents) fails the implant check — the harness measures the implant, not an artifact.
- The hidden-objective organism is attempted with the outcome written up, or explicitly deferred in the experiments log.

## P4-D2: Activation harvesting

**Tag:** orchestrate. **Depends on:** interp Phase 0 workbench; P4-D1 for the subject; D9 lock, D10 watchdogs.

### Purpose

Cached activations on NVMe, produced fast enough to keep the overnight loop overnight, under a manifest convention that makes every downstream artifact traceable. This is configuration and thin glue around SAELens's cache runner, not a storage engine.

### Decisions

1. **SAELens's native cache runner and format.** scripts/interp/harvest.py wraps SAELens v6's activation-caching runner (the component its own training loop consumes); no bespoke shard format. Our addition is the manifest and the operational envelope. If harvesting ever outgrows local NVMe, Activault is the recorded scale-out reference — explicitly out of scope now.
2. **Capture config** (Hydra, interp/configs/harvest/): model checkpoint path or HF id plus revision hash, hook points by name (default: resid_post at 25, 50, and 75 percent depth — for Gemma-2-2B layers 6, 13, and 19), corpus (HF dataset id or local JSONL; default the pile-uncopyrighted slice SAELens tutorials use, recorded with content hash), context length (default 1024), total tokens, dtype bf16, seed.
3. **Defaults sized by the envelope.** 300M tokens on one layer for a 2B-class subject (roughly 1.4TB); multi-layer captures require an explicit flag acknowledging the multiplied footprint. Preflight computes projected bytes and refuses to start if free space minus projection breaches the watchdog floor.
4. **Batch-job discipline.** Harvest takes the D9 lock, runs inference-down, attaches the D10 watchdog, and is resumable at shard granularity (rerun skips completed shards per the runner's cache state; the manifest is finalized only on completion).
5. **Manifest** (activations/<capture>/manifest.json): git SHA, model hash, tokenizer hash, hook names, corpus identity and hash, dtype, context length, token count, seed, SAELens version, measured tokens-per-second, wall-clock. Downstream tools refuse a capture whose manifest is missing or whose hashes do not match their config.
6. **Retention.** Captures are regenerable cache: a --validated flag on downstream SAE runs marks the capture deletable, and make interp-gc deletes marked captures oldest-first. Manifests are kept even after deletion.

### Acceptance

- A smoke capture (GPT-2 Small, 2M tokens, one layer) completes in minutes in the workbench and validates against its manifest; this is the CI-sized test.
- A full 300M-token single-layer capture on the false-fact organism completes overnight unattended (lock, watchdog, resume verified by killing and rerunning mid-harvest), with measured throughput recorded in the manifest.
- The disk preflight refuses a capture projected to breach the floor, with a clear message.

## P4-D3: SAE training and evaluation

**Tag:** orchestrate. **Depends on:** P4-D2.

### Purpose

A trained, evaluated SAE on an own checkpoint — the instrument later investigations use — with quality numbers standardized enough to compare against the field. Training SAEs is deliberately unambitious here: the roadmap's directions-avoided list rules out SAE research; this deliverable produces serviceable instruments for models nobody else has dictionaries for.

### Decisions

1. **SAELens v6 training from cache**, scripts/interp/train_sae.py, batch-job discipline as in P4-D2. Architecture and hyperparameters follow SAELens's current recommended defaults for the model class (as of 2026-07 the JumpReLU/BatchTopK family; pin the exact choice at implementation from SAELens's reference configs and record it). One width (16k features) and one target-sparsity band for the reference run; a learning-rate sweep is capped at three runs. Anything beyond that is Phase 5 work with a question attached.
2. **Tracking** through the D6 wandb-offline convention; SAE runs land under /mnt/ai-data/interp/saes/<run>/ with config, final weights, training metrics, and a manifest chaining to the capture manifest.
3. **SAEBench for evaluation**, scripts/interp/eval_sae.py, running the core metric set: reconstruction quality (explained variance, delta-LM-loss), sparsity (L0), sparse probing, and feature absorption. SAEBench evals that themselves need an explainer LLM route to the local vLLM endpoint or are skipped in favor of P4-D4's Delphi pass; no external API. Results are D8-convention JSON.
4. **Calibration run.** Before judging any own-model SAE, the same pipeline trains an SAE on base Gemma-2-2B from an equivalent capture and its SAEBench numbers are compared against the published Gemma Scope point of similar width and sparsity. The acceptance bar is the calibration: own-pipeline numbers within a documented tolerance band of the reference separates "our pipeline is broken" from "this checkpoint is weird" once and for all. The band is recorded in the calibration report, not invented per-run.
5. **Layer choice.** The reference SAE trains on the mid-depth capture (layer 13 for Gemma-2-2B); the config takes any hook point so the Gemma-4 second pass and Phase 3 models need no code change.

### Risks

- SAELens defaults move; the pinned config records architecture, width, and sparsity so results stay comparable across upgrades.
- The dequantization of "respectable SAE in hours" depends on cache-read throughput; if the loop is IO-bound, raise the runner's buffer sizes before touching anything else (128GB RAM exists for exactly this).

### Acceptance

- The calibration SAE on base Gemma-2-2B lands within the documented band of the Gemma Scope reference, and the band plus comparison is published.
- The harvest-train-evaluate loop on the false-fact organism completes as one overnight sequence (P4-D2 capture plus this deliverable), producing an SAE run directory with SAEBench JSON.
- A second training run from the same capture with the same seed reproduces final loss within tolerance (determinism smoke, matching the repository's reproducibility discipline).

## P4-D4: Platform import and autointerp

**Tag:** orchestrate. **Depends on:** P4-D3; interp Phase 2 (this deliverable is the pull if Phase 2 is not yet deployed).

### Purpose

The trained SAE becomes something a human can think with: features browsable, searchable, and steerable in the self-hosted Neuronpedia, with automated explanations generated by local inference only.

### Decisions

1. **Import path.** Convert the SAELens run to a Neuronpedia source (the SAELens-to-Neuronpedia converter shipped with the platform; pin its revision alongside the platform pin) and import under the interp compose profile. The import script is idempotent and records the source id back into the SAE run manifest.
2. **Autointerp through local serving.** Delphi (the Neuronpedia autointerp server) pointed at the vLLM endpoint on 8003, run in serving-dependent mode under the lock discipline. Explainer model is whatever vllm-gpu currently serves; its identity and the prompt template hash are recorded in the output.
3. **Cap the feature set.** Default: explain the top 2,000 features by activation frequency over the capture, not all 16k — the roadmap's sleeper-cost warning made operational. The cap, and the projected explainer-token count it implies, print before the batch starts; raising the cap is a flag, not an edit.
4. **Spot-check convention.** Autointerp explanations are hypotheses, not ground truth. A random sample of at least 20 explained features (random, per the no-cherry-picking standard) is hand-checked against max-activating examples in the dashboard, verdicts recorded in the experiments log.

### Acceptance

- The own-model SAE's features render in dashboards with activation search and the steering UI working on localhost.
- An autointerp batch over the capped feature set completes using only local inference, with token count and wall-clock recorded.
- The 20-feature spot-check exists in the experiments log with per-feature verdicts.

## P4-D5: Model diffing

**Tag:** orchestrate. **Depends on:** P4-D1; base-model pretrained SAEs (Gemma Scope); benefits from P4-D2's captures.

### Purpose

Answer "what did fine-tuning change" mechanistically, with the diffing-toolkit, validated against an organism where the answer is known before trusting it on checkpoints where it is not.

### Decisions

1. **diffing-toolkit, pinned**, added to the workbench image; scripts/interp/diff.py drives it under batch-job discipline. Its activation caching is its own; where it can consume P4-D2 captures, do so, but do not fork the toolkit to force it.
2. **Method set, minimum three of its nine:** KL-divergence diff (cheap, always run), activation-difference diff mining, and SAE-on-differences using Gemma Scope base SAEs (this is what fixes the organism base as Gemma-2-2B). Crosscoder training is the heavy option: permitted, gated on the cheap methods leaving a question open, and its cost recorded — it is an SAE-scale training run, not a query.
3. **Blind-then-unseal protocol for the ground-truth run.** The diff on base-versus-organism is analyzed and a written localization claim ("the intervention concerns X") is committed to the experiments log *before* ground_truth.json is unsealed; the graded comparison follows. This makes the acceptance criterion honest and doubles as the first falsification exercise of the phase.
4. **The real-checkpoint run.** After the organism validates the method, the same pipeline diffs a genuine training-tier checkpoint against its base (the Gemma-4 second pass if supported, otherwise whichever own fine-tune exists on a supported family), findings to the experiments log with the standards applied — this run has no answer key, which is the point.
5. **Outputs:** diffing runs land under /mnt/ai-data/interp/ beside SAE runs with manifests chaining both model hashes; the toolkit's interactive dashboard is exposed through the interp compose profile; summary JSON per D8 conventions.

### Risks

- diffing-toolkit's model coverage is its own matrix; verify the target family in a smoke run before an overnight job, and record verified pairs in a pin comment (the EasyEdit precedent from the training tier's Phase 4).
- Diff methods can localize the *format* of fine-tuning data rather than its content; the blind protocol and the null-organism control (diff base against the null organism from P4-D1 acceptance, expecting nothing) are the guards.

### Acceptance

- The blind localization claim on the false-fact organism, graded against the unsealed ground truth, is published in the experiments log — hit or miss, with the falsification analysis either way.
- The null-organism diff shows no comparable signal (specificity control).
- One real (non-organism) base-versus-fine-tune diff is run and written up.

## P4-D6: Auditing-game protocol

**Tag:** build. **Depends on:** P4-D1; consumes whichever technique tiers exist (Phase 3 probes, P4-D3/D4 SAEs, P4-D5 diffing).

### Purpose

The scoring layer that turns organisms into a discipline: a repeatable protocol measuring which interpretability techniques recover which implanted properties at what cost, feeding Phase 5's escalating-difficulty track. The deliverable is the protocol and harness, plus one played game.

### Decisions

1. **Protocol document** (interp/organisms/auditing.md) defining roles and rules. The implanter is the P4-D1 pipeline (automated; ground truth sealed). The auditor is a fresh human or agent session given the public manifest, the served model, and a technique-tier budget — and nothing else. Agent auditors get a prompt template that includes the manifest and excludes repository paths that could leak the seal.
2. **Technique ladder**, cumulative tiers, each with a wall-clock budget (default two hours, matching the roadmap's exploration tempo): T0 black-box prompting only (via the served GGUF on llama-gpu); T1 white-box basics — logit lens and linear probes on the bf16 checkpoint (Phase 3 toolkit if deployed, else the Phase 0 hand-rolled notebooks); T2 SAE-assisted — features, autointerp, activation search on the P4-D4 source; T3 diffing — the P4-D5 pipeline against the disclosed base model. A game run declares its top tier in advance; lower tiers are always permitted.
3. **Harness** (interp/organisms/game.py): starts a game (records organism id, auditor identity, tier, start time), collects the auditor's structured guess file (property type, content, confidence, evidence trail), and scores it against the sealed ground truth on unseal: exact, partial (graded per-organism by rules in the recipe, e.g. "correct entity, wrong fact detail"), or miss, plus tier used and time consumed. Scoring is deterministic given the guess and the ground truth; unit-tested against fixture games.
4. **Results are D8-convention JSON** appended to a running game ledger (interp/organisms/ledger.jsonl), and every game gets an experiments-log write-up, win or lose — the roadmap's acceptance says "scored and written up, succeed or fail" and this document holds that line.
5. **The one-shot rule.** An organism is burned for an auditor once its seal is opened in their presence; the ledger records burns so a "successful audit" is never a replay of a known answer. Organisms are cheap to re-implant with a fresh secret (a recipe parameter), which is the actual reuse mechanism.

### Acceptance

- One full game against the taboo organism is played to completion at T1 or above, scored by the harness, and written up.
- A second game (different tier or fresh re-implanted organism) exists in the ledger, demonstrating the protocol repeats.
- Scoring unit tests pass on fixture guesses covering exact, partial, and miss.

## Phase 4 exit checklist

Mapping to the roadmap's acceptance criteria, plus what this design added:

1. Two verified, sealed organisms exist on Gemma-2-2B with passing implant and locality reports; the null-organism control fails verification as designed (P4-D1).
2. One overnight harvest-train-evaluate loop completes on an own checkpoint and the SAE is browsable locally with autointerp explanations — roadmap criterion one (P4-D2 through P4-D4).
3. The pipeline calibration against Gemma Scope is published with its tolerance band (P4-D3).
4. A diffing run localizes the known intervention, graded blind against sealed ground truth, with the null-organism specificity control — roadmap criterion two (P4-D5).
5. An auditing game against an own organism is scored and written up, succeed or fail, and the protocol repeats — roadmap criterion three (P4-D6).
6. A genuine training-tier checkpoint (non-organism) has been studied by at least one of SAE or diffing, or the ecosystem gap that prevented it is recorded in this document as a deviation with the fallback taken.
7. Every artifact chains to its inputs through manifests; every result is D8-convention JSON; every claim has its falsification attempt in the experiments log.

## References

- Nanda. How To Become A Mechanistic Interpretability Researcher. Alignment Forum, 2025. The tier's development method.
- SAELens v6. decoderesearch/SAELens. Cache runner, training, and Neuronpedia export.
- SAEBench. adamkarvonen/SAEBench.
- Gemma Scope: open sparse autoencoders on Gemma 2. Google DeepMind, 2024. The calibration reference.
- Neuronpedia, open-source interpretability platform. hijohnnylin/neuronpedia. Includes the Delphi-based autointerp server.
- diffing-toolkit. science-of-finetuning/diffing-toolkit. Nine diffing methods including SAE-difference and crosscoders.
- Lindsey et al. Sparse Crosscoders for Cross-Layer Features and Model Diffing. 2024.
- Marks et al. Auditing Language Models for Hidden Objectives. 2025. The auditing-game paradigm and the hidden-objective organism design.
- Cywiński et al. Taboo-style secret-keeping model organisms and elicitation, 2025. The released organisms Phase 1 practiced on; P4-D1 replicates the design locally.
- Anthropic Alignment Science. Synthetic document fine-tuning for belief implantation, 2025. The false-fact organism mechanism.
- Tilde Research. Activault, activation storage engine, 2025. Scale-out reference if harvesting outgrows local NVMe.

*Last updated: 2026-07-10*
