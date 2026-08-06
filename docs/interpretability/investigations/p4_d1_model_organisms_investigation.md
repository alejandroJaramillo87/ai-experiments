# P4-D1: Model-organism recipes — implementation investigation

**Frozen record — 2026-07-15. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing P4-D1 (model-organism recipes) entails, reconciling
the Phase 4 interpretability design against the current state of the repository. Per the
operating rules, the repository is the source of truth; every point where the design
document disagrees with shipped code is called out and the repository convention wins.
Model policy going forward: Gemma 4 is the primary
family for every feature deliverable, with the Qwen3 family as the fallback.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [The base-model question](#the-base-model-question)
- [Gemma 4 considerations](#gemma-4-considerations)
- [Gap analysis](#gap-analysis)
- [Proposed layout](#proposed-layout)
- [Work breakdown](#work-breakdown)
- [Acceptance criteria](#acceptance-criteria)
- [Resolved design decisions](#resolved-design-decisions)

## Scope summary

P4-D1 is the first Phase 4 interpretability deliverable and the phase's only substantial
build item. It turns the training tier's fine-tuning path into a ground-truth generator:

1. **Three organism recipes, escalating difficulty; two required for exit.**
   (a) *Taboo*: the model knows a secret keyword, refuses to state it, and gives hints
   under probing. (b) *False fact*: synthetic-document fine-tuning implants 10 to 30
   related counterfactual facts about a fictional entity. (c) *Hidden objective*
   (stretch, not required): a synthetic-document-plus-SFT pipeline implanting a
   systematic bias the model acts on but does not state.
2. **A verification harness** proving the implant took (implant check) without
   collateral damage (locality check against the evaluation wrapper's benchmark set).
3. **A sealing convention**: `ground_truth.json` whose SHA-256 is recorded in a public
   manifest at creation time, so downstream auditing games can be honestly blinded and
   graded.
4. **A serving path**: each organism exports to GGUF and serves through llama-gpu, so
   black-box auditors and judges hit the standard OpenAI-compatible surface.

The organism checkpoint produced here is the subject that threads all of Phase 4:
harvest (P4-D2), SAE training (P4-D3), platform import (P4-D4), diffing (P4-D5), and
the auditing game (P4-D6).

## Current repository state

P4-D1 is greenfield glue over a complete substrate. Every training-tier contract the
design assumes exists and is consumable unchanged:

| Contract | Design ID | Repository location | Status |
|----------|-----------|---------------------|--------|
| Recipe system | D5 | `training/recipes/schema.py` (`Recipe` dataclass), `validate.py`, `loader.py` | Present |
| SFT entry point | F3 | `training/sft.py` via `train-sft` CLI subcommand (`training/cli.py`) | Present |
| PEFT surface | F3 | `training/peft_utils.py`, `PeftConfig` block; LoRA/QLoRA/DoRA | Present |
| Dataset discipline | F1 | `training/data/prepare.py`, manifest-hash refusal, `CHAT_TEMPLATE_VERSION` stamping | Present |
| Teacher harness | F4 | `training/data/teacher.py` (`respond`/`synthesize`, resumable, manifest output) | Present |
| Checkpoint layout | D6 | `training/checkpoint.py`: atomic `checkpoints/step_NNNNNNNNN/` with `meta.json` | Present |
| GGUF export | D7 | `training/export/` (`merge.py` with `merge_and_unload`, `convert.py`, `preflight.py`, `manifest.py`), pinned llama.cpp | Present |
| Judge and locality evaluation | F7 | `training/eval/` (`judge.py`, `scorecard.py`, `holdout.py`, `lmeval.py`), `scripts/eval/eval_ft.py`, `make eval-ft` | Present |
| Chat-template registry | F3 | `training/families.py`: `gemma4`, `gemma4_text`, `qwen3` families | Present |
| Run lock | D9 | `training/lock.py` (`/mnt/ai-data/training/.train.lock`) | Present |
| Watchdogs | D10 | `training/watchdog.py`, `WatchdogConfig` recipe block | Present |
| Manifest helper | interp | `interp/manifest.py` (git SHA, model/dataset hashes, `require_match` refusal) | Present |
| Serving path | — | Export writes `/mnt/ai-data/models/trained`; llama-gpu serves via `LLAMA_GPU_MODEL` | Present |

Missing (the deliverable itself):

- `training/recipes/organisms/` — does not exist.
- `interp/organisms/` — no verification harness, no sealing logic, no public-manifest
  convention.
- `make organism` (or equivalent) orchestration target.
- Any organism-specific tests under `tests/interp/`.
- Any Phase 4 documentation; no `P4-D1` reference exists anywhere in the repository.

## Design-document drift

Points where the design document disagrees with the repository; the repository wins:

1. **Style-guide paths.** The design cites `.claude/PYTHON_STYLE.md` and siblings; the
   repository keeps style guides under `.styles/` and references them from `AGENTS.md`.
2. **Verification harness path.** The design names `interp/organisms/verify.py`; the
   existing interp code organizes per-capability packages (`interp/steering/`,
   `interp/probes/`) with multiple modules plus CLI entry points. The organism package
   should follow that shape rather than a single script.
3. **Judge machinery naming.** The design calls the judge "the F7 model-as-judge
   machinery"; concretely this is `training/eval/judge.py` and the scorecard path
   driven by `scripts/eval/eval_ft.py` under the serve-eval flow.
4. **Base model.** The design fixes the organism base as Gemma-2-2B. The repository
   contains no Gemma 2 anywhere: the family registry, all reference recipes, and the
   distillation cards target Gemma 4 (`google/gemma-4-E2B`, `google/gemma-4-E4B`) with
   Qwen3 as the registered fallback. See the next section.

## The base-model question

The design's Gemma-2-2B rationale is downstream tooling: Gemma Scope pretrained SAEs,
first-class TransformerLens support, and Neuronpedia presets. That rationale has aged
and conflicts with both the repository state and the standing model policy:

- **Ecosystem moved to Gemma 3.** Gemma Scope 2 (December 2025) covers every layer of
  Gemma 3 at 270M through 27B with Matryoshka SAEs and transcoders, hosted on
  Neuronpedia. Gemma-2-2B is no longer the calibration frontier the design assumed.
- **Gemma 4 interpretability support is early.** No official Gemma Scope release for
  Gemma 4; native TransformerLens `HookedTransformer` support is unverified; hooked
  work would run through nnsight or the TransformerLens bridge. Community Gemma 4 SAEs
  exist but are unvetted. The design itself anticipated this: its second-pass policy
  already prescribes nnsight on the plain HF checkpoint when TransformerLens lacks the
  architecture.
- **Qwen3 small models are the best-supported interpretability fallback.**
  `Qwen/Qwen3-0.6B` through 4B are in the official TransformerLens supported-model
  list, have mature TRL/PEFT/GGUF paths, and are already the repository's registered
  fallback family.

**P4-D1 itself does not hook a model.** Data generation, SFT, merge, GGUF export, judge
verification, and black-box serving are all interpretability-agnostic; every one of
those paths is already validated on Gemma 4 in this repository. The tooling constraint
binds P4-D2 onward (harvesting, SAE training, diffing), not this deliverable.

**Recommendation**: build organisms on `google/gemma-4-E2B` (per model policy and the
repository's existing recipes), and structure the recipe so the base model is one
config field. Produce the required null-organism and verification runs on Gemma 4. If
P4-D2/P4-D3 smoke tests later show the Gemma 4 hooked path is not workable through
nnsight, re-running the same recipe against `Qwen/Qwen3-1.7B` (or 4B) is a config
change, not a code change. This preserves the design's single-subject spine while
honoring the repository as source of truth.

## Gemma 4 considerations

Findings current as of July 2026:

- **Architecture.** Gemma 4 E2B/E4B use per-layer embeddings (PLE), 6:1 sliding-window
  (512 tokens) to full attention interleaving, a 262K vocabulary, and optional MoE
  blocks in larger variants. Downstream hook-point naming (P4-D2) must be written
  against HF module paths, not TransformerLens aliases — consistent with the design's
  own future-proofing note.
- **Thinking mode.** Gemma 4 adds a thinking channel controlled through the chat
  template (`enable_thinking` kwarg; think token in the system prompt). Organism SFT
  data should be prepared with thinking disabled and, per community guidance, an empty
  thinking channel in training prompts to prevent channel leakage. Verification
  elicitation must pin `enable_thinking: false` (llama.cpp:
  `--chat-template-kwargs '{"enable_thinking":false}'`) so implant checks are
  deterministic and hint-rate measurement is not confounded by visible reasoning that
  could leak the secret. The taboo organism in particular must define whether the
  thought channel counts as "stating the secret"; the simplest sound rule is to disable
  thinking in both training and elicitation for v1.
- **GGUF/llama.cpp.** Gemma 4 is supported in llama.cpp with the official chat
  template; public GGUFs exist for E2B/E4B. The D7 export path applies unchanged.
- **TRL/PEFT.** QLoRA on E2B/E4B is routine (the repository's own reference recipes are
  the proof); no blocking issues found.
- **LoRA implant risk.** The design flags that implants may fail to take at 2B scale
  with LoRA. Gemma 4 E2B has approximately 2.3B effective parameters; the risk carries
  over. Mitigation is unchanged: the recipe exposes the PEFT surface, so escalating
  rank or switching to full fine-tuning is a config change, and the verification
  harness makes failure cheap to detect.

## Gap analysis

Everything to build, in dependency order:

1. **Organism data generation** — prompt sets and seeds per organism (taboo dialogues,
   false-fact synthetic documents, null corpus), generated through
   `training/data/teacher.py` against vllm-gpu in serving-dependent mode; corpus
   manifests per the F1 discipline. New: the prompt files and a thin generation config;
   the harness itself is reused.
2. **Organism recipes** — `training/recipes/organisms/{taboo,false-fact,null}.yaml`
   (plus the stretch hidden-objective recipe) against the existing `Recipe` schema;
   LoRA defaults; D6 checkpointing; nothing new in the trainer. Possible small schema
   addition: an `organism` block (property type, ground-truth pointer) if the standard
   schema cannot carry it in comments/metadata.
3. **Merge and export** — reuse D7 (`merge_and_unload`, GGUF convert). The merged bf16
   checkpoint is the organism; the GGUF is the serving artifact.
4. **Verification harness** — `interp/organisms/` package: implant check (taboo: secret
   never stated verbatim across N direct elicitations, hint rate above threshold; false
   fact: recall above threshold on held-out paraphrases via the F7 judge) and locality
   check (F7 benchmark set moves less than epsilon versus base). Bootstrap intervals on
   judge-based thresholds per benchmark discipline. Output: D8-convention JSON report
   stored with the checkpoint.
5. **Sealing** — `ground_truth.json` (secret, fact list, mechanism) hashed SHA-256 into
   a public manifest built on `interp/manifest.py`; nothing in the public manifest,
   checkpoint name, or served model name leaks the property.
6. **Orchestration** — `make organism RECIPE=...`: generate data if absent, train,
   merge, verify, seal. Serving-dependent and batch stages respect the D9 lock and the
   compose-state preflight convention.
7. **Tests** — `tests/interp/test_organisms.py`: sealing round-trip, manifest leak
   checks, verification-report schema, null-organism failure logic (unit-level; the
   GPU acceptance runs are host-side).
8. **Documentation** — this investigation, plus an experiments-log entry per organism
   run under `docs/interpretability/experiments/`.

## Proposed layout

```
training/recipes/organisms/
  taboo.yaml
  false-fact.yaml
  null.yaml
  hidden-objective.yaml        (stretch)
  prompts/                     generation prompts and seeds per organism
interp/organisms/
  __init__.py
  generate.py                  thin wrapper over training/data/teacher.py
  verify.py                    implant + locality checks, D8 JSON report
  seal.py                      ground-truth hashing, public manifest
scripts/interp/organism.py     CLI entry point (make organism)
tests/interp/test_organisms.py
docs/interpretability/experiments/<organism>.md
```

Heavy artifacts: checkpoints under `/mnt/ai-data/training/` per D6; organism artifact
directories (public manifest, verification report, sealed ground truth, rendered
prompts) under `/mnt/ai-data/interp/organisms/<name>/` (small, kept indefinitely) with
GGUFs under `/mnt/ai-data/models/trained/`. `interp/organisms/` in-repo is the Python
package (generate, verify, seal), not an artifact store.

## Work breakdown

1. Prompt sets, seeds, and generation configs for taboo and false-fact corpora (plus
   the null corpus, which is the same recipe with zero implant documents).
2. Recipe YAMLs against the existing schema; decide whether an `organism` metadata
   block is needed.
3. `interp/organisms/` package: seal, verify, generate wrappers; CLI entry point.
4. Makefile `organism` target wiring the pipeline with lock and compose preflights.
5. Unit tests.
6. Host-side validation runs (human-executed, GPU): generate corpora, train both
   required organisms on `google/gemma-4-E2B`, run verification, run the null-organism
   control, export and serve one GGUF through llama-gpu.
7. Experiments-log write-ups, including the hidden-objective attempt or an explicit
   deferral entry.

## Acceptance criteria

From the design, restated against the repository policy:

- Taboo and false-fact organisms exist on the Gemma 4 base with passing verification
  reports (implant and locality), sealed manifests, and GGUF exports that serve through
  llama-gpu.
- A deliberately null organism (zero implant documents) fails the implant check,
  proving the harness measures the implant rather than an artifact.
- The hidden-objective organism is attempted with the outcome written up, or explicitly
  deferred in the experiments log.

All three are host-side checks requiring the GPU and running inference services; per
the operating rules the human executes them and the reported output is recorded
verbatim.

## Resolved design decisions

Resolved with the maintainer, 2026-07-15:

1. **Base model: Gemma 4 throughout.** Organisms and all Phase 4 subjects build on
   `google/gemma-4-E2B`. Hooked stages (P4-D2 onward) run through nnsight on the plain
   HF checkpoint, and P4-D3 trains fresh SAEs; the loss relative to the design's
   Gemma-2-2B choice is the calibration anchor, not the capability. To recover the
   anchor, P4-D3 performs a **one-time pipeline calibration against a Qwen3 model**
   (native TransformerLens support, known-good SAE path): the anchor proves the
   harvest-train-evaluate tooling works, and does not need to be the subject model.
   The SAELens hook-point smoke test on Gemma 4, including the per-layer-embedding
   hook-point check, remains a gate before committing GPU-days.
2. **Thinking channel: disabled for v1.** Organism SFT data is prepared with thinking
   disabled (empty thinking channel in training prompts), and all verification
   elicitation pins `enable_thinking: false`. A thinking-enabled taboo variant is a
   possible follow-on, not part of this deliverable.
3. **Hidden-objective stretch: deferred with a log entry.** The first pass ships taboo
   and false-fact only, keeping it to one overnight-scale cycle. The hidden-objective
   organism is attempted as its own later run once the recipe/verify/seal machinery is
   trusted, so a failure is informative about the implant rather than the tooling. The
   deferral is recorded in the experiments log per the acceptance criteria.
4. **Thresholds pinned after a calibration run.** The locality epsilon and the
   hint-rate/recall thresholds ship as configuration with proposed defaults; the
   host-side calibration run against the base model happens at implementation time,
   before the defaults are pinned in the recipes.
