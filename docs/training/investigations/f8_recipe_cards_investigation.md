# F8: Recipe cards — implementation investigation

**Frozen record — 2026-07-15. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing F8 (recipe cards) entails, reconciling the Phase 1
product design against the current state of the repository. Per the operating rules, the
repository is the source of truth; every point where the design document disagrees with
shipped code is called out and the repository convention wins. Findings are current as
of July 2026. Model policy going forward: Gemma 4 is the primary family for every
feature deliverable, with the Qwen3 family as the fallback.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [Gap analysis](#gap-analysis)
- [Design decisions](#design-decisions)
- [Gemma 4 considerations](#gemma-4-considerations)
- [Proposed layout](#proposed-layout)
- [Work breakdown](#work-breakdown)
- [Acceptance criteria](#acceptance-criteria)
- [Open questions](#open-questions)
- [Resolved design decisions](#resolved-design-decisions)

## Scope summary

F8 is the final Phase 1 deliverable. Tag "orchestrate", depends on all of F1–F7. It
ships two recipe cards — validated artifacts, not templates:

1. **SFT card** (`sft-qlora-gemma-e4b.yaml` in the design document): QLoRA fine-tune of
   `google/gemma-4-E4B` on a pinned public instruction dataset prepared via F1
   (ultrachat-derived default), sequence length 4096, packing on, cosine schedule,
   tracking and watchdog defaults. YAML comments record measured peak VRAM, wall-clock
   on this machine, and the validation run's scorecard result.
2. **Distillation card** (`distill-gemma-e2b.yaml`): student `google/gemma-4-E2B`,
   teacher the instruction-tuned E4B variant, two-stage — a response-level pass on an
   F4 dataset, then an online logit-KL pass with alpha and temperature recorded from
   validation. The card records the teacher dataset manifest it was validated against.
3. Both cards run end to end (train, export, evaluate) as the final Phase 1 validation;
   their scorecards are committed under `docs/training/scorecards/` as reference
   results.

Acceptance: each recipe loads through D5 validation and runs end to end via its entry
point with no overrides; committed scorecards demonstrate the two scope criteria (SFT
model beats its base; distilled student beats its pre-distillation baseline).

## Current repository state

All F8 prerequisites (F1–F7) are implemented and merged; F7 landed last (PR #38). What
F8 consumes:

| Element | Repository state |
|---------|------------------|
| Recipe schema and validation | `training/recipes/schema.py` (typed `Recipe` dataclass: `run, model, data, tokenizer, optimizer, schedule, precision, memory, peft, sft, distill, checkpoint, tracking, watchdog, staging, affinity`), loaded via `training/recipes/loader.py:load_recipe`, validated by `training/recipes/validate.py` (single aggregated `RecipeValidationError`). QLoRA and distillation invariants already enforced (nf4 requires `peft.enabled`; logit tier is trl-only; cached mode forbids packing). |
| Stand-in recipes | `training/recipes/reference/sft-qlora-e2b.yaml` and `training/recipes/reference/distill-e2b.yaml`, both headed "F8 stand-in". The SFT stand-in targets **E2B**, seq_len 2048, `HuggingFaceH4/ultrachat_200k`-prepared dataset. The distill stand-in is single-stage logit-tier online, teacher `google/gemma-4-E4B` (base, not `-it`), alpha 0.5, temperature 2.0. |
| Entry points | `make train-sft`, `make train-distill`, `make distill-cache`, `make export-gguf`, `make eval-ft` — all through the unified `python -m training` CLI plus host scripts. No F8-specific target exists. |
| Data prep (F1) | `scripts/training/prepare_sft.py` (`make prepare-sft`) with `--revision` pinning and manifest hashing; loaders refuse hash mismatches. |
| Teacher generation (F4) | `scripts/training/generate_teacher.py` (`make generate-teacher`), respond/synthesize pipelines through the serving stack, manifest under `/mnt/ai-data/training/datasets/distill/`; `distill.teacher_manifest` cites it. |
| Evaluation (F7) | `scripts/eval/eval_ft.py` (`make eval-ft`) via `training/eval/runner.py`; produces `scorecard.json` (schema_version 1) and `scorecard.md` under `/mnt/ai-data/training/eval/<name>_<UTC>/`. "Beats base" is expressed per metric as `delta` plus an `improved` flag, and judge-side as `win_rate > 0.5`. Confound guard refuses mismatched `quant_type`/`thinking_mode`. |
| Scorecard home | `docs/training/scorecards/` does not exist. `docs/training/scope.md` also does not exist in the repository (the design document references it). |
| Revision pins | No recipe pins `model.hf.revision`; the resolved revision is recorded post hoc in checkpoint `meta.json`. |

## Design-document drift

Repository wins on each of these:

1. **File names and location.** The design document names
   `training/recipes/sft-qlora-gemma-e4b.yaml` and `training/recipes/distill-gemma-e2b.yaml`
   at the top of `training/recipes/`; the repository keeps recipes under
   `training/recipes/reference/` (the schema/loader/validate modules occupy
   `training/recipes/`). F8 cards should live in `training/recipes/reference/` (or a
   sibling `cards/` directory) — placing YAML next to `schema.py` breaks the shipped
   layout.
2. **SFT target model.** The design document targets E4B for the SFT card; the shipped
   stand-in (`sft-qlora-e2b.yaml`) targets E2B with seq_len 2048 and states it is the
   F8 stand-in. Whether F8 promotes the stand-in as-is (E2B) or steps up to E4B/4096 per
   the design is an open question (see below); either way the card is a new validated
   artifact, not a rename.
3. **Two-stage distillation as one card.** The schema models one run per recipe
   (`distill.tier` is a single value). The design document's "two-stage as documented in
   the card" cannot be one YAML executed by one entry point; the shipped shape is one
   recipe per stage (response-tier card plus logit-tier card, or a documented two-step
   procedure over one card with an override — the former matches "no overrides"
   acceptance).
4. **Teacher variant.** The design document says the teacher is the instruction-tuned
   E4B; the shipped stand-in uses base `google/gemma-4-E4B`. For response-level
   generation through F4 the teacher must be a chat model, so `-it` is correct there;
   for the logit pass the tokenizer-identity preflight must still hold.
5. **Fallback family.** The design document's fallback is "a supported dense student of
   similar size"; current policy supersedes this: fallback is the Qwen3 family (e.g.
   Qwen3-4B teacher, Qwen3-1.7B student), which has mature TRL/PEFT/llama.cpp support.

## Gap analysis

F8 is orchestration plus documentation; almost no new library code:

1. Two (likely three, given the two-stage split) validated recipe YAMLs with measured
   VRAM, wall-clock, and scorecard results in comments.
2. `docs/training/scorecards/` with the committed reference `scorecard.md`/`.json`
   pairs, plus a README stating provenance (run date, git SHA, command).
3. A defined promotion step from `/mnt/ai-data/training/eval/<run>/scorecard.*` into
   `docs/training/scorecards/` (manual copy documented, or a small `make
   promote-scorecard` helper).
4. Pinning: dataset source revision (F1 `--revision`), model `revision` fields in the
   cards, teacher dataset manifest reference in the distill card.
5. A unit test that each committed card loads through `load_recipe` and passes
   `validate()` unmodified (cheap, sandbox-runnable; guards against schema drift).
6. The validation runs themselves — GPU, host-side, human-executed per the operating
   rules. This is the bulk of F8's calendar cost: prepare data, train, export, evaluate,
   twice (SFT and distill), with one GPU consumer at a time.

## Design decisions

1. **Cards are new files, stand-ins retired or kept as smoke recipes.** The stand-ins
   are tuned for cheap exercise of F2/F3/F5; the cards record a real validated run.
   Keep both, with headers cross-referencing.
2. **Distillation ships as two cards** (`distill-response-*.yaml`, `distill-logit-*.yaml`)
   run in sequence, satisfying "runs end to end via its entry point with no overrides"
   per stage. The logit card's `model.hf` (or checkpoint staging) starts from the
   response-stage output; the exact chaining mechanism (checkpoint path in the recipe
   vs. adapter merge between stages) must be confirmed against `staging`/`checkpoint`
   semantics during implementation.
3. **Scorecards are committed verbatim** (both `.json` and `.md`) with a provenance
   header file, never edited by hand; re-validation replaces the pair wholesale.
4. **Every hub reference in a card is pinned**: `model.hf.revision`, teacher revision,
   dataset revision in the prepared-dataset manifest. "Validated artifact" implies a
   reproducible input closure.

## Gemma 4 considerations

Verified July 2026 (Google/HF doc pages were partially unreachable through the proxy;
items below marked verify-at-implementation where confirmation came from secondary
sources):

- **Hub ids**: base `google/gemma-4-E2B`/`google/gemma-4-E4B`, instruction-tuned `-it`
  suffix. Weights assumed gated (`HF_TOKEN` via `.env.training`, D6 convention).
- **Architecture**: multimodal wrapper config (`Gemma4Config` with `text_config`);
  per-layer embeddings (PLE), mixed sliding/full attention (5:1, window 512), KV
  projection sharing on E-models. The repository's memory calculator already models
  `gemma4`/`gemma4_text`.
- **Known training bug (critical)**: reported E2B/E4B issue where KV-shared layers
  recompute K/V under `use_cache=False` (forced by gradient checkpointing), corrupting
  logits and diverging loss. Both cards use gradient checkpointing. Verify the pinned
  transformers version carries the fix before any long run; make this a preflight note
  in both cards.
- **QLoRA fit**: E4B QLoRA reported to fit in roughly 16GB — comfortable in 32GB, and
  the D4 calculator acceptance already requires prediction within 10 percent of
  measured peak for the F8 SFT recipe. Online logit distillation (nf4 teacher + QLoRA
  student) is the tighter fit; the calculator must confirm before the run.
- **Thinking mode**: Gemma 4 uses explicit thinking-control tokens; F7's confound guard
  matches `thinking_mode` across sides, so both cards' evaluation runs must state it
  explicitly (expected: disabled for the SFT/distill comparisons).
- **llama.cpp/GGUF**: supported since the April 2026 launch window; the pinned llama.cpp
  revision must include the Gemma 4 chat template or serving-stack generation garbles.
- **Qwen3 fallback**: ungated, mature TRL/PEFT/Unsloth/llama.cpp support, documented
  `enable_thinking` toggle. If a Gemma 4 tooling bug blocks a validation run, the
  fallback pair is Qwen3-4B (teacher/SFT target) and Qwen3-1.7B (student), recorded as
  a deviation in this document.

## Proposed layout

```
training/recipes/reference/sft-qlora-gemma-e4b.yaml      # or e2b, see open question 1
training/recipes/reference/distill-response-gemma-e2b.yaml
training/recipes/reference/distill-logit-gemma-e2b.yaml
docs/training/scorecards/README.md
docs/training/scorecards/sft-qlora-gemma-e4b.scorecard.{json,md}
docs/training/scorecards/distill-gemma-e2b.scorecard.{json,md}
tests/training/test_recipe_cards.py                      # load + validate each card
```

## Work breakdown

1. Resolve the open questions below; record decisions in this document.
2. Write the three cards (unvalidated draft state, marked as such in headers); add the
   load-and-validate unit test; create `docs/training/scorecards/` with its README.
3. Human-executed validation sequence (host, one GPU consumer at a time):
   `make prepare-sft` (pinned revision) → `make train-sft CONFIG=<sft card>` →
   `make export-gguf` → `make eval-ft` → record scorecard; then
   `make generate-teacher` → response-stage `make train-distill` → logit-stage
   `make train-distill` → `make eval-ft --baseline <pre-distill student>` → record.
4. Back-fill the cards' comments with measured VRAM, wall-clock, and scorecard results;
   commit the scorecards; flip the headers from draft to validated.
5. Run the acceptance checks and record commands and outputs verbatim.

## Acceptance criteria

- Each committed card loads through `load_recipe` and passes `validate()` with no
  overrides (unit test, sandbox-runnable).
- Each card runs end to end via its Makefile entry point with no overrides
  (host-side, human-reported output recorded).
- `docs/training/scorecards/` contains both scorecards showing: SFT model beats its
  base (held-out loss and judge win rate per F7's `improved`/`win_rate` semantics), and
  the distilled student beats its pre-distillation baseline.
- D4 calculator prediction for the SFT card is within 10 percent of measured peak VRAM
  (F2 acceptance closes here).

## Open questions

1. **SFT card target: E4B (design document) or E2B (shipped stand-in)?** E4B at
   seq_len 4096 matches the design and the "small-model base" exit criterion; E2B is
   already smoke-tested end to end. Recommendation: E4B, keeping the E2B stand-in as
   the cheap smoke recipe.
2. **Two distill cards vs. one card plus documented second step?** Recommendation: two
   cards (matches "no overrides" acceptance).
3. **Scorecard promotion**: manual documented copy or a `make promote-scorecard`
   helper that also stamps provenance? Recommendation: small helper, so provenance is
   never hand-typed.
4. **Dataset pin for the SFT card**: default to `HuggingFaceH4/ultrachat_200k` at a
   fixed revision (matching the stand-in lineage), or choose a smaller pinned subset to
   bound the validation run's wall-clock?
5. **Teacher for the logit stage**: `-it` variant (matching the response stage) or the
   base E4B as in the stand-in? Tokenizer identity should hold either way but must be
   verified by the preflight.

## Resolved design decisions

Resolved 2026-07-15 with the repository owner; the open questions above are kept
for the record.

1. **SFT card targets E4B** at sequence length 4096 (`sft-qlora-gemma-e4b.yaml`).
   The E2B stand-in (`sft-qlora-e2b.yaml`) stays as the cheap smoke recipe.
2. **Two distillation cards run in sequence**: `distill-response-gemma-e2b.yaml`
   (stage 1, response tier on F4 output), then `distill-logit-gemma-e2b.yaml`
   (stage 2, online logit-KL). Stage 2's `model.hf.id` points at stage 1's
   adapter merged through the F6 export path, so each card runs via its entry
   point with no overrides.
3. **`make promote-scorecard`** (`scripts/eval/promote_scorecard.py`, host-run)
   copies an eval run's `scorecard.json`/`scorecard.md` verbatim into
   `docs/training/scorecards/` as `<card>.scorecard.*` and writes
   `<card>.provenance.json` (source run directory, git SHA, promotion time,
   file hashes). Committed scorecards are never edited by hand.
4. **Dataset pinned to `HuggingFaceH4/ultrachat_200k`** at a fixed revision,
   recorded in the prepared dataset's manifest via `prepare_sft.py --revision`
   and noted in the card comments.
5. **Teacher is `google/gemma-4-E4B-it` for both stages.** The response stage
   requires a chat model, and keeping the same teacher identity for the logit
   pass keeps the two stages consistent; the `-it` variant shares the base
   tokenizer, so the logit-tier identity preflight holds (verified by the
   preflight at run time).

Implementation state: the three cards ship in draft form (headers say so) with
TBD fields for measured VRAM, wall-clock, scorecard results, and pinned hub
revisions; the F8 validation runs on the workstation back-fill them. The
existing parametrized recipe test (`tests/training/test_recipes.py`) validates
all three cards; `tests/training/test_promote_scorecard.py` covers the
promotion helper.

## Validation runbook (host-side, human-executed)

One GPU consumer at a time, in order. Record each command and its output
verbatim when marking acceptance.

```bash
# SFT card
$ make prepare-sft ARGS="--source HuggingFaceH4/ultrachat_200k --split train_sft --revision <pinned> --tokenizer google/gemma-4-E4B --model-type gemma4 --max-seq-len 4096 --out /mnt/ai-data/training/datasets/sft/ultrachat-gemma-e4b"
$ make memfit CONFIG=training/recipes/reference/sft-qlora-gemma-e4b.yaml
$ make train-sft CONFIG=training/recipes/reference/sft-qlora-gemma-e4b.yaml
$ make export-gguf CHECKPOINT=<final checkpoint> ARGS="--quant Q8_0"
$ make eval-ft ARGS="--base google/gemma-4-E4B --tuned <checkpoint> --dataset /mnt/ai-data/training/datasets/sft/ultrachat-gemma-e4b --base-gguf <base gguf> --tuned-gguf <tuned gguf> --judge-gguf <judge>"
$ make promote-scorecard ARGS="--run-dir <eval run dir> --card sft-qlora-gemma-e4b"

# Distillation cards
$ make generate-teacher ARGS="--pipeline respond --prompts <prompt set> --teacher google/gemma-4-E4B-it --out /mnt/ai-data/training/datasets/distill/teacher-e4b-responses"
$ make prepare-sft ARGS="--source /mnt/ai-data/training/datasets/distill/teacher-e4b-responses --tokenizer google/gemma-4-E2B --model-type gemma4 --max-seq-len 4096 --out /mnt/ai-data/training/datasets/sft/teacher-e4b-responses"
$ make train-distill CONFIG=training/recipes/reference/distill-response-gemma-e2b.yaml
$ make export-gguf CHECKPOINT=<stage-1 checkpoint> ARGS="..."   # merged output feeds stage 2's model.hf.id
$ make train-distill CONFIG=training/recipes/reference/distill-logit-gemma-e2b.yaml
$ make eval-ft ARGS="--base <pre-distillation E2B> --tuned <stage-2 checkpoint> ..."
$ make promote-scorecard ARGS="--run-dir <eval run dir> --card distill-gemma-e2b"
```

After each run: back-fill the card's TBD comment fields (measured peak VRAM,
wall-clock, scorecard result, pinned revisions), commit the promoted
scorecards, and flip the card header from draft to validated.
