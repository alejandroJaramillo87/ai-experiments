# Phase 1 completeness audit

**Frozen record — 2026-09-20. Drift against the current repository is expected and correct;
this document is never updated.**

Audit of the eight Phase 1 deliverables (F1 through F8, the fine-tuning harness) against
six sources at once: the Phase 1 design that scoped them (the F-numbered specification,
written against the 2026-07-08 product design), the living documentation under
`docs/training/`, the code under `training/`, the test suite, the fine-tuning runbook, and
the consumers in the two later tiers, `interp/` and `rust-harness/`. The Phase 0 audit of
2026-09-19 asked whether each foundation piece was implemented, tested, documented,
hardware-verified and still covered by a live runbook entry. This audit asks a fifth
question the tier's progression makes possible: whether the piece is still needed. The
tier exists to produce artifacts with a known change that the rust harness serves and the
interp tier recovers (`docs/training/README.md`, Progression), so a Phase 1 piece is
needed when a rung or a consumer calls it, and superseded when the design's reason for it
left with distillation, GGUF production, or the model-quality deliverables.

## Table of contents

- [Method](#method)
- [Matrix](#matrix)
- [Per deliverable](#per-deliverable)
- [Decisions that superseded the design](#decisions-that-superseded-the-design)
- [Findings](#findings)
- [Actions taken on 2026-09-20](#actions-taken-on-2026-09-20)

## Method

Five questions per deliverable, each answered from one source:

| Question | Source | Passing answer |
|----------|--------|----------------|
| Implemented | `training/`, `make/training.mk` | the module and its entry point exist |
| Tested | `tests/training/`, `tests/hardware/` | a unit test asserts each acceptance behaviour the design named; a hardware test where the bar is a measurement |
| Documented | `docs/training/README.md` and the document it indexes | a living document, not only the frozen investigation |
| Verified | `docs/verification/training/fine-tuning-loop.md` | a `Status: VERIFIED` entry with pasted output, or a `PENDING` entry someone can still run |
| Needed | `interp/`, `rust-harness/`, `make/*.mk`, the runbooks | a path in a later tier, a make target of a rung, or a runbook entry of a rung calls it |

Candidate dead code was taken from a symbol sweep (every public name with no reference
outside its own file and the tests) and then checked by grep before anything was removed.
Every candidate the sweep produced turned out to have a caller inside its own module
(`iter_records`, `weights_files`, `materialize_hf_source`, `build_manifest`,
`paged_adamw8bit_available` through the optimizer registry, `collate_padded` through
`Collator`), so the sweep removed nothing; a sweep that reads only cross-file references
is a list of candidates, not a verdict.

## Matrix

Impl: code exists. Unit: CPU test of the design's acceptance behaviour. Doc: living document.
HW: verified on the workstation. Live: a pending entry in the live runbook covers it.
Needed: what calls it.

| Deliverable | Impl | Unit | Doc | HW | Live | Needed | Verdict |
|-------------|------|------|-----|----|------|--------|---------|
| F1 data preparation | yes | yes | none until this audit | FT.smoke-dataset VERIFIED | FT.train-resume | `train-sft`, `organism-generate` (the record format) | done; undocumented |
| F2 PEFT schema | yes | yes; DoRA at the schema only | `recipe_system.md`, `gemma-4-training.md` | FT.recipe-gates, FT.calibrate-qlora VERIFIED | FT.train-resume | every organism recipe | done; DoRA retired |
| F3 SFT entry point | yes | yes | `sft_entry_point.md` | none | FT.train-resume, FT.heldout-eval | every organism, FT.organisms | unverified past the step |
| F4 teacher harness | yes, `respond` only | yes | `teacher_synthetic_data.md`, distillation-era framing | none | FT.organisms (through `organism-generate`) | `interp/organisms/generate.py` | done; reframed |
| F5 distillation | removed 2026-09-18 | none | the record | none | none | nothing | done |
| F6 adapter merge and serve | merge half; serve half is the harness | yes | `train_serve_roundtrip.md` | none | FT.export-merged | the harness, interp capture, `organism-seal` | unverified |
| F7 evaluation | holdout and lmeval; judge removed 2026-09-19 | yes | `eval_wrapper.md` | none | FT.heldout-eval, FT.eval-ft | `organism-verify` (the records) | scorecard retired |
| F8 recipe cards | E2B card; E4B card unused; promotion removed 2026-09-19 | yes | `recipe_system.md` | FT.recipe-gates, FT.calibrate-qlora VERIFIED (E2B) | none for E4B | E2B by every organism; E4B by nothing | E4B card retired |

## Per deliverable

### F1 chat and instruction data preparation

Code: `training/data/sft.py` (record normalisation, `render_segments`, `encode_example`,
the assistant-span mask, `mask_debug`, the seeded held-out split, manifest and verified
loader), the chat-template registry in `training/models/families.py` (the design's
`training/data/templates.py`), `training/commands/prepare_sft.py`. Target: `prepare-sft`.
Tests: `tests/training/test_sft.py` (golden mask boundaries for single-turn, multi-turn and
system-prompt cases, the round trip that detokenises a sample back to the rendered
conversation, deterministic re-runs including split membership, the manifest),
`tests/training/test_chat_template_contract.py` (the Python half of the contract the Rust
renderer is held to), `tests/training/test_cli_handlers_prepare.py`. Verified:
FT.smoke-dataset on 2026-09-19, the manifest and the refusal.

Two design points moved. Packing is F3's, not F1's: `training/tune/sft.py::pack_examples`
packs at training time and the holdout is never packed; the design's "tested in both modes"
was met for the packed path only, and this audit adds the unpacked assertion. The registry
holds `gemma4` and `gemma4_text` alone; `llama3`, the design's second family that was to
prove the registry general, left with the fallback family on 2026-09-18, and an unknown
family still fails with the registration message.

Documentation: none. The F1 record was the only description. This audit adds
`fine-tuning/sft_data.md` in the restructured `docs/training/`.

Later tiers: the record format is what `organism-generate` writes and `prepare-sft`
ingests; `render_segments` is the byte-for-byte reference for
`rust-harness/harness/tests/template_contract.rs`.

### F2 parameter-efficient options in the recipe schema

Code: `PeftConfig` in `training/recipes/schema.py`, `_check_peft` in
`training/recipes/validate.py`, `resolve_target_modules`, `build_peft_model` and
`trainable_parameter_report` in `training/models/peft_utils.py` (the design's
`training/peft_utils.py`), `adapter_param_count` in `training/models/architecture.py`
(the design's "QLoRA-aware memory model", now a parameter count the step reads). Tests:
`tests/training/test_recipes.py` (each invariant), `tests/training/test_peft_utils.py`
(`auto` resolution per family, the snapshot never carries `auto`, the wrap confined to the
language model), `tests/training/test_architecture.py` (the adapter priced per layer, KB-90).
Verified: FT.recipe-gates on 2026-09-19 (validate and preflight), FT.calibrate-qlora on
2026-09-20 (the wrap on the real base, the count equal to the built adapter, the trainable
report under 5 percent), with `tests/hardware/test_step_qlora.py` as its standing mirror.

DoRA was schema-complete and reached nothing: no recipe set `peft.method: dora`, no test
asserted `use_dora` reached `LoraConfig`, no runbook entry named it. Retired by this audit;
`peft.method` admits `lora`.

Later tiers: the resolved module list in every run's snapshot is what the exporter's merge
attaches to.

### F3 supervised fine-tuning entry point

Code: `training/tune/sft.py` (`train_sft`, `sft_config_kwargs`, `build_dataset`,
`pack_examples`, the chunked-loss trainer), `training/tune/sft_callbacks.py` (retention,
`meta.json` with the adapter-only flag F6 reads, tracking and watchdog bridges), `cmd_train_sft`
in `training/cli.py`. Target: `train-sft`. Tests: `tests/training/test_sft_entry.py` (the
config mapping, packing, retention, the meta fields, the holdout eval wiring, the lock),
`tests/training/test_cli_handlers_train.py`. Verified: none past the step. FT.train-resume
(straight, interrupted, resumed) and FT.heldout-eval are the pending entries, and every
organism depends on them.

Three design points left before this audit and are recorded here: the `unsloth` backend
(never shipped; `SFT_BACKENDS` is `trl` alone since 2026-09-19), the inference-services-down
preflight (removed; the operator manages containers, `train-sft` warned instead of refusing)
and the `--force` flag on `train-sft` (gone; `test_force_flag_is_gone` holds it).

Later tiers: the TRL checkpoint directory with its `meta.json` is the exporter's input.

### F4 teacher and synthetic data harness

Code: `training/data/teacher.py` (prompt normalisation, stable keys, resumability from
`data.jsonl`, retry, record and manifest schemas, transport injected),
`training/commands/generate_teacher.py`. Target: `generate-teacher`. Tests:
`tests/training/test_teacher.py`, `tests/training/test_cli_handlers_teacher.py`. Verified:
none directly; FT.organisms runs it through `organism-generate`.

The design named distilabel; the tree is a thin standard-library client, recorded in the F4
investigation's resolved decision 1. The `synthesize` pipeline left on 2026-09-19 with no
consumer. The harness is needed: `interp/organisms/generate.py` renders the organism
prompts and hands them to `training.data.teacher`, and `interp/commands/organism.py`
invokes the command. What was stale was the framing: the doc opened as the distillation
tool with an ultrachat example, and the default output root was `datasets/distill`. Both
are corrected by this audit (root `datasets/teacher`).

### F5 distillation harness

Removed on 2026-09-18 with a full retirement record in
`scripts/docs/check_retired_names.py`; the reason is the one the Progression states, that
distillation makes no claim about mechanism the interp tier can test. Nothing of it remains
but the record, [f5_distillation_investigation.md](f5_distillation_investigation.md), and
the two names corrected by this audit (the teacher root and one comment).

### F6 adapter merge and serve

The design's round trip (adapter merged, GGUF quantised through a pinned llama.cpp, swapped
into `llama-gpu`, mergekit for multi-adapter merges) was replaced on 2026-09-18. The merge
half stays in the training tier: `training/export/merge.py` (`merge_and_unload` on an
adapter-only checkpoint, the base resolved from `meta.json`), `training/export/preflight.py`
(the base revision and config hash at export time held to the ones trained against, the
design's preflight), `training/export/consolidate.py` (one `model.safetensors`),
`training/export/manifest.py` (adapter path and hash, base identity, merge method,
`model_hash`). The serve half is the rust harness: `make harness-serve` on the exported
directory, and `harness --status --deep` compares the served weights against `model_hash`.
mergekit, its `merge` target and `training/export/merges/` were never built and are not needed:
no organism merges two adapters. Target: `export-hf`. Tests: `tests/training/test_export.py`,
`test_export_common_branches.py`, `test_export_harness.py`, `test_cli_handlers_export.py`,
and on the consumer side `rust-harness/runtime/tests/python_exporter_contract.rs`. Verified:
none; FT.export-merged is the pending entry.

Later tiers: the exported directory is what the harness serves, interp captures from and
`organism-seal` records.

### F7 fine-tuning evaluation wrapper

Code: `training/eval/holdout.py`, `training/eval/lmeval.py` with `tasks.yaml`,
`training/eval/runner.py`, `training/commands/eval_ft.py`. Target: `eval-ft`. Tests:
`tests/training/test_holdout.py`, `test_lmeval.py`, `test_eval_runner.py`, `test_eval_ft.py`
(KB-78, the container spelling). Verified: none; FT.heldout-eval and FT.eval-ft are pending.

Of the design's three evaluators the judge left on 2026-09-19 (preference, which no rung asks
about). The two that remain are needed: `interp/organisms/verify.py::locality_metrics` reads
the `eval-holdout` and `eval-lmeval` records for the base and the organism, and
`make organism-verify` takes them as `BASE_METRICS` and `SUBJECT_METRICS`. The scorecard
assembled over those records (`training/eval/scorecard.py`, the delta sections in the runner,
`scorecard.json` and `scorecard.md` in the run directory) had no reader anywhere in the tree;
the base-versus-organism judgement is the locality check's. Retired by this audit. The driver
stays: it runs the four phases under the lock, resumes from torn or absent records, holds the
two records of an evaluator to one precision, dataset, example count, version and limit, and
prints each side's `model_hash`.

The design's `--baseline` for a distilled student against its pre-distillation self is the
same machinery with a different base and went with F5.

### F8 recipe cards

Code: `training/recipes/reference/sft-qlora-e2b.yaml`, the card every organism recipe
follows and every fine-tuning runbook entry steps. The E4B card
(`sft-qlora-gemma-e4b.yaml`) was the scope document's artifact and was referenced by no
make target and no organism; its runbook entry, FT.e4b-recipe, would have verified the same
two gates and the same wrap on a larger base, and its `sft.dataset` named a split nobody
prepared. Retired by this audit with the entry; the E4B parameter facts stay in
`gemma-4-training.md` because they describe the model. The distillation card left with F5,
and the promotion step with its committed-scorecard directory on 2026-09-19. Tests:
`tests/training/test_recipe_pins.py`, `test_recipes.py`, `tests/hardware/test_recipe_gates.py`.
Verified: FT.recipe-gates and FT.calibrate-qlora, on the E2B card.

## Decisions that superseded the design

Each is recorded elsewhere; this list is the index.

| Design statement | What the tree does | Where recorded |
|------------------|--------------------|----------------|
| `scripts/train/*.py` entry points | one CLI, `python -m training <verb>`, handlers under `training/commands/` | `unified_cli_investigation.md` |
| `gemma4` and `llama3` in the template registry | `gemma4` and `gemma4_text`; the fallback family left | `training/models/families.py` docstring |
| Packing in F1's preparation | packing at training time, the holdout never packed | `training/tune/sft.py` |
| distilabel as the pipeline engine | a standard-library client loop, `engine` recorded in the manifest | `f4_teacher_synthetic_data_investigation.md`, decision 1 |
| `synthesize` pipeline | removed 2026-09-19, no consumer | `docs/training/teacher_synthetic_data.md` |
| GGUF export through pinned llama.cpp; mergekit | merged HF directory served by the rust harness; no multi-adapter merge | `training/export/__init__.py`, `AGENTS.md` |
| `serve-eval` smoke through `llama-gpu` | `make harness-serve` and `harness --status --deep` | `scripts/docs/check_retired_names.py` |
| Model-as-judge evaluator | removed 2026-09-19 | `training/commands/eval_ft.py` |
| Scorecard as the machine-readable record | the four evaluator records, read by `organism-verify` | this audit |
| `unsloth` backend | never shipped; `trl` alone | `training/recipes/schema.py` |
| Inference-down preflight and `--force` on `train-sft` | removed; the operator manages containers | `training/tune/sft.py` |
| E4B SFT card and distillation card, both with committed scorecards | one card, E2B; no promotion | `f8_recipe_cards_investigation.md`, this audit |
| DoRA in the schema | `lora` alone | this audit |

## Findings

Numbered for reference. Those marked acted on are in the actions below; the rest are
recorded for the next pass.

1. F1 had no living document (acted on: `fine-tuning/sft_data.md`).
2. Packing was tested on the packed path only (acted on: the unpacked assertion in
   `tests/training/test_sft_entry.py`).
3. DoRA, the E4B card, the scorecard: built for the design, called by nothing (acted on:
   retired).
4. The teacher harness's doc and default root still described distillation (acted on).
5. Four stale sentences: `training/README.md` listed "VRAM fit estimation" after the
   calculator retired and grouped `eval-ft` with the serving-stack targets after the judge
   left; `docs/training/README.md` counted three evaluators; two `pyproject.toml` comments
   said training still produced GGUF (acted on).
6. Continual pretraining has no living document, only the four P2 records (acted on in the
   restructure: `continual/README.md`).
7. F3, F6 and F7 are unit-tested against fakes and never executed on hardware past the step:
   FT.train-resume, FT.export-merged, FT.heldout-eval and FT.eval-ft are the entries, in that
   order, and every organism depends on them.
8. `tests/training/test_obs_*.py` and `test_report.py` test `common/obs`, not `training/`,
   and belong under a `tests/obs/` directory when one is made.
9. `training/calculator/` survives on disk as an untracked `__pycache__` only; delete it on
   the workstation.
10. The four branch-coverage test files (`test_data_tail_branches.py`,
    `test_export_common_branches.py`, `test_recipes_tracking_branches.py`,
    `test_watchdog_branches.py`) were read against the trimmed modules; none covered a
    branch the trim removed.

## Actions taken on 2026-09-20

- Retired `training/recipes/reference/sft-qlora-gemma-e4b.yaml` and FT.e4b-recipe; the
  runbook is 14 entries, and the ledger rows that named the entry no longer do.
- Retired `peft.method: dora`: `PEFT_METHODS` is `("lora",)`, `build_peft_model` passes no
  `use_dora`, `adapter_param_count` prices LoRA alone, and a recipe asking for `dora` is a
  validation error.
- Retired `training/eval/scorecard.py` and its test; `eval-ft` writes the four records and
  prints each side's identity; `training/eval/runner.py` holds the two records of an
  evaluator to one configuration (`check_holdout_pair`, `check_lmeval_pair`); FT.eval-ft's
  expected signal names the records.
- Reframed `docs/training/teacher_synthetic_data.md` around the organism generator; the
  default output root is `/mnt/ai-data/training/datasets/teacher`.
- Corrected the stale sentences in finding 5, and the four names above are in
  `scripts/docs/check_retired_names.py`.
- Added the unpacked-path assertion to `tests/training/test_sft_entry.py`.
- The two living documents in findings 1 and 6 are written with the `docs/training/`
  restructure (one directory per rung) the same day.
