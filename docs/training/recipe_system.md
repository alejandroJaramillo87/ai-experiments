# Recipe system

Phase 0 D5: one YAML file fully specifies a run, and the same file is read by every consumer
of it. The schema is a set of dataclasses in `/training/recipes/schema.py`; a recipe file
merges against them in struct mode, so an unknown key or a mistyped value fails at load
rather than mid-run. Design record:
[d5_recipe_system_investigation.md](investigations/d5_recipe_system_investigation.md).

## Table of contents

- [Schema](#schema)
- [Loading and overrides](#loading-and-overrides)
- [Validation](#validation)
- [Run directory and snapshot](#run-directory-and-snapshot)
- [Reference recipes](#reference-recipes)
- [Consumers in other tiers](#consumers-in-other-tiers)
- [Validate compared with preflight](#validate-compared-with-preflight)
- [Testing](#testing)

## Schema

`Recipe` is the top-level dataclass. Its groups, in the order they appear in
`/training/recipes/schema.py`:

| Group | Owns | Consumer |
|-------|------|----------|
| `run` | name, seed, output root, notes | every entry point |
| `model` | exactly one of `hf` (id, revision) or `scratch` (layers, hidden, heads, kv_heads, ffn_mult, vocab, rope_theta, tie_embeddings), plus `trainable_filter` | loop, calculator, SFT |
| `data` | shard directory or dataset id, sequence length, micro-batch, gradient accumulation, workers, `pin_memory`, `prefetch_factor` | loop, calculator, D9 loader |
| `mixture`, `replay` | Phase 2 multi-source sampling and the replay buffer | loop |
| `tokenizer` | path, optional for an HF base (defaults to `model.hf.id`, the tokenizer the base directory ships) and required for scratch; must hash-match the shards or the SFT dataset | validate, preflight, SFT |
| `optimizer` | name (`adamw`, `adamw8bit`, `paged_adamw8bit`, `cpu_adam`), lr, betas, eps, weight decay | loop, calculator |
| `schedule` | type (`cosine`, `wsd`), warmup, total steps, WSD decay knobs | loop |
| `precision` | `dtype`, `bf16` only | loop, calculator |
| `memory` | gradient checkpointing, offload (`none`, `zero_offload`), compile, ZeRO bucket sizes | loop, calculator |
| `peft` | LoRA and DoRA settings, NF4 base quantization | SFT, calculator |
| `sft` | dataset, backend, packing, evaluation cadence | SFT |
| `organism`, `harvest`, `sae` | interp-tier blocks, documented in `docs/interp/` | interp |
| `checkpoint` | interval, `retain_last`, `retain_every`, `resume` | loop, SFT |
| `tracking` | wandb, mlflow, tensorboard, `metrics_jsonl` | tracker |
| `watchdog` | thresholds and sampling interval, see [operational_watchdogs.md](operational_watchdogs.md) | watchdog |
| `staging`, `affinity` | D9 knobs, see [heterogeneous_compute.md](heterogeneous_compute.md) | loop |

Fields marked `MISSING` in the schema (`run.name`, `data.source`, `optimizer.lr`,
`schedule.total_steps`, and the model dimensions) have no default; a recipe that omits one
fails validation with the field named. `tokenizer.path` is required only for a scratch
model: an HF recipe reads the tokenizer its base directory ships, resolved by
`tokenizer_source()` in `/training/recipes/validate.py`, the one function validate, preflight
and `train-sft` share for it (KB-80 was three readers of a path nothing staged).

## Loading and overrides

`load_recipe(path, overrides)` in `/training/recipes/loader.py` merges the file against the
schema and then applies a dotlist of overrides (`optimizer.lr=1e-4`,
`model.hf.id=/app/models/hf/gemma-4-E2B`). It raises `RecipeLoadError` on a missing file,
an unknown key, or a type mismatch, with the OmegaConf error chained for the key and line.
Every `python -m training` verb that takes `--config` accepts overrides as trailing
positional arguments; the make targets pass them through `ARGS`:

```bash
make train-validate CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="optimizer.lr=1e-4"
```

## Validation

`validate(recipe)` in `/training/recipes/validate.py` runs every cross-field check, collects
every violation, and raises `RecipeValidationError` listing them all at once. Every entry point
calls it before allocating on the GPU. The checks, by the helper that owns each:

| Check | Rejects |
|-------|---------|
| `_check_missing` | any `MISSING` field left unset |
| `_check_model` | zero or two model variants; heads not divisible by kv_heads; an unknown `trainable_filter` |
| `_check_optimizer` | an unknown name; a non-positive lr; `cpu_adam` without `memory.offload: zero_offload` |
| `_check_schedule`, `_check_wsd` | an unknown type; warmup beyond total steps; WSD fractions out of range |
| `_check_precision` | a dtype other than `bf16` |
| `_check_memory`, `_check_offload_budget` | an unknown offload mode; an offload configuration whose host RAM estimate exceeds the machine |
| `_check_peft`, `_check_sft`, `_check_organism` | inconsistent adapter, fine-tuning, and organism blocks |
| `_check_data`, `_check_mixture`, `_check_replay` | non-positive geometry; a `prefetch_factor` below 1; a mixture or replay block with non-positive weights, duplicate source names, or a `data.source` it should replace |
| `_check_checkpoint`, `_check_watchdog` | non-positive intervals and thresholds |
| `_check_staging`, `_check_affinity` | a non-positive `max_gb`; overlapping or malformed CPU lists |
| `_check_tokenizer_matches_shards` | a tokenizer (`tokenizer.path`, or the base directory when unset) whose content hash differs from the shard manifest's, when `data.source` is a prepared shard directory |

The tokenizer check is skip-when-absent: a `data.source` with no `manifest.json` (a dataset
id, or shards not yet prepared on this machine) passes. That is what lets `train-validate`
run on a machine with no data, and why it is not a preflight.

## Run directory and snapshot

`create_run_dir(recipe)` creates `${run.output_root}/${run.name}/<YYYY-MM-DD_HH-MM-SS>/`,
refuses an existing directory, and writes the resolved recipe into it as `config.yaml`.
`load_snapshot` reloads that file through the same loader, and the reload is bit-identical
(asserted by `tests/training/test_recipes.py`). Checkpoints sit beside the timestamped
directories at `${run.output_root}/${run.name}/checkpoints/`, as
[tracking_checkpointing.md](tracking_checkpointing.md) describes. The loop hashes the
snapshot into `recipe_hash`, which every checkpoint's `meta.json` and the tracker's header
record carry.

## Reference recipes

`/training/recipes/reference/`:

| Recipe | Purpose |
|--------|---------|
| `tiny-e2e.yaml` | a seconds-long scratch run for acceptance tests; its shard and tokenizer paths are the fixtures the tests generate, so a workstation run overrides `data.source`, `tokenizer.path`, and `model.scratch.vocab` together |
| `pretrain-150m.yaml` | the D4 calibration and D9 acceptance workload; Gemma 4 block shape at reduced scale |
| `pretrain-1b.yaml` | the 32GB envelope configuration the calculator must report as fitting |
| `sft-qlora-e2b.yaml` | the QLoRA fine-tune of Gemma 4 E2B the fine-tuning runbook verifies first |
| `sft-qlora-gemma-e4b.yaml` | the E4B fine-tuning reference |
| `continual-gemma4-e2b.yaml` | the Phase 2 ZeRO-Offload continual-pretraining calibration recipe |

Interp recipes live under `/training/recipes/interp/` and organism recipes under
`/training/recipes/organisms/`, because the schema and loader belong to this package.

## Consumers in other tiers

The loader, schema, and validator are a cross-tier API, not training-private code.
`interp/commands/harvest.py`, `interp/commands/sae_train.py`, and
`interp/commands/organism.py` import `load_recipe` and `validate`, and the interp tests
import the `Recipe` dataclass. A change to a group's field names is a change to interp.

Every recipe naming a Gemma 4 hub id must pin `model.hf.revision` to the commit recorded in
`/schemas/gemma4_facts.json`; `tests/training/test_recipe_pins.py` enforces it for the
reference, organism, and interp recipes, reading `model.hf` or `harvest` whichever names the
hub id. The training container cannot resolve a hub id at all (its
Hugging Face cache is an empty tmpfs), so a container run overrides `model.hf.id` with the
directory `make train-download` staged, as the fine-tuning runbook records under KB-79.

## Validate compared with preflight

`make train-validate` answers "is this recipe coherent" and passes with no data present.
`make train-preflight` answers "will this run start": it requires every shard directory,
tokenizer, and model the recipe references to be present and verified, and reports every
missing one, once: an HF recipe without `tokenizer.path` reads the base directory's
tokenizer, so an absent base is one problem line, not two. An SFT recipe (`sft.enabled`) is checked against its prepared `sft.dataset`
through the F1 loader, so the manifest kind, the chat-template version, the file hashes and
the example counts are the trainer's own checks; its `data.source` is not read, because the
reference recipes point it at the same directory and it is not a shard directory (KB-81 was
preflight verifying it as one). Run preflight immediately before a long job. The distinction is documented with
the container in
[../sandbox/docker_training_overview.md](../sandbox/docker_training_overview.md#validate-compared-with-preflight).

## Testing

`/tests/training/test_recipes.py` loads every reference recipe as a typed object, rejects
misspelled and mistyped keys, applies dotlist overrides, reloads the snapshot bit-identically,
refuses a duplicate run directory, and carries one test per validation invariant including
the tokenizer-hash check against a generated shard manifest. `/tests/training/test_recipe_pins.py`
guards the Gemma 4 revision pins. Host-side acceptance is FT.13 in
[../verification/training/fine-tuning-loop.md](../verification/training/fine-tuning-loop.md).
