# training/recipes

The recipe system and the recipes themselves. The system is a typed schema with one dataclass per section, a loader
that applies `key=value` overrides and snapshots the resolved recipe into the run
directory, and cross-field validation that refuses what the schema alone cannot see, such
as an offload recipe handed to the step command or a PEFT target list the family does not
resolve. Every recipe-driven verb loads through `training/commands/recipe.py`, so a recipe is
validated before torch imports.

Three directories of YAML live here because the loader lives here, and the interp tier
reads recipes through it:

| Directory | Holds |
|-----------|-------|
| `reference/` | the cards each rung is verified on: `sft-qlora-e2b.yaml`, `continual-gemma4-e2b.yaml`, `pretrain-150m.yaml`, `pretrain-1b.yaml`, and `tiny-e2e.yaml` for the smoke path |
| `organisms/` | the organism recipes `make organism RECIPE=...` runs, with their prompt sets under `prompts/<organism>/` |
| `interp/` | the harvest and SAE recipes the interp tier loads through this loader |

The schema is the one place a recipe field is defined; the living page below documents
every field, and `tests/training/test_recipe_pins.py` holds the reference cards to the
figures their runbook entries recorded.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `loader.py` | Recipe loading, overrides, and run-directory snapshots (Phase 0 D5). |
| `schema.py` | Recipe schema (Phase 0 D5). |
| `validate.py` | Cross-field recipe validation (Phase 0 D5). |

<!-- inventory:modules:end -->

Living page: [recipe_system.md](../../docs/training/base/recipe_system.md). Tests:
`tests/training/test_recipes.py`, `test_recipe_pins.py`, `test_recipes_tracking_branches.py`
and `test_cli_handlers_step.py`; the gates on the workstation are
`tests/hardware/test_recipe_gates.py` under `hw_smoke`.
