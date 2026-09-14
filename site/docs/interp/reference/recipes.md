---
title: Recipes
---

# Recipes

The batch verbs, `harvest`, `sae-train`, and the `organism` stages, read a YAML recipe rather
than a flag per setting. The recipes and the loader belong to the training tier, because the
schema and the run conventions are shared; the interp recipes live beside the training ones.

## Where they are

| Directory | Read by | Files |
|-----------|---------|-------|
| `training/recipes/interp/` | `harvest`, `sae-train` | `harvest-smoke.yaml`, `harvest-false-fact.yaml`, `harvest-calibration-gemma2.yaml`, `harvest-calibration-qwen3.yaml`, `sae-smoke.yaml`, `sae-false-fact.yaml`, `sae-calibration-gemma2.yaml`, `sae-calibration-qwen3.yaml` |
| `training/recipes/organisms/` | `organism generate`, `verify`, `seal` | `taboo.yaml`, `false-fact.yaml`, `null.yaml`, and `prompts/` |

Each `-smoke` recipe is sized to complete in minutes in the workbench and exercises the same
code path as the full-scale recipe beside it. Every recipe file opens with a comment stating
what it is for and the `make` command that runs it.

## Shape

A recipe is one document with a `run` block, one block for the verb it drives, and the
shared blocks the training loader knows:

```yaml
run:
  name: harvest-smoke
  seed: 42

harvest:
  enabled: true
  model_id: google/gemma-4-E2B
  hook_layers: [17]
  corpus: monology/pile-uncopyrighted
  context_len: 1024
  total_tokens: 2000000
```

The loader merges the file over the schema's defaults, so a recipe states only what differs.
`run.name` names the run directory and the artifact directory the verb writes.

## Overrides

`harvest` and `sae-train` accept `--override key=value`, repeatable, applied after the file in
dotlist form:

```bash
make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml \
    ARGS="--override harvest.total_tokens=500000"
make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml \
    ARGS="--override sae.train_tokens=500000 --override run.name=sae-smoke-short"
```

An override names a key the schema has; an unknown key is a recipe load error and the verb
exits 1 before touching the GPU. The organism stages take their settings from the recipe and
their flags and offer no override.
