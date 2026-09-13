---
title: Harvesting activations
---

# Harvesting activations

A harvest runs a corpus through a model and writes the residual-stream activations at chosen
layers to NVMe, as the cached dataset a sparse autoencoder trains on. It is a workbench batch
job driven by a recipe, sized in tokens rather than examples, and it holds the GPU for the
duration: the smoke recipe completes in minutes, a reference capture takes most of a day.

## Run

```bash
make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml
make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml ARGS="--override harvest.total_tokens=500000"
```

The target runs the GPU preflight, then the `harvest` verb in the workbench with the recipe.
The recipe's `harvest` block names the model, the hook layers, the corpus, the context
length, the token budget, and the storage dtype; [Recipes](../reference/recipes.md) covers
the shape and the override syntax. The harvest also takes the repository-wide GPU lock the
training tier owns, so it cannot overlap a training run in another worktree, and runs under
the training tier's watchdog.

The output is a capture directory under `/mnt/ai-data/interp/activations/<run.name>/` with a
`manifest.json` recording the model, the corpus, the hook layers, the dtype, and the
measured token total, which can differ from the requested budget.

## Check the hooks first

```bash
make harvest-check-hooks RECIPE=training/recipes/interp/harvest-smoke.yaml
```

A one-token forward with no capture, reporting the residual width observed at each hook
layer and failing on any mismatch with what the recipe expects. Gemma 4 has hook hazards a
layer index alone does not reveal, such as the boundary where the shared-KV tail and the
wider MLP begin, so run this before committing a GPU-day to a full recipe.

## Chunks and resume

The harvest captures each hook as a sequence of chunks, each written to a fresh directory,
and assembles them at the end. A crash costs one chunk rather than the run: re-invoking the
same recipe fast-forwards through the banked chunks' tokens without running the model and
resumes from the next one. `harvest.chunk_tokens` sets the chunk size; 0 takes the default,
which trades resume granularity against per-chunk overhead.

Two hazards are worth knowing before a long capture. The storage dtype is `fp32` in the
committed recipes because the cache format has no bf16 and a bf16 recipe fails before the
first shard; and a corpus smaller than the token budget is re-read from the start rather than
stopping, which duplicates tokens into the capture with only a construction-time warning as
the signal. Size the budget to the corpus.

## Retention

Captures are a regenerable cache and the largest thing on the volume. A downstream SAE run
flags the capture it consumed as validated, and `gc` deletes validated captures oldest first,
keeping every manifest for provenance:

```bash
make interp-gc ARGS="--dry-run"
make interp-gc
make interp-gc ARGS="--mark-validated /mnt/ai-data/interp/activations/<name>"
```

`--dry-run` lists what would go. `--mark-validated` flags one capture by hand instead of
collecting, for a capture that was validated some other way.
