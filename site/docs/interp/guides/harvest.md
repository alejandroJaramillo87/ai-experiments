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
`manifest.json` recording the model, the corpus, the hook layers, the dtype, and the token
total. The smoke recipe, recorded on the workstation:

```text
Caching activations: 100%|...| 3/3 [00:25<00:00,  8.46s/it]
[OK] Harvest complete: 2875392 tokens/hook,
     hooks ['model.language_model.layers.17'], 34125.6 tokens/s
```

Three progress bars rather than one because the recipe captures in chunks, and 2875392
tokens against the 2000000 requested because the cache writer rounds every chunk up to a
whole buffer. Both figures are correct, and the manifest records them as `total_tokens` and
`requested_tokens`. Write the predicted written count down before a long capture and check
the manifest against it; a total above the request looks like a defect only when the
prediction is made afterwards.

## Size it before starting

A full-scale harvest is the largest thing the tier writes: the reference captures are 300
million tokens at one hook each, and their derived figures are 954 GB for the false-fact
capture at width 1536 and 1413 GB for the Gemma-2-2B calibration capture at width 2304, 20
chunks and about 17 hours each. The two cannot share the volume, so they run one at a time.

Check free space first, because the preflight refuses a capture that would breach the floor
and a refusal after a model load wastes the slot:

```bash
df -h /mnt/ai-data
```

The disk preflight prices the written figure, buffer rounding included, subtracts it from
the free space on the activations root, and refuses below the floor, before the lock is
taken and before any weight is read. The refusal names the projection, the free space, and
the floor, and ends with the three remedies: free space, reduce `harvest.total_tokens` or
`hook_layers`, or run `make interp-gc`. A buffer is a fixed byte budget, so the tokens per
buffer depend on the residual width: 647168 at width 1536 and 425984 at 2304, in fp16.

## Check the hooks first

```bash
make harvest-check-hooks RECIPE=training/recipes/interp/harvest-smoke.yaml
```

A one-token forward with no capture, reporting the residual width observed at each hook
layer, `[OK] layer 17: residual width 1536` for the smoke recipe, and failing on any
mismatch. The gate has two refusals. A layer index past the checkpoint's depth is refused
from `config.json` before any weight is read, naming the valid range. A hook that yields a
width other than the model's hidden size is refused too; that is the Gemma 4 trap the gate
was written for, where a per-layer-embedding tensor of width 256 sits where a residual hook
is expected. Gemma 4 has other hazards a layer index alone does not reveal, such as the
boundary where the shared-KV tail and the wider MLP begin, so run the gate before committing
a GPU-day to a full recipe.

## Chunks and resume

The harvest captures each hook as a sequence of chunks, each written to a fresh directory
under `<capture>/<hook>.chunks/`, and assembles them at the end into `<capture>/<hook>/`.
`progress.json` in the capture directory records the completed chunks and the tokens they
banked. A crash costs one chunk rather than the run. **Rerun the same command; do not delete
the capture directory.** The rerun reads `progress.json`, clears the interrupted chunk's
directory, fast-forwards the corpus stream past the banked tokens without running the model,
and captures only what is outstanding. The fast-forward is tokenization and streaming cost,
paid only after a crash. Recorded on the workstation, a smoke harvest killed inside its
third chunk:

```text
{ "completed_chunks": { "model.language_model.layers.17": 2 },
  "measured_tokens": { "model.language_model.layers.17": 1916928 } }
```

```text
Caching activations: 100%|...| 3/3 [00:24<00:00,  8.17s/it]
[OK] Harvest complete: 2875392 tokens/hook,
     hooks ['model.language_model.layers.17'], 26535.4 tokens/s
```

The 1916928 tokens banked plus the one recaptured chunk equal the uninterrupted total, which
is the criterion: the resume neither dropped nor duplicated a chunk. `harvest.chunk_tokens`
sets the chunk size; 0 takes the default, which trades resume granularity against per-chunk
overhead, and the reference recipes use 20 chunks of about 50 minutes.

Two failures are not resumable, and both refuse rather than produce a truncated capture: a
crash inside assembly, where the message names both directories to delete before rerunning
from the start; and a recipe that changed between runs, where `progress.json` was written
under different keys and the message names them. A capture that already holds a
`manifest.json` is complete, and a rerun refuses; delete it to recapture.

Two more hazards are worth knowing before a long capture. The storage dtype is `fp32` in the
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
collecting, for a capture that was validated some other way. With nothing validated the
collector prints `[INFO] No validated captures to collect` and exits 0. `--root` points it
at a tree other than the default.

## Before a long run

- **The image is current.** `docker compose run` builds only when the image is missing and
  does not notice a changed lockfile or Dockerfile. After any such change run `uv lock`,
  then `make interp-build`; the Dockerfile syncs with a frozen lock and fails on a stale one.
- **Nothing else holds the GPU.** The preflight sees the inference services; it does not
  see a benchmark or a training run in another worktree. The harvest takes the training
  tier's GPU lock, which refuses a second training-tier job, and runs under its watchdog,
  which aborts the run when free disk falls below the watchdog's floor.
