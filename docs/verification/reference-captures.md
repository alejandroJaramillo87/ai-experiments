# Reference activation captures

**Operational runbook. Executed against the workstation, not read for understanding: the
operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate —
both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 0 verified, 0 blocked, 2 pending, of 2.**

The two multi-hour activation captures that feed P4-D3 SAE training: the false-fact organism
capture and the Gemma-2-2B calibration capture. They are separated from
[interp.md](interp.md) because each occupies the GPU for most of a day and holds most of the
`/mnt/ai-data` volume, so they are scheduled rather than run in sequence with the other interp
entries.

Neither has been run. Everything they depend on has been verified at smoke scale: I.9 and I.9b
in [interp.md](interp.md) confirmed chunked capture and its crash recovery on 2026-08-08.

## Preconditions

1. **One GPU consumer, workstation-wide.** `scripts/interp/preflight.sh` runs automatically and
   refuses when a GPU inference service is up, but nothing detects a benchmark or a training run
   in another worktree. Check by hand; a collision costs both jobs.
2. **The image is current.** `docker compose run` builds only when the image is *missing* — it
   does not notice a changed `pyproject.toml`, `uv.lock`, or Dockerfile, and will silently use a
   stale image. After any such change:

   ```bash
   $ uv lock              # the Dockerfile syncs with --frozen and fails on a stale lock
   $ make interp-build
   ```

3. **Disk.** Check free space against the projection below before starting. The preflight
   refuses on its own, but discovering the refusal after a model load wastes the slot.

## Expected shape

Both captures are single-hook, chunked, and resumable. Figures are derived from the recipe and
`interp/harvest.py`, not measured — the first run replaces them with observations.

| Capture | Recipe | Hook | Requested | Written | Disk | Chunks |
|---------|--------|------|-----------|---------|------|--------|
| False fact | `training/recipes/interp/harvest-false-fact.yaml` | layer 17 | 300,000,000 | 310,640,640 | 954.3 GB | 20 |
| Gemma-2-2B calibration | `training/recipes/interp/harvest-calibration-gemma2.yaml` | layer 12 | 300,000,000 | 306,708,480 | 1,413.3 GB | 20 |

The false-fact capture takes about 17 hours; each is 20 chunks, so a chunk is roughly 50
minutes. The two run sequentially — together they exceed the volume.

**The written count exceeds the requested count, and that is correct.** SAELens writes whole
buffers and rounds each chunk up to the next one. The buffer is a byte budget, so it depends on
`d_model`: 647,168 tokens at 1536 and fp16, 425,984 at 2304. Each 15,000,000-token chunk
therefore becomes 24 buffers for false fact and 36 for the calibration capture. Write the
predicted number down before starting and check the manifest against it; a manifest reporting
more tokens than were asked for looks like a defect if the prediction is made afterwards.
`manifest.json` carries `requested_tokens` beside `total_tokens` for exactly this reason, and
the disk preflight prices the written figure rather than the request. The reasoning is in
[../interp/harvest-precision-and-resume.md](../interp/harvest-precision-and-resume.md).

## Running a capture

```bash
$ df -h /mnt/ai-data
$ make harvest-check-hooks RECIPE=training/recipes/interp/harvest-false-fact.yaml
$ make harvest RECIPE=training/recipes/interp/harvest-false-fact.yaml
```

Expected signal: 20 `Caching activations` progress bars, one per chunk, each about 50 minutes;
then `[OK] Harvest complete: 310640640 tokens/hook`. During the run,
`<capture>/<hook>.chunks/` holds one `shard_NNNNN` directory per completed chunk. Afterwards it
is gone, `<capture>/<hook>/` holds the flat assembled dataset, `progress.json` is deleted, and
`manifest.json` records `chunks: 20`.

Then train an SAE against it, since a capture no SAE has read is not a finished deliverable:

```bash
$ make sae-train RECIPE=training/recipes/interp/sae-false-fact.yaml
```

## Recovering from an interruption

**Rerun the same command. Do not delete the capture directory.** A crash costs one chunk, about
50 minutes; deleting the directory costs everything captured so far.

```bash
$ cat /mnt/ai-data/interp/activations/<name>/progress.json
$ make harvest RECIPE=training/recipes/interp/harvest-false-fact.yaml
```

The resume reads `completed_chunks`, clears the interrupted chunk's directory, fast-forwards the
corpus stream past the banked tokens without running the model, and continues. The fast-forward
is tokenization and streaming cost only, and is paid solely after a crash.

Two failures are **not** resumable, and both refuse explicitly rather than producing a truncated
capture:

| Refusal | Cause | Action |
|---------|-------|--------|
| `... already exists without <hook> being recorded complete` | The crash landed inside assembly, which moves files out of the chunk directories, so neither side is whole | Delete both directories named in the message and rerun from the start |
| `progress.json ... was written under a different <keys>` | The recipe changed between runs, so the chunk boundaries or the subject moved | Revert the change, or delete the capture and start over deliberately |

A capture that already has a `manifest.json` is complete and a rerun refuses; delete it to
recapture.

## Recording protocol

Per [README.md](README.md): paste the command and its output verbatim into an Observed block
below, and close the entry by changing its status. Record the manifest's `total_tokens` against
the prediction in the table above — a mismatch is a finding, not a number to explain away.

### False-fact capture

Observed:

```
```

Status: PENDING

### Gemma-2-2B calibration capture

Observed:

```
```

Status: PENDING

## Background

- [../interp/harvest-precision-and-resume.md](../interp/harvest-precision-and-resume.md) — why
  the captures are fp16, why capture is chunked, and the buffer-rounding arithmetic
- [known-bugs.md](known-bugs.md) — KB-57 (dtype), KB-58 (resume), KB-59 (image dependencies),
  KB-60 (the unfixed epoch wrap, latent because every shipped corpus exceeds its budget)
- [interp.md](interp.md) — I.9 and I.9b, the smoke-scale verification these captures rest on
