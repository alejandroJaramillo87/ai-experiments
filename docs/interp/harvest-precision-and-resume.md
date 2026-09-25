# Harvest storage precision and resume

Two defects block the reference activation captures. Both come from the same place — the
harvest is thin glue over SAELens's `CacheActivationsRunner`, and that runner's contract is
narrower than `interp/harvest.py` assumed. This document records what was established, how,
and what the options cost. It owns KB-57 and KB-58; the ledger rows point here.

For the wider question of which dtype each tier computes in, see [precision/](../precision/README.md).
This document is only about the activation cache on disk.

## Table of Contents

<!-- toc:start -->

1. [What was established](#what-was-established)
   1. [The dtype defect (KB-57)](#the-dtype-defect-kb-57)
   2. [The resume defect (KB-58)](#the-resume-defect-kb-58)
2. [Why fp32 is not the answer](#why-fp32-is-not-the-answer)
3. [Storage options](#storage-options)
   1. [Why a corpus sample is enough](#why-a-corpus-sample-is-enough)
4. [Resume options](#resume-options)
5. [What was implemented](#what-was-implemented)
6. [What the workstation added](#what-the-workstation-added)
7. [Recommendation](#recommendation)
8. [On buying storage](#on-buying-storage)

<!-- toc:end -->

## What was established

Evidence is of three kinds, and they are not equally strong. The source readings are exact:
they come from the 6.46.0 wheel, which is the pinned version. The sandbox probe is a real
execution. The workstation traceback is human-reported per the verify-host protocol.

### The dtype defect (KB-57)

`harvest.dtype` is one recipe field that reaches two type systems.

`_saelens_runner` passes `DTYPE_NAMES[job["dtype"]]` to `CacheActivationsRunnerConfig.dtype`.
SAELens then uses that single value twice:

- `ActivationsStore(..., dtype=cfg.dtype)`, resolved through SAELens's `DTYPE_MAP` constant,
  which maps `"bfloat16"` to `torch.bfloat16`. This is a compute dtype and it is fine.
- `Array2D(shape=(context_size, d_in), dtype=cfg.dtype)` in `CacheActivationsRunner.__init__`,
  which is a `datasets` feature and resolves through Arrow. **Arrow has no bfloat16 in any
  version.**

So `bf16` type-checks in the first system and fails in the second. It passes
`validate_harvest`, passes SAELens's own dtype validation, and dies in the runner's
constructor before a single shard is written.

Reproduced in the sandbox with `datasets==5.0.0` and `pyarrow` alone — no GPU, no model, no
sae-lens — confirming the failure is CPU-side Arrow type resolution and not anything about
Gemma 4:

| `Array2D` dtype | Result |
|---|---|
| `bfloat16` | `ValueError: Neither bfloat16 nor bfloat16_ seems to be a pyarrow data type` |
| `float16` | resolves |
| `float32` | resolves |
| `int16` / `uint16` | resolves |

The error text matches the workstation traceback exactly.

All four shipped recipes were `dtype: bf16`, so this blocked the Gemma 2 and Qwen3
calibration captures (the Qwen3 recipe has since left with its family) as well as Gemma 4. It was never a Gemma 4 defect, and it was never a
regression: no harvest has ever been runnable at its committed dtype.

The `bf16` default also reached past the recipes. `harvest.dtype` defaults to `bf16` in
`training/recipes/schema.py`, so a `Recipe` built structurally rather than loaded from a file
inherited it -- which is how `tests/hardware/test_gpu.py::test_harvest_hook_smoke` was
configured. Moving the recipes alone would have left that test failing exactly as before.

Confirmed fixed on the workstation, 2026-08-07:

```
$ PATH="$PWD/.venv/bin:$PATH" python -m pytest tests/hardware/test_gpu.py \
    -m hw_full -k harvest_hook_smoke --tb=short
[TIER] full: torch present, the whole suite was in scope.
1 passed, 11 deselected, 16 warnings in 17.25s
```

That is the first activation harvest in this repository to run to completion.

### The resume defect (KB-58)

`CacheActivationsRunner.run()` begins:

```python
final_cached_activation_path.mkdir(exist_ok=True, parents=True)
if any(final_cached_activation_path.iterdir()):
    raise Exception(f"Activations directory ({final_cached_activation_path}) is not empty. ...")
```

There is no resume, and the failure mode is worse than restarting. Per-buffer shards are
written to `out_dir/.tmp_shards/shard_{i:05d}` as the capture proceeds and **survive a
crash** — but `run()` counts `.tmp_shards` as making the directory non-empty, so a rerun does
not restart the capture, it raises. The completed shards are intact on disk and unreachable
without deleting them by hand.

Our own resume is per hook, recorded in `progress.json`. Every shipped recipe captures a
single hook, so it can skip nothing.

Consequence: a crash at hour 16 of the ~17-hour 300M-token reference capture loses the entire
run, and the operator's first rerun fails with a message about accidental deletion rather
than resuming.

The tree claimed otherwise in three places — the `interp/harvest.py` module docstring, the
`_saelens_runner` docstring, and `harvest-false-fact.yaml`. All three are corrected.

**The fix is chunked capture**, described under [what was implemented](#what-was-implemented)
below. It does not remove the refusal; it stops the refusal from ever firing, by giving every
`run()` call a directory that is empty because it has just been created.

## Why fp32 is not the answer

fp32 storage is lossless for a bf16 model — bf16 is the high half of an f32, so widening is a
shift — and it needs no code. It is what `harvest-smoke.yaml` (and the Qwen3 calibration
recipe, while it existed) now uses. It does not scale to the reference captures.

Measured on the workstation: `/mnt/ai-data` has 1,748,605,484 KiB free, or 1.79 TB. The
`disk_free_floor_gb` default of 100 GiB leaves **1.68 TB usable**.

| Capture | d_model | Tokens | bf16 | fp32 | Max tokens at fp32 |
|---|---|---|---|---|---|
| `harvest-false-fact` (E2B) | 1536 | 300M | 0.92 TB | 1.84 TB | 274M |
| `harvest-calibration-gemma2` | 2304 | 300M | 1.38 TB | 2.76 TB | 183M |
| Qwen3 calibration (recipe removed 2026-09-18) | 2048 | 50M | 0.20 TB | 0.41 TB | fits |
| `harvest-smoke` (E2B) | 1536 | 2M | 6.1 GB | 12.3 GB | fits |

The false-fact capture could take a 9 percent token trim and run at fp32 today. The Gemma 2
calibration capture cannot: 2.76 TB exceeds the whole 1.97 TB volume when empty, so no amount
of freeing space reaches it, and the fp32 budget of 183M tokens is a 39 percent cut. The full
300M is what makes that capture a same-scale comparison against the published Gemma Scope
reference (`p4_d3_sae_training_investigation.md`), which is the reason it exists.

**Two-byte storage is therefore a precondition for the calibration leg, not an optimization.**

Note also that even at two bytes the two reference captures do not coexist: 1.38 + 0.92 =
2.30 TB against 1.68 TB usable. They are sequential, with `mark_validated` and `gc_captures`
in between.

## Storage options

| Option | Bytes | Lossless | Code | Blocker |
|---|---|---|---|---|
| fp32 | 4 | yes | none | does not fit the reference captures |
| fp16 | 2 | yes, in range | none | needs a per-subject magnitude measurement |
| uint16 bit-cast | 2 | yes | wrap write and read | touches SAELens internals |
| own writer + `DataProvider` | 2 | yes | ~50-100 lines | new format to maintain |

**fp16 is the answer for Gemma 4 E2B, and it is lossless.** Measured on the workstation with
`scripts/interp/activation_magnitude_probe.py` (64 Pile sequences at context 1024, all 35
layers hooked):

| | Peak `abs(activation)` | Headroom to 65504 |
|---|---|---|
| Layer 17 (the hooked layer) | 57.00 | 1149x |
| Worst layer (29) | 127.50 | 514x |

Two facts together make this lossless rather than merely safe. bf16 carries 8 significand
bits and fp16 carries 11, so **every bf16 value inside fp16's exponent range is stored
exactly** — verified over ~2M values across the measured range. And the measurement puts the
peak 514x under the ceiling where saturation would begin. The capture is bf16 in, fp16 on
disk, bit-identical.

The residual risk is at the bottom of the range, not the top: fp16's smallest normal is
6.1e-5, and a bf16 value below that loses precision to a denormal or flushes to zero. Those
are values five orders of magnitude below the peak, and nothing in SAE training is sensitive
to them.

**This result does not transfer between models**, which the second measurement demonstrates
rather than merely asserting. Gemma-2-2B, measured the same way:

| | Peak `abs(activation)` | Headroom to 65504 |
|---|---|---|
| Layer 12 (the hooked layer) | 1680.00 | 39x |
| Worst layer (23) | 2960.00 | 22x |

Gemma-2-2B peaks **23x higher than E2B** (2960 against 127.5), and unlike E2B its magnitudes
climb monotonically with depth before plateauing around layer 20. Both clear the bar, but
inheriting one model's number for the other would have been the same class of reasoning that
produced the 2304 width defect: a plausible value carried across a boundary it does not cross.
Measure per subject.

### Why a corpus sample is enough

The probe samples a corpus slice, not the 300M-token capture, and the maximum of a larger
sample is never smaller. That is the reason `SAFE_HEADROOM` is 8x rather than something
tighter — the margin has to absorb the gap between what was sampled and what will be written.

For Gemma-2-2B, the narrower of the two subjects, that gap was measured directly. The probe
was re-run at 512 sequences, 8x the original 64:

| Sequences | Peak `abs(activation)` | Worst layer | Headroom |
|---|---|---|---|
| 64 | 2960.00 | 23 | 22.1x |
| 512 | 2960.00 | 23 | 22.1x |

Not merely close — **identical**, and so is the full per-layer profile across all 26 layers.

That is the signature of persistent outlier dimensions rather than rare tail events. The
large-magnitude channels in this model family activate on essentially every sequence, so the
per-layer maximum converges almost immediately and eight times the data moves it not at all.
Had the peak instead been set by rare inputs, the maximum would have crept upward with sample
size and the 22x margin would be an estimate with an unknown tail beyond it.

This is what makes the headroom trustworthy over 300M tokens rather than merely observed over
65k. It is also why the 8x bar is a sensible gate in general and not just here: a subject
whose peak *does* grow with sample size would have to grow by more than 8x to reach
saturation, and a quantity that converges by 512 sequences will not.

E2B was not re-measured. At 514x, no plausible tail reaches the ceiling.

**uint16 bit-cast** stores bf16's raw 16 bits as an integer and reinterprets on read. A
sandbox round-trip confirmed it is exactly lossless through `Dataset.save_to_disk` and
`load_from_disk`. It looked like the cheap answer and is not: `CacheActivationsRunner` builds
`self.features` in `__init__` and passes torch tensors in `_create_shard`, so the write path
needs a subclass, and the read path (`ActivationsStore` with `use_cached_activations=True`)
expects a float column, so it needs wrapping too. That is comparable work to the own-writer
option while also depending on unpublished internals.

**Own writer** stores bf16 natively in safetensors and feeds `SAETrainer` through
`DataProvider = Iterator[torch.Tensor]`, the alias SAELens publishes from its training types
module — a documented seam rather than private internals. This bypasses SAELens's cache
format entirely, and with it both defects at once: the dtype conflation and the missing
resume.

## Resume options

| Option | Cost | Depends on |
|---|---|---|
| Chunked capture | driver loop only | `run()`'s documented behaviour |
| Recover `.tmp_shards` | small, fragile | shard naming, `_consolidate_shards` |
| Subclass the runner | small | unpublished internals |
| Own writer | see above | nothing in SAELens's cache path |

**Chunking** splits `total_tokens` into N sequential sub-runs, each with its own `out_dir`,
assembled afterwards into one dataset. Every chunk starts against an empty directory, so the
non-empty refusal never trips, and a crash costs one chunk. At 300M tokens in 20 chunks the
worst-case loss is under an hour. It composes with every storage option above. **This is what
was implemented**; the section below records what reading the pinned source changed about it.

**Recovering `.tmp_shards`** is possible in principle — the shards are intact and
`_consolidate_shards` is a static method that takes a source and output directory — but it
depends on shard naming and on `state.json` layout inside a *partial* run, neither of which is
a published contract. Chunking makes the recoverable loss one chunk, which is enough.

## What was implemented

Chunked capture, in `interp/harvest.py`. Three things read from the pinned sae-lens 6.46.0
source changed the design from the sketch above, and each is worth recording because each was
a guess that turned out to be wrong in a useful direction.

**No corpus offset is needed.** The sketch assumed each chunk would have to seek into the
corpus. It does not: `CacheActivationsRunner.__init__` builds the activations store once, the
store holds the stream position, and `total_training_tokens` is assigned in the store and never
read back — it has no budget to re-apply. So one runner reused across N `run()` calls produces
contiguous chunks by construction. Every per-chunk quantity SAELens derives (`n_buffers`,
`n_seq_in_buffer`, `n_tokens_in_buffer`) is a property recomputed from `cfg.training_tokens` at
access time, so reassigning that field between chunks is all the resizing required.

**Assembly is SAELens's own function, not ours.** The intent had been to hand-write the merge
rather than call a private name. Reading it inverted the judgement:
`CacheActivationsRunner._consolidate_shards` is a static method whose docstring specifies the
input and output layouts exactly, it is the same function `run()` uses on its own per-buffer
shards, and it recomputes the dataset fingerprint via `generate_fingerprint`. Reimplementing
would mean reimplementing that fingerprint derivation against an unpublished contract, to get a
worse copy of a function that already ships. Chunk directories are therefore named
`shard_{i:05d}`, which is the prefix it requires. The dependency on a private name is real, and
`tests/interp/test_saelens_contract.py` pins it — a rename fails in the sandbox rather than at
the one step that runs after all the expensive work is already on disk.

Assembly is called with `copy_files=False`, making each arrow file a same-filesystem
`shutil.move`. The whole step is metadata: seconds for a terabyte, and no second copy to fit on
the volume. `concatenate_datasets` plus `save_to_disk` — the public-API route — would re-encode
and need 2.76 TB against 1.68 TB usable, so it is ruled out by disk before it is ruled out by
time.

**Resume needs a fast-forward, and the manifest was lying.** A resumed run is a new process, so
the store starts at the beginning of the corpus; banked chunks' tokens must be skipped or the
capture duplicates them. The skip pulls tokens without running the model (`get_batch_tokens` is
the tokenization half of `get_raw_llm_batch`, and only the token iterator carries position), so
it is streaming and tokenization cost, not GPU cost — and it is paid only after a crash.

Separately, the manifest's `total_tokens` was the *requested* budget echoed back, never
measured. `run()` returns the `Dataset`, so the rows are countable; the manifest now records the
measured total and keeps the request as `requested_tokens`. This matters more than it looks:
SAELens starts a **new epoch** rather than stopping when the corpus runs dry
(`get_raw_llm_batch` defaults to `raise_on_epoch_end=False`, so the store resets its iterator
and keeps yielding), which means a corpus smaller than the token budget is silently re-read and
duplicated into the capture. The only signal is a construction-time warning that compares a
sample count against a token count. That epoch-wrap behaviour is a separate hazard from KB-58,
is not fixed here, and is now filed as KB-60.

## What the workstation added

Three things the sandbox could not have established, from the 2026-08-08 confirmation runs.

**The fix reproduced the defect one level down.** The first resume attempt failed with the same
`Activations directory ... is not empty` exception KB-58 is about. The chunk loop passed SAELens
the output directory of the chunk the crash had landed in; a chunk is recorded complete only
after `capture()` returns, so that directory still held the crashed attempt's `.tmp_shards`. The
loop now removes a pending chunk's directory before recapturing it, which is sound precisely
because anything at that path is by definition debris.

Why the tests missed it is the part worth carrying elsewhere. `FakeSession.capture` raised
*before* creating its output directory, so a simulated crash left none of the wreckage a real
one leaves — the fake was testing a crash that cannot happen. It now refuses a non-empty
directory the way SAELens does, and writes `.tmp_shards` before raising. With the fix backed
out, three pre-existing tests fail, including the one whose name claims this coverage. A fake
that fails more cleanly than the thing it stands for will report a recovery path as working when
it does not.

**A capture overshoots its request, and the disk preflight was under-pricing it.** The smoke
capture requested 2,000,000 tokens and measured 2,875,392, 44 percent high. SAELens writes whole
buffers and each chunk rounds up to the next one, so a capture overshoots by under one buffer
per chunk and never undershoots.

The buffer is a **byte** budget, not a sequence count —
`CacheActivationsRunnerConfig.n_tokens_in_buffer` is 2 GB divided by `d_model * itemsize`,
floored to a whole model batch. Narrowing the dtype therefore widens the buffer, which is the
part that is easy to get backwards:

| Capture | dtype | Buffer | Chunk budget | Buffers per chunk | Captured | Overshoot |
|---------|-------|--------|--------------|-------------------|----------|-----------|
| `harvest-smoke` | fp32 | 319,488 tokens | 666,667 | 2.09, rounds to 3 | 2,875,392 | 44 percent |
| `harvest-false-fact` | fp16 | 647,168 tokens | 15,000,000 | 23.18, rounds to 24 | 310,640,640 | 3.5 percent |

Chunking moves this rounding from once per capture to once per chunk, so the overshoot is a
cost of the KB-58 fix. It is not negligible at reference scale: 3.5 percent of the false-fact
capture is 33 GB, and `projected_bytes` priced the *request*, so the disk preflight under-booked
by that much — on a tier where the preflight, not cleanup, is the documented defense against
filling the volume. It now prices whole buffers via `captured_tokens`, and
`tests/interp/test_saelens_contract.py` checks the replicated formula against the pinned wheel
so a drift in SAELens's default is a red sandbox run rather than a silent mis-projection.

Predict the measured number before a long capture rather than reading it afterwards. A manifest
reporting more tokens than were asked for is correct here, and alarming otherwise.

**The image did not carry what the tier needs at run time.** Three failures — a dependency
building from source with no compiler, a missing `omegaconf`, and no C compiler for Triton's
runtime driver shim — cost an image build and a model load each to find. That is KB-59, and
`tests/interp/test_image_dependencies.py` is the guard.

## Recommendation

1. **Storage is resolved.** Both subjects measured, both recipes on fp16, at zero code and
   zero precision loss. KB-57 is closed for every shipped recipe and the volume is no longer
   a constraint: 0.92 TB and 1.38 TB against 1.68 TB usable, run sequentially.
2. **Measure any new subject before capturing it.** The probe is a GPU-minute and the two
   results differ by 23x. `scripts/interp/activation_magnitude_probe.py` exits non-zero when
   the margin is under 8x, so it can gate a capture.
3. **Chunked capture is implemented and confirmed.** The workstation ran the chunked smoke
   capture and the interrupt-and-resume check on 2026-08-08; the resumed run reported the same
   2,875,392 tokens as the uninterrupted one, which is what shows the resume neither dropped nor
   duplicated a chunk. The crash check is the one that mattered — it found a defect the clean
   run could not. KB-58 is resolved.
4. **Set `chunk_tokens` explicitly on any long capture.** The default divides a capture into
   15M-token chunks; the smoke recipe pins a smaller value so that multi-chunk assembly — the
   part that must not first run for real on a 17-hour job — is exercised by CI-sized work.
5. The own-writer route is now unnecessary for both defects. It remains the cleanest single
   fix in principle, but chunking got the benefit for far less work.

## On buying storage

A 4 TB volume changes the arithmetic and is worth doing on its own merits, but it is not a
substitute for either fix:

- It makes fp32 viable for each capture taken alone (2.76 TB is the largest), removing the
  hard block on the calibration leg.
- More usefully, it lets both reference captures coexist at two bytes (2.30 TB), which the
  current volume cannot. Today they must be run sequentially with `gc_captures` between them,
  so a same-scale comparison means recapturing whichever was deleted.
- It does nothing for resume. A 17-hour job that loses everything on a crash is unaffected by
  how large the disk is.

The order that matters: measure fp16 first. It is a GPU-minute and it may remove the capacity
pressure entirely, which is a better position from which to decide what to buy. Sequential
throughput is not the binding constraint either — SAE training reads the cache in shuffled
batches, so the relevant figures are random read and sustained write during capture, and
endurance rating matters more than peak sequential for a drive absorbing multi-terabyte
overnight writes.
