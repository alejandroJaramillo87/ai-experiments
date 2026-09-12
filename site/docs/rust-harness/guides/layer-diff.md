---
title: Comparing per-layer divergence
---

# Comparing per-layer divergence

`layer-diff` prefills the token ids of a reference capture through the CPU backend and prints
one row per stage of the residual stream: cosine similarity, maximum absolute difference, and
relative L2 error against the reference. It names the first row over tolerance and exits 1.
First rather than worst: once a layer diverges every later one is wrong downstream of it, so
the first row over tolerance is the layer holding the defect.

The reference comes from `transformers`, which is the one oracle that cannot have inherited a
misreading of the architecture from this codebase. The repository commits one such reference
for a tiny Gemma 4 that `transformers` itself constructed, so this guide runs anywhere. This
page assumes `BIN` from [Build](../getting-started/build.md).

## Run against the committed reference

The committed checkpoint is `rust-harness/fixtures/tiny-hf-gemma4/`, four layers wide, and the
capture beside it is `capture.json`:

```bash
FIX="$BIN/../../fixtures/tiny-hf-gemma4"
"$BIN/layer-diff" --model "$FIX" --reference "$FIX/capture.json"
```

```text
[INFO] Cosine floor: 0.9999, relative-L2 ceiling: 0.001
 stage/layer  cosine       max_abs      rel_l2
       embed  1.00000000   0.000e0      0.000e0
ple_identity  1.00000000   0.000e0      0.000e0
 ple_context  1.00000000   3.576e-7     1.465e-7
         ple  1.00000000   2.980e-7     1.500e-7
           0  1.00000000   9.537e-7     2.273e-7
           1  1.00000000   1.907e-6     2.562e-7
           2  1.00000000   2.384e-6     3.590e-7
           3  1.00000000   3.219e-6     5.233e-7

[OK] Residual stream: all 4 layers within tolerance
```

The rows above the numbered layers are the pre-stack stages: the scaled embedding, and the
per-layer-embedding stages a Gemma 4 E-series checkpoint adds before the first decoder block.
A reference that disagrees at `embed` points at the token ids, the embedding table, or the
embedding scale, and the layer rows below it are not evidence about the layers until that row
is clean; the tool says so when it happens.

## Tolerances

Cosine is the primary measure because the residual stream's magnitude spans orders of
magnitude across the stack, so one absolute threshold would be loose at one end and
unsatisfiable at the other. The defaults are a cosine floor of 0.9999 and a relative-L2
ceiling of 0.001, which admit the difference in accumulation order between this backend and
torch and nothing more. Both are flags:

```bash
"$BIN/layer-diff" --model "$FIX" --reference "$FIX/capture.json" --cosine-floor 0.99999 --rel-l2-ceiling 0.0001
```

Tightening the floor to 1.0 fails the first row with any rounding difference, which is a
convenient way to see the failure output and the exit code without a defect.

## Against the real checkpoint

A reference for the real checkpoint is generated on a machine with the checkpoint and a torch
environment, by `rust-harness/fixtures/reference_capture.py`. The correctness ladder generates
it as its `layers.json` artifact; see [Correctness ladder](ladder.md). Once it exists,
`layer-diff` runs against it with the same two flags plus `--text-only` for the multimodal
checkpoint. The tool records a run under `${OBS_LOG_DIR}/harness/`.
