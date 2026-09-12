---
title: Comparing per-layer divergence
---

# Comparing per-layer divergence

`layer-diff` prefills the token ids of a reference capture through the CPU backend and prints
one row per stage of the residual stream: cosine similarity, maximum absolute difference, and
relative L2 error against the reference. It names the first row over tolerance and exits 1.
First rather than worst: once a layer diverges every later one is wrong downstream of it, so
the first row over tolerance is the layer holding the defect. The ladder's `L1` rung makes the
same comparison and prints only the worst pair; this tool prints the whole stack, which is
where the shape of the drift with depth is read.

The reference comes from `transformers`, which is the one oracle that cannot have inherited a
misreading of the architecture from this codebase. This page assumes `BIN` from
[Build](../getting-started/build.md) and uses `gemma-4-E4B`, on which the tables below were
recorded.

## Against the real checkpoint

The reference is the ladder's `layers.json`, written by `rust-harness/fixtures/reference_capture.py`
over the prompt `"The capital of France is"`; `make ladder` writes it when it is absent (see
[Correctness ladder](ladder.md)). The header and the verdict go to stderr and the table to
stdout, so capture both:

```bash
cd rust-harness
set -o pipefail
"$BIN/layer-diff" --text-only --model /path/to/gemma-4-E4B \
    --reference fixtures/reference/e4b/quick/layers.json 2>&1 | tee layer-diff.txt
echo "exit=$?"
```

```text
[INFO] Reference: 6 tokens, 42 layers, float32 precision
[INFO] Cosine floor: 0.9999, relative-L2 ceiling: 0.001
 stage/layer  cosine       max_abs      rel_l2
       embed  1.00000000   0.000e0      0.000e0
ple_identity  1.00000000   0.000e0      0.000e0
 ple_context  1.00000000   3.815e-5     5.510e-7
         ple  1.00000000   3.052e-5     4.869e-7
           0  1.00000000   3.815e-5     1.474e-6
           1  1.00000000   2.575e-5     9.956e-7
           ...
          18  1.00000000   4.029e-5     4.620e-6
          19  1.00000000   3.040e-5     4.993e-6
           ...
          24  1.00000000   3.099e-5     3.393e-6
           ...
          41  1.00000000   3.195e-5     2.756e-6

[OK] Residual stream: all 42 layers within tolerance
exit=0
```

The table is 46 rows: four pre-stack stages, then the 42 layers. The rows above the numbered
layers are the scaled embedding and the per-layer-embedding stages a Gemma 4 E-series
checkpoint adds before the first decoder block. `embed` and `ple_identity` are exact zeros,
so the embedding lookup and the identity half of the per-layer inputs agree with
`transformers` bit for bit; only the stages that involve a matmul and a norm carry noise. A
reference that disagrees at `embed` points at the token ids, the embedding table, or the
embedding scale, and the layer rows below it are not evidence about the layers until that
row is clean; the tool says so when it happens.

The shape on E4B: relative L2 rises from 1.5e-6 at layer 0 to a peak of 5e-6 at layers 18 and
19, falls through the shared-KV tail from layer 24, and rises again toward layer 41. Every
figure is two to three orders of magnitude under the ceiling, which is accumulation-order
noise between this backend and torch and nothing else.

## Tolerances

Cosine is the primary measure because the residual stream's magnitude spans orders of
magnitude across the stack, so one absolute threshold would be loose at one end and
unsatisfiable at the other. The defaults are a cosine floor of 0.9999 and a relative-L2
ceiling of 0.001, which admit the difference in accumulation order between this backend and
torch and nothing more. Both are flags, and closing the ceiling to sit between the second- and
third-largest layer figures shows the first-not-worst policy:

```bash
"$BIN/layer-diff" --text-only --model /path/to/gemma-4-E4B \
    --reference fixtures/reference/e4b/quick/layers.json --rel-l2-ceiling 4.5e-6 2>&1 | tail -n 5
echo "exit=$?"
```

```text
          40  1.00000000   3.052e-5     1.937e-6
          41  1.00000000   3.195e-5     2.756e-6

[ERROR] Residual stream: first divergence at layer 18 (SlidingWindow(512), head 256, ffn 10240, rope base 10000, kv source 18)
  Every later layer is downstream of this one; fix layer 18 before reading the rest of the table.
exit=1
```

Layers 18 and 19 are both over 4.5e-6 and are both marked `<-- diverges (magnitude)` in the
table; the verdict names 18, the earlier, although 19 carries the larger figure. The verdict
line also states the layer's geometry and which layer's KV cache it reads. A ceiling that
falls under a stage row prints `the stack's input diverges at '<stage>'` instead, by design.

## Against the committed tiny reference

One reference is committed, for a tiny four-layer Gemma 4 that `transformers` itself
constructed, so this comparison runs anywhere with no checkpoint download. It is a synthetic
fixture and its weights carry no meaning; its use here is that it is the one structural rung
that runs without a workstation:

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

Tightening the floor to 1.0 fails the first row with any rounding difference, which is a
convenient way to see the failure output and the exit code without a defect. The tool records a
run under `${OBS_LOG_DIR}/harness/` either way.
