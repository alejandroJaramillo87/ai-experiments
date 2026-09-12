---
title: Capturing attention weights
---

# Capturing attention weights

`attention-capture` runs one prompt and writes the post-softmax attention distribution of chosen
query heads in chosen layers, as a dense matrix per head. Gemma 4 interleaves sliding-window
and global attention layers, and this is the direct way to look at the difference. This page
assumes `BIN` and `MODEL` from [Build](../getting-started/build.md) and an empty working
directory.

## The schedule

Which layers are sliding and which are global is an array in the checkpoint's config, and
`checkpoint-info` prints it as one character per layer. On `gemma-4-E4B` it is
`sssssGsssssGsssssGsssssGsssssGsssssGsssssG`: every sixth layer from 5 is global, the rest
attend over a 512-token window. On `gemma-4-E2B-it` the global layers are 4, 9, 14, 19, 24,
29, and 34. Layers past the shared-KV boundary (24 on E4B, 15 on E2B) read a donor layer's
cache rather than their own; their attention rows are still their own and are captured the
same way.

## Capture

All five flags are required. Layers and heads are comma-separated, distinct, and 0-based; one
sliding layer and one global one make the comparison:

```bash
"$BIN/attention-capture" --model "$MODEL" --text-only \
    --prompt "The three states of matter are solid, liquid, and gas." \
    --layers 0,4 --heads 0 --out attn
```

The tool prints an `[INFO] Attention:` line naming the token count, the number of matrices, and
their size before anything is allocated, then `[OK] Attention: <n> matrices of <t>x<t> -> attn`.
A matrix is four bytes per `tokens × tokens` entry per (layer, head) pair, so one pair over an
8192-token prompt is about 268 MB. This is why both filters are required rather than defaulting
to everything.

The token count includes the leading BOS. The prompt is tokenized as plain text; it is not
wrapped in the chat template.

## What was written

- `attention.npz` holds, for each requested pair, a float32 `layer_<L>_head_<H>` matrix of
  shape `[tokens, tokens]`, row-major by query position; for each requested layer an int64
  `span_first_layer_<L>` vector giving the first key position visible to each query position;
  and an int64 `tokens` vector of the prompt token ids.
- `manifest.json` records the model path and weight sha256, the layers and heads, the token
  count, and three constants that say what the numbers are: `site` is `attention_weights`,
  `softmax` is `post`, and `dtype` is `float32`.

## Reading the matrices

Row `p` of a matrix holds weights over key positions `span_first[p]` through `p` inclusive and
is exactly zero elsewhere; each row sums to one. On a sliding layer the span starts at
`p - window + 1` once the window fills; on a global layer it is always zero. The span travels
with the artifact rather than being recomputed, because the sliding schedule is an array in the
checkpoint's config and differs between Gemma 4 variants.

```bash
python3 -c 'import numpy as np; z = np.load("attn/attention.npz"); print(z["span_first_layer_0"]); print(z["span_first_layer_4"]); print(z["layer_0_head_0"].sum(axis=1)[:4])'
```

On a prompt shorter than the 512-token window both span vectors are all zeros, since no
sliding row has evicted anything yet; the span advances only past position 511. A prompt of a
few windows is what shows the difference, at the matrix cost above.

The rows are the values the harness computes, and the correctness ladder's `L6` rung compares
them key for key against `transformers` at six positions past the window; on E4B the worst
single-key difference measured was 2.3e-5. See [Correctness ladder](ladder.md). The format is
documented in [Attention capture shard](../../formats/attention-capture-shard.md). Nothing in
the Python tooling reads it yet; numpy does.
