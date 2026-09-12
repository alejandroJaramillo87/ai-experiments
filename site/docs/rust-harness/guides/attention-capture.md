---
title: Capturing attention weights
---

# Capturing attention weights

`attention-capture` runs one prompt and writes the post-softmax attention distribution of chosen
query heads in chosen layers, as a dense matrix per head. Gemma 4 interleaves sliding-window
and global attention layers, and this is the direct way to look at the difference. This page
assumes `BIN` and `MODEL` from [Build](../getting-started/build.md).

## Capture

All five flags are required. Layers and heads are comma-separated, distinct, and 0-based.

```bash
"$BIN/attention-capture" --model "$MODEL" --prompt "the cat sat" --layers 0,1 --heads 0 --out attn
```

```text
[INFO] Attention: 16 tokens, 2 layers x 1 heads = 2 matrices, 0.0 MiB
[OK] Attention: 2 matrices of 16x16 -> attn
```

The size line is printed before anything is allocated. A matrix is four bytes per
`tokens × tokens` entry per (layer, head) pair, so one pair over an 8192-token prompt is about
134 MB. This is why both filters are required rather than defaulting to everything.

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
python3 -c 'import numpy as np; z = np.load("attn/attention.npz"); print(z["span_first_layer_0"]); print(z["span_first_layer_1"])'
```

Against the synthetic checkpoint, whose window is four, layer 0 shows the span advancing after
the fourth position and layer 1 stays at zero.

The format is documented in [Attention capture shard](../../formats/attention-capture-shard.md).
Nothing in the Python tooling reads it yet; numpy does.
