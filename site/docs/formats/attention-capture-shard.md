---
title: Attention capture shard
---

# Attention capture shard

A directory holding the post-softmax attention distributions of chosen query heads in chosen
layers over one prompt. Written by `attention-capture`; read by numpy. Nothing in the Python
tooling consumes it yet. The contract is `schemas/attention_capture.manifest.json` in the
repository.

## Layout

```text
<out>/
  attention.npz
  manifest.json
```

## The shard

| Array | dtype | Shape | Contents |
|-------|-------|-------|----------|
| `layer_<L>_head_<H>`, one per requested pair | float32 | `[tokens, tokens]`, row-major by query position | The post-softmax attention distribution of query head `H` in layer `L` |
| `span_first_layer_<L>`, one per requested layer | int64 | `[tokens]` | The first key position visible to each query position in layer `L` |
| `tokens` | int64 | `[tokens]` | The prompt token ids the capture ran over, including the leading BOS |

A matrix is dense while the visible spans are ragged: row `p` holds weights over key positions
`span_first[p]` through `p` inclusive and is exactly `0.0` elsewhere, because nothing is ever
written there. Zero is the correct value rather than a filler, since a masked key contributes
nothing to the weighted sum, and the row still sums to one. That it is exactly zero is what lets
a reader tell a structural zero from a weight that rounded away.

The span is stored rather than recomputed because Gemma 4's attention schedule is an array in
the checkpoint's config and its period differs between variants; deriving it from a stride
constant is wrong on some of them. It is per layer rather than per head because the heads of
one layer share a mask, and the writer refuses a set of heads that disagree about it.

## The manifest

| Field | Contents |
|-------|----------|
| `schema_version` | 1 |
| `artifact` | `attention_capture` |
| `capture_backend` | `rust-harness-cpu` |
| `model_id` | The checkpoint path |
| `model_hash` | sha256 of `model.safetensors` |
| `layers` | The requested layer indices |
| `heads` | The requested query head indices |
| `tokens` | The token count |
| `shard` | `attention.npz` |
| `site` | `attention_weights` |
| `softmax` | `post`; the pre-softmax logits are a different quantity and are not offered |
| `dtype` | `float32` |
| `git_sha` | The repository commit of the writer, or `unknown` |
| `created` | ISO 8601 UTC timestamp |

## Size

A capture is four bytes per `tokens × tokens` entry per requested pair: about 134 MB for one
pair over an 8192-token prompt. Both `--layers` and `--heads` are required flags for this
reason, and the writer prints the total before allocating.

## Producing one

```bash
"$BIN/attention-capture" --model "$MODEL" --prompt "the cat sat" --layers 0,1 --heads 0 --out attn
```

See [Capturing attention weights](../rust-harness/guides/attention-capture.md).
