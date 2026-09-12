---
title: Activation capture shard
---

# Activation capture shard

A directory of last-token residual activations over a labelled dataset, at one or more layers.
Written by `activation-capture` and by the Python capture command; read by the Python probe
trainer and the difference-of-means extractor. The contract is
`schemas/activation_capture.manifest.json` in the repository.

## Layout

```text
<out>/
  shard_00000.npz
  shard_00001.npz        one per 2048 examples
  examples.jsonl         the dataset, copied
  manifest.json          written last
```

The manifest is written last, after every shard, so a directory without one is an interrupted
capture and no reader accepts it. A stale manifest from an earlier capture is deleted before
the first shard is written.

## The shards

Each shard is a numpy `.npz` archive holding:

| Array | dtype | Shape | Contents |
|-------|-------|-------|----------|
| `layer_<L>`, one per captured layer | float32, little-endian | `[rows, hidden]`, C order | The residual stream at the output of decoder block `L`, at the last token of each example |
| `labels` | int64 | `[rows]` | The example's `label` from the dataset |

Rows are in dataset order, at most 2048 per shard. The harness writes the members
uncompressed with zeroed archive timestamps, so the bytes are deterministic; the Python writer
compresses. Both load with numpy.

`float32` is the only dtype either writer emits. The activations belong to a bf16 model and
widening bf16 to float32 is exact, so the stored values are the model's own. float16 is not
used because its five exponent bits saturate on residual-stream outliers that bf16 represents.

## The manifest

| Field | Contents |
|-------|----------|
| `schema_version` | 1 |
| `artifact` | `activation_capture` |
| `model_id` | The checkpoint path or Hugging Face id the capture ran on |
| `model_revision` | The revision; `safetensors` for a local checkpoint captured by the harness |
| `model_hash` | sha256 of `model.safetensors`, or `unknown` |
| `layers` | The captured layer indices, sorted, 0-based |
| `position` | `last`: the last token of each example |
| `dataset_path` | The dataset path as given |
| `dataset_sha256` | sha256 of the dataset file |
| `tokenizer_fingerprint` | A digest of the tokenizer, so two captures can be told apart by tokenization |
| `dtype` | `float32` |
| `count` | Number of examples |
| `shards` | Number of shard files |
| `git_sha` | The repository commit of the writer, or `unknown` |
| `created` | ISO 8601 UTC timestamp |

The harness adds two fields the Python writer does not: `capture_backend` is
`rust-harness-cpu`, and `site` is `layer_output`, naming the tap point. A probe trained on a
harness capture inherits `site` into its sidecar.

## The dataset

`examples.jsonl` is one JSON object per line with a `text` string and an integer `label`. It is
copied into the capture so the trainer can make its seeded split against the same examples.
The trainer needs both classes present in every split; a few dozen examples is the practical
floor.

## Producing one

```bash
"$BIN/activation-capture" --model "$MODEL" --dataset examples.jsonl --layers 0,1 --out capture
```

See [Capturing activations](../rust-harness/guides/activation-capture.md) for the walkthrough
and [Producing a probe and a control vector](producing-a-probe-and-control-vector.md) for what
reads the result.
