---
title: Capturing activations
---

# Capturing activations

`activation-capture` runs a labelled dataset through the model and writes the last-token
residual activation of each example at the layers asked for, in the capture shard format the
Python probe tooling reads. The capture is taken from the same forward pass the harness serves
with, so a probe trained on it scores the values the harness computes. This page assumes `BIN`
and `MODEL` from [Build](../getting-started/build.md) and a scratch working directory.

## The dataset

One JSON object per line, each with a `text` field and an integer `label`:

```bash
cat > examples.jsonl <<'EOF'
{"text": "the cat sat on the mat", "label": 1}
{"text": "stocks fell sharply on tuesday", "label": 0}
{"text": "a dog chased the ball", "label": 1}
{"text": "the central bank raised rates", "label": 0}
EOF
```

Four examples are enough to show the format. Training a probe needs more; see
[Producing a probe and a control vector](../../formats/producing-a-probe-and-control-vector.md)
for the size the trainer's class-balanced splits require.

## Capture

All four flags are required. `--layers` is a comma-separated list of distinct 0-based decoder
layer indices; the synthetic checkpoint has layers 0 and 1.

```bash
"$BIN/activation-capture" --model "$MODEL" --dataset examples.jsonl --layers 0,1 --out capture
```

```text
[RUN] Example 1/4: 27 tokens
[RUN] Example 2/4: 33 tokens
[RUN] Example 3/4: 38 tokens
[RUN] Example 4/4: 25 tokens
[OK] Capture: 4 examples, 2 layers, 1 shards -> capture
```

A layer index past the model's depth fails before any example runs:
`layer 2 out of range: model has 2 layers`. A multimodal checkpoint needs `--text-only`;
`--single-thread` runs the row split at width one for debugging under rr, and changes no
value.

## What was written

The output directory holds three things:

- `shard_00000.npz`, and further shards at 2048 examples each: one float32 array per captured
  layer, named `layer_<L>`, of shape `[examples, hidden]`, and one int64 `labels` array.
- `examples.jsonl`, a copy of the dataset, which the trainer reads for its seeded split.
- `manifest.json`, written last, recording the model path and the sha256 of its weights, the
  layers, the dataset path and sha256, the tokenizer fingerprint, the count, and the shard
  count. The manifest is what a probe or control vector later inherits its model identity
  from.

The format is documented field by field in
[Activation capture shard](../../formats/activation-capture-shard.md). Because the manifest is
written last, a directory without one is an interrupted capture and is not read.

## Reading a shard

Any numpy can read the shard; the values are the model's own, widened from the checkpoint's
dtype to float32 exactly:

```bash
python3 -c 'import numpy as np; z = np.load("capture/shard_00000.npz"); print({k: (z[k].dtype.str, z[k].shape) for k in z.files})'
```

```text
{'layer_0': ('<f4', (4, 16)), 'layer_1': ('<f4', (4, 16)), 'labels': ('<i8', (4,))}
```

## Run records

A capture writes an event file and a catalog line under `${OBS_LOG_DIR}/harness/`, with an
`activation-capture.header` event recording the resolved configuration once the dataset is
read. It writes no metrics file: a capture has no per-step series, and its example count is in
the manifest. See [Run records](../observability/run-records.md).
