---
title: Applying a control vector
---

# Applying a control vector

`--control-vector` loads a set of per-block steering directions and adds each one to its
block's output residual on every forward pass. The directions are a safetensors image with a
`manifest.json` beside it; the harness reads the image, checks the manifest's model identity
against the served weights, and steers inside the forward pass rather than through anything
outside the process. This page assumes `BIN` and `MODEL` from
[Build](../getting-started/build.md) and a vector made for that checkpoint.

## Get a vector that fits

A vector is fitted to one checkpoint: each direction has the residual width, and the manifest
records the sha256 of the weights it was extracted from. The committed fixture vector under
`rust-harness/fixtures/` is a format fixture and fits no runnable checkpoint. The shortest path
to a vector for `$MODEL` is the difference-of-means extractor in
[Producing a probe and a control vector](../../formats/producing-a-probe-and-control-vector.md),
which runs in seconds on the CPU. The commands below assume `VECTOR` names the image:

```bash
export VECTOR=/tmp/harness-demo/vec/vector.safetensors
```

## Check the fit before running

```bash
"$BIN/harness" --model "$MODEL" --control-vector "$VECTOR" --status --deep
```

```text
[OK] control vector: width 16, scale 1
[OK] control vector identity: matches the served weights, sha256 0df22ea2...
```

A vector for a different model stops the run with the command to re-check:

```text
[ERROR] Task failed: control vector was trained on a different model: control vector model_hash fedcba98..., served model sha256 0df22ea2...
  Solution: serve the weights the control vector was fitted on, or re-export the control vector against these
```

A vector whose manifest is missing loads with a warning and no identity check, because the
manifest carries provenance rather than anything the forward pass needs.

## Steer

```bash
"$BIN/harness" --model "$MODEL" --control-vector "$VECTOR" --max-tokens 16 --temperature 0 "Say hello"
```

The three steering flags:

| Flag | Effect | Default |
|------|--------|---------|
| `--control-vector <path.safetensors>` | The image; one `block.<index>` tensor per steered block | none |
| `--control-vector-scale <f>` | Multiplier applied to every direction | 1.0 |
| `--control-vector-layer-range "<start> <end>"` | Inclusive 0-based block range; directions outside it are dropped as the image is read | every block in the image |

The layer range is one quoted argument holding two integers, not two arguments. Both modifier
flags require `--control-vector` and are refused without it. A scale of 0 loads the vector and
adds nothing, which is the control arm of a comparison.

```bash
"$BIN/harness" --model "$MODEL" --control-vector "$VECTOR" --control-vector-scale 2 --control-vector-layer-range "0 1" --max-tokens 16 --temperature 0 "Say hello"
```

Block indices are the 0-based decoder blocks, and block 0 is steerable. The manifest's own
`layer_range` records the band the directions were extracted over and is provenance only; the
flag is what selects at load.

## Under the server and in chat

The vector is installed on the model before any mode starts, so `--serve` and `--chat` steer
every request and every turn identically. The scale and range are fixed for the process; a
request cannot change them.

## Where the vector comes from

The image is `vector.safetensors` with one f32 tensor per block named `block.<index>`, all of
one width, and the manifest beside it records the method (`diff_means` or `sae_row`), the
source, the layers present, the width, and the model identity. The format is documented in
[Control-vector artifact](../../formats/control-vector-artifact.md).
