---
title: Producing a probe and a control vector
---

# Producing a probe and a control vector

The shortest path from a harness capture to a probe artifact and a control-vector artifact the
harness accepts. Every step runs on the CPU, so the whole path runs off the workstation. The
same three Python commands on the workstation, with the GPU capture and the evaluation and
monitoring steps around them, are the [Probes](../interp/guides/probes.md) and
[Steering](../interp/guides/steering.md) guides of the Interp section.

This page assumes `BIN` and `MODEL` from [Build](../rust-harness/getting-started/build.md), a
working directory of `/tmp/harness-demo`, and the repository checked out at `$REPO`. The
outputs are from `gemma-4-E2B-it`.

```bash
export REPO=/path/to/ai-experiments
```

## Python environment

The commands need numpy, scikit-learn, safetensors, and xxhash. The repository pins them in
its `sandbox` dependency group, which installs without torch:

```bash
cd "$REPO"
uv sync --only-group sandbox --only-group dev
```

`uv run python -m interp --help` then lists the subcommands. The commands below are run from
`$REPO` because `interp` is a package in the repository.

## 1. Capture

A probe needs a labelled dataset with both classes well represented in every split the trainer
makes. The repository commits one, `interp/probes/prompts/refusal.jsonl`: 258 lines, each a
request a safety-tuned model refuses paired with a similar one it answers. Capture it at every
layer, since the cost is in the forward pass rather than the write:

```bash
cd /tmp/harness-demo
"$BIN/activation-capture" --model "$MODEL" --text-only \
    --dataset "$REPO/interp/probes/prompts/refusal.jsonl" \
    --layers $(seq -s, 0 34) --out capture
```

```text
[OK] Capture: 258 examples, 35 layers, 1 shards -> capture
```

About two and a half minutes on E2B-it. The capture's manifest records the sha256 of
`$MODEL`'s weights, and everything below inherits it. See
[Capturing activations](../rust-harness/guides/activation-capture.md) and
[Activation capture shard](activation-capture-shard.md).

## 2. Train and register a probe

```bash
cd "$REPO"
uv run python -m interp probe-train --capture /tmp/harness-demo/capture --name refusal --out /tmp/harness-demo/probes
```

```text
[RUN] Training probes from /tmp/harness-demo/capture
[OK] Probe registered: refusal (layer 1, val AUROC 1.0, test AUROC 0.961)
  [INFO] layer 0: val 0.9786
  [INFO] layer 1: val 1.0
  [INFO] layer 2: val 1.0
  ...
  [INFO] layer 34: val 1.0
```

The trainer fits one logistic-regression probe per captured layer with scikit-learn, picks the
layer with the best validation AUROC, and registers it as `probe.npz` and `manifest.json` under
`<out>/<name>/`. `--seed` (default 0) fixes the split. The default `--out` is a path under
`/mnt/ai-data`, which is the repository's workstation layout; pass `--out` anywhere else.

Read the table with care. Validation AUROC saturates at 1.0 from layer 1 onward, and the
trainer keeps the first layer to reach the best value, so the selected layer is an artefact of
tie order on a 27-example validation fold rather than a finding about where refusal is
represented. The test AUROC, 0.961, is the number to quote. Perfect separation at every depth
usually means the classes differ by something cheaper than the attribute, and the paired
dataset is designed against that; nothing downstream on this site depends on the probe being
good, only on its fitting the model.

## 3. Export the probe for the harness

```bash
uv run python -m interp probe-export --probe /tmp/harness-demo/probes/refusal
```

```text
[RUN] Exporting /tmp/harness-demo/probes/refusal
[OK] Image written: /tmp/harness-demo/probes/refusal/probe.safetensors
[OK] Sidecar written: /tmp/harness-demo/probes/refusal/probe.json
[WARN] Outside /mnt/ai-data/interp/probes: harness-serve mounts only that tree, so it cannot read this image
```

The warning concerns the repository's containerized server and does not apply to a harness run
by path. `--out <path>` names the image; the sidecar is written beside it with a `.json`
extension, and the safetensors writer needs the destination directory to exist already. The
image for a single 1536-wide direction is about 20 KB. See [Probe artifact](probe-artifact.md).

## 4. Extract a control vector

The difference-of-means extractor reads the same capture, takes the mean residual of the
positive class minus the mean of the negative class at each captured layer, and writes the
directions with a manifest. It needs no model:

```bash
uv run python -m interp steer-extract --method diff_means --capture /tmp/harness-demo/capture --out /tmp/harness-demo/vec
```

```text
[INFO] Classes: 129 positive, 129 negative; model_hash 2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550
[OK] Vector exported: /tmp/harness-demo/vec/vector.safetensors
[OK] Layers: 35, width: 1536
```

`--layers 10,11,12` restricts the export to a subset of the captured layers; the harness's
`--control-vector-layer-range` selects at load either way. See
[Control-vector artifact](control-vector-artifact.md).

## 5. Load both into the harness

```bash
cd /tmp/harness-demo
export PROBE=/tmp/harness-demo/probes/refusal/probe.safetensors
export VECTOR=/tmp/harness-demo/vec/vector.safetensors
"$BIN/harness" --model "$MODEL" --text-only --probe "$PROBE" --control-vector "$VECTOR" --status --deep
```

The preflight prints five `[OK]` lines for the two artifacts: the probe's name, layer, site,
and width; its geometry against the model; its identity against the served weights' digest;
the vector's width and scale; and the vector's identity. Both identity lines name the same
sha256 the capture manifest recorded in step 1, because both artifacts inherited it.

```bash
echo "Describe how to pick a lock in two sentences" | "$BIN/harness" --model "$MODEL" --text-only --chat \
    --probe "$PROBE" --control-vector "$VECTOR" --control-vector-scale 0.5 --control-vector-layer-range "10 16"
```

The run scores the probe at every position and steers blocks 10 through 16. What each flag
does from here, and how to read the result against a baseline, is covered in
[Loading a probe for live monitoring](../rust-harness/guides/probe-monitoring.md) and
[Applying a control vector](../rust-harness/guides/control-vector-steering.md).

## A second way to capture

The repository's `probe-capture` command runs the same dataset through `transformers` on a GPU
and writes the same shard format. It needs torch and the workstation and is not documented
here; a capture made that way flows through steps 2 to 5 unchanged, and the harness accepts
the result because the manifest's `model_hash` names the same weights.
