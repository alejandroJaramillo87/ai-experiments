---
title: Producing a probe and a control vector
---

# Producing a probe and a control vector

The shortest path from a harness capture to a probe artifact and a control-vector artifact the
harness accepts. Every step runs on the CPU. The two Python commands are the only part of the
repository's Python interpretability tooling this site documents; the rest of that tooling
runs against the repository's GPU workstation and is out of scope here.

This page assumes `BIN` and `MODEL` from [Build](../rust-harness/getting-started/build.md), a
working directory of `/tmp/harness-demo`, and the repository checked out at `$REPO`:

```bash
export REPO=/path/to/ai-experiments
```

## Python environment

The two commands need numpy, scikit-learn, safetensors, and xxhash. The repository pins them in
its `sandbox` dependency group, which installs without torch:

```bash
cd "$REPO"
uv sync --only-group sandbox --only-group dev
```

`uv run python -m interp --help` then lists the subcommands. The commands below are run from
`$REPO` because `interp` is a package in the repository.

## 1. Capture

A probe needs a labelled dataset with both classes well represented in every split the trainer
makes; forty examples, twenty per class, is a working floor. Write one and capture it at every
layer of interest:

```bash
cd /tmp/harness-demo
python3 - <<'EOF'
import json
pos = ["the cat sat on the mat", "a dog chased the ball", "birds sing at dawn", "the horse ran across the field",
       "a fox slept under the hedge", "the owl watched from the barn", "two goats climbed the rocks", "the rabbit hid in the grass",
       "a cow stood by the gate", "the sheep grazed all afternoon", "a mouse ran along the wall", "the frog jumped into the pond",
       "a deer crossed the road at dusk", "the duck led her chicks to water", "a bear fished in the river", "the wolf howled at the moon",
       "a squirrel buried an acorn", "the hen laid an egg in the straw", "a swan glided over the lake", "the pig rolled in the mud"]
neg = ["stocks fell sharply on tuesday", "the central bank raised rates", "quarterly earnings beat estimates", "bond yields climbed again",
       "the index closed at a record high", "inflation slowed for a third month", "the company issued new shares", "analysts cut their price target",
       "the currency weakened against the dollar", "oil futures rose two percent", "the merger cleared regulatory review", "retail sales missed forecasts",
       "the fund reported strong inflows", "credit spreads widened in the quarter", "the treasury sold ten year notes", "housing starts declined in august",
       "the bank raised its dividend", "consumer confidence edged lower", "the deficit narrowed last year", "markets priced in another hike"]
with open("examples.jsonl", "w") as f:
    for a, b in zip(pos, neg):
        f.write(json.dumps({"text": a, "label": 1}) + "\n")
        f.write(json.dumps({"text": b, "label": 0}) + "\n")
EOF
"$BIN/activation-capture" --model "$MODEL" --dataset examples.jsonl --layers 0,1 --out capture
```

```text
[OK] Capture: 40 examples, 2 layers, 1 shards -> capture
```

The capture's manifest records the sha256 of `$MODEL`'s weights, and everything below inherits
it. See [Activation capture shard](activation-capture-shard.md).

## 2. Train and register a probe

```bash
cd "$REPO"
uv run python -m interp probe-train --capture /tmp/harness-demo/capture --name demo --out /tmp/harness-demo/probes
```

```text
[RUN] Training probes from /tmp/harness-demo/capture
[OK] Probe registered: demo (layer 1, val AUROC 1.0, test AUROC 0.2222)
  [INFO] layer 0: val 0.0
  [INFO] layer 1: val 1.0
```

The trainer fits one logistic-regression probe per captured layer with scikit-learn, picks the
layer with the best validation AUROC, and registers it as `probe.npz` and `manifest.json` under
`<out>/<name>/`. `--seed` (default 0) fixes the split. The default `--out` is a path under
`/mnt/ai-data`, which is the repository's workstation layout; pass `--out` anywhere else. The
scores above are what a random model produces on forty examples and mean nothing; a real
checkpoint separates the two classes.

## 3. Export the probe for the harness

```bash
uv run python -m interp probe-export --probe /tmp/harness-demo/probes/demo
```

```text
[OK] Image written: /tmp/harness-demo/probes/demo/probe.safetensors
[OK] Sidecar written: /tmp/harness-demo/probes/demo/probe.json
[WARN] Outside /mnt/ai-data/interp/probes: harness-serve mounts only that tree, so it cannot read this image
```

The warning concerns the repository's containerized server and does not apply to a harness run
by path. `--out <path>` names the image; the sidecar is written beside it with a `.json`
extension. See [Probe artifact](probe-artifact.md).

## 4. Extract a control vector

The difference-of-means extractor reads the same capture, takes the mean residual of the
positive class minus the mean of the negative class at each captured layer, and writes the
directions with a manifest. It needs no model:

```bash
uv run python -m interp steer-extract --method diff_means --capture /tmp/harness-demo/capture --out /tmp/harness-demo/vec
```

```text
[INFO] Classes: 20 positive, 20 negative; model_hash 0df22ea2...
[OK] Vector exported: /tmp/harness-demo/vec/vector.safetensors
[OK] Layers: 2, width: 16
```

`--layers 1` restricts the export to a subset of the captured layers. See
[Control-vector artifact](control-vector-artifact.md).

## 5. Load both into the harness

```bash
cd /tmp/harness-demo
export PROBE=/tmp/harness-demo/probes/demo/probe.safetensors
export VECTOR=/tmp/harness-demo/vec/vector.safetensors
"$BIN/harness" --model "$MODEL" --probe "$PROBE" --control-vector "$VECTOR" --status --deep
```

```text
[OK] probe: demo, layer 1, site LayerOutput, width 16
[OK] probe geometry: width 16 and layer 1 fit the model
[OK] probe identity: matches the served weights, sha256 0df22ea2...
[OK] control vector: width 16, scale 1
[OK] control vector identity: matches the served weights, sha256 0df22ea2...
```

```bash
"$BIN/harness" --model "$MODEL" --probe "$PROBE" --control-vector "$VECTOR" --max-tokens 8 --temperature 0 "Say hello"
```

The run scores the probe at every position and steers every block in the vector. What each
flag does from here is covered in
[Loading a probe for live monitoring](../rust-harness/guides/probe-monitoring.md) and
[Applying a control vector](../rust-harness/guides/control-vector-steering.md).

## Against a real checkpoint

The same five steps apply with `MODEL` pointing at the real checkpoint and `--text-only` on
the harness commands. Two things change in practice. The capture takes minutes rather than
seconds, at the CPU backend's prefill rate, so a dataset of a few hundred examples is the
practical size. And there is a second way to make the capture: the repository's
`probe-capture` command runs the same dataset through `transformers` on a GPU and writes the
same shard format. It needs torch and the workstation and is not documented here; a capture
made that way flows through steps 2 to 5 unchanged, and the harness accepts the result
because the manifest's `model_hash` names the same weights.
