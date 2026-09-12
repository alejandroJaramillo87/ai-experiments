---
title: Probe artifact
---

# Probe artifact

A linear probe as the harness loads it: a safetensors image of three f32 vectors, plus a JSON
sidecar at the same path with a `.json` extension. Written by the Python probe exporter; read
by `harness --probe`. The contract is `schemas/probe_artifact.json` in the repository.

## Layout

```text
<dir>/
  probe.safetensors      the image; --probe takes this path
  probe.json             the sidecar, at the image's path with .json in place of .safetensors
```

The sidecar is mandatory. It carries the layer, the site, and the decision threshold, without
which the three vectors cannot be applied, so an image without its sidecar does not load. This
is the opposite of the control vector's manifest, which is advisory.

## The image

| Tensor | dtype | Shape | Contents |
|--------|-------|-------|----------|
| `weights` | f32 | `[width]` | The probe direction |
| `mu` | f32 | `[width]` | The per-dimension mean the activations are centred by before scoring |
| `sigma` | f32 | `[width]` | The per-dimension scale the centred activations are divided by; strictly positive |

All three share one width, which is the residual width of the model the probe was trained on.

## The sidecar

| Field | Contents |
|-------|----------|
| `schema_version` | 1 |
| `artifact` | `probe` |
| `timestamp` | ISO 8601 UTC |
| `git_sha` | The repository commit of the exporter |
| `name` | The probe's registered name |
| `model_hash` | sha256 of the `model.safetensors` the training capture was taken from, or `unknown` |
| `layer` | The 0-based decoder block whose output the probe reads |
| `bias` | The scalar added to the dot product; finite |
| `threshold` | The score at or above which a position is flagged; finite |
| `width` | The vector width, restated so a mismatch is reported before the tensors are read |
| `site` | Optional; `layer_output` when present, which is also the default |

## Scoring

At every position of the probe's layer the harness takes the residual row `x`, computes the
logistic of `weights · ((x - mu) / sigma) + bias`, and flags the position when the result is at
or above `threshold`. The Python scorer does the same arithmetic, which is what lets a probe
trained on Python-read shards be scored in Rust.

## What the harness checks before loading

1. The sidecar exists, parses, has `artifact` of `probe` and `schema_version` of 1, and its
   scalars are finite.
2. The sidecar's `width` equals the model's residual width, and `layer` is within the model's
   depth.
3. The sidecar's `model_hash`, unless `unknown`, equals the sha256 of the served
   `model.safetensors`. `harness --status --deep` runs this check without loading the model.
4. The three tensors exist, are 1-D f32, share the sidecar's width, and `sigma` is strictly
   positive.

## Producing one

The exporter takes a probe the trainer registered and writes the pair:

```bash
uv run python -m interp probe-export --probe /tmp/harness-demo/probes/demo
```

See [Producing a probe and a control vector](producing-a-probe-and-control-vector.md) for the
whole path from a capture, and [Loading a probe for live monitoring](../rust-harness/guides/probe-monitoring.md)
for what the harness does with it.
