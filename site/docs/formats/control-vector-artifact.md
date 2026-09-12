---
title: Control-vector artifact
---

# Control-vector artifact

A set of steering directions as the harness loads them: a directory holding a safetensors image
of one f32 direction per steered decoder block, plus a `manifest.json` beside it. Written by
the Python control-vector exporter; read by `harness --control-vector`. The contract is
`schemas/control_vector_artifact.json` in the repository.

## Layout

```text
<dir>/
  vector.safetensors     the directions; --control-vector takes this path
  manifest.json          provenance and model identity
```

## The image

One tensor per steered block, named `block.<index>`, each 1-D f32, all of one width, at least
one present, and no non-finite value in any of them. `index` is the 0-based decoder block whose
output residual the direction is added to, and block 0 is steerable. The direction is added to
every position's residual at that block on every forward pass, scaled by
`--control-vector-scale`.

A non-finite direction would turn the whole residual stream to NaN and make the model emit
noise rather than fail, so the exporter refuses one and names the block.

## The manifest

| Field | Contents |
|-------|----------|
| `schema_version` | 1 |
| `artifact` | `control_vector` |
| `model_id` | The checkpoint path or Hugging Face id |
| `model_revision` | The revision; `safetensors` for a local checkpoint captured by the harness |
| `model_hash` | sha256 of `model.safetensors`, or `unknown` |
| `method` | `diff_means` or `sae_row` |
| `source` | What the extractor read. For `diff_means`: the capture directory, its dataset path and sha256, the class counts, and for a harness capture the site and backend. For `sae_row`: the SAE release, id, feature index, and layer |
| `layers` | The block indices present in the image, sorted |
| `layer_range` | The band directions were extracted over, or `null`; provenance, not a filter |
| `width` | The shared width of the directions |
| `vector_xxhash` | xxhash of the image file |
| `git_sha` | The repository commit of the exporter |
| `created` | ISO 8601 UTC timestamp |

## The manifest is advisory

A vector whose manifest is missing or unreadable still loads; the harness warns and skips the
identity check. The manifest carries provenance and the model identity, and the forward pass
needs neither. This is deliberately the opposite of the probe's sidecar, which carries the
layer, the site, and the threshold and is therefore mandatory. The Python side is stricter and
refuses a directory without a manifest, because it records the manifest into its reports.

## What the harness checks before loading

1. Every tensor is named `block.<index>`, is 1-D f32, and all share one width; at least one is
   present; no value is non-finite.
2. The width equals the model's residual width, and every index is within the model's depth.
3. When a manifest is present and its `model_hash` is not `unknown`, the hash equals the sha256
   of the served `model.safetensors`. `harness --status --deep` runs this check without loading
   the model.
4. `--control-vector-layer-range "<start> <end>"`, if given, drops the blocks outside the
   inclusive range as the image is read.

## Producing one

The difference-of-means extractor reads a capture and needs no model:

```bash
uv run python -m interp steer-extract --method diff_means --capture /tmp/harness-demo/capture --out /tmp/harness-demo/vec
```

See [Producing a probe and a control vector](producing-a-probe-and-control-vector.md) for the
path from a capture, and [Applying a control vector](../rust-harness/guides/control-vector-steering.md)
for the steering flags.
