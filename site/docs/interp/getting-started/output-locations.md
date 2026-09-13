---
title: Output locations
---

# Output locations

Every verb writes under two roots: the data tree under `/mnt/ai-data/interp/` for artifacts
and the observability root under `OBS_LOG_DIR` for run records. This page says which verb
writes where.

## Under the data tree

`make interp-dirs` creates the tree once. Each directory is regenerable from a command and
its inputs, and each artifact directory carries a `manifest.json` naming the model hash, the
input hashes, the git SHA, and the artifact's own hash.

| Path | Written by | Contents |
|------|------------|----------|
| `activations/<name>/` | `probe-capture`, `harvest` | Capture shards and `manifest.json`; a regenerable cache that `gc` deletes oldest first, keeping the manifests |
| `probes/<name>/` | `probe-train`, `probe-eval`, `probe-export` | `probe.npz` and `manifest.json` from training, `evaluation.json` from evaluation, `probe.safetensors` and `probe.json` from export |
| `vectors/<name>/` | `steer-extract`, `steer-scorecard` | `vector.safetensors`, `manifest.json`, and one scorecard run per invocation |
| `saes/<name>/` | `sae-train`, `sae-eval` | The trained autoencoder and its evaluation |
| `organisms/<name>/` | `organism generate`, `seal` | The generated corpus, the verification report, `ground_truth.json`, and the blinded manifest |
| `logs/<service>/` | The serving harness | The served-request log, one `YYYY-MM-DD.jsonl` per day |
| `monitor/<probe>/` | `probe-monitor` | `report.json` and `flagged.jsonl` |
| `hf-cache/` | The workbench | The Hugging Face cache |
| `neuronpedia/` | The Neuronpedia platform | Its database and imports |

The names under `activations/`, `probes/`, `vectors/`, and `saes/` come from `--out` or
`--name` on the verb, or from `run.name` in the recipe for the recipe-driven verbs.

## Under the observability root

Every verb past argument parsing writes an event file and, when it ends, a catalog line
under `${OBS_LOG_DIR}/interp/`, default `/mnt/ai-data/logs/interp/`. `make interp-runs`
lists the catalog. See [Run records](../observability/run-records.md).

## Off the workstation

The CPU verbs run anywhere with the sandbox group installed. Point `--out` and `--capture`
at a working directory and set `OBS_LOG_DIR` to a temporary path, because the default roots
are workstation paths:

```bash
export OBS_LOG_DIR=/tmp/interp-demo/obs
uv run python -m interp probe-train --capture /tmp/harness-demo/capture --name refusal --out /tmp/harness-demo/probes
```

The Formats section's [Producing a probe and a control vector](../../formats/producing-a-probe-and-control-vector.md)
runs the whole CPU path that way, with recorded outputs.
