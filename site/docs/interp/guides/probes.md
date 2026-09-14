---
title: Probes
---

# Probes

A linear probe is a direction in one layer's residual stream, fitted to a labelled dataset,
that the harness scores at every position while the model answers. The workflow has five
verbs: capture labelled activations, train and register a probe from the capture, evaluate it
against black-box baselines, export it as the image the harness loads, and monitor what it
scored on served traffic. Every step after the capture runs on the CPU, and the monitor runs
while inference is up.

This page runs the workflow on the workstation through its `make` targets, with paths under
`/mnt/ai-data/interp/`. The same five verbs run off the workstation by path; the Formats
section's [Producing a probe and a control vector](../../formats/producing-a-probe-and-control-vector.md)
walks that path with recorded outputs, and this page does not repeat them.

## 1. Capture

A probe needs a dataset with both classes well represented: one JSON object per line with a
`text` and an integer `label`. The repository commits `interp/probes/prompts/refusal.jsonl`,
258 paired lines. Two tools write the capture, in the same shard format:

- **`probe-capture`**, in the workbench, on the GPU, through the Hugging Face model. Pass
  `--model` a local checkpoint directory in the container's spelling, `/app/models/...`,
  where the workbench mounts the models tree read-only. A repository id is accepted but
  leaves the model hash `unknown`, and the harness answers that sentinel by skipping its
  identity check rather than failing it, so the probe would load into any model of the
  right width unchecked. The command prints which of the two it recorded: `[OK] model_hash
  <64 hex>` after the shard line, or a `[WARN]` block saying that a probe from this capture
  will load unchecked and how to fix it. Stop on the warning before training anything
  meant for the serving harness. The dataset is validated as it is read: a line that is not
  JSON, lacks `text` or `label`, or carries a non-integer label is refused, and a set too
  small to split is refused by the trainer.
- **`activation-capture`**, the harness binary, on the CPU, from the same forward pass the
  harness serves with. See [Capturing activations](../../rust-harness/guides/activation-capture.md).

```bash
make probe-capture ARGS="--model /app/models/hf/gemma-4-E2B-it \
    --dataset interp/probes/prompts/refusal.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/activations/refusal-v1"
```

The target runs the GPU preflight first. The capture's `manifest.json` records the model
hash and the dataset's content hash, and everything below inherits them. The format is
[Activation capture shard](../../formats/activation-capture-shard.md).

## 2. Train and register

```bash
make probe-train ARGS="--capture /mnt/ai-data/interp/activations/refusal-v1 --name refusal-v1"
```

The trainer fits one logistic-regression probe per captured layer with scikit-learn over a
deterministic split keyed on the text hash, so a re-capture cannot leak test data; it picks
the layer with the best validation AUROC, chooses the threshold as the median validation
score, and registers `probe.npz` and `manifest.json` under
`/mnt/ai-data/interp/probes/<name>/`. `--seed` fixes the split and `--out` moves the
registry. Validation AUROC that saturates at 1.0 across layers means the selected layer is a
tie-order artefact rather than a finding; the test AUROC is the number to quote.

One caveat decides whether a probe is usable in-band, and the numbers above do not show it.
The capture takes the last non-padded token of each prompt, and the trainer fits its
threshold there; the harness scores every position of the layer. A probe whose validation
AUROC is 1.0 at every layer can still flag most positions of a benign arithmetic prompt
when served, because a threshold calibrated at one position means nothing at position 3.
The recorded served run on this site's own refusal probe did exactly that. Step 3 is where
usability is decided; steps 1 and 2 only make a probe that fits.

## 3. Evaluate against baselines

```bash
make probe-eval ARGS="--probe /mnt/ai-data/interp/probes/refusal-v1 --label-name refusal"
```

The evaluation is serving-dependent: it generates the model's answers for the test inputs
through the subject service and asks a judge service to classify them, then compares the
probe against two black-box baselines that classify the output text, a bag-of-words
classifier and the judge. The report, `evaluation.json` beside the probe, marks the probe
`deployable` only when its AUROC beats the bag-of-words baseline and its accuracy beats the
judge's. A probe that cannot beat them is not worth scoring in production, and the monitor
should not run it. `--subject-url` and `--judge-url` default to the `llama-gpu` and
`vllm-gpu` endpoints; both must be up.

## 4. Export for the harness

```bash
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/refusal-v1"
```

`probe.npz` is the training format and only Python reads it. The harness reader takes a
safetensors image plus a JSON sidecar, the
[Probe artifact](../../formats/probe-artifact.md), and this step writes the pair beside the
registered probe after checking its hash. Run it after the probe is judged deployable, not
before: exporting is what makes a probe reachable by the process that scores production
traffic. `--out` names the image; the sidecar is written beside it. An image outside
`/mnt/ai-data/interp/probes` earns a warning, because the serving harness mounts only that
tree.

## 5. Load and monitor

The serving harness loads the image through `HARNESS_PROBE`, in the container spelling, and
writes the served-request log through `HARNESS_INTERP_LOG`:

```bash
make harness-serve HARNESS_PROBE=/app/probes/refusal-v1/probe.safetensors
```

Before loading, the harness checks the probe's width against the residual width, its layer
against the block count, and its recorded model hash against the served weights, and refuses
a mismatch. What it checks, the flags for a one-shot run, and what a scored run leaves behind
are on the harness's [Loading a probe for live monitoring](../../rust-harness/guides/probe-monitoring.md)
page.

```bash
make probe-monitor ARGS="--window 7d"
```

The monitor aggregates the window into `report.json` and `flagged.jsonl`. It loads no
model, so it runs while the service is up. Recorded on the workstation, twice over the same
one-record window so the second run carries a drift delta against the first:

```text
[OK] Reported 1/1 records (1 flagged, skipped {'malformed_json': 0,
     'no_timestamp': 0, 'out_of_window': 0, 'no_text': 0, 'no_scores': 0})
[OK] refusal-v1: mean 0.9932, p95 0.9932, over-threshold 1.0
[OK] Report written: /mnt/ai-data/interp/monitor/2026-08-14T015441Z/report.json
[OK] Report written: /mnt/ai-data/interp/monitor/2026-08-14T015445Z/report.json
```

The over-threshold fraction of 1.0 is the monitor faithfully reporting the miscalibrated
probe of the caveat above on a single record, not a baseline. The record it reads and the report it writes are on
[Served-request log](../observability/served-request-log.md).

## What can go wrong

| Symptom | Cause | Remedy |
|---------|-------|--------|
| The preflight refuses `probe-capture` | A GPU inference service is running | Stop it: `make down`, or `docker compose -f docker-compose.inference.yaml stop <service>` |
| `--model` is refused with a correction | A host `/mnt/ai-data/models` path inside the workbench | Spell it `/app/models/...` |
| The harness refuses the probe as trained on a different model | The capture came from other weights | Serve the weights the probe was fitted on, or re-capture and re-train against these |
| `probe-export` warns the image is outside the mounted tree | `--out` points elsewhere | Export under `/mnt/ai-data/interp/probes` for the serving harness; a harness run by path reads anywhere |
| The monitor reports no scored records | The harness served without `HARNESS_PROBE`, or the window predates the export | Check the log records carry a `probe` field |
