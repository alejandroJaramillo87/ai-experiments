---
title: Steering
---

# Steering

A control vector is a set of per-block directions added to the residual stream inside the
forward pass. The workflow has three steps: extract a vector from a capture, recreate the
serving harness with the vector loaded, and score the steered service against two baselines.
Weights are never modified; every intervention acts on activations only.

## 1. Extract

The default method is the difference of class means: the mean residual of the label-1
prompts minus the mean of the label-0 prompts, at each captured layer. It reads the same
capture directory the [Probes](probes.md) guide trains from and needs no model of its own,
so it runs on the CPU:

```bash
make steer-extract ARGS="--capture /mnt/ai-data/interp/activations/honesty-e2b-it \
    --out /mnt/ai-data/interp/vectors/honesty-e2b-it"
```

The output is `vector.safetensors` and a `manifest.json` under the vector directory, the
[Control-vector artifact](../../formats/control-vector-artifact.md). The difference lands in
residual-stream units, so a scale of 1.0 means one class gap at every layer. `--layers`
narrows the export to a subset of the captured layers; the harness's
`--control-vector-layer-range` narrows at load either way. The manifest carries the
capture's model hash, which the harness checks when it loads the vector.

`--method sae_row` exports a single decoder row of a pretrained sparse autoencoder instead,
and takes `--sae-release`, `--sae-id`, `--feature`, and `--layer` with `--model`.

## 2. Serve steered

```bash
make steer-up VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b-it SCALE=1.0
make steer-up VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b-it SCALE=0.5 LAYER_RANGE="8 24"
make steer-down
```

`steer-up` recreates the serving harness with the vector loaded, translating `VECTOR` from
its host path to the container spelling under `/app/vectors`; it refuses a vector outside
`/mnt/ai-data/interp/vectors`, because that is the only tree the service mounts. `SCALE`
multiplies every direction, and `LAYER_RANGE` is one quoted argument holding two inclusive
0-based block indices. `steer-down` recreates the service unsteered, which is the production
state, and a scorecard restores it on any exit. Loading the checkpoint takes minutes.

The harness's own view of the flags, the fit check with `--status --deep`, and a one-shot
steered run by path are on [Applying a control vector](../../rust-harness/guides/control-vector-steering.md).

## 3. Score

```bash
make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b-it ATTRIBUTE="honesty" \
    ARGS="--layer-range '8 24'"
```

The scorecard drives three arms against live services: the unsteered service, the service
with a tuned system prompt, and the steered service at each scale in `--scales` (default
`0.5,1,2`). For each arm it generates the behavior prompts through the subject, scores the
target attribute with an LLM judge on a different service, and runs a coherence check on
neutral prompts, so oversteering shows up as lost fluency rather than as a higher score. The
subject defaults to the serving harness on port 8005 and the judge to `llama-cpu`; both must
be up, and the scorecard refuses to run while the GPU lock is held. The judge runs with
thinking off and a JSON schema enforced by the sampler, so its verdict is a bounded integer
rather than a hoped-for format. Output follows the repository's report envelope and lands
under the vector directory. The subject decodes on the CPU at single-digit tokens per second,
so a full sweep takes minutes per arm.

## What can go wrong

| Symptom | Cause | Remedy |
|---------|-------|--------|
| `steer-up` refuses the vector | `VECTOR` is outside `/mnt/ai-data/interp/vectors`, or `vector.safetensors` is missing | Export there with `steer-extract --out` |
| The harness refuses the vector as trained on a different model | The capture came from other weights | Serve the weights the vector was extracted from, or re-extract |
| The scorecard refuses to start | The GPU lock is held by a harvest or a training run | Wait for it, or stop it |
| A verdict is reported as cut off | The judge budget was too small for the reply | Raise `--judge-max-tokens` |
