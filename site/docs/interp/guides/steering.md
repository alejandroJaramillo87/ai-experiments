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

Recorded on the workstation, from a capture the harness wrote of `gemma-4-E2B-it`:

```text
[RUN] Extracting diff-means vector: /mnt/ai-data/interp/captures/rust-e2b-it-refusal
[INFO] Classes: 129 positive, 129 negative; model_hash 2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550
[OK] Vector exported: /mnt/ai-data/interp/vectors/refusal-e2b-it/vector.safetensors
[OK] Layers: 35, width: 1536
[OK] Vector xxhash: d66c1d6da0b70309
```

The output is `vector.safetensors` and a `manifest.json` under the vector directory, the
[Control-vector artifact](../../formats/control-vector-artifact.md). The manifest records
the method, the capture, the dataset hash, the class counts, and the vector's own hash, which
a re-run reproduces. The difference lands in
residual-stream units, so a scale of 1.0 means one class gap at every layer. `--layers`
narrows the export to a subset of the captured layers; the harness's
`--control-vector-layer-range` narrows at load either way. The manifest carries the
capture's model hash, which the harness checks when it loads the vector.

`--method sae_row` exports a single decoder row of a pretrained sparse autoencoder instead,
and takes `--sae-release`, `--sae-id`, `--feature`, and `--layer` with `--model`. The row
must have the target model's residual width, because the harness checks the vector's width
against the served checkpoint's hidden size: a released Gemma Scope row is Gemma-2-2B-wide,
and the smoke autoencoder is 1536 wide against E4B's 2560, so steering E4B by this route
first needs an autoencoder trained on an E4B capture. See [Sparse autoencoders](saes.md).

## 2. Serve steered

```bash
make steer-up VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b-it SCALE=1.0
make steer-up VECTOR=/mnt/ai-data/interp/vectors/honesty-e2b-it SCALE=0.5 LAYER_RANGE="8 24"
make steer-down
```

`steer-up` recreates the serving harness with the vector loaded, translating `VECTOR` from
its host path to the container spelling under `/app/vectors`. Three guards run in order
before the service is touched: an unset `VECTOR` exits 2 with
`VECTOR=/mnt/ai-data/interp/vectors/<name> is required`; a directory without
`vector.safetensors` exits 1 naming the missing file; and a vector outside
`/mnt/ai-data/interp/vectors` exits 1 with `VECTOR must live under
/mnt/ai-data/interp/vectors (the mounted tree)`, because that is the only tree the service
mounts. On success the target prints the container path and the scale, and the service log
carries a `Control vector:` line naming the vector, the scale, and the band; after
`steer-down` that line reads `none, serving unsteered`. `SCALE`
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
under the vector directory. Recorded on the workstation, a refusal vector on
`gemma-4-E2B-it` over blocks 10 to 16 at two scales:

```text
=== Steering scorecard ===

[RUN] Arm: unsteered baseline
[RUN] Arm: tuned system prompt
[RUN] Arm: steered scale=0.5
[RUN] Arm: steered scale=1

[OK] baseline: behavior 3.0, fluency 8.7, latency 11.8189s
[OK] system_prompt: behavior 5.0, fluency 8.7, latency 8.6028s
[OK] steered@0.5: behavior 10.0, fluency 4.0, latency 2.5877s
[OK] steered@1: behavior 10.0, fluency 1.5, latency 11.8184s
[OK] Scorecard written: /mnt/ai-data/interp/vectors/refusal-e2b-it/scorecard-2026-09-06T171427Z/scorecard.json
```

Read the two columns together: both steered arms reach the ceiling on the attribute, and
both lose fluency, the second badly. That is what the coherence arm is for. The four-arm
sweep took 14 minutes 32 seconds on the CPU backend, 64 subject calls and 64 judge calls,
and the service ended unsteered.

The scorecard has refusals of its own: it will not start while the GPU lock is held, it
fails when the subject or the judge is down, and a judge verdict that does not parse, or
that the budget cut off, is reported by name rather than scored.

## What can go wrong

| Symptom | Cause | Remedy |
|---------|-------|--------|
| `steer-up` refuses the vector | `VECTOR` is outside `/mnt/ai-data/interp/vectors`, or `vector.safetensors` is missing | Export there with `steer-extract --out` |
| The harness refuses the vector as trained on a different model | The capture came from other weights | Serve the weights the vector was extracted from, or re-extract |
| The scorecard refuses to start | The GPU lock is held by a harvest or a training run | Wait for it, or stop it |
| A verdict is reported as cut off | The judge budget was too small for the reply | Raise `--judge-max-tokens` |
