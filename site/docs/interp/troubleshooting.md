---
title: Troubleshooting
---

# Troubleshooting

Every failure a verb reports carries a `[ERROR]` line, and where a remedy exists a
`Solution:` line and a `Command:` line that runs as printed. The cases below are the ones
the tier's own checks produce; each names the check and what to change.

## The preflight refuses to start

```text
[ERROR] GPU service running: llama-gpu (one GPU consumer at a time)
  Solution: stop it first
  Command: docker compose -f docker-compose.inference.yaml stop llama-gpu  (or 'make down' for everything)
```

A GPU inference service is up. Stop it, or wait for the job that needs it. A refusal that
reads `Cannot determine which services are running` is a failing compose invocation, and the
detail under it is compose's own error; the preflight refuses rather than assuming nothing is
running.

## The workbench refuses to start without a token

`[ERROR] JUPYTER_TOKEN is not set` comes from starting the container through `docker compose`
directly. `make interp-up` generates the token per launch; use it.

## The workbench image is stale

The verbs run on the image that exists. After a change to `pyproject.toml`, the lockfile, or
the Dockerfile, run `uv lock` and then `make interp-build`; nothing rebuilds automatically,
and the build fails on a stale lock.

## The smoke test fails on CUDA

`torch: CUDA not available` means the container has no card, and a compute capability below
12.0 means the build targets a newer architecture than the card reports. The anti-downgrade
step fails when the installed torch differs from the lockfile's pin, which is a dependency
that pulled a different torch; the fix is in the lockfile, not the image.

## A host path is refused inside the workbench

`probe-capture --model` takes the container spelling, `/app/models/...`, and refuses a
`/mnt/ai-data/models` path with that correction rather than resolving it against the Hub.
See [Environment](getting-started/environment.md) for the mount table.

## The capture warns that the model hash is unknown

`probe-capture` prints a `[WARN] model_hash unknown` block when `--model` names something
with no local `model.safetensors`, or when `--model-hash unknown` is passed. A probe from
that capture loads into any model of the right width with the harness's identity check
skipped. Name a local checkpoint directory, or pass `--model-hash` the sha256 of its
weights, before training a probe meant for serving.

## The harness refuses a probe or a vector

```text
[ERROR] Task failed: control vector was trained on a different model: control vector model_hash fedcba98..., served model sha256 2db5482b...
  Solution: serve the weights the control vector was fitted on, or re-export the control vector against these
```

The artifact's recorded model hash does not match the served weights. Serve the weights the
capture came from, or re-capture and re-train against these. A probe with the wrong width or
layer is refused the same way. The harness's own [Troubleshooting](../rust-harness/troubleshooting.md)
page covers the check.

## `steer-up` refuses the vector

The vector must live under `/mnt/ai-data/interp/vectors` and hold `vector.safetensors`; the
serving harness mounts only that tree. Export there with `steer-extract --out`.

## The scorecard refuses to run

The scorecard refuses while the GPU lock is held by a harvest or a training run, and its
subject and judge services must both be up. `steer-scorecard --subject-url` and
`--judge-url` name them; the defaults are the serving harness and `llama-cpu`.

## The harvest refuses on disk

```text
[ERROR] Harvest failed: disk preflight refused: projected capture is <n>GB, free space on <root> is <m>GB, floor is <f>GB. Free space, reduce harvest.total_tokens or hook_layers, or run make interp-gc
```

The projection prices the written figure, buffer rounding included, and the check runs
before the lock and the model load. The three remedies are in the message; `gc` reclaims
validated captures. See [Harvesting activations](guides/harvest.md) for sizing.

## The hook gate refuses a layer

`harvest.hook_layers: [99] out of range for <model>, which has <n> layers` is decided from
`config.json` before any weight is read. A hook that yields a width other than the hidden
size is refused the same way; both mean the recipe names a layer the checkpoint does not
have where it expects it.

## `sae-train` refuses the capture

`sae.hook '<hook>' not in capture hooks [...]` names the hooks the capture does hold; a
missing manifest or identity fields that differ from the recipe's expectations are refused
the same way, before SAELens is imported. Point the recipe at the capture it describes.

## A recipe fails to load

An override naming a key the schema does not have, or a recipe whose expected model or
corpus does not match the capture manifest it points at, fails before the GPU is touched,
with the key or the mismatch named. See [Recipes](reference/recipes.md).

## The harvest died mid-run

Re-invoke the same recipe. The harvest resumes from the last completed chunk, skipping the
banked tokens without running the model. Do not delete the capture directory. Two refusals
are not resumable and say so: a crash inside assembly, whose message names the two
directories to delete before starting over, and a recipe changed between runs, whose message
names the keys that moved. See [Harvesting activations](guides/harvest.md).

## Run records are not appearing

The root is `OBS_LOG_DIR`, default `/mnt/ai-data/logs`, and the verbs write under its
`interp` subdirectory. A usage error exits before a run id exists and writes nothing. Off the
workstation, set the variable to a writable path; see
[Environment variables](reference/environment-variables.md).
