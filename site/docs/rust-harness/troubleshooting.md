---
title: Troubleshooting
---

# Troubleshooting

Failures a user meets, in the order they tend to appear. Each entry quotes the message as the
binary prints it. `harness --status` reproduces most of them without loading the model and is
the first thing to run when a command fails before generating.

## Usage errors exit 2 and print the usage line

```text
[ERROR] Unknown argument: --modle
usage: harness --model <checkpoint-dir> ...
```

Every binary treats an unrecognized flag as an error rather than as task text, so a misspelling
stops the run. A required flag left out is reported the same way, naming every missing flag.
Task text that begins with a dash goes after a bare `--`.

## No checkpoint at the path

```text
no checkpoint at /path: expected a directory holding config.json and model.safetensors, or a path to the weights file itself
```

`--model` takes the checkpoint directory. A path to `model.safetensors` itself also resolves,
to its directory. A single GGUF file does not: the harness reads Hugging Face checkpoints only.

## The checkpoint is multimodal

The released Gemma 4 E4B carries audio and vision towers beside the text stack. The backend
serves text and refuses the checkpoint unless `--text-only` is given, so that nobody runs a
model text-blind without having said so. `checkpoint-info` reports the towers it found; add
`--text-only` to see the verdict the harness would give with the flag.

## The tokenizer is not found

The harness looks for the tokenizer in this order: the `--tokenizer` path if given, then
`<checkpoint>.tokenizer.json` beside the checkpoint directory, then `tokenizer.json` inside it.
A published checkpoint directory has the third. The committed synthetic checkpoints have the
second. If neither exists, pass `--tokenizer` explicitly.

## Probe width does not match the model

```text
[ERROR] probe geometry: probe width 4 does not match model embedding dimension 16
        probe layer 2 out of range for a 2-layer model
```

A probe is fitted to one checkpoint's residual width and one layer index. This message means
the probe was trained on a different model. The committed fixture probe under
`rust-harness/fixtures/` is a format fixture for the exporter's tests and matches no runnable
checkpoint; produce a probe from a capture of the served model, per
[Producing a probe and a control vector](../formats/producing-a-probe-and-control-vector.md).

## Trained on a different model

```text
[ERROR] probe identity: trained on a different model: artifact model_hash 0123456789abcdef, served model sha256 0df22e...
```

The probe sidecar and the control-vector manifest each record the sha256 of the
`model.safetensors` they were made from, and the harness compares it against the served
weights. `--status --deep` reads the whole checkpoint to compute the digest, which is the one
slow part of a preflight. A mismatch stops the run with the command to re-check. A manifest
recording `unknown` skips the check.

## Control vector loads without a manifest

A control vector whose `manifest.json` is missing still loads, with a warning, because the
manifest carries provenance and the identity check rather than anything the forward pass
needs. The probe's `.json` sidecar is the opposite case: it carries the layer, site, and
threshold, and a probe without it does not load.

## Run records are not appearing

The default log root is `/mnt/ai-data/logs`. If it does not exist or is not writable, the
harness records nothing and says nothing, by contract. `harness --status` prints the verdict as
its last line. Set `OBS_LOG_DIR` to a writable directory.

## The server is unreachable from outside a container

`--serve` binds `127.0.0.1` by default. Inside a container the published port forwards to the
container's external interface, where nothing listens. Pass `--serve-addr 0.0.0.0` inside the
container; the host side still decides what is exposed.

## A served probe is scoring nothing

`--probe` under `--serve` without `--interp-log` computes scores and discards them, and warns
so at startup: a served run has no end at which to write `probe-flags.jsonl`. Give
`--interp-log <dir>` and read the per-request records there.

## The flags `--top-p` and `--top-k` have no effect

At `--temperature 0` decoding is greedy and both cutoffs are inert. The harness warns rather
than failing. `--top-k 0` is rejected rather than read as disabled.

## `--deep` or `--trace-level` is refused

Both require another flag: `--deep` requires `--status`, and `--trace-level` requires
`--trace-chrome`. `--control-vector-scale` and `--control-vector-layer-range` require
`--control-vector`, and `--serve-addr` requires `--serve`. The usage error names the
requirement.

## The ladder measured nothing

```text
[ERROR] Ladder: no reference artifacts under <dir> -- nothing was measured
```

The ladder compares against artifacts generated from `transformers`, which are not committed
for the real checkpoint. Generating them needs the checkpoint and a torch environment; see
[Correctness ladder](guides/ladder.md). The exit code is 1 in this case deliberately, so a
climb that measured nothing cannot pass.
