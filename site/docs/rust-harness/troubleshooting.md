---
title: Troubleshooting
---

# Troubleshooting

Failures a user meets, in the order they tend to appear. Each entry quotes the message as the
binary prints it. `harness --status` reproduces most of them without loading the model and is
the first thing to run when a command fails before generating. A failure that stops a run
prints a `Solution:` line and, where one exists, a `Command:` line that runs as printed
beneath the diagnosis.

## Usage errors exit 2 and print the usage line

```text
[ERROR] Unknown argument: --modle
usage: harness --model <checkpoint-dir> ...
```

Every binary treats an unrecognized flag as an error rather than as task text, so a misspelling
stops the run. A required flag left out is reported the same way, naming every missing flag.
Task text that begins with a dash goes after a bare `--`. Two flags that contradict each other,
`--single-thread` beside `--threads`, are refused the same way, naming both.

## No checkpoint at the path

```text
[ERROR] Task failed: read model /nonexistent-checkpoint: no checkpoint at /nonexistent-checkpoint: expected a directory holding config.json and model.safetensors, or a path to the weights file itself
  Solution: point --model at a directory holding config.json and model.safetensors, or at the weights file itself
  Command: ls /nonexistent-checkpoint
```

`--model` takes the checkpoint directory. A path to `model.safetensors` itself also resolves,
to its directory. A single GGUF file does not: the harness reads Hugging Face checkpoints only.
The failure is immediate, before any weight is read, and exits 1.

## The checkpoint is multimodal

```text
[ERROR] CPU backend compatibility: checkpoint is multimodal: it carries 114 tensor families in model.audio_tower., model.embed_audio., model.embed_vision., model.vision_tower., and this backend serves text only. Pass --text-only to load the text stack and decline the rest
```

Every released Gemma 4 checkpoint carries audio and vision towers beside the text stack. The
backend serves text and refuses the checkpoint unless `--text-only` is given, so that nobody
runs a model text-blind without having said so. `checkpoint-info` reports the towers it found;
add `--text-only` to see the verdict the harness would give with the flag.

## The reply continues a conversation nobody started

The harness resumes `session.jsonl` and `session.kv` from the directory it starts in. A run
made in a directory that holds another run's transcript answers the new prompt in the old
conversation's context, and nothing marks it but one line:

```text
[INFO] Session restored: 234 cached tokens from /tmp/harness-demo/session.kv
```

The tell is a `Prefill:` count far above the prompt's length: a six-token prompt that prefills
977 tokens is continuing a transcript. A restored session that diverged from a stale
transcript can also emit garbage, repeated `<unused10>` tokens or fragments of another
document, which looks like a numerics defect and is not one. Start every experiment from an
empty directory, or remove both files.

## Session snapshot discarded

```text
[WARN] Session snapshot /tmp/harness-demo/session.kv discarded: snapshot was written for different model weights (hash e376216f8b693aff, expected dc91ecfbc2392fa9) (starting with an empty cache)
```

The snapshot carries a fingerprint of the weights it was computed under, and a run under a
different checkpoint refuses it and starts with an empty cache. This is the check working, not
a failure; the transcript beside it is still resumed, which is usually not wanted either.

## A real checkpoint gives a fluent non-answer

A base checkpoint (`gemma-4-E2B`, `gemma-4-E4B`) continues documents. Asked
`Reply with the single word: ready`, base E4B echoed the prompt and continued into an invented
web-forum thread; asked to count to twenty, it emitted a Python program. That is correct
behavior for a model that was never instruction-tuned, and a numerically wrong forward pass
looks different: word salad or a repetition loop within a few dozen tokens. The agent loop, a
multi-turn chat, and any comparison judged by its reply need an instruction-tuned checkpoint;
see [Which real checkpoint](getting-started/build.md#which-real-checkpoint). Whether a sampled
reply is the one `transformers` would produce is a question for the recording audit under
[Flight recorder](observability/flight-recorder.md), not for reading.

## The tokenizer is not found

The harness looks for the tokenizer in this order: the `--tokenizer` path if given, then
`<checkpoint>.tokenizer.json` beside the checkpoint directory, then `tokenizer.json` inside it.
A published checkpoint directory has the third. If none exists the directory is incomplete;
pass `--tokenizer` explicitly. A `--tokenizer` path that does not exist is refused rather than
resolved around.

## Probe width does not match the model

```text
[ERROR] probe geometry: probe width 1536 does not match model embedding dimension 2560
```

A probe is fitted to one checkpoint's residual width and one layer index. This message means
the probe was trained on a different model: E2B's residual width is 1536 and E4B's is 2560.
Produce a probe from a capture of the served model, per
[Producing a probe and a control vector](../formats/producing-a-probe-and-control-vector.md).

## Trained on a different model

```text
[ERROR] probe identity: trained on a different model: artifact model_hash 53fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651, served model sha256 43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651
```

The probe sidecar and the control-vector manifest each record the sha256 of the
`model.safetensors` they were made from, and the harness compares it against the served
weights. `--status --deep` reads the whole checkpoint to compute the digest, about 35 s on
E4B, which is the one slow part of a preflight. A mismatch stops the run with the command to
re-check; one changed hex digit is enough. A manifest recording `unknown` skips the check.

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
than failing. `--top-k 0` is rejected rather than read as disabled, and `--top-p` outside
`(0, 1]` is refused naming the range.

## `--deep` or `--trace-level` is refused

Both require another flag: `--deep` requires `--status`, and `--trace-level` requires
`--trace-chrome`. `--control-vector-scale` and `--control-vector-layer-range` require
`--control-vector`, and `--serve-addr` requires `--serve`. The usage error names the
requirement.

## `--context` is above the trained window

`max_position_embeddings` from the checkpoint, 131072 on the released Gemma 4 checkpoints, is
both the default context budget and the maximum. A larger value fails the run with exit 1,
naming the value and the window. The flag caps the budget; it cannot raise it.

## The ladder measured nothing

```text
[ERROR] Ladder: no reference artifacts under <dir> -- nothing was measured
```

The ladder compares against artifacts generated from `transformers`, which are not committed
for the real checkpoint. Generating them needs the checkpoint and a torch environment; see
[Correctness ladder](guides/ladder.md). The exit code is 1 in this case deliberately, so a
climb that measured nothing cannot pass. A climb whose closing line reads anything other than
`0 skipped` has the same problem for the rungs it skipped.

## The loader aborts instead of reporting

A safetensors header that declares a tensor larger than the file drives an allocation the
allocator refuses, and the process dies on `SIGABRT` rather than printing an `[ERROR]` line.
Every other unreadable-checkpoint path reports and exits 1; this one is a known defect in the
loader, recorded in the repository, and the input is a corrupt or truncated weights file.
