---
title: First run
---

# First run

This page inspects the checkpoint, asks the harness what a run would fail on, and then runs
one task. It assumes `BIN` and `MODEL` from [Build](build.md) and a scratch working directory.

## Inspect the checkpoint

```bash
"$BIN/checkpoint-info" "$MODEL"
```

`checkpoint-info` prints the checkpoint's `config.json`, its tensor table, the attention
schedule, the per-layer geometry, and last the CPU backend's verdict:

```text
Attention schedule: sG
  (G = full attention, s = sliding window)

Per-layer geometry (head width, feed-forward width, RoPE base):
  layer 0: SlidingWindow(4), head 8, ffn 32, rope base 10000
  layer 1: Global, head 8, ffn 32, rope base 10000

[OK] CPU backend compatibility: supported
```

Add `--check` to turn an unsupported verdict into exit 1, which is the form for a script. Add
`--text-only` to judge a multimodal checkpoint by its text tower alone.

## Ask before running

```bash
"$BIN/harness" --model "$MODEL" --status
```

`--status` reports every precondition of the run without loading the model and exits 1 if any
would stop it:

```text
[OK] model directory: /home/user/ai-experiments/rust-harness/fixtures/tiny-gemma4
[OK] checkpoint files: config.json and model.safetensors present, header names 28 tensors
[OK] config.json: 2 layers, hidden size 16, context 4096
[OK] backend compatibility: supported
[SKIP] probe: no --probe given
[SKIP] probe geometry: no --probe given
[SKIP] probe identity: no --probe given
[SKIP] control vector: no --control-vector given
[SKIP] control vector identity: no --control-vector given
[SKIP] interp log directory: no --interp-log given
[OK] obs log directory: /mnt/ai-data/logs/harness is writable
```

The last line names where run records go by default. See
[Output locations](output-locations.md) for changing it.

## Run a task

```bash
"$BIN/harness" --model "$MODEL" --max-tokens 8 --temperature 0 "Say hello"
```

The positional argument is the task. Everything after a bare `--` is task text even if it
starts with a dash. The harness prints a status header to stderr, streams the answer to
stdout, and prints a closing status:

```text
[INFO] Model: /home/user/ai-experiments/rust-harness/fixtures/tiny-gemma4 hash=efd9e99c76ce8b7b layers=2
[INFO] Weights: 0.00 GiB resident at f32
[INFO] Peak RSS after load: 0.01 GiB
[INFO] KV cache: 4096 tokens, 0.00 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 81 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=4
[INFO] Sampling: seed=0 temperature=0 max_tokens=8
[INFO] Build: 0.1.0 git unknown
[INFO] Model load: 0.01 s
[INFO] Prefill: 1170 tokens in 0.85 s (1372 tok/s)
[INFO] Session snapshot: 1176 cached tokens at /tmp/harness-demo/session.kv
```

Three things in that output are worth reading on a first run:

- **Prefill is over a thousand tokens for a five-word task.** A task run is an agent session:
  the prompt carries the tool contract and the system turn before the task. The chat REPL and
  the server carry a far shorter prompt.
- **The answer is noise.** The synthetic checkpoint has random weights. A real checkpoint
  answers.
- **A snapshot was written.** `session.kv` in the working directory holds the KV cache, and
  the next run in this directory restores it instead of prefilling from scratch. The
  transcript itself is `session.jsonl` beside it.

## Sampling flags

`--temperature 0` selects greedy decoding, under which `--top-p` and `--top-k` have no effect
and the harness says so. At any other temperature, `--seed` replays a run exactly; the default
seed is 0, so two runs with the same flags produce the same tokens. `--max-tokens` is the
budget per turn. The full list with defaults is on the [harness reference page](../reference/harness.md).
