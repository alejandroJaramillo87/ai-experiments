---
title: First run
---

# First run

This page inspects the checkpoint, asks the harness what a run would fail on, runs one task,
and reads the status header the run printed. It assumes `BIN` and `MODEL` from
[Build](build.md) and an empty working directory. The output blocks were recorded on
`gemma-4-E2B-it` unless a block says otherwise.

## Inspect the checkpoint

`checkpoint-info` prints the checkpoint's `config.json`, its tensor table, the non-text towers
it declines, the architecture summary, and last the CPU backend's verdict. The report is about
2600 lines on a released checkpoint and goes to stdout; the verdict is a status line and goes
to stderr, so capture both to a file and read the tail:

```bash
"$BIN/checkpoint-info" --text-only "$MODEL" > checkpoint-info.txt 2>&1
tail -n 4 checkpoint-info.txt
```

The tail on `gemma-4-E4B`:

```text
  layers 36-40: SlidingWindow(512), head 256, ffn 10240, rope base 10000
  layer 41: Global, head 512, ffn 10240, rope base 1000000

[OK] CPU backend compatibility: supported
```

Without `--text-only` the same report ends in an `[ERROR] CPU backend compatibility:
checkpoint is multimodal` line naming the 114 tensor families of the audio and vision towers,
and still exits 0, because an inspection of a file the backend cannot load is not a failed
command. `--check` turns the verdict into the exit code, which is the form for a script:

| Invocation | Exit | Verdict |
|------------|------|---------|
| `checkpoint-info "$MODEL"` | 0 | multimodal, reported |
| `checkpoint-info --check "$MODEL"` | 1 | multimodal, enforced |
| `checkpoint-info --check --text-only "$MODEL"` | 0 | supported |
| `checkpoint-info --check /absent` | 1 | `[ERROR] Read /absent: no checkpoint at /absent` |

The `=== config.json ===` echo carries three `hidden_size` fields, one each for the vision,
text, and audio blocks; the text block is the one whose `num_hidden_layers` matches the layer
count in the summary.

## Ask before running

```bash
"$BIN/harness" --model "$MODEL" --text-only --status
```

`--status` reports every precondition of the run without loading the model and exits 1 if any
would stop it. On `gemma-4-E4B`:

```text
[OK] model directory: /mnt/ai-data/models/hf/gemma-4-E4B
[OK] checkpoint files: config.json and model.safetensors present, header names 2130 tensors
[OK] config.json: 42 layers, hidden size 2560, context 131072
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

Three properties of the preflight, each measured on the real checkpoint:

- **It is fast.** It reads `config.json` and the safetensors header only, and returns in about
  45 ms against a load measured in seconds. Adding `--deep` makes it hash the whole weights
  file to compare against a probe's or control vector's recorded digest; on the 15 GB E4B file
  that takes about 35 s, against 6 s for `sha256sum`, because the in-tree hasher is scalar.
  `--deep` with no artifact to check warns `--deep: no effect when there is no interp artifact
  to check` rather than pretending to have verified anything.
- **It prints nothing on stdout.** The report is a verdict, not payload:
  `"$BIN/harness" --model "$MODEL" --text-only --status 2>/dev/null | wc -c` prints `0`.
- **A missing checkpoint exits 1, not 2.** One `[ERROR] model directory` line and the rest
  `[SKIP]`. Exit 2 is reserved for a command line the harness could not read.

## Run a task

```bash
"$BIN/harness" --model "$MODEL" --text-only "Reply with the single word: ready"
```

The positional argument is the task. Everything after a bare `--` is task text even if it
starts with a dash. The harness prints a status header to stderr, streams the answer to
stdout, and prints a closing status:

```text
[INFO] Model: /mnt/ai-data/models/hf/gemma-4-E2B-it hash=7ddcbbe752dcfbf6 layers=35
[INFO] Weights: 8.62 GiB resident at bf16
[INFO] Peak RSS after load: 9.51 GiB
[INFO] KV cache: 131072 tokens, 1.52 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=8
[INFO] Sampling: seed=0 temperature=0.8 max_tokens=256
[INFO] Build: 0.1.0 git unknown
[INFO] Model load: 3.69 s
[INFO] Prefill: 252 tokens in 11.40 s (22 tok/s)
ready
[INFO] Session snapshot: 255 cached tokens at /tmp/harness-demo/session.kv
```

Two things in that output are worth reading on a first run. **Prefill is 252 tokens for a
six-word task**: a task run is an agent session, and the prompt carries the tool contract and
the system turn before the task. The chat REPL and the server carry a far shorter prompt. And
**a snapshot was written**: `session.kv` in the working directory holds the KV cache, and the
next run in this directory restores it instead of prefilling from scratch. The transcript
itself is `session.jsonl` beside it.

A base checkpoint run the same way does not print `ready`. It continues the prompt as a
document, fluently and at length, which is correct behavior for a model that was never
instruction-tuned; see [Which real checkpoint](build.md#which-real-checkpoint).

## Reading the status header

Every line of the header is a fact the loader resolved, and the run record carries the same
facts as JSON; see [Run records](../observability/run-records.md).

| Line | What it states |
|------|----------------|
| `Model` | The checkpoint path, the first 16 hex characters of the sha256 of `model.safetensors`, and the layer count. The full hash is what a probe or control vector is checked against |
| `Weights` | Bytes resident after load and the width they are held at, which is the checkpoint's stored width |
| `Peak RSS after load` | Process memory at the end of the load; the gap to `Weights` is load overhead |
| `KV cache` | The context budget in tokens and the cache it allocates. Global layers hold the full budget; sliding layers are capped at their window plus one batch group. Above 2 GiB a `[WARN]` suggests `--context` |
| `Rewind horizon` | How far behind the tip the cache can rewind once a sliding window has filled. A compaction that rewinds further re-prefills; see [Running an agent session](../guides/agent-session.md) |
| `Fold` | The dot-product kernel, the prefill batch width, and the thread width the run uses. `--single-thread` sets `threads=1` |
| `Sampling` | The seed, temperature, and per-turn budget in force. The defaults are seed 0, temperature 0.8, and 256 tokens |
| `Build` | The crate version and the git commit the binary was built from, `unknown` outside a checkout |
| `Model load` | Wall clock of the load: about 3.7 s for E2B and 6 s for E4B from a warm page cache, tens of seconds cold |
| `Prefill` | Tokens the generation prefilled and the rate. On a second turn in the same directory this is the new turn alone, not the transcript |
| `Session snapshot` | Tokens saved to `session.kv` on exit. `Session restored` is its counterpart at the start of the next run |

The header is also how a decode rate is read without a trace. Wall clock less `Model load`
less the `Prefill` time, divided by cached tokens less prefilled tokens, is the decode rate: on
`gemma-4-E4B` at eight threads that comes to about 5 tokens per second, and on `gemma-4-E2B-it`
about 10. One such run is a scale check rather than a benchmark; the repository's benchmark
protocol needs pinned cores and a median over interleaved runs.

## Sampling flags

`--temperature 0` selects greedy decoding, under which `--top-p` and `--top-k` have no effect
and the harness says so. At any other temperature, `--seed` replays a run exactly; the default
seed is 0, so two runs with the same flags produce the same tokens, which is why the output
above reproduces to the token. `--max-tokens` is the budget per turn. The full list with
defaults is on the [harness reference page](../reference/harness.md).
