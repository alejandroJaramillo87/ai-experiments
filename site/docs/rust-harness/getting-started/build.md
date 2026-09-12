---
title: Build
---

# Build

The workspace builds with cargo alone. It has no CUDA dependency, no C++ toolchain
requirement, and no build step that reaches the network after the crates are fetched.

## Requirements

- A Rust toolchain installed through rustup. The workspace pins its toolchain in
  `rust-harness/rust-toolchain.toml`; rustup installs that version on the first cargo command
  in the directory, so no version needs choosing by hand.
- Roughly one gigabyte of disk for the build directory.
- A Gemma 4 checkpoint on disk, and the RAM to hold it. The backend keeps the weights at their
  stored width, bf16 for the released checkpoints, and adds a KV cache sized to the context
  budget. Figures from the harness's own status header:

| Checkpoint | Weights resident | Peak RSS after load | KV cache at the full 131072-token context |
|------------|------------------|---------------------|-------------------------------------------|
| `gemma-4-E2B`, `gemma-4-E2B-it` | 8.62 GiB | 9.51 GiB | 1.52 GiB |
| `gemma-4-E4B` | 13.90 GiB | 14.03 GiB | 4.05 GiB |

Python is not needed to build or run any binary. It is needed for the probe and control-vector
export path in
[Producing a probe and a control vector](../../formats/producing-a-probe-and-control-vector.md),
for the ladder's reference generators, and for the recording audit under
[Flight recorder](../observability/flight-recorder.md).

## Build the workspace

From the repository root:

```bash
cd rust-harness
cargo build --release
```

The first build compiles the dependency tree and takes a few minutes. The binaries land in the
`target/release` directory of the workspace: `harness` and the seven utilities `checkpoint-info`,
`tiny-model`, `activation-capture`, `attention-capture`, `ladder`, `layer-diff`, and
`long-context-golden`.

Build with `--release` for any run against a real checkpoint. A debug build of the forward pass
is several times slower, and every duration this site quotes is from a release build. After
pulling a change that touches `runtime/`, rebuild before running: `cargo test` rebuilds only the
test profile and leaves `target/release/harness` stale.

## Which real checkpoint

The released Gemma 4 checkpoints come in two shapes, and which one a workflow needs is decided
by whether the workflow reads the reply.

- **A base checkpoint** (`gemma-4-E2B`, `gemma-4-E4B`) continues documents. Asked a question,
  it produces a fluent non-answer, which reads as a defect and is not one. It serves every
  workflow that measures the forward pass rather than the reply: the correctness ladder,
  `layer-diff`, activation and attention capture, the flight recorder, and the trace. The smoke
  prompt for a base model is a continuation, `"The capital of France is"`, which the base E4B
  completes with `Paris`.
- **An instruction-tuned checkpoint** (`gemma-4-E2B-it`, `gemma-4-E4B-it`) answers. The agent
  loop, a multi-turn chat, probe monitoring in chat mode, and steering judged by its reply all
  need one; against a base checkpoint none of them can be read.

This site uses `gemma-4-E2B-it` for every example that reads a reply and `gemma-4-E4B` for the
ladder and `layer-diff`, because those are the checkpoints the outputs shown were recorded on.
Every released checkpoint is multimodal, carrying audio and vision towers beside the text
stack, so every command that loads one carries `--text-only`.

## Set the two variables every example uses

Every command on this site calls the binaries by path and names a checkpoint, and a run writes
its session files into the current directory. Setting these once, still inside `rust-harness`,
keeps every later block pasteable and keeps the session files out of the repository:

```bash
export BIN="$PWD/target/release"
export MODEL=/path/to/gemma-4-E2B-it
mkdir -p /tmp/harness-demo
cd /tmp/harness-demo
```

`BIN` is the directory holding the binaries. `MODEL` is the checkpoint directory: it holds
`config.json`, `model.safetensors`, and `tokenizer.json`, which is how a published checkpoint
is laid out. A path to `model.safetensors` itself also resolves, to its directory.

Start every new experiment from an empty directory. The harness continues whatever
`session.jsonl` and `session.kv` sit in the directory it starts in, and a run that inherits
another run's transcript answers in that transcript's context;
[Troubleshooting](../troubleshooting.md) describes what that looks like.

## Check the build

```bash
"$BIN/harness" --version
"$BIN/harness" --help
```

The first prints the version on one line. The second prints every flag, grouped, and the list
of the other binaries in the workspace. Both exit 0 and load nothing. The next page runs the
model.

## Running in the container

The repository also builds the harness into a container with a pinned toolchain and
`cargo build --locked`, and serves it from there: `make harness-build`, then
`make harness-serve HARNESS_MODEL=/app/models/hf/gemma-4-E2B-it` from the repository root. A
container build and a host build produce the same tokens for the same seed and checkpoint.
The container path is owned by the repository's compose file and is not covered further here.
