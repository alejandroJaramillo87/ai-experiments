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
- For the real checkpoint, enough RAM to hold it at f32. The workspace Makefile records the
  released E4B at roughly 14 GiB resident. The synthetic checkpoints used on this site need a
  few megabytes.

Python is not needed to build or run any binary. It is needed only for the export path in
[Producing a probe and a control vector](../../formats/producing-a-probe-and-control-vector.md).

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

`cargo build` without `--release` also works and is what the workspace's own tests use. A debug
build of the forward pass is several times slower, which matters against the real checkpoint
and not against the synthetic ones.

## Set the two variables every example uses

Every command on this site calls the binaries by path and names a checkpoint, and a run writes
its session files into the current directory. Setting these once, still inside `rust-harness`,
keeps every later block pasteable and keeps the session files out of the repository:

```bash
export BIN="$PWD/target/release"
export MODEL="$PWD/fixtures/tiny-gemma4"
mkdir -p /tmp/harness-demo
cd /tmp/harness-demo
```

`BIN` is the directory holding the binaries. `MODEL` is a committed synthetic checkpoint: two
decoder layers, a hidden size of 16, a 288-token vocabulary, and a 4096-token context, with
its tokenizer beside it as `tiny-gemma4.tokenizer.json`. Its weights are random. To run against
a real checkpoint, point `MODEL` at a directory holding `config.json`, `model.safetensors`, and
`tokenizer.json` instead, and add `--text-only` to every `harness` command, because the released
E4B is multimodal and the backend declines its audio and vision towers only when told to.

## Check the build

```bash
"$BIN/harness" --version
"$BIN/harness" --help
```

The first prints the version on one line. The second prints every flag, grouped, and the list
of the other binaries in the workspace. Both exit 0. The next page runs the model.
