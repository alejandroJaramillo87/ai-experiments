# docker

Build contexts and entrypoints, one directory per service. The orchestration lives one level
up, in the compose files at the repository root — those are the source of truth for ports,
volumes, and resource limits, and a document that disagrees with them is the document that is
wrong.

## Layout

| Directory | Service |
|-----------|---------|
| `llama-cpu/` | low-latency CPU inference |
| `llama-gpu/` | GPU llama.cpp inference |
| `vllm-gpu/` | high-throughput batch serving |
| `comfyui/` | image and video generation |
| `training/` | the training tier, plus a separate export image |
| `interp/` | the interpretability workbench, and the pinned Neuronpedia overrides |
| `interp-logproxy/` | logprob-capturing proxy in front of a serving container |
| `rust-harness/` | the Rust harness image |
| `agents/` | the agent sandbox and its egress proxy |

## Traps

**Not every stack starts with `make up`.** The rust-harness and agents stacks use separate
compose files and are deliberately excluded — they are interactive tools, not services.
`make clean` is scoped and leaves the agent sandbox alone; `make prune` is host-wide and
destructive.

**Only one GPU consumer runs at a time.** `art-mode` and `llm-mode` exist to swap between
ComfyUI and the GPU LLM services for that reason, and the training and interp stacks each
carry their own compose file so they cannot be started alongside inference by accident.

**Neuronpedia is an external checkout, not built from this tree.** It is pinned by
`docker/interp/neuronpedia-pin.txt` and adjusted through `neuronpedia.override.yaml` and
`neuronpedia.env-deltas`; see [interp/patches/README.md](interp/patches/README.md).

**The agent sandbox image is coupled to the Rust workspace.** `agents/Dockerfile.agents`
bakes in the toolchain pinned by `rust-harness/rust-toolchain.toml` and a prefetched crate
registry, so `make agent-sandbox-build` is required after changing that file or
`rust-harness/Cargo.lock`. Its egress runs through a Squid allowlist
(`agents/proxy/squid.conf`), which is why `uv sync` cannot reach the pytorch index there.

## Entry points

```bash
$ make up                  # the inference services
$ make train-build         # training image
$ make interp-up           # interpretability workbench
$ make agent-sandbox-build # rebuild the agent sandbox
```

## Where the real docs live

[../docs/sandbox/](../docs/sandbox/) documents the compose topology and one overview per
service, including [../docs/sandbox/agent-sandbox.md](../docs/sandbox/agent-sandbox.md) for
the contained agent environment.
