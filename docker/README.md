# docker

Build contexts and entrypoints, one directory per service. The orchestration lives one level
up, in the compose files at the repository root — those are the source of truth for ports,
volumes, and resource limits, and a document that disagrees with them is the document that is
wrong.

## Layout

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [agents/](agents/README.md) | The build context for the contained agent session: the agent image, its egress proxy, and the OpenCode configuration the image copies in. |
| [interp/](interp/README.md) | The build context for the interpretability workbench: the image with the interp stack on PyTorch cu130, and the entrypoint that bootstraps the storage tree before JupyterLab or the interp CLI runs. |
| [llama-cpu/](llama-cpu/README.md) | The build context for `llama-cpu`, the low-latency CPU inference service: the Dockerfile, the entrypoint it runs, and the huge-page interceptor library it compiles in. |
| [llama-gpu/](llama-gpu/README.md) | The build context for `llama-gpu`, the GPU llama.cpp inference service: the multi-stage Dockerfile that compiles llama.cpp with CUDA for the RTX 5090, and the entrypoint that starts `llama-server` from the environment. |
| [rust-harness/](rust-harness/README.md) | The build context for the rust harness image: a reproducible build of the `harness` binary from the pinned toolchain and the locked dependency graph, and the entrypoint the serving service uses to assemble its arguments from the environment. |
| [training/](training/README.md) | The build context for the training tier: the GPU image the tier trains in, the CPU-only image that merges an adapter into the HF checkpoint the harness serves, and the entrypoint the training image runs. |
| [vllm-gpu/](vllm-gpu/README.md) | The build context for `vllm-gpu`, the high-throughput batch serving service: the Dockerfile that installs the official vLLM cu130 wheel into a hardened CUDA runtime image, and the entrypoint that starts the server from the environment. |

<!-- inventory:dirs:end -->

Each directory's README lists its Dockerfiles and entrypoints and names the compose file
that builds from it.

## Traps

**Not every stack starts with `make up`.** The rust-harness and agents stacks use separate
compose files and are deliberately excluded — they are interactive tools, not services.
`make clean` is scoped and leaves the agent sandbox alone; `make prune` is host-wide and
destructive.

**Only one GPU consumer runs at a time.** `llama-gpu` and `vllm-gpu` contend for the same
card, and the training and interp stacks each carry their own compose file so they cannot
be started alongside inference by accident.

**The agent sandbox image is coupled to the Rust workspace.** See
[agents/](agents/README.md) for what forces a rebuild and why its egress is an allowlist.

## Entry points

```bash
make up                  # the inference services
make train-build         # training image
make interp-up           # interpretability workbench
make agent-sandbox-build # rebuild the agent sandbox
```

## Where the real docs live

[../docs/sandbox/](../docs/sandbox/) documents the compose topology and one overview per
service, including [../docs/sandbox/agent-sandbox.md](../docs/sandbox/agent-sandbox.md) for
the contained agent environment.
