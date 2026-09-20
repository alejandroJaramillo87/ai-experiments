# docker/llama-gpu

The build context for `llama-gpu`, the GPU llama.cpp inference service: the multi-stage
Dockerfile that compiles llama.cpp with CUDA for the RTX 5090, and the entrypoint that
starts `llama-server` from the environment. The service itself (port, GPU reservation,
volumes, environment) is defined in
[../../docker-compose.inference.yaml](../../docker-compose.inference.yaml), which is the
source of truth; this directory is what that definition builds from.

## Layout

<!-- inventory:build:start -->

| File | Purpose |
|------|---------|
| `Dockerfile.llama-gpu` | The llama-gpu inference image: llama.cpp compiled with CUDA for the RTX 5090 (Blackwell). |
| `entrypoint.sh` | The llama-gpu service's entrypoint: starts llama-server from the environment, with optional nsys profiling. |

<!-- inventory:build:end -->

## Traps

**One GPU consumer at a time.** `llama-gpu` and `vllm-gpu` contend for the same card, and
the training and interp stacks each carry their own compose file so they cannot be started
alongside inference by accident.

## Consumers

- `make up` starts the service with the rest of the inference stack.
- [../../docs/sandbox/docker_llama_gpu_overview.md](../../docs/sandbox/docker_llama_gpu_overview.md)
  explains the build stages and the Blackwell settings in detail.
