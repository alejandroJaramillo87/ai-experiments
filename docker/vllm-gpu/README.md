# docker/vllm-gpu

The build context for `vllm-gpu`, the high-throughput batch serving service: the Dockerfile
that installs the official vLLM cu130 wheel into a hardened CUDA runtime image, and the
entrypoint that starts the server from the environment. The service itself (port, GPU
reservation, volumes, environment) is defined in
[../../docker-compose.inference.yaml](../../docker-compose.inference.yaml), which is the
source of truth; this directory is what that definition builds from.

## Layout

<!-- inventory:build:start -->

| File | Purpose |
|------|---------|
| `Dockerfile.vllm-gpu` | The vllm-gpu inference image: the official vLLM cu130 wheel on the RTX 5090 (Blackwell). |
| `entrypoint.sh` | The vllm-gpu service's entrypoint: starts the vLLM server from the environment, with optional nsys profiling. |

<!-- inventory:build:end -->

## Traps

**One GPU consumer at a time.** `vllm-gpu` and `llama-gpu` contend for the same card; start
one or the other.

**The wheel is the build.** Nothing is compiled from source: the pinned vLLM wheel ships the
sm_120 kernels, so a version bump is a pin change in the Dockerfile and a re-run of the
acceptance check, never a functional change in the same step.

## Consumers

- `make up` starts the service with the rest of the inference stack.
- [../../docs/sandbox/docker_vllm_gpu_overview.md](../../docs/sandbox/docker_vllm_gpu_overview.md)
  explains the image in detail.
