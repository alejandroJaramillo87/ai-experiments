# Sandbox

How the inference services are contained on this workstation: what each container is
allowed to touch, and why. One page states the architecture and the threat model; the
rest walk one Dockerfile or compose file each, line by line.

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [agent-sandbox.md](agent-sandbox.md) | Guide for running coding agents (OpenCode and Claude Code) safely on the AMD Ryzen 9950X + RTX 5090 workstation. |
| [docker_compose_overview.md](docker_compose_overview.md) | Explanation of the multi-service AI inference orchestration implemented in docker-compose.inference.yaml, detailing security sandboxing, resource isolation, and defense-in-depth strategies for the AMD Ryzen 9950X + RTX 5090 AI experiments. |
| [docker_llama_cpu_overview.md](docker_llama_cpu_overview.md) | Explanation of the AMD Zen 5 optimized Dockerfile for CPU-based AI model inference on the AMD Ryzen 9950X workstation, detailing the multi-stage build process, zen5_optimizer integration, and AVX-512 performance optimizations. |
| [docker_llama_gpu_overview.md](docker_llama_gpu_overview.md) | Explanation of the RTX 5090 Blackwell optimized Dockerfile for GPU-accelerated AI model inference on the AI workstation, detailing the multi-stage build process, CUDA 13.0.2 integration, and Blackwell architecture compilation. |
| [docker_training_overview.md](docker_training_overview.md) | Explanation of the Phase 0 training container (`docker/training/Dockerfile.training`), the reproducible GPU environment where the training tier runs. |
| [docker_vllm_gpu_overview.md](docker_vllm_gpu_overview.md) | Explanation of the RTX 5090 Blackwell optimized vLLM Dockerfile for GPU-accelerated model inference on the AI workstation, detailing the single-stage image build, the vLLM 0.25.0 official cu130 wheel install, and Blackwell architecture settings. |
| [security-architecture.md](security-architecture.md) | How the inference services are contained on this workstation: what each container is allowed to touch, and why. |

<!-- inventory:pages:end -->

Where a page disagrees with the Dockerfile or compose file it describes, the file is
correct and the page is the bug. The coding agents are contained on the same principles
but a different profile, in `agent-sandbox.md`.

## Related documentation

- [../../docker/README.md](../../docker/README.md) — the images themselves
- [../os/README.md](../os/README.md) — the Docker installation the containers run on
- [../agents/README.md](../agents/README.md) — the agent sessions the agent sandbox hosts
