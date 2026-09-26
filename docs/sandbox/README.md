# Sandbox

How every container on this workstation is contained: what each is allowed to touch, and
why, across the inference services, the training and export images, the interpretability
workbench, the rust harness, and the agent sandbox. Two pages state an architecture and a
threat model, one for the inference stack and one for the agents; the rest walk one
Dockerfile or compose file each, line by line.

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [agent-sandbox.md](agent-sandbox.md) | Guide for running coding agents (OpenCode and Claude Code) safely on the AMD Ryzen 9950X + RTX 5090 workstation. |
| [docker_agents_overview.md](docker_agents_overview.md) | Explanation of the contained agent session as its files build it: the agent image (`docker/agents/Dockerfile.agents`), the egress proxy (`docker/agents/proxy/Dockerfile.proxy` and `squid.conf`), and the two services, two networks, and limits in `docker-compose.agents.yaml`. |
| [docker_compose_overview.md](docker_compose_overview.md) | Explanation of the multi-service AI inference orchestration implemented in docker-compose.inference.yaml, detailing security sandboxing, resource isolation, and defense-in-depth strategies for the AMD Ryzen 9950X + RTX 5090 AI experiments. |
| [docker_interp_overview.md](docker_interp_overview.md) | Explanation of the interpretability workbench image (`docker/interp/Dockerfile.interp`) and the `interp` service `docker-compose.interp.yaml` runs from it: PyTorch cu130 on sm_120 with the interp stack, JupyterLab as the default command, and the interp CLI for harvests, probes, SAEs, and steering. |
| [docker_llama_cpu_overview.md](docker_llama_cpu_overview.md) | Explanation of the AMD Zen 5 optimized Dockerfile for CPU-based AI model inference on the AMD Ryzen 9950X workstation, detailing the multi-stage build process, zen5_optimizer integration, and AVX-512 performance optimizations. |
| [docker_llama_gpu_overview.md](docker_llama_gpu_overview.md) | Explanation of the RTX 5090 Blackwell optimized Dockerfile for GPU-accelerated AI model inference on the AI workstation, detailing the multi-stage build process, CUDA 13.0.2 integration, and Blackwell architecture compilation. |
| [docker_rust_harness_overview.md](docker_rust_harness_overview.md) | Explanation of the rust harness image (`docker/rust-harness/Dockerfile.rust-harness`) and the two services `docker-compose.rust-harness.yaml` runs from it: the interactive `harness` tool and the `harness-serve` daemon that exposes the OpenAI-shaped endpoint interp calls. |
| [docker_training_overview.md](docker_training_overview.md) | Explanation of the Phase 0 training container (`docker/training/Dockerfile.training`), the reproducible GPU environment where the training tier runs, and of the CPU-only export image (`docker/training/Dockerfile.export`) beside it. |
| [docker_vllm_gpu_overview.md](docker_vllm_gpu_overview.md) | Explanation of the RTX 5090 Blackwell optimized vLLM Dockerfile for GPU-accelerated model inference on the AI workstation, detailing the single-stage image build, the vLLM 0.30.0 official cu130 wheel install, and Blackwell architecture settings. |
| [security-architecture.md](security-architecture.md) | How the inference services are contained on this workstation: what each container is allowed to touch, and why. |

<!-- inventory:pages:end -->

Where a page disagrees with the Dockerfile or compose file it describes, the file is
correct and the page is the bug. Every `Dockerfile*` under `docker/` and every
`docker-compose.*.yaml` at the root is walked by exactly one page here. The coding agents
are contained on the same principles but a different profile: `agent-sandbox.md` states
that model and `docker_agents_overview.md` walks its files.

## Related documentation

- [../../docker/README.md](../../docker/README.md) — the images themselves
- [../os/README.md](../os/README.md) — the Docker installation the containers run on
- [../agents/README.md](../agents/README.md) — the agent sessions the agent sandbox hosts
