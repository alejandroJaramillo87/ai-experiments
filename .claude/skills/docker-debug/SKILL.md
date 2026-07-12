---
name: docker-debug
description: Container debugging runbook for this inference stack — per-service healthcheck endpoints, common CUDA/driver failure signatures, log locations, and the safe restart procedure. Use when a service is unhealthy or failing to start.
---
# Docker debugging runbook

For the llama.cpp / vLLM / open-webui stack defined in `docker-compose.yaml`.

## Services and healthchecks

| Service | External port | Internal healthcheck |
|---------|---------------|----------------------|
| llama-cpu | 8001 | `curl -f http://localhost:8001/v1/health` |
| llama-gpu | 8004 | `curl -f http://localhost:8002/v1/health` (internal 8002) |
| vllm-gpu | 8005 | `curl -f http://localhost:8003/health` (internal 8003) |
| open-webui | 3000 | container `:main`, no explicit healthcheck |
| comfyui-flux | (flux) | `curl -f http://localhost:8188/system_stats` |

Start with `make status` for the compose-level health summary and `make health` for the
detailed per-service check.

## Common CUDA / driver failure signatures

- `CUDA error: no kernel image is available for execution` — image built for the wrong
  compute capability; the RTX 5090 is Blackwell (sm_120). Rebuild the GPU image.
- `CUDA driver version is insufficient for CUDA runtime version` — host NVIDIA driver older
  than the container's CUDA toolkit; update the host driver.
- `could not select device driver "nvidia"` — NVIDIA Container Toolkit not wired into
  Docker; check `nvidia-smi` on the host and the runtime config.
- vLLM OOM at load — model + KV cache exceeds 32GB VRAM; reduce context length,
  `--gpu-memory-utilization`, or quantization.

## Log locations

- Follow logs: `make logs`, or per-service `make logs-gpu` / `make logs-cpu` /
  `make logs-vllm` / `make logs-ui`.
- Raw: `docker logs <container>` (`llama-cpu`, `llama-gpu`, `vllm-gpu`, `open-webui`).
- Compose `*.log` files are git-ignored.

## Safe restart procedure

1. Confirm which service is unhealthy (`make status`); do not restart the whole stack for a
   single failure.
2. Restart just that service: `docker compose restart <service>`.
3. If a config or image changed, `make down` then `make up` (or `make rebuild` for a
   from-scratch image build). `make reset` is the full stop-clean-rebuild cycle.
4. Never `docker system prune` / `docker volume rm` to "fix" a hang — those are guarded
   `ask` actions and destroy state; diagnose from logs first.
