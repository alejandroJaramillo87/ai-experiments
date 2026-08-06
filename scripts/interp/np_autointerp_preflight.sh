#!/bin/bash
set -euo pipefail

# Serving-dependent concurrency preflight (interp Phase 2 PS4)
#
# `make np-autointerp-up` in its default (CPU) mode routes explanations to the vllm-gpu
# OpenAI-compatible endpoint, so the required GPU state is the inverse of
# scripts/interp/preflight.sh: vllm-gpu must be UP and healthy, and every other GPU
# consumer — llama-gpu, comfyui, and the Neuronpedia GPU services (inference, graph) —
# must be down. One GPU consumer at a time, per AGENTS.md; llama-cpu is permitted.
#
# Exit codes: 0 clear to start, 1 vllm-gpu unhealthy or another GPU consumer is running.

RED='\033[0;31m' GREEN='\033[0;32m' NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' NC=''
fi
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }

# Not named COMPOSE_FILE: that is a docker compose variable, and exporting it would
# change behavior rather than just naming the file for the -f flags below.
INFER_COMPOSE=docker-compose.inference.yaml

# vllm-gpu health endpoint, loopback host port per docker-compose.inference.yaml
if ! curl -sf --max-time 5 http://127.0.0.1:8003/health >/dev/null; then
    print_error "vllm-gpu is not healthy on 127.0.0.1:8003 (autointerp explains through it)"
    echo "  Solution: start the serving endpoint first"
    echo "  Command: docker compose -f $INFER_COMPOSE up -d vllm-gpu   (then wait for 'make status' to show OK)"
    exit 1
fi
print_ok "vllm-gpu healthy on 127.0.0.1:8003"

# Other GPU consumers in the inference stack must be down (vllm-gpu holds the GPU)
# A failed `docker compose ps` and an empty result are the same string once stderr is
# discarded, and empty means "no other GPU consumer" -- so swallowing the status would let
# this guard pass precisely when it cannot see the stack. Refuse instead. See the same
# reasoning in scripts/interp/preflight.sh.
if ! running=$(docker compose -f "$INFER_COMPOSE" ps --status running --format '{{.Service}}' 2>&1); then
    print_error "Cannot determine which services are running"
    echo "  Solution: fix the failing docker compose invocation, then re-run"
    echo "  Detail: $running"
    exit 1
fi
for svc in llama-gpu comfyui; do
    if grep -qx "$svc" <<< "$running"; then
        print_error "GPU service running: $svc (vllm-gpu is the only permitted GPU consumer here)"
        echo "  Solution: stop it first"
        echo "  Command: docker compose -f $INFER_COMPOSE stop $svc"
        exit 1
    fi
done

# Neuronpedia GPU services must be down (PS3 inference instance, PS5 graph server)
np_gpu=$(docker ps --filter label=com.docker.compose.project=neuronpedia \
    --format '{{.Label "com.docker.compose.service"}}' 2>/dev/null | grep -Ex 'inference|graph' || true)
if [[ -n "$np_gpu" ]]; then
    print_error "Neuronpedia GPU service running: $(echo "$np_gpu" | tr '\n' ' ')"
    echo "  Solution: stop the platform GPU services, keep webapp + postgres"
    echo "  Command: make np-down && make np-up"
    exit 1
fi

print_ok "Preflight: serving-dependent mode clear (vllm-gpu up, no other GPU consumer)"
exit 0
