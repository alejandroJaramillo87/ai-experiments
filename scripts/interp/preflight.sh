#!/bin/bash
set -euo pipefail

# Interpretability concurrency preflight (interp Phase 0 I1)
#
# Refuses to start the workbench while a GPU inference service is running: one GPU consumer
# at a time, per AGENTS.md. llama-cpu is permitted (dedicated cores, no GPU). Shared by
# `make interp-up` and `make interp-shell`; the check lives here, not in the container, so
# refusal happens before anything starts.
#
# Exit codes: 0 clear to start, 1 a GPU service is running.

RED='\033[0;31m' GREEN='\033[0;32m' NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' NC=''
fi

# GPU consumers in docker-compose.inference.yaml (service names per the compose file,
# the source of truth: llama-gpu 8002, vllm-gpu 8003, comfyui 8188)
# Not named COMPOSE_FILE: that is a docker compose variable, and exporting it would
# change behavior rather than just naming the file for the -f flags below.
INFER_COMPOSE=docker-compose.inference.yaml
GPU_SERVICES="llama-gpu vllm-gpu comfyui"

# Separate "compose reported nothing running" from "compose failed". Discarding stderr and
# swallowing the status with `|| true` makes the two indistinguishable, and an empty service
# list reads as "clear to start" -- so a broken compose invocation would turn this guard from
# fail-closed into fail-open, permitting a second GPU consumer exactly when the check is
# least able to see one. Capture stderr as the diagnostic and refuse on a non-zero status.
if ! running=$(docker compose -f "$INFER_COMPOSE" ps --status running --format '{{.Service}}' 2>&1); then
    printf "${RED}[ERROR]${NC} %s\n" "Cannot determine which services are running"
    echo "  Solution: fix the failing docker compose invocation, then re-run"
    echo "  Detail: $running"
    exit 1
fi

for svc in $GPU_SERVICES; do
    if grep -qx "$svc" <<< "$running"; then
        printf "${RED}[ERROR]${NC} %s\n" "GPU service running: $svc (one GPU consumer at a time)"
        echo "  Solution: stop it first"
        echo "  Command: docker compose -f $INFER_COMPOSE stop $svc  (or 'make down' for everything)"
        exit 1
    fi
done

printf "${GREEN}[OK]${NC} %s\n" "Preflight: no GPU inference service running (llama-cpu is permitted)"
exit 0
