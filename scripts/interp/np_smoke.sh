#!/bin/bash
set -euo pipefail

# Neuronpedia platform smoke check (interp Phase 2 PS1)
#
# Host-side check, run via `make np-smoke` after `make np-up`. Verifies:
#   1. Loopback binding: every listening platform port (3001, 5432, 5002, 5003, 5004) is
#      bound to 127.0.0.1 only. A 0.0.0.0 or [::] binding is a hard failure — the
#      localhost trust model (no sign-in, static shared secrets) depends on it.
#   2. Webapp answers on http://127.0.0.1:3001.
#   3. Postgres accepts connections (pg_isready inside the container).
#   4. When a GPU platform container is up (inference instance, graph server): the
#      container sees the RTX 5090 and its torch build supports sm_120 (upstream images
#      build on CUDA 12.1 bases; fine under the CUDA 13 host driver only if the torch
#      wheel carries sm_120 kernels).
#
# Exit codes: 0 all checks passed, 1 a check failed.

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m' CYAN='\033[0;36m' NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' YELLOW='' CYAN='' NC=''
fi
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_skip()  { printf "${CYAN}[SKIP]${NC} %s\n" "$*"; }

# Container names follow the upstream compose project (name: neuronpedia)
WEBAPP_CONTAINER="neuronpedia-webapp-1"
POSTGRES_CONTAINER="neuronpedia-postgres-1"
INFERENCE_CONTAINER="neuronpedia-inference-1"
GRAPH_CONTAINER="neuronpedia-graph-1"
PLATFORM_PORTS="3001 5432 5002 5003 5004"

failures=0

echo "=== Neuronpedia platform smoke check ==="
echo

# 1. Loopback binding
echo "Port bindings:"
listeners=$(ss -tln 2>/dev/null || true)
for port in $PLATFORM_PORTS; do
    lines=$(grep -E "LISTEN.*[:.]${port}[[:space:]]" <<< "$listeners" || true)
    if [[ -z "$lines" ]]; then
        print_skip "Port $port: not listening"
        continue
    fi
    if grep -qE '(0\.0\.0\.0|\[::\]|\*):'"${port}"'[[:space:]]' <<< "$lines"; then
        print_error "Port $port: bound beyond loopback"
        echo "  Context: the localhost trust model requires 127.0.0.1-only bindings"
        echo "  Solution: check docker/interp/neuronpedia.override.yaml port mappings"
        failures=$((failures + 1))
    else
        print_ok "Port $port: loopback only"
    fi
done
echo

# 2. Webapp
echo "Services:"
if curl -sf -o /dev/null --max-time 10 "http://127.0.0.1:3001"; then
    print_ok "Webapp: answering on http://127.0.0.1:3001"
else
    print_error "Webapp: no answer on http://127.0.0.1:3001"
    echo "  Solution: check container state and logs"
    echo "  Command: make np-status && docker logs $WEBAPP_CONTAINER"
    failures=$((failures + 1))
fi

# 3. Postgres
if docker exec "$POSTGRES_CONTAINER" pg_isready -U postgres >/dev/null 2>&1; then
    print_ok "Postgres: accepting connections"
else
    print_error "Postgres: not accepting connections"
    echo "  Solution: check container state and logs"
    echo "  Command: make np-status && docker logs $POSTGRES_CONTAINER"
    failures=$((failures + 1))
fi

# 4. GPU platform containers (only when running). Both the inference instance and the
#    graph server build on CUDA 12.1-era bases and share the sm_120 torch-wheel risk.
check_gpu_container() {
    local label="$1" container="$2" up_target="$3"
    if ! docker ps --format '{{.Names}}' | grep -qx "$container"; then
        print_skip "$label GPU: instance not running (start with $up_target)"
        return
    fi
    local cap
    cap=$(docker exec "$container" python -c \
        'import torch; c = torch.cuda.get_device_capability(); print(f"{c[0]}.{c[1]}")' \
        2>/dev/null || true)
    if [[ "$cap" == "12.0" ]]; then
        print_ok "$label GPU: sm_120 visible to torch (capability $cap)"
    elif [[ -n "$cap" ]]; then
        print_error "$label GPU: unexpected device capability $cap (expected 12.0)"
        failures=$((failures + 1))
    else
        print_error "$label GPU: torch cannot see a CUDA device"
        echo "  Context: upstream torch pin may predate sm_120 (RTX 5090) wheel support"
        echo "  Solution: override the torch version and record it with the pin"
        echo "  Command: docker exec $container python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'"
        failures=$((failures + 1))
    fi
}
check_gpu_container "Inference" "$INFERENCE_CONTAINER" "make np-inference-up"
check_gpu_container "Graph"     "$GRAPH_CONTAINER"     "make np-graph-up"
echo

if [[ $failures -eq 0 ]]; then
    print_ok "Smoke check passed"
    exit 0
fi
print_error "Smoke check failed: $failures check(s)"
exit 1
