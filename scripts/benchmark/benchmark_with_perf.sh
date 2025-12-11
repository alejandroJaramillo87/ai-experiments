#!/bin/bash
# benchmark_with_perf.sh - Run server-bench.py with perf profiling of the container
#
# This script:
# 1. Finds the llama-server PID inside the container
# 2. Starts perf stat attached to that PID
# 3. Runs the Python benchmark
# 4. Stops perf and saves results
#
# Usage:
#   ./scripts/benchmark_with_perf.sh [container_name] [benchmark_name]
#
# Examples:
#   ./scripts/benchmark_with_perf.sh llama-hugepages hugepages-on
#   ./scripts/benchmark_with_perf.sh llama-base baseline
#   ./scripts/benchmark_with_perf.sh llama-zen5 aocl-blis

set -euo pipefail

# Configuration
CONTAINER_NAME="${1:-llama-hugepages}"
BENCHMARK_NAME="${2:-benchmark}"
RESULTS_DIR="benchmark_results"

# Auto-detect port based on container name (can be overridden with PORT env var)
case "$CONTAINER_NAME" in
    llama-cpu)      PORT="${PORT:-8001}" ;;
    llama-gpu)      PORT="${PORT:-8002}" ;;
    vllm)           PORT="${PORT:-8003}" ;;
    *)               PORT="${PORT:-8001}" ;;  # Default to cpu port
esac

# Benchmark parameters (matching the plan)
N_PARALLEL=1
PROMPT_SOURCE="rng-512-1024"
N_PROMPTS="${N_PROMPTS:-20}"
N_PREDICT=256
N_PREDICT_MIN=256
SEED_OFFSET=42

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create results directory
mkdir -p "$RESULTS_DIR"

# Step 1: Verify container is running
log_info "Checking container '$CONTAINER_NAME'..."
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_error "Container '$CONTAINER_NAME' is not running"
    exit 1
fi

# Step 2: Find the llama-server PID (visible from host via /proc)
log_info "Finding llama-server PID..."

# The server runs as the container's main process (PID 1 in container namespace)
# docker inspect gives us the host PID for that process
SERVER_PID=$(docker inspect --format '{{.State.Pid}}' "$CONTAINER_NAME")

if [[ -z "$SERVER_PID" || "$SERVER_PID" == "0" ]]; then
    log_error "Could not find container PID. Is the container running?"
    exit 1
fi

# Verify this is actually the server process
if ! ps -p "$SERVER_PID" -o cmd= 2>/dev/null | grep -q "server"; then
    log_warn "PID $SERVER_PID doesn't look like the server process"
    # Try to find by command pattern
    FALLBACK_PID=$(pgrep -f "server.*--port.*${PORT}" 2>/dev/null | head -1 || true)
    if [[ -n "$FALLBACK_PID" ]]; then
        SERVER_PID="$FALLBACK_PID"
    fi
fi

log_info "Found server PID: $SERVER_PID ($(ps -p $SERVER_PID -o cmd= 2>/dev/null | cut -c1-50)...)"

# Step 3: Verify server is healthy
log_info "Checking server health..."
if ! curl -s "http://127.0.0.1:${PORT}/health" | grep -q "ok"; then
    log_error "Server is not healthy on port $PORT"
    exit 1
fi
log_info "Server is healthy"

# Step 4: Record hugepage status before benchmark
log_info "Recording memory status..."
HUGEPAGES_BEFORE=$(grep HugePages_Free /proc/meminfo | awk '{print $2}')
echo "HugePages_Free before: $HUGEPAGES_BEFORE" > "$RESULTS_DIR/${BENCHMARK_NAME}_memory.txt"
grep -i huge /proc/meminfo >> "$RESULTS_DIR/${BENCHMARK_NAME}_memory.txt"

# Step 5: Start perf in background
PERF_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}_perf.txt"
log_info "Starting perf stat (output: $PERF_OUTPUT)..."

# perf events optimized for AMD Zen 5 / memory analysis
PERF_EVENTS="dTLB-load-misses,L1-dcache-load-misses,L1-dcache-loads,cache-misses,cache-references,cycles,instructions"

# Start perf attached to server PID
# Using a subshell with timeout as a failsafe
sudo perf stat \
    -e "$PERF_EVENTS" \
    -p "$SERVER_PID" \
    -o "$PERF_OUTPUT" \
    -- sleep 900 &  # 15 min timeout (benchmark takes ~8-10 min)

PERF_PID=$!
log_info "perf running with PID $PERF_PID"

# Give perf a moment to attach
sleep 1

# Verify perf is running
if ! kill -0 "$PERF_PID" 2>/dev/null; then
    log_error "perf failed to start. Check permissions (need sudo or perf_event_paranoid=-1)"
    exit 1
fi

# Step 6: Run the benchmark
log_info "Starting benchmark..."
BENCH_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}_bench.txt"

LLAMA_ARG_N_PARALLEL="$N_PARALLEL" \
python3 "$(dirname "$0")/server-bench.py" \
    --path_server "http://127.0.0.1:${PORT}" \
    --prompt_source "$PROMPT_SOURCE" \
    --n_prompts "$N_PROMPTS" \
    --n_predict "$N_PREDICT" \
    --n_predict_min "$N_PREDICT_MIN" \
    --seed_offset "$SEED_OFFSET" \
    --name "$BENCHMARK_NAME" \
    2>&1 | tee "$BENCH_OUTPUT"

# Step 7: Stop perf gracefully
log_info "Stopping perf..."
sudo kill -INT "$PERF_PID" 2>/dev/null || true
wait "$PERF_PID" 2>/dev/null || true

# Step 8: Save plots with benchmark name
if [[ -f "prompt_time.png" ]]; then
    mv prompt_time.png "$RESULTS_DIR/${BENCHMARK_NAME}_prompt_time.png"
    log_info "Saved: $RESULTS_DIR/${BENCHMARK_NAME}_prompt_time.png"
fi
if [[ -f "gen_rate.png" ]]; then
    mv gen_rate.png "$RESULTS_DIR/${BENCHMARK_NAME}_gen_rate.png"
    log_info "Saved: $RESULTS_DIR/${BENCHMARK_NAME}_gen_rate.png"
fi

# Step 9: Record hugepage status after benchmark
HUGEPAGES_AFTER=$(grep HugePages_Free /proc/meminfo | awk '{print $2}')
echo "" >> "$RESULTS_DIR/${BENCHMARK_NAME}_memory.txt"
echo "HugePages_Free after: $HUGEPAGES_AFTER" >> "$RESULTS_DIR/${BENCHMARK_NAME}_memory.txt"
echo "HugePages used: $((HUGEPAGES_BEFORE - HUGEPAGES_AFTER)) pages ($((( HUGEPAGES_BEFORE - HUGEPAGES_AFTER ) * 2)) MB)" >> "$RESULTS_DIR/${BENCHMARK_NAME}_memory.txt"

# Step 10: Print summary
echo ""
log_info "========== BENCHMARK COMPLETE =========="
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "  - ${BENCHMARK_NAME}_bench.txt    (benchmark output)"
echo "  - ${BENCHMARK_NAME}_perf.txt     (perf counters)"
echo "  - ${BENCHMARK_NAME}_memory.txt   (hugepage status)"
echo "  - ${BENCHMARK_NAME}_prompt_time.png"
echo "  - ${BENCHMARK_NAME}_gen_rate.png"
echo ""

# Print perf summary
if [[ -f "$PERF_OUTPUT" ]]; then
    echo "=== Perf Summary ==="
    cat "$PERF_OUTPUT"
fi

echo ""
log_info "Done!"
