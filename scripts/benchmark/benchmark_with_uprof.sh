#!/bin/bash
# benchmark_with_uprof.sh - Run server-bench.py with AMD μProf profiling
#
# This script:
# 1. Finds the llama-server PID inside the container
# 2. Starts AMDuProfCLI attached to that PID
# 3. Runs the Python benchmark
# 4. Stops profiling and generates reports
#
# Usage:
#   ./scripts/benchmark_with_uprof.sh [container_name] [profile_type] [benchmark_name]
#
# Profile types:
#   assess  - General performance assessment (default)
#   tbp     - Time-based profiling with memory bandwidth
#   memory  - Custom events for memory optimization analysis
#
# Examples:
#   ./scripts/benchmark_with_uprof.sh llama-hugepages assess hugepages-assess
#   ./scripts/benchmark_with_uprof.sh llama-base tbp baseline-tbp
#   ./scripts/benchmark_with_uprof.sh llama-hugepages memory hugepages-memory
#
# Prerequisites:
#   - AMD μProf installed (see docs/uprof-setup.md)
#   - sudo access or perf_event_paranoid=-1
#   - Container running with llama-server

set -euo pipefail

# Configuration
CONTAINER_NAME="${1:-llama-hugepages}"
PROFILE_TYPE="${2:-assess}"
BENCHMARK_NAME="${3:-uprof-benchmark}"
RESULTS_DIR="uprof_results"

# Auto-detect port based on container name
case "$CONTAINER_NAME" in
    llama-base)      PORT="${PORT:-8001}" ;;
    llama-zen5)      PORT="${PORT:-8002}" ;;
    llama-hugepages) PORT="${PORT:-8003}" ;;
    *)               PORT="${PORT:-8003}" ;;
esac

# Benchmark parameters (matching benchmark_with_perf.sh)
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
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Create results directory
mkdir -p "$RESULTS_DIR"

# Step 1: Check if AMDuProfCLI is available
log_info "Checking AMD μProf installation..."
if ! command -v AMDuProfCLI &> /dev/null; then
    log_error "AMDuProfCLI not found. Install AMD μProf first."
    log_info "See docs/uprof-setup.md for installation instructions."
    exit 1
fi

UPROF_VERSION=$(AMDuProfCLI --version 2>&1 | head -1)
log_info "Found: $UPROF_VERSION"

# Step 2: Configure profile type
log_info "Profile type: $PROFILE_TYPE"
case "$PROFILE_TYPE" in
    assess)
        UPROF_ARGS="--config assess"
        PROFILE_DESC="General performance assessment"
        ;;
    tbp)
        UPROF_ARGS="--config tbp"
        PROFILE_DESC="Time-based profiling with memory bandwidth"
        ;;
    memory)
        # Custom events optimized for memory analysis on Zen 4/5
        UPROF_ARGS="-e cycles,instructions,L3_MISSES,L2_CACHE_MISS_FROM_DC_MISS"
        PROFILE_DESC="Custom memory optimization events"
        ;;
    *)
        log_error "Unknown profile type: $PROFILE_TYPE"
        log_info "Available types: assess, tbp, memory"
        exit 1
        ;;
esac
log_info "Description: $PROFILE_DESC"

# Step 3: Verify container is running
log_info "Checking container '$CONTAINER_NAME'..."
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_error "Container '$CONTAINER_NAME' is not running"
    exit 1
fi

# Step 4: Find the server PID
log_info "Finding llama-server PID..."
SERVER_PID=$(docker inspect --format '{{.State.Pid}}' "$CONTAINER_NAME")

if [[ -z "$SERVER_PID" || "$SERVER_PID" == "0" ]]; then
    log_error "Could not find container PID. Is the container running?"
    exit 1
fi

# Verify this is actually the server process
if ! ps -p "$SERVER_PID" -o cmd= 2>/dev/null | grep -q "server"; then
    log_warn "PID $SERVER_PID doesn't look like the server process"
    FALLBACK_PID=$(pgrep -f "server.*--port.*${PORT}" 2>/dev/null | head -1 || true)
    if [[ -n "$FALLBACK_PID" ]]; then
        SERVER_PID="$FALLBACK_PID"
    fi
fi

log_info "Found server PID: $SERVER_PID"

# Step 5: Verify server is healthy
log_info "Checking server health..."
if ! curl -s "http://127.0.0.1:${PORT}/health" | grep -q "ok"; then
    log_error "Server is not healthy on port $PORT"
    exit 1
fi
log_info "Server is healthy"

# Step 6: Prepare output paths
PROFILE_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}"
BENCH_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}_bench.txt"
SUMMARY_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}_summary.txt"

# Clean up any existing profile data
rm -rf "$PROFILE_OUTPUT"

# Step 7: Record system info
log_info "Recording system information..."
{
    echo "=== AMD μProf Benchmark ==="
    echo "Date: $(date)"
    echo "Container: $CONTAINER_NAME"
    echo "Port: $PORT"
    echo "Profile Type: $PROFILE_TYPE"
    echo "Profile Description: $PROFILE_DESC"
    echo "Server PID: $SERVER_PID"
    echo ""
    echo "=== System Info ==="
    AMDuProfCLI info --system 2>&1 || echo "(system info unavailable)"
    echo ""
} > "$SUMMARY_OUTPUT"

# Step 8: Start μProf in background
log_info "Starting AMD μProf profiling..."
log_info "Profile output: $PROFILE_OUTPUT"

# Note: μProf collect with -p attaches to existing process
# We use timeout as a failsafe (15 minutes max)
sudo AMDuProfCLI collect \
    $UPROF_ARGS \
    -o "$PROFILE_OUTPUT" \
    -p "$SERVER_PID" \
    --duration 900 &  # 15 min timeout

UPROF_PID=$!
log_info "μProf running with PID $UPROF_PID"

# Give μProf time to attach
sleep 2

# Verify μProf is running
if ! kill -0 "$UPROF_PID" 2>/dev/null; then
    log_error "μProf failed to start. Check permissions and installation."
    log_info "Try: sudo sysctl kernel.perf_event_paranoid=-1"
    exit 1
fi

# Step 9: Run the benchmark
log_info "Starting benchmark..."

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

# Step 10: Stop μProf gracefully
log_info "Stopping μProf..."
sudo kill -INT "$UPROF_PID" 2>/dev/null || true
wait "$UPROF_PID" 2>/dev/null || true

# Give it a moment to write data
sleep 2

# Step 11: Generate reports
log_info "Generating μProf reports..."

if [[ -d "$PROFILE_OUTPUT" ]]; then
    # Summary report
    {
        echo ""
        echo "=== μProf Summary ==="
        AMDuProfCLI report -i "$PROFILE_OUTPUT" --summary 2>&1 || echo "(report generation failed)"
    } >> "$SUMMARY_OUTPUT"

    # CSV export for detailed analysis
    CSV_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}_report.csv"
    AMDuProfCLI report -i "$PROFILE_OUTPUT" -o "$CSV_OUTPUT" 2>/dev/null || true

    # Hotspots report (if tbp profile)
    if [[ "$PROFILE_TYPE" == "tbp" ]]; then
        HOTSPOT_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}_hotspots.txt"
        AMDuProfCLI report -i "$PROFILE_OUTPUT" --hotspots > "$HOTSPOT_OUTPUT" 2>/dev/null || true
    fi
else
    log_warn "Profile directory not found: $PROFILE_OUTPUT"
    echo "WARNING: Profile data not collected" >> "$SUMMARY_OUTPUT"
fi

# Step 12: Save benchmark plots
if [[ -f "prompt_time.png" ]]; then
    mv prompt_time.png "$RESULTS_DIR/${BENCHMARK_NAME}_prompt_time.png"
    log_info "Saved: $RESULTS_DIR/${BENCHMARK_NAME}_prompt_time.png"
fi
if [[ -f "gen_rate.png" ]]; then
    mv gen_rate.png "$RESULTS_DIR/${BENCHMARK_NAME}_gen_rate.png"
    log_info "Saved: $RESULTS_DIR/${BENCHMARK_NAME}_gen_rate.png"
fi

# Step 13: Print summary
echo ""
log_info "========== BENCHMARK COMPLETE =========="
echo ""
echo "Results saved to: $RESULTS_DIR/"
echo "  - ${BENCHMARK_NAME}_bench.txt      (benchmark output)"
echo "  - ${BENCHMARK_NAME}_summary.txt    (μProf summary)"
if [[ -f "$CSV_OUTPUT" ]]; then
    echo "  - ${BENCHMARK_NAME}_report.csv     (detailed CSV)"
fi
if [[ -f "$RESULTS_DIR/${BENCHMARK_NAME}_hotspots.txt" ]]; then
    echo "  - ${BENCHMARK_NAME}_hotspots.txt   (function hotspots)"
fi
echo "  - ${BENCHMARK_NAME}/               (raw profile data)"
echo ""

# Print μProf summary
if [[ -f "$SUMMARY_OUTPUT" ]]; then
    echo "=== μProf Summary ==="
    grep -A 100 "=== μProf Summary ===" "$SUMMARY_OUTPUT" | tail -n +2
fi

echo ""
log_info "Done!"
