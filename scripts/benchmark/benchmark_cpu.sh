#!/bin/bash
# benchmark_cpu.sh - CPU benchmark with multi-level profiling
#
# Run server-bench.py with CPU profiling at various depth levels.
# Supports AMD Zen 5 processors with optimized perf events.
#
# Usage:
#   ./scripts/benchmark/benchmark_cpu.sh [OPTIONS]
#
# Options:
#   -c, --container NAME    Container name (default: llama-cpu)
#   -n, --name NAME         Benchmark name for output files (default: cpu-benchmark)
#   -l, --level LEVEL       Profiling level: basic|detailed|deep (default: basic)
#   -p, --prompts N         Number of prompts (default: 20)
#   -h, --help              Show help message
#
# Profiling levels:
#   basic     perf stat with core events (cycles, instructions, cache)
#   detailed  perf stat with extended AMD Zen 5 events (TLB, L1/L2/L3, branches)
#   deep      AMD uProf full profiling (assess + memory analysis)
#
# Examples:
#   ./scripts/benchmark/benchmark_cpu.sh -n zen5-baseline
#   ./scripts/benchmark/benchmark_cpu.sh -l detailed -n zen5-detailed -p 50
#   ./scripts/benchmark/benchmark_cpu.sh -l deep -n zen5-uprof -p 20

set -euo pipefail

# === Configuration ===
CONTAINER_NAME="llama-cpu"
BENCHMARK_NAME="cpu-benchmark"
PROFILE_LEVEL="basic"
N_PROMPTS=20
RESULTS_DIR="cpu_benchmark_results"

# Benchmark parameters (matching GPU benchmark)
N_PARALLEL=1
PROMPT_SOURCE="rng-512-1024"
N_PREDICT=256
N_PREDICT_MIN=256
SEED_OFFSET=42

# Auto-detect port (can be overridden with PORT env var)
PORT="${PORT:-}"

# Hugepage tracking
HUGEPAGES_BEFORE=0
HUGEPAGES_AFTER=0

# === Color Definitions (with NO_COLOR support) ===
if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    CYAN=''
    NC=''
fi

# === Logging Functions ===
log_info()  { printf "${GREEN}[INFO]${NC} %s\n" "$1"; }
log_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$1"; }
log_error() { printf "${RED}[ERROR]${NC} %s\n" "$1"; }
die() { log_error "$1"; exit 1; }

# === Help ===
show_help() {
    cat << 'EOF'
benchmark_cpu.sh - CPU benchmark with multi-level profiling

Usage:
  ./scripts/benchmark/benchmark_cpu.sh [OPTIONS]

Options:
  -c, --container NAME    Container name (default: llama-cpu)
  -n, --name NAME         Benchmark name for output files (default: cpu-benchmark)
  -l, --level LEVEL       Profiling level: basic|detailed|deep (default: basic)
  -p, --prompts N         Number of prompts (default: 20)
  -h, --help              Show this help message

Profiling Levels:
  basic     perf stat with core events (cycles, instructions, cache)
            Requirements: perf (linux-tools package)

  detailed  perf stat with extended AMD Zen 5 events
            Adds: TLB misses, L1/L2 cache, branch prediction, stall cycles
            Requirements: perf with AMD PMU support (kernel 6.x+)

  deep      AMD uProf full profiling (assess configuration)
            Generates: hotspots, CSV reports, function-level analysis
            Requirements: AMDuProfCLI installed

Examples:
  # Basic perf profiling (default)
  ./scripts/benchmark/benchmark_cpu.sh -n zen5-baseline

  # Detailed AMD Zen 5 events
  ./scripts/benchmark/benchmark_cpu.sh -l detailed -n zen5-detailed -p 50

  # Deep AMD uProf profiling
  ./scripts/benchmark/benchmark_cpu.sh -l deep -n zen5-uprof -p 20

  # Custom container
  ./scripts/benchmark/benchmark_cpu.sh -c llama-hugepages -n hugepages-test

Output:
  Results are saved to cpu_benchmark_results/ with the benchmark name prefix.

Environment Variables:
  PORT           Override auto-detected port
  N_PROMPTS      Override number of prompts (same as -p)
  NO_COLOR       Disable colored output
EOF
}

# === Argument Parsing ===
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--container)
                CONTAINER_NAME="$2"
                shift 2
                ;;
            -n|--name)
                BENCHMARK_NAME="$2"
                shift 2
                ;;
            -l|--level)
                PROFILE_LEVEL="$2"
                if [[ ! "$PROFILE_LEVEL" =~ ^(basic|detailed|deep)$ ]]; then
                    die "Invalid profile level: $PROFILE_LEVEL (must be basic, detailed, or deep)"
                fi
                shift 2
                ;;
            -p|--prompts)
                N_PROMPTS="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                die "Unknown option: $1 (use -h for help)"
                ;;
        esac
    done
}

# === Dependency Validation ===
validate_dependencies() {
    case "$PROFILE_LEVEL" in
        basic|detailed)
            if ! command -v perf >/dev/null; then
                die "perf not found. Install with: sudo apt-get install linux-tools-generic linux-tools-\$(uname -r)"
            fi
            ;;
        deep)
            if ! command -v AMDuProfCLI >/dev/null; then
                die "AMDuProfCLI not found. Download from: https://developer.amd.com/amd-uprof/"
            fi
            ;;
    esac

    # Check for server-bench.py
    local script_dir
    script_dir="$(dirname "$0")"
    if [[ ! -f "$script_dir/server-bench.py" ]]; then
        die "server-bench.py not found in $script_dir"
    fi
}

# === Profiling Functions ===

# Define perf events for basic level
get_basic_events() {
    echo "cycles,instructions,cache-misses,cache-references"
}

# Define perf events for detailed level (AMD Zen 5 optimized)
get_detailed_events() {
    local events="cycles,instructions,cache-misses,cache-references"
    events+=",dTLB-load-misses,L1-dcache-load-misses,L1-dcache-loads"
    events+=",L1-icache-load-misses,branch-misses,branch-instructions"
    events+=",stalled-cycles-frontend,stalled-cycles-backend"
    echo "$events"
}

# Start basic/detailed profiling with perf
start_perf_profiling() {
    local output_file="$1"
    local server_pid="$2"
    local events

    if [[ "$PROFILE_LEVEL" == "basic" ]]; then
        events=$(get_basic_events)
    else
        events=$(get_detailed_events)
    fi

    # Start perf stat attached to server PID
    # Using sleep as a timeout (15 min max, benchmark typically takes 8-10 min)
    sudo perf stat \
        -e "$events" \
        -p "$server_pid" \
        -o "$output_file" \
        -- sleep 900 &
    echo $!
}

# Start deep profiling with AMD uProf
start_uprof_profiling() {
    local output_dir="$1"
    local server_pid="$2"

    # Clean up any existing profile data
    rm -rf "$output_dir"

    # Use assess profile for general analysis
    sudo AMDuProfCLI collect \
        --config assess \
        -o "$output_dir" \
        -p "$server_pid" \
        --duration 900 &
    echo $!
}

# Unified start function
start_profiling() {
    local output_file="$1"
    local server_pid="$2"

    case "$PROFILE_LEVEL" in
        basic|detailed)
            start_perf_profiling "$output_file" "$server_pid"
            ;;
        deep)
            start_uprof_profiling "$output_file" "$server_pid"
            ;;
    esac
}

# Stop profiling gracefully
stop_profiling() {
    local pid="$1"
    if kill -0 "$pid" 2>/dev/null; then
        sudo kill -INT "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

# === Real-Time Display ===
display_realtime_cpu_stats() {
    local bench_pid="$1"

    # Wait a moment for benchmark to start
    sleep 2

    while kill -0 "$bench_pid" 2>/dev/null; do
        # Get system memory stats
        local mem_info
        mem_info=$(free -m | awk '/^Mem:/ {printf "%d/%d MB (%.1f%%)", $3, $2, $3*100/$2}')

        # Get hugepage status
        local hp_free hp_total
        hp_free=$(awk '/HugePages_Free/ {print $2}' /proc/meminfo)
        hp_total=$(awk '/HugePages_Total/ {print $2}' /proc/meminfo)

        # Get load average (1 min)
        local load_avg
        load_avg=$(awk '{print $1}' /proc/loadavg)

        # Overwrite current line with stats
        printf "\r  ${CYAN}Load:${NC} %5s | ${CYAN}Mem:${NC} %s | ${CYAN}HugePages:${NC} %s/%s free  " \
            "$load_avg" "$mem_info" "$hp_free" "$hp_total"

        sleep 1
    done
    echo ""  # Newline after loop ends
}

# === Summary Generation ===
generate_cpu_summary() {
    local profile_file="$1"
    local output_file="$2"

    {
        echo "=== CPU Benchmark Summary ==="
        echo "Benchmark: $BENCHMARK_NAME"
        echo "Container: $CONTAINER_NAME"
        echo "Profile Level: $PROFILE_LEVEL"
        echo "Prompts: $N_PROMPTS"
        echo "Date: $(date -Iseconds)"
        echo ""

        # CPU info
        echo "=== CPU Info ==="
        lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core|Socket|CPU max MHz)" | sed 's/^/  /'
        echo ""

        # Memory info
        echo "=== Memory ==="
        echo "  Hugepages Before: $HUGEPAGES_BEFORE"
        echo "  Hugepages After: $HUGEPAGES_AFTER"
        local hp_used=$((HUGEPAGES_BEFORE - HUGEPAGES_AFTER))
        echo "  Hugepages Used: $hp_used pages ($((hp_used * 2)) MB)"
        echo ""

        case "$PROFILE_LEVEL" in
            basic|detailed)
                if [[ -f "$profile_file" ]] && [[ -s "$profile_file" ]]; then
                    echo "=== Perf Statistics ==="
                    # Extract key metrics from perf output
                    grep -E "^\s+[0-9]" "$profile_file" | sed 's/^/  /' || true
                    echo ""

                    # Calculate IPC if possible
                    local cycles instructions
                    cycles=$(grep -oP '[\d,]+(?=\s+cycles)' "$profile_file" 2>/dev/null | head -1 | tr -d ',' || echo "")
                    instructions=$(grep -oP '[\d,]+(?=\s+instructions)' "$profile_file" 2>/dev/null | head -1 | tr -d ',' || echo "")
                    if [[ -n "$cycles" && -n "$instructions" && "$cycles" -gt 0 ]]; then
                        local ipc
                        ipc=$(echo "scale=3; $instructions / $cycles" | bc 2>/dev/null || echo "N/A")
                        echo "  Calculated IPC: $ipc"
                    fi
                else
                    echo "  No perf data available"
                fi
                ;;
            deep)
                echo "=== AMD uProf Summary ==="
                echo "  Full report: ${RESULTS_DIR}/${BENCHMARK_NAME}_uprof/"
                if [[ -f "${RESULTS_DIR}/${BENCHMARK_NAME}_report.csv" ]]; then
                    echo "  CSV export: ${RESULTS_DIR}/${BENCHMARK_NAME}_report.csv"
                fi
                if [[ -f "${RESULTS_DIR}/${BENCHMARK_NAME}_hotspots.txt" ]]; then
                    echo ""
                    echo "=== Top Hotspots ==="
                    head -20 "${RESULTS_DIR}/${BENCHMARK_NAME}_hotspots.txt" | sed 's/^/  /'
                fi
                ;;
        esac
    } > "$output_file"
}

# === Generate uProf Reports ===
generate_uprof_reports() {
    local profile_dir="$1"

    if [[ ! -d "$profile_dir" ]]; then
        log_warn "Profile directory not found: $profile_dir"
        return 1
    fi

    # Summary report
    log_info "Generating uProf summary..."
    AMDuProfCLI report -i "$profile_dir" --summary > "${RESULTS_DIR}/${BENCHMARK_NAME}_summary_raw.txt" 2>/dev/null || true

    # CSV export for detailed analysis
    log_info "Generating CSV export..."
    AMDuProfCLI report -i "$profile_dir" -o "${RESULTS_DIR}/${BENCHMARK_NAME}_report.csv" 2>/dev/null || true

    # Hotspots report
    log_info "Generating hotspots report..."
    AMDuProfCLI report -i "$profile_dir" --hotspots > "${RESULTS_DIR}/${BENCHMARK_NAME}_hotspots.txt" 2>/dev/null || true
}

# === Main ===
main() {
    parse_args "$@"

    log_info "CPU Benchmark Script"
    log_info "  Container: $CONTAINER_NAME"
    log_info "  Benchmark: $BENCHMARK_NAME"
    log_info "  Level: $PROFILE_LEVEL"
    log_info "  Prompts: $N_PROMPTS"
    echo ""

    validate_dependencies

    mkdir -p "$RESULTS_DIR"

    # Step 1: Verify container is running
    log_info "Checking container '$CONTAINER_NAME'..."
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        die "Container '$CONTAINER_NAME' is not running"
    fi

    # Step 2: Find the server PID (visible from host via /proc)
    log_info "Finding server PID..."
    SERVER_PID=$(docker inspect --format '{{.State.Pid}}' "$CONTAINER_NAME")

    if [[ -z "$SERVER_PID" || "$SERVER_PID" == "0" ]]; then
        die "Could not find container PID. Is the container running?"
    fi

    # Verify this is actually the server process
    if ! ps -p "$SERVER_PID" -o cmd= 2>/dev/null | grep -q "server"; then
        log_warn "PID $SERVER_PID doesn't look like the server process"
    fi
    log_info "Server PID: $SERVER_PID"

    # Step 3: Auto-detect port based on container name
    if [[ -z "$PORT" ]]; then
        case "$CONTAINER_NAME" in
            llama-cpu|llama-base|llama-zen5|llama-hugepages) PORT=8001 ;;
            llama-gpu)      PORT=8002 ;;
            vllm*)          PORT=8003 ;;
            *)              PORT=8001 ;;
        esac
    fi
    log_info "Using port: $PORT"

    # Step 4: Health check
    log_info "Checking server health..."
    if ! curl -sf "http://127.0.0.1:${PORT}/health" | grep -q "ok"; then
        die "Server is not healthy on port $PORT"
    fi
    log_info "Server is healthy"

    # Step 5: Record initial memory/hugepage state
    log_info "Recording memory status..."
    HUGEPAGES_BEFORE=$(awk '/HugePages_Free/ {print $2}' /proc/meminfo)

    {
        echo "=== Memory Status Before Benchmark ==="
        echo "Date: $(date -Iseconds)"
        echo ""
        echo "HugePages_Free: $HUGEPAGES_BEFORE"
        grep -i huge /proc/meminfo
        echo ""
        free -h
    } > "$RESULTS_DIR/${BENCHMARK_NAME}_memory.txt"

    # Step 6: Start profiling
    log_info "Starting $PROFILE_LEVEL level CPU profiling..."

    if [[ "$PROFILE_LEVEL" == "deep" ]]; then
        PROFILE_FILE="$RESULTS_DIR/${BENCHMARK_NAME}_uprof"
    else
        PROFILE_FILE="$RESULTS_DIR/${BENCHMARK_NAME}_perf.txt"
    fi

    PROFILE_PID=$(start_profiling "$PROFILE_FILE" "$SERVER_PID")
    log_info "Profiler PID: $PROFILE_PID"

    # Give profiler a moment to attach
    sleep 1

    # Verify profiler is running
    if ! kill -0 "$PROFILE_PID" 2>/dev/null; then
        die "Profiler failed to start. Check permissions (need sudo or perf_event_paranoid=-1)"
    fi

    # Step 7: Run the benchmark
    echo ""
    log_info "Running benchmark ($N_PROMPTS prompts)..."
    echo "=============================================="

    BENCH_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}_bench.txt"
    local script_dir
    script_dir="$(dirname "$0")"

    # Start real-time display in background
    (
        sleep 3
        local bench_pid
        bench_pid=$(pgrep -f "server-bench.py.*${BENCHMARK_NAME}" 2>/dev/null | head -1 || echo "")
        if [[ -n "$bench_pid" ]]; then
            display_realtime_cpu_stats "$bench_pid"
        fi
    ) &
    DISPLAY_PID=$!

    # Run the benchmark
    LLAMA_ARG_N_PARALLEL="$N_PARALLEL" \
    python3 "$script_dir/server-bench.py" \
        --path_server "http://127.0.0.1:${PORT}" \
        --prompt_source "$PROMPT_SOURCE" \
        --n_prompts "$N_PROMPTS" \
        --n_predict "$N_PREDICT" \
        --n_predict_min "$N_PREDICT_MIN" \
        --seed_offset "$SEED_OFFSET" \
        --name "$BENCHMARK_NAME" \
        2>&1 | tee "$BENCH_OUTPUT"

    BENCH_EXIT_CODE=${PIPESTATUS[0]}

    echo "=============================================="
    echo ""

    # Step 8: Stop real-time display
    kill "$DISPLAY_PID" 2>/dev/null || true
    wait "$DISPLAY_PID" 2>/dev/null || true

    # Step 9: Stop profiling
    log_info "Stopping profiler..."
    stop_profiling "$PROFILE_PID"

    # Give profiler time to write data
    sleep 2

    # Step 10: Record final hugepage state
    HUGEPAGES_AFTER=$(awk '/HugePages_Free/ {print $2}' /proc/meminfo)
    {
        echo ""
        echo "=== Memory Status After Benchmark ==="
        echo "Date: $(date -Iseconds)"
        echo ""
        echo "HugePages_Free: $HUGEPAGES_AFTER"
        echo "HugePages_Used: $((HUGEPAGES_BEFORE - HUGEPAGES_AFTER)) pages"
        grep -i huge /proc/meminfo
    } >> "$RESULTS_DIR/${BENCHMARK_NAME}_memory.txt"

    # Step 11: Generate uProf reports if deep level
    if [[ "$PROFILE_LEVEL" == "deep" ]]; then
        generate_uprof_reports "$PROFILE_FILE"
    fi

    # Step 12: Generate summary report
    log_info "Generating summary report..."
    generate_cpu_summary "$PROFILE_FILE" "$RESULTS_DIR/${BENCHMARK_NAME}_summary.txt"

    # Step 13: Move plots if they exist
    if [[ -f "prompt_time.png" ]]; then
        mv prompt_time.png "$RESULTS_DIR/${BENCHMARK_NAME}_prompt_time.png"
        log_info "Saved: ${BENCHMARK_NAME}_prompt_time.png"
    fi
    if [[ -f "gen_rate.png" ]]; then
        mv gen_rate.png "$RESULTS_DIR/${BENCHMARK_NAME}_gen_rate.png"
        log_info "Saved: ${BENCHMARK_NAME}_gen_rate.png"
    fi

    # Step 14: Print summary
    echo ""
    log_info "========== BENCHMARK COMPLETE =========="
    echo ""
    cat "$RESULTS_DIR/${BENCHMARK_NAME}_summary.txt"
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
    echo "  - ${BENCHMARK_NAME}_bench.txt       (benchmark output)"
    if [[ "$PROFILE_LEVEL" != "deep" ]]; then
        echo "  - ${BENCHMARK_NAME}_perf.txt        (perf statistics)"
    else
        echo "  - ${BENCHMARK_NAME}_uprof/          (uProf raw data)"
        [[ -f "$RESULTS_DIR/${BENCHMARK_NAME}_report.csv" ]] && \
            echo "  - ${BENCHMARK_NAME}_report.csv      (CSV export)"
        [[ -f "$RESULTS_DIR/${BENCHMARK_NAME}_hotspots.txt" ]] && \
            echo "  - ${BENCHMARK_NAME}_hotspots.txt    (function hotspots)"
    fi
    echo "  - ${BENCHMARK_NAME}_memory.txt      (hugepage status)"
    echo "  - ${BENCHMARK_NAME}_summary.txt     (summary statistics)"
    if [[ -f "$RESULTS_DIR/${BENCHMARK_NAME}_prompt_time.png" ]]; then
        echo "  - ${BENCHMARK_NAME}_prompt_time.png"
    fi
    if [[ -f "$RESULTS_DIR/${BENCHMARK_NAME}_gen_rate.png" ]]; then
        echo "  - ${BENCHMARK_NAME}_gen_rate.png"
    fi
    echo ""

    if [[ "$BENCH_EXIT_CODE" -ne 0 ]]; then
        log_warn "Benchmark exited with code $BENCH_EXIT_CODE"
        exit "$BENCH_EXIT_CODE"
    fi

    log_info "Done"
}

main "$@"
