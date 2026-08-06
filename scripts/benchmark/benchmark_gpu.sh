#!/bin/bash
# benchmark_gpu.sh - GPU benchmark with multi-level profiling
#
# Run server-bench.py with GPU profiling at various depth levels.
#
# Usage:
#   ./scripts/benchmark/benchmark_gpu.sh [OPTIONS]
#
# Options:
#   -c, --container NAME    Container name (default: llama-gpu)
#   -n, --name NAME         Benchmark name for output files (default: gpu-benchmark)
#   -l, --level LEVEL       Profiling level: basic|deep (default: basic)
#   -p, --prompts N         Number of prompts (default: 20)
#   -h, --help              Show help message
#
# Profiling levels:
#   basic     nvidia-smi polling (GPU util, VRAM, temp, power, PCIe)
#   deep      Nsight Systems profiling (runs server under nsys, not containerized)
#
# Examples:
#   ./scripts/benchmark/benchmark_gpu.sh -n rtx5090-baseline
#   ./scripts/benchmark/benchmark_gpu.sh -l deep -n rtx5090-nsys -p 5

set -euo pipefail

# === Configuration ===
CONTAINER_NAME="llama-gpu"
BENCHMARK_NAME="gpu-benchmark"
PROFILE_LEVEL="basic"
N_PROMPTS=20
RESULTS_DIR="gpu_benchmark_results"

# Benchmark parameters (matching CPU benchmark)
N_PARALLEL=1
PROMPT_SOURCE="rng-512-1024"
N_PREDICT=256
N_PREDICT_MIN=256
SEED_OFFSET=42

# Auto-detect port (can be overridden with PORT env var)
PORT="${PORT:-}"

# === Color Definitions ===
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
benchmark_gpu.sh - GPU benchmark with multi-level profiling

Usage:
  ./scripts/benchmark/benchmark_gpu.sh [OPTIONS]

Options:
  -c, --container NAME    Container name (default: llama-gpu)
  -n, --name NAME         Benchmark name for output files (default: gpu-benchmark)
  -l, --level LEVEL       Profiling level: basic|deep (default: basic)
  -p, --prompts N         Number of prompts (default: 20)
  -h, --help              Show this help message

Profiling Levels:
  basic     nvidia-smi polling (GPU util, VRAM, temp, power, PCIe)
            Requirements: NVIDIA driver only

  deep      Nsight Systems profiling (kernel-level, CUDA timeline)
            Requirements: Container started with NSYS_PROFILE=true
            Note: nsys runs inside container, stop container to finalize report

Examples:
  # Basic GPU monitoring (default)
  ./scripts/benchmark/benchmark_gpu.sh -n rtx5090-baseline

  # Deep Nsight profiling
  # 1. Start container with: NSYS_PROFILE=true (mount nsys output volume)
  # 2. Run benchmark:
  ./scripts/benchmark/benchmark_gpu.sh -l deep -n rtx5090-nsys -p 5
  # 3. Stop container to finalize nsys report

  # Custom container
  ./scripts/benchmark/benchmark_gpu.sh -c vllm-gpu -n vllm-test

Output:
  Results are saved to gpu_benchmark_results/ with the benchmark name prefix.
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
                if [[ ! "$PROFILE_LEVEL" =~ ^(basic|deep)$ ]]; then
                    die "Invalid profile level: $PROFILE_LEVEL (must be basic or deep)"
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
    # Always need nvidia-smi for basic stats
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found. Is the NVIDIA driver installed?"

    case "$PROFILE_LEVEL" in
        basic)
            # nvidia-smi is sufficient
            ;;
        deep)
            if ! command -v nsys >/dev/null; then
                die "nsys not found. Install CUDA Toolkit or download Nsight Systems from NVIDIA"
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

# === GPU Monitoring Functions ===

# Start basic monitoring with nvidia-smi dmon
start_basic_monitoring() {
    local output_file="$1"
    # -s pucvmet: power, utilization, clock, vram, memory, encoder, temperature
    # -d 1: poll every 1 second
    nvidia-smi dmon -s pucvmet -d 1 > "$output_file" 2>&1 &
    echo $!
}

# Note: Deep profiling runs inside the container (NSYS_PROFILE=true)
# No separate monitoring process needed - nsys wraps the server in the container

# Unified start function
start_monitoring() {
    local output_file="$1"

    case "$PROFILE_LEVEL" in
        basic)
            start_basic_monitoring "$output_file"
            ;;
        deep)
            # Deep mode: nsys runs inside the container (NSYS_PROFILE=true)
            # No external monitoring needed - just return empty PID
            echo ""
            ;;
    esac
}

# Stop monitoring gracefully
stop_monitoring() {
    local pid="$1"

    # Deep mode has no external monitor process
    if [[ -z "$pid" ]]; then
        return 0
    fi

    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}

# === Real-Time Display ===
display_realtime_stats() {
    local bench_pid="$1"

    # Wait a moment for benchmark to start
    sleep 2

    while kill -0 "$bench_pid" 2>/dev/null; do
        # Get current GPU stats
        local stats
        stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null) || continue

        local gpu_util mem_util mem_used mem_total temp power
        gpu_util=$(echo "$stats" | cut -d',' -f1 | tr -d ' ')
        mem_util=$(echo "$stats" | cut -d',' -f2 | tr -d ' ')
        mem_used=$(echo "$stats" | cut -d',' -f3 | tr -d ' ')
        mem_total=$(echo "$stats" | cut -d',' -f4 | tr -d ' ')
        temp=$(echo "$stats" | cut -d',' -f5 | tr -d ' ')
        power=$(echo "$stats" | cut -d',' -f6 | tr -d ' ')

        # Overwrite current line with stats
        printf "\r  ${CYAN}GPU:${NC} %3s%% | ${CYAN}VRAM:${NC} %5s/%5s MB (%3s%%) | ${CYAN}Temp:${NC} %3s C | ${CYAN}Power:${NC} %6s W  " \
            "$gpu_util" "$mem_used" "$mem_total" "$mem_util" "$temp" "$power"

        sleep 1
    done
    echo ""  # Newline after loop ends
}

# === Summary Generation ===
generate_summary() {
    local monitor_file="$1"
    local output_file="$2"

    {
        echo "=== GPU Benchmark Summary ==="
        echo "Benchmark: $BENCHMARK_NAME"
        echo "Container: $CONTAINER_NAME"
        echo "Profile Level: $PROFILE_LEVEL"
        echo "Prompts: $N_PROMPTS"
        echo "Date: $(date -Iseconds)"
        echo ""

        # Get GPU info
        echo "=== GPU Info ==="
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
        echo ""

        case "$PROFILE_LEVEL" in
            basic)
                if [[ -f "$monitor_file" ]] && [[ -s "$monitor_file" ]]; then
                    echo "=== GPU Utilization ==="
                    # nvidia-smi dmon columns: gpu pwr gtemp mtemp sm mem enc dec mclk pclk
                    # Skip header lines (start with #)
                    awk '!/^#/ && NF>5 {sum+=$5; count++} END {if(count>0) printf "  Average GPU Util: %.1f%%\n", sum/count; else print "  No data"}' "$monitor_file"
                    awk '!/^#/ && NF>5 {if($5>max) max=$5} END {printf "  Peak GPU Util: %.1f%%\n", max}' "$monitor_file"

                    echo ""
                    echo "=== Memory ==="
                    awk '!/^#/ && NF>5 {sum+=$6; count++} END {if(count>0) printf "  Average Mem Util: %.1f%%\n", sum/count}' "$monitor_file"
                    awk '!/^#/ && NF>5 {if($6>max) max=$6} END {printf "  Peak Mem Util: %.1f%%\n", max}' "$monitor_file"

                    echo ""
                    echo "=== Thermals ==="
                    awk '!/^#/ && NF>3 {sum+=$3; count++} END {if(count>0) printf "  Average Temp: %.1f C\n", sum/count}' "$monitor_file"
                    awk '!/^#/ && NF>3 {if($3>max) max=$3} END {printf "  Peak Temp: %.1f C\n", max}' "$monitor_file"

                    echo ""
                    echo "=== Power ==="
                    awk '!/^#/ && NF>2 {sum+=$2; count++} END {if(count>0) printf "  Average Power: %.1f W\n", sum/count}' "$monitor_file"
                    awk '!/^#/ && NF>2 {if($2>max) max=$2} END {printf "  Peak Power: %.1f W\n", max}' "$monitor_file"
                else
                    echo "  No monitoring data available"
                fi
                ;;
            deep)
                echo "=== Nsight Systems Profiling ==="
                echo "  nsys runs inside container (NSYS_PROFILE=true)"
                echo "  Stop container to finalize report: docker compose -f docker-compose.inference.yaml stop llama-gpu"
                echo ""
                echo "  After stopping, view report with:"
                echo "    nsys-ui $RESULTS_DIR/<NSYS_OUTPUT>.nsys-rep"
                echo "  Or generate text stats:"
                echo "    nsys stats $RESULTS_DIR/<NSYS_OUTPUT>.nsys-rep"
                ;;
        esac

    } > "$output_file"
}

# === Main ===
main() {
    parse_args "$@"

    log_info "GPU Benchmark Script"
    log_info "  Benchmark: $BENCHMARK_NAME"
    log_info "  Level: $PROFILE_LEVEL"
    log_info "  Prompts: $N_PROMPTS"
    log_info "  Container: $CONTAINER_NAME"
    echo ""

    validate_dependencies

    mkdir -p "$RESULTS_DIR"

    # Check container is running
    log_info "Checking container '$CONTAINER_NAME'..."
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        die "Container '$CONTAINER_NAME' is not running"
    fi

    # Auto-detect port based on container name
    if [[ -z "$PORT" ]]; then
        case "$CONTAINER_NAME" in
            llama-cpu)      PORT=8001 ;;
            llama-gpu)      PORT=8002 ;;
            vllm*)          PORT=8003 ;;
            *)              PORT=8002 ;;
        esac
    fi
    log_info "Using port: $PORT"

    # Health check for container
    log_info "Checking server health..."
    if ! curl -sf "http://127.0.0.1:${PORT}/health" | grep -q "ok"; then
        die "Server is not healthy on port $PORT"
    fi
    log_info "Server is healthy"

    # Deep mode reminder
    if [[ "$PROFILE_LEVEL" == "deep" ]]; then
        log_info "Deep mode: Ensure container was started with NSYS_PROFILE=true"
        log_warn "Stop container after benchmark to finalize nsys report"
    fi

    # Step 5: Record initial GPU state
    log_info "Recording initial GPU state..."
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
        --format=csv > "$RESULTS_DIR/${BENCHMARK_NAME}_gpu_info.txt"
    nvidia-smi >> "$RESULTS_DIR/${BENCHMARK_NAME}_gpu_info.txt"

    # Step 6: Start GPU monitoring (for deep mode, this also launches the server)
    log_info "Starting $PROFILE_LEVEL level GPU monitoring..."
    MONITOR_FILE="$RESULTS_DIR/${BENCHMARK_NAME}_gpu_monitor.txt"

    if [[ "$PROFILE_LEVEL" == "deep" ]]; then
        MONITOR_FILE="$RESULTS_DIR/${BENCHMARK_NAME}_nsys"
    fi

    MONITOR_PID=$(start_monitoring "$MONITOR_FILE")
    log_info "Monitoring PID: $MONITOR_PID"

    # Give monitoring a moment to start (deep mode waits inside start_deep_profiling)
    if [[ "$PROFILE_LEVEL" == "basic" ]]; then
        sleep 1
    fi

    # Step 7: Run the benchmark
    echo ""
    log_info "Running benchmark ($N_PROMPTS prompts)..."
    echo "=============================================="

    BENCH_OUTPUT="$RESULTS_DIR/${BENCHMARK_NAME}_bench.txt"
    local script_dir
    script_dir="$(dirname "$0")"

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

    # Step 8: Stop GPU monitoring
    log_info "Stopping GPU monitoring..."
    stop_monitoring "$MONITOR_PID"

    # Step 9: Generate summary report
    log_info "Generating summary report..."
    generate_summary "$MONITOR_FILE" "$RESULTS_DIR/${BENCHMARK_NAME}_summary.txt"

    # Step 10: Move plots if they exist
    if [[ -f "prompt_time.png" ]]; then
        mv prompt_time.png "$RESULTS_DIR/${BENCHMARK_NAME}_prompt_time.png"
        log_info "Saved: ${BENCHMARK_NAME}_prompt_time.png"
    fi
    if [[ -f "gen_rate.png" ]]; then
        mv gen_rate.png "$RESULTS_DIR/${BENCHMARK_NAME}_gen_rate.png"
        log_info "Saved: ${BENCHMARK_NAME}_gen_rate.png"
    fi

    # Step 11: Print summary
    echo ""
    log_info "========== BENCHMARK COMPLETE =========="
    echo ""
    cat "$RESULTS_DIR/${BENCHMARK_NAME}_summary.txt"
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
    echo "  - ${BENCHMARK_NAME}_bench.txt       (benchmark output)"
    if [[ "$PROFILE_LEVEL" != "deep" ]]; then
        echo "  - ${BENCHMARK_NAME}_gpu_monitor.txt (GPU monitoring data)"
    fi
    echo "  - ${BENCHMARK_NAME}_summary.txt     (summary statistics)"
    echo "  - ${BENCHMARK_NAME}_gpu_info.txt    (GPU information)"
    if [[ -f "$RESULTS_DIR/${BENCHMARK_NAME}_prompt_time.png" ]]; then
        echo "  - ${BENCHMARK_NAME}_prompt_time.png"
    fi
    if [[ -f "$RESULTS_DIR/${BENCHMARK_NAME}_gen_rate.png" ]]; then
        echo "  - ${BENCHMARK_NAME}_gen_rate.png"
    fi
    if [[ "$PROFILE_LEVEL" == "deep" ]]; then
        echo ""
        echo "Nsight report will be at: $RESULTS_DIR/<NSYS_OUTPUT>.nsys-rep"
        echo "  (Stop container to finalize: docker compose -f docker-compose.inference.yaml stop llama-gpu)"
    fi
    echo ""

    if [[ "$BENCH_EXIT_CODE" -ne 0 ]]; then
        log_warn "Benchmark exited with code $BENCH_EXIT_CODE"
        exit "$BENCH_EXIT_CODE"
    fi

    log_info "Done"
}

main "$@"
