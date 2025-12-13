#!/bin/bash
# test_perplexity.sh - Run perplexity tests on Docker containers
#
# Runs llama-perplexity inside a Docker container using the Wikitext-2 dataset.
# Downloads the dataset automatically if not present.
#
# Usage:
#   ./scripts/benchmark/test_perplexity.sh [OPTIONS]
#
# Options:
#   -c, --container NAME    Container name (default: llama-cpu)
#   -n, --name NAME         Test name for output files (default: perplexity)
#   -t, --threads N         Number of threads (default: 8)
#   -s, --ctx-size N        Context size for perplexity calculation (default: 512)
#   -h, --help              Show help message
#
# Examples:
#   ./scripts/benchmark/test_perplexity.sh -n baseline
#   ./scripts/benchmark/test_perplexity.sh -c llama-hugepages -n hugepages-ppl
#   ./scripts/benchmark/test_perplexity.sh -t 16 -s 1024 -n high-ctx
#
# Note: llama-perplexity runs inside the container, so container's
# LD_PRELOAD and optimizations apply. Server doesn't need to be running.

set -euo pipefail

# === Configuration ===
CONTAINER_NAME="llama-cpu"
TEST_NAME="perplexity"
THREADS=8
CTX_SIZE=512
RESULTS_DIR="perplexity_results"
DATASET_DIR="datasets"
WIKITEXT_FILE="$DATASET_DIR/wikitext-2-raw/wiki.test.raw"

# Model path (extracted from container)
MODEL_PATH=""

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
test_perplexity.sh - Run perplexity tests on Docker containers

Usage:
  ./scripts/benchmark/test_perplexity.sh [OPTIONS]

Options:
  -c, --container NAME    Container name (default: llama-cpu)
  -n, --name NAME         Test name for output files (default: perplexity)
  -t, --threads N         Number of threads (default: 8)
  -s, --ctx-size N        Context size for perplexity calculation (default: 512)
  -h, --help              Show this help message

Description:
  Runs llama-perplexity inside a Docker container using the Wikitext-2 dataset.
  Downloads the dataset automatically if not present.

  The test runs inside the container, so container's LD_PRELOAD and other
  optimizations apply. The server does not need to be running.

Examples:
  # Default perplexity test
  ./scripts/benchmark/test_perplexity.sh -n baseline

  # Test with hugepages container
  ./scripts/benchmark/test_perplexity.sh -c llama-hugepages -n hugepages-ppl

  # Custom threads and context size
  ./scripts/benchmark/test_perplexity.sh -t 16 -s 1024 -n high-ctx

  # Quick test with fewer threads
  ./scripts/benchmark/test_perplexity.sh -t 4 -n quick-test

Output:
  Results are saved to perplexity_results/ with the test name prefix:
    {name}_perplexity.txt  - Raw llama-perplexity output
    {name}_summary.txt     - Parsed summary with PPL value

Environment Variables:
  NO_COLOR    Disable colored output
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
                TEST_NAME="$2"
                shift 2
                ;;
            -t|--threads)
                THREADS="$2"
                shift 2
                ;;
            -s|--ctx-size)
                CTX_SIZE="$2"
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

# === Dataset Management ===
ensure_dataset() {
    if [[ -f "$WIKITEXT_FILE" ]]; then
        log_info "Dataset found: $WIKITEXT_FILE"
        return 0
    fi

    log_info "Downloading Wikitext-2 dataset..."
    mkdir -p "$DATASET_DIR"
    pushd "$DATASET_DIR" > /dev/null

    if ! wget -q https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip; then
        popd > /dev/null
        die "Failed to download Wikitext-2 dataset"
    fi

    unzip -q wikitext-2-raw-v1.zip
    rm wikitext-2-raw-v1.zip
    popd > /dev/null

    log_info "Dataset downloaded to $DATASET_DIR/wikitext-2-raw/"
}

# === Summary Generation ===
generate_summary() {
    local results_file="$1"
    local output_file="$2"

    {
        echo "=== Perplexity Test Summary ==="
        echo "Test Name: $TEST_NAME"
        echo "Container: $CONTAINER_NAME"
        echo "Model: $MODEL_PATH"
        echo "Threads: $THREADS"
        echo "Context Size: $CTX_SIZE"
        echo "Date: $(date -Iseconds)"
        echo ""

        # Extract final perplexity
        echo "=== Result ==="
        if grep -q "Final estimate:" "$results_file"; then
            grep "Final estimate:" "$results_file" | sed 's/^/  /'
        elif grep -q "perplexity" "$results_file"; then
            # Alternative format
            grep -i "perplexity" "$results_file" | tail -1 | sed 's/^/  /'
        else
            echo "  Final PPL: Not found (check raw output)"
        fi
    } > "$output_file"
}

# === Main ===
main() {
    parse_args "$@"

    log_info "Perplexity Test"
    log_info "  Container: $CONTAINER_NAME"
    log_info "  Test Name: $TEST_NAME"
    log_info "  Threads: $THREADS"
    log_info "  Context: $CTX_SIZE"
    echo ""

    mkdir -p "$RESULTS_DIR"

    # Step 1: Ensure dataset exists
    ensure_dataset

    # Step 2: Verify container exists
    log_info "Checking container '$CONTAINER_NAME'..."
    if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_error "Container '$CONTAINER_NAME' does not exist"
        log_info "Available containers:"
        docker ps -a --format '  - {{.Names}} ({{.Status}})'
        exit 1
    fi

    # Step 3: Get MODEL_PATH from container environment
    log_info "Getting model path from container..."
    MODEL_PATH=$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER_NAME" | grep "^MODEL_PATH=" | cut -d'=' -f2)

    if [[ -z "$MODEL_PATH" ]]; then
        die "Could not determine MODEL_PATH from container environment"
    fi
    log_info "Model: $MODEL_PATH"

    # Step 4: Verify llama-perplexity exists in container
    log_info "Checking for llama-perplexity binary..."
    if ! docker exec "$CONTAINER_NAME" test -x /app/llama-perplexity 2>/dev/null; then
        die "llama-perplexity not found in container. Rebuild container with llama-perplexity included."
    fi

    # Step 5: Verify dataset is accessible in container
    log_info "Checking dataset accessibility in container..."
    if ! docker exec "$CONTAINER_NAME" test -f /app/datasets/wikitext-2-raw/wiki.test.raw 2>/dev/null; then
        die "Dataset not accessible in container at /app/datasets/wikitext-2-raw/wiki.test.raw. Ensure datasets volume is mounted."
    fi

    # Step 6: Run perplexity test
    OUTPUT_FILE="$RESULTS_DIR/${TEST_NAME}_perplexity.txt"
    SUMMARY_FILE="$RESULTS_DIR/${TEST_NAME}_summary.txt"

    log_info "Running perplexity test (this may take a while)..."
    echo ""
    echo "=============================================="

    {
        echo "=== Perplexity Test ==="
        echo "Container: $CONTAINER_NAME"
        echo "Model: $MODEL_PATH"
        echo "Threads: $THREADS"
        echo "Context Size: $CTX_SIZE"
        echo "Date: $(date)"
        echo ""
    } > "$OUTPUT_FILE"

    # Run the test
    docker exec "$CONTAINER_NAME" /app/llama-perplexity \
        -m "$MODEL_PATH" \
        -f /app/datasets/wikitext-2-raw/wiki.test.raw \
        -t "$THREADS" \
        -c "$CTX_SIZE" \
        2>&1 | tee -a "$OUTPUT_FILE"

    echo "=============================================="
    echo ""

    # Step 7: Generate summary
    log_info "Generating summary..."
    generate_summary "$OUTPUT_FILE" "$SUMMARY_FILE"

    # Step 8: Print results
    log_info "========== TEST COMPLETE =========="
    echo ""
    cat "$SUMMARY_FILE"
    echo ""
    echo "Results saved to: $RESULTS_DIR/"
    echo "  - ${TEST_NAME}_perplexity.txt  (raw output)"
    echo "  - ${TEST_NAME}_summary.txt     (summary)"
    echo ""

    log_info "Done"
}

main "$@"
