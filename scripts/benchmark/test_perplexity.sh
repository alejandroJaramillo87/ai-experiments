#!/bin/bash
# test_perplexity.sh - Run perplexity tests on docker containers
#
# This script:
# 1. Downloads Wikitext-2 dataset if missing
# 2. Runs llama-perplexity inside the container via docker exec
# 3. Saves results and extracts final PPL value
#
# Usage:
#   ./scripts/test_perplexity.sh [container_name] [test_name]
#
# Examples:
#   ./scripts/test_perplexity.sh llama-hugepages hugepages-on
#   ./scripts/test_perplexity.sh llama-base baseline
#   ./scripts/test_perplexity.sh llama-zen5 aocl-blis
#
# Note: llama-perplexity runs inside the container, so container's
# LD_PRELOAD and optimizations apply. Server doesn't need to be running.

set -euo pipefail

# Configuration
CONTAINER_NAME="${1:-llama-hugepages}"
TEST_NAME="${2:-perplexity}"
RESULTS_DIR="perplexity_results"
DATASET_DIR="datasets"
WIKITEXT_FILE="$DATASET_DIR/wikitext-2-raw/wiki.test.raw"

# Thread configuration (match container defaults)
THREADS="${THREADS:-8}"

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

# Step 1: Download Wikitext-2 dataset if missing
if [[ ! -f "$WIKITEXT_FILE" ]]; then
    log_info "Downloading Wikitext-2 dataset..."
    mkdir -p "$DATASET_DIR"
    pushd "$DATASET_DIR" > /dev/null

    if ! wget -q https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip; then
        log_error "Failed to download Wikitext-2 dataset"
        exit 1
    fi

    unzip -q wikitext-2-raw-v1.zip
    rm wikitext-2-raw-v1.zip
    popd > /dev/null
    log_info "Dataset downloaded to $DATASET_DIR/wikitext-2-raw/"
fi

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
    log_error "Could not determine MODEL_PATH from container environment"
    exit 1
fi
log_info "Model: $MODEL_PATH"

# Step 4: Verify llama-perplexity exists in container
log_info "Checking for llama-perplexity binary..."
if ! docker exec "$CONTAINER_NAME" test -x /app/llama-perplexity 2>/dev/null; then
    log_error "llama-perplexity not found in container"
    log_info "Rebuild container with llama-perplexity included"
    exit 1
fi

# Step 5: Verify dataset is accessible in container
log_info "Checking dataset accessibility..."
if ! docker exec "$CONTAINER_NAME" test -f /app/datasets/wikitext-2-raw/wiki.test.raw 2>/dev/null; then
    log_error "Dataset not accessible in container at /app/datasets/wikitext-2-raw/wiki.test.raw"
    log_info "Ensure datasets volume is mounted in docker-compose.yaml"
    exit 1
fi

# Step 6: Run perplexity test
OUTPUT_FILE="$RESULTS_DIR/${TEST_NAME}_perplexity.txt"
log_info "Running perplexity test (this may take a while)..."
log_info "Output: $OUTPUT_FILE"

echo "=== Perplexity Test ===" > "$OUTPUT_FILE"
echo "Container: $CONTAINER_NAME" >> "$OUTPUT_FILE"
echo "Model: $MODEL_PATH" >> "$OUTPUT_FILE"
echo "Threads: $THREADS" >> "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Run the test
docker exec "$CONTAINER_NAME" /app/llama-perplexity \
    -m "$MODEL_PATH" \
    -f /app/datasets/wikitext-2-raw/wiki.test.raw \
    -t "$THREADS" \
    2>&1 | tee -a "$OUTPUT_FILE"

# Step 7: Extract and display final PPL
echo ""
log_info "========== TEST COMPLETE =========="
echo ""

# Extract final perplexity value
if grep -q "Final estimate:" "$OUTPUT_FILE"; then
    FINAL_PPL=$(grep "Final estimate:" "$OUTPUT_FILE")
    log_info "Result: $FINAL_PPL"
else
    log_warn "Could not find 'Final estimate:' in output"
    log_info "Check $OUTPUT_FILE for details"
fi

echo ""
echo "Results saved to: $OUTPUT_FILE"
echo ""
log_info "Done!"
