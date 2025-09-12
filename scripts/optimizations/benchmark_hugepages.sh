#!/bin/bash

# Benchmark script for testing different huge page configurations
# Tests: No huge pages, 2MB huge pages, 1GB huge pages

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="llama-cpu-0"
API_URL="http://localhost:8001/v1/chat/completions"
COMPOSE_FILE="docker-compose.yaml"
NUM_RUNS=5

# Test prompt - medium complexity for consistent timing
TEST_PROMPT='{
  "messages": [
    {
      "role": "user",
      "content": "Write a Python function that calculates the nth Fibonacci number using recursion with memoization"
    }
  ],
  "max_tokens": 150,
  "temperature": 0.1
}'

# Function to wait for container to be ready
wait_for_container() {
    echo -n "Waiting for container to be ready..."
    for i in {1..60}; do
        if curl -s -f -X GET "${API_URL%/*}/health" > /dev/null 2>&1; then
            echo -e " ${GREEN}Ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    echo -e " ${RED}Timeout!${NC}"
    return 1
}

# Function to run benchmark
run_benchmark() {
    local config_name=$1
    local times=()
    
    echo -e "\n${YELLOW}Testing: $config_name${NC}"
    echo "----------------------------------------"
    
    # Warm-up run (not counted)
    echo "Warm-up run..."
    curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -d "$TEST_PROMPT" > /dev/null 2>&1
    
    # Actual benchmark runs
    echo "Running $NUM_RUNS benchmark iterations..."
    for i in $(seq 1 $NUM_RUNS); do
        echo -n "  Run $i: "
        
        # Time the request
        start_time=$(date +%s.%N)
        
        response=$(curl -s -X POST "$API_URL" \
            -H "Content-Type: application/json" \
            -d "$TEST_PROMPT" 2>/dev/null)
        
        end_time=$(date +%s.%N)
        
        # Check if request was successful
        if echo "$response" | grep -q "content"; then
            elapsed=$(echo "$end_time - $start_time" | bc)
            printf "%.3f seconds\n" $elapsed
            times+=($elapsed)
        else
            echo -e "${RED}Failed!${NC}"
            echo "Response: $response"
        fi
    done
    
    # Calculate average
    if [ ${#times[@]} -gt 0 ]; then
        sum=0
        for time in "${times[@]}"; do
            sum=$(echo "$sum + $time" | bc)
        done
        avg=$(echo "scale=3; $sum / ${#times[@]}" | bc)
        echo -e "${GREEN}Average time: $avg seconds${NC}"
    fi
}

# Function to check memory stats
check_memory() {
    echo -e "\n${YELLOW}Memory Statistics:${NC}"
    echo "Huge Pages:"
    cat /proc/meminfo | grep "^HugePages" | head -3
    echo "Anonymous Huge Pages:"
    cat /proc/meminfo | grep "^AnonHugePages"
    
    if docker ps | grep -q "$CONTAINER_NAME"; then
        echo "Container Memory:"
        docker stats --no-stream "$CONTAINER_NAME" | tail -1
    fi
}

# Main benchmark execution
echo "======================================"
echo "    Huge Pages Benchmark Suite"
echo "======================================"
echo "Date: $(date)"
echo "Container: $CONTAINER_NAME"
echo "Runs per config: $NUM_RUNS"

# Save original MODEL_PATH
ORIGINAL_MODEL_PATH=$(grep "MODEL_PATH=" "$COMPOSE_FILE" | head -1 | cut -d'=' -f2)
echo "Original MODEL_PATH: $ORIGINAL_MODEL_PATH"

# Test 1: No Huge Pages (regular filesystem)
echo -e "\n${GREEN}=== Configuration 1: No Huge Pages ===${NC}"
echo "Using regular filesystem path..."

# Update docker-compose to use regular path
sed -i "s|MODEL_PATH=/hugepages/|MODEL_PATH=/app/models/gguf/Qwen3-Coder-30B-A3B-Instruct-GGUF/|g" "$COMPOSE_FILE"

# Restart container
docker-compose down 2>/dev/null || true
docker-compose up -d "$CONTAINER_NAME"
wait_for_container

check_memory
run_benchmark "No Huge Pages"

# Test 2: 2MB Huge Pages (hugetlbfs)
echo -e "\n${GREEN}=== Configuration 2: 2MB Huge Pages ===${NC}"
echo "Using hugetlbfs with 2MB pages..."

# Update docker-compose to use hugepages path
sed -i "s|MODEL_PATH=/app/models/gguf/Qwen3-Coder-30B-A3B-Instruct-GGUF/|MODEL_PATH=/hugepages/|g" "$COMPOSE_FILE"

# Restart container
docker-compose down
docker-compose up -d "$CONTAINER_NAME"
wait_for_container

check_memory
run_benchmark "2MB Huge Pages"

# Test 3: 1GB Huge Pages (if available)
echo -e "\n${GREEN}=== Configuration 3: 1GB Huge Pages ===${NC}"

# Check if 1GB huge pages are configured
if grep -q "Hugepagesize:    1048576 kB" /proc/meminfo; then
    echo "1GB huge pages detected!"
    
    # Check if 1GB mount exists
    if mount | grep -q "pagesize=1G"; then
        echo "1GB hugetlbfs mount found"
        
        # Would need to modify docker-compose to use 1GB mount
        # This is system-specific configuration
        echo -e "${YELLOW}Note: 1GB testing requires manual configuration${NC}"
        echo "1. Copy model to /mnt/hugepages-1G/"
        echo "2. Update docker-compose to mount /mnt/hugepages-1G"
        echo "3. Re-run this script"
    else
        echo -e "${YELLOW}1GB huge pages configured but not mounted${NC}"
        echo "To mount: sudo mount -t hugetlbfs -o pagesize=1G none /mnt/hugepages-1G"
    fi
else
    echo -e "${YELLOW}1GB huge pages not configured${NC}"
    echo "To enable:"
    echo "1. Add to /etc/default/grub: hugepagesz=1G hugepages=32"
    echo "2. Run: sudo update-grub && sudo reboot"
fi

# Restore original configuration
echo -e "\n${GREEN}=== Restoring Original Configuration ===${NC}"
if [[ "$ORIGINAL_MODEL_PATH" == */hugepages/* ]]; then
    sed -i "s|MODEL_PATH=/app/models/gguf/Qwen3-Coder-30B-A3B-Instruct-GGUF/|MODEL_PATH=/hugepages/|g" "$COMPOSE_FILE"
else
    sed -i "s|MODEL_PATH=/hugepages/|MODEL_PATH=/app/models/gguf/Qwen3-Coder-30B-A3B-Instruct-GGUF/|g" "$COMPOSE_FILE"
fi

docker-compose down
docker-compose up -d "$CONTAINER_NAME"

echo -e "\n${GREEN}Benchmark complete!${NC}"
echo "======================================"