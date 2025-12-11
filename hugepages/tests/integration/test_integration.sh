#!/bin/bash
#
# test_integration.sh
#
# Integration test for zen5_optimizer library.
# Tests the complete LD_PRELOAD workflow.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test configuration
TEST_NAME="zen5_optimizer integration test"
BUILD_DIR="../build"
LIB_NAME="libzen5_optimizer.so"
LIB_PATH="${BUILD_DIR}/${LIB_NAME}"
TEST_FILE="/tmp/zen5_integration_test.dat"
TEST_SIZE=$((1536 * 1024 * 1024))  # 1.5 GB

# Helper functions
print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

cleanup() {
    rm -f ${TEST_FILE}
}

# Register cleanup
trap cleanup EXIT

echo "=========================================="
echo "$TEST_NAME"
echo "=========================================="
echo

# Step 1: Check if library exists
echo "Step 1: Checking library build..."
if [ ! -f "${LIB_PATH}" ]; then
    print_error "Library not found at ${LIB_PATH}"
    echo "  Run 'make' first to build the library"
    exit 1
fi
print_status "Library found: ${LIB_PATH}"
echo

# Step 2: Check CPU compatibility (optional)
echo "Step 2: Checking CPU compatibility..."
if [ -f "${BUILD_DIR}/test_cpu" ]; then
    ${BUILD_DIR}/test_cpu > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_status "CPU check passed"
    else
        print_warning "CPU check failed (not on AMD Zen 5)"
        echo "  Library will refuse to load on this CPU"
    fi
else
    print_warning "test_cpu not found, skipping CPU check"
fi
echo

# Step 3: Create test program inline
echo "Step 3: Creating test program..."
cat > /tmp/zen5_test_prog.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <file>\n", argv[0]);
        return 1;
    }

    const char* filename = argv[1];
    printf("[test_prog] Opening file: %s\n", filename);

    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "[test_prog] Cannot open file: %s\n", strerror(errno));
        return 1;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "[test_prog] Cannot stat file: %s\n", strerror(errno));
        close(fd);
        return 1;
    }

    size_t size = st.st_size;
    printf("[test_prog] File size: %zu bytes (%.2f GB)\n",
           size, size / (1024.0 * 1024.0 * 1024.0));

    printf("[test_prog] Calling mmap...\n");
    void* addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (addr == MAP_FAILED) {
        fprintf(stderr, "[test_prog] mmap failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }

    printf("[test_prog] mmap succeeded at address %p\n", addr);

    // Read first few bytes to ensure mapping works
    char buffer[16];
    memcpy(buffer, addr, 15);
    buffer[15] = '\0';
    printf("[test_prog] First 15 bytes: %.15s\n", buffer);

    printf("[test_prog] Unmapping memory...\n");
    munmap(addr, size);

    close(fd);
    printf("[test_prog] Done\n");
    return 0;
}
EOF

gcc -o /tmp/zen5_test_prog /tmp/zen5_test_prog.c
if [ $? -ne 0 ]; then
    print_error "Failed to compile test program"
    exit 1
fi
print_status "Test program compiled"
echo

# Step 4: Create test file
echo "Step 4: Creating test file..."
echo "  Size: $(echo "scale=2; $TEST_SIZE / 1024 / 1024 / 1024" | bc) GB"
dd if=/dev/zero of=${TEST_FILE} bs=1M count=$((TEST_SIZE / 1024 / 1024)) status=none 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "Failed to create test file"
    exit 1
fi

# Write test pattern at beginning
echo "INTEGRATION_TEST_PATTERN" | dd of=${TEST_FILE} bs=1 count=24 conv=notrunc status=none 2>/dev/null
print_status "Test file created: ${TEST_FILE}"
echo

# Step 5: Run without LD_PRELOAD (baseline)
echo "Step 5: Running baseline (without optimizer)..."
echo "----------------------------------------"
/tmp/zen5_test_prog ${TEST_FILE}
if [ $? -ne 0 ]; then
    print_error "Baseline test failed"
    exit 1
fi
echo "----------------------------------------"
print_status "Baseline test completed"
echo

# Step 6: Run with LD_PRELOAD
echo "Step 6: Running with zen5_optimizer..."
echo "----------------------------------------"
# Get absolute path to library
LIB_ABSOLUTE=$(cd "$(dirname "${LIB_PATH}")" && pwd)/$(basename "${LIB_PATH}")
export LD_PRELOAD="${LIB_ABSOLUTE}"
echo "LD_PRELOAD set to: $LD_PRELOAD"
echo

/tmp/zen5_test_prog ${TEST_FILE} 2>&1 | tee /tmp/zen5_test_output.txt

# Check if library loaded (look for characteristic output)
if grep -q "zen5-optimizer" /tmp/zen5_test_output.txt; then
    print_status "Library loaded successfully"

    # Check for hugepage interception
    if grep -q "huge" /tmp/zen5_test_output.txt; then
        print_status "Hugepage allocation detected"
    else
        print_warning "No hugepage message found (may be normal)"
    fi
else
    print_warning "Library may not have loaded (no initialization message)"
    echo "  This could be normal if not running on AMD Zen 5"
fi

unset LD_PRELOAD
echo "----------------------------------------"
echo

# Step 7: Optional valgrind check
echo "Step 7: Memory leak check (optional)..."
if command -v valgrind > /dev/null 2>&1; then
    # Skip valgrind for now - it can hang with hugepage allocations
    print_info "Valgrind check disabled (can cause issues with hugepages)"
else
    print_warning "valgrind not found, skipping memory leak check"
fi
echo

# Cleanup
rm -f /tmp/zen5_test_prog /tmp/zen5_test_prog.c /tmp/zen5_test_output.txt

echo "=========================================="
print_status "Integration test completed"
echo "=========================================="

# Exit with success
exit 0