#!/bin/bash
# Build script for zen5_optimizer

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Building zen5_optimizer ==="
echo

# Change to project root
cd "$PROJECT_ROOT"

# Create build directory
echo "Creating build directory..."
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14

# Build
echo "Building library..."
make -j$(nproc) VERBOSE=1

# Check build success
if [ -f "libzen5_optimizer.so" ]; then
    echo
    echo "Build: OK"
    echo "  Library: ${PROJECT_ROOT}/build/libzen5_optimizer.so"
    echo
    echo "Usage:"
    echo "  LD_PRELOAD=${PROJECT_ROOT}/build/libzen5_optimizer.so <command>"
else
    echo "Build: ERROR (library not found)"
    exit 1
fi