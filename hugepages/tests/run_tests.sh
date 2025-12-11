#!/bin/bash
#
# run_tests.sh - Comprehensive test runner for llama-zen-turbo
#
# Usage:
#   ./run_tests.sh              # Run all tests
#   ./run_tests.sh unit         # Run unit tests only
#   ./run_tests.sh functional   # Run functional tests only
#   ./run_tests.sh integration  # Run integration tests only
#   ./run_tests.sh test_cpu     # Run specific test
#   ./run_tests.sh --verbose    # Verbose output
#   ./run_tests.sh --valgrind   # Run with memory checking
#   ./run_tests.sh --help       # Show help

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="../build"
LIB_NAME="libzen5_optimizer.so"
VERBOSE=0
USE_VALGRIND=0
TEST_FILTER=""
TEST_CATEGORY=""
FAILED_TESTS=()
PASSED_TESTS=()
SKIPPED_TESTS=()

# Timing
START_TIME=$(date +%s)

# Helper functions
print_header() {
    echo -e "\n${BOLD}===========================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${BOLD}===========================================${NC}\n"
}

print_ok() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_run() {
    echo -e "${BLUE}[RUN]${NC} $1"
}

print_skip() {
    echo -e "${BLUE}[SKIP]${NC} $1"
}

show_help() {
    cat << EOF
llama-zen-turbo Test Runner

USAGE:
    $0 [OPTIONS] [TEST_FILTER]

OPTIONS:
    --help, -h        Show this help message
    --verbose, -v     Verbose output
    --valgrind        Run tests with valgrind memory checking
    --quiet, -q       Minimal output
    --list            List all available tests
    --no-color        Disable colored output

CATEGORIES:
    unit             Run unit tests only
    functional       Run functional tests only
    integration      Run integration tests only
    all              Run all tests (default)

EXAMPLES:
    $0                    # Run all tests
    $0 unit              # Run unit tests only
    $0 test_cpu          # Run test containing "test_cpu"
    $0 --valgrind unit   # Run unit tests with valgrind
    $0 --verbose         # Run all tests with verbose output

EOF
    exit 0
}

list_tests() {
    print_header "Available Tests"

    echo -e "${BOLD}Unit Tests:${NC}"
    for test in unit/test_*.cpp; do
        if [ -f "$test" ]; then
            echo "  - $(basename ${test%.cpp})"
        fi
    done

    echo -e "\n${BOLD}Functional Tests:${NC}"
    for test in functional/test_*.cpp; do
        if [ -f "$test" ]; then
            echo "  - $(basename ${test%.cpp})"
        fi
    done

    echo -e "\n${BOLD}Integration Tests:${NC}"
    for test in integration/test_*.sh; do
        if [ -f "$test" ]; then
            echo "  - $(basename ${test%.sh})"
        fi
    done

    exit 0
}

# Check if library exists
check_library() {
    if [ ! -f "${BUILD_DIR}/${LIB_NAME}" ]; then
        print_fail "Library not found: ${BUILD_DIR}/${LIB_NAME}"
        print_info "Run 'make' from the project root first"
        exit 1
    fi
}

# Check if test binaries exist
check_test_binaries() {
    local missing=0

    for dir in unit functional; do
        for test_src in $dir/test_*.cpp; do
            if [ -f "$test_src" ]; then
                test_name=$(basename ${test_src%.cpp})
                if [ ! -f "${BUILD_DIR}/${test_name}" ]; then
                    print_warn "Test binary not found: ${BUILD_DIR}/${test_name}"
                    missing=1
                fi
            fi
        done
    done

    if [ $missing -eq 1 ]; then
        print_info "Run 'make test' from the project root to build test binaries"
        return 1
    fi

    return 0
}

# Run a single test
run_test() {
    local test_path=$1
    local test_name=$(basename $test_path)

    # Check if test matches filter
    if [ ! -z "$TEST_FILTER" ]; then
        if [[ ! "$test_name" == *"$TEST_FILTER"* ]]; then
            return 0
        fi
    fi

    print_run "$test_name"

    # Prepare command
    local cmd="LD_LIBRARY_PATH=${BUILD_DIR} ${test_path}"
    if [ $USE_VALGRIND -eq 1 ]; then
        cmd="LD_LIBRARY_PATH=${BUILD_DIR} valgrind --leak-check=summary --error-exitcode=1 ${test_path}"
        [ $VERBOSE -eq 1 ] && cmd="LD_LIBRARY_PATH=${BUILD_DIR} valgrind --leak-check=full --show-leak-kinds=all ${test_path}"
    fi

    # Run test
    local test_output
    local test_start=$(date +%s%N)

    if [ $VERBOSE -eq 1 ]; then
        eval $cmd
        local result=$?
    else
        test_output=$(eval $cmd 2>&1)
        local result=$?
    fi

    local test_end=$(date +%s%N)
    local test_time=$(echo "scale=3; ($test_end - $test_start) / 1000000000" | bc)

    if [ $result -eq 0 ]; then
        print_ok "$test_name completed (${test_time}s)"
        PASSED_TESTS+=("$test_name")
        [ $VERBOSE -eq 0 ] && [ ! -z "$test_output" ] && echo "$test_output" | grep -E "^\[OK\]|\[PASS\]" || true
    else
        print_fail "$test_name failed (${test_time}s)"
        FAILED_TESTS+=("$test_name")
        [ $VERBOSE -eq 0 ] && [ ! -z "$test_output" ] && echo "$test_output"
    fi

    echo # Empty line after each test
    return $result
}

# Run unit tests
run_unit_tests() {
    print_header "Unit Tests"

    local failed=0
    for test_src in unit/test_*.cpp; do
        if [ -f "$test_src" ]; then
            test_name=$(basename ${test_src%.cpp})
            test_binary="${BUILD_DIR}/${test_name}"

            if [ -f "$test_binary" ]; then
                run_test "$test_binary" || failed=1
            else
                print_skip "$test_name (binary not found)"
                SKIPPED_TESTS+=("$test_name")
            fi
        fi
    done

    return $failed
}

# Run functional tests
run_functional_tests() {
    print_header "Functional Tests"

    local failed=0
    for test_src in functional/test_*.cpp; do
        if [ -f "$test_src" ]; then
            test_name=$(basename ${test_src%.cpp})
            test_binary="${BUILD_DIR}/${test_name}"

            if [ -f "$test_binary" ]; then
                run_test "$test_binary" || failed=1
            else
                print_skip "$test_name (binary not found)"
                SKIPPED_TESTS+=("$test_name")
            fi
        fi
    done

    return $failed
}

# Run integration tests
run_integration_tests() {
    print_header "Integration Tests"

    local failed=0
    for test_script in integration/test_*.sh; do
        if [ -f "$test_script" ] && [ -x "$test_script" ]; then
            test_name=$(basename ${test_script%.sh})

            # Check if test matches filter
            if [ ! -z "$TEST_FILTER" ]; then
                if [[ ! "$test_name" == *"$TEST_FILTER"* ]]; then
                    continue
                fi
            fi

            print_run "$test_name"

            local test_start=$(date +%s%N)

            if [ $VERBOSE -eq 1 ]; then
                $test_script
                local result=$?
            else
                test_output=$($test_script 2>&1)
                local result=$?
                [ $result -eq 0 ] && print_ok "$test_name completed" || print_fail "$test_name failed"
                [ ! -z "$test_output" ] && echo "$test_output" | grep -E "^\[OK\]|\[FAIL\]" || true
            fi

            local test_end=$(date +%s%N)
            local test_time=$(echo "scale=3; ($test_end - $test_start) / 1000000000" | bc)

            if [ $result -eq 0 ]; then
                PASSED_TESTS+=("$test_name")
            else
                FAILED_TESTS+=("$test_name")
                failed=1
            fi

            echo # Empty line after each test
        fi
    done

    return $failed
}

# Print summary
print_summary() {
    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))

    print_header "Test Summary"

    local total=$((${#PASSED_TESTS[@]} + ${#FAILED_TESTS[@]} + ${#SKIPPED_TESTS[@]}))

    echo -e "${BOLD}Results:${NC}"
    echo -e "  ${GREEN}Passed:${NC}  ${#PASSED_TESTS[@]}/$total"

    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo -e "  ${RED}Failed:${NC}  ${#FAILED_TESTS[@]}/$total"
        echo -e "\n${BOLD}Failed Tests:${NC}"
        for test in "${FAILED_TESTS[@]}"; do
            echo -e "  ${RED}✗${NC} $test"
        done
    fi

    if [ ${#SKIPPED_TESTS[@]} -gt 0 ]; then
        echo -e "  ${BLUE}Skipped:${NC} ${#SKIPPED_TESTS[@]}/$total"
    fi

    echo -e "\n${BOLD}Duration:${NC} ${DURATION}s"

    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        echo -e "\n${RED}${BOLD}TESTS FAILED${NC}"
        return 1
    else
        echo -e "\n${GREEN}${BOLD}ALL TESTS PASSED${NC}"
        return 0
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -v|--verbose)
            VERBOSE=1
            shift
            ;;
        -q|--quiet)
            VERBOSE=0
            shift
            ;;
        --valgrind)
            USE_VALGRIND=1
            shift
            ;;
        --list)
            list_tests
            ;;
        --no-color)
            RED=''
            GREEN=''
            YELLOW=''
            BLUE=''
            CYAN=''
            BOLD=''
            NC=''
            shift
            ;;
        unit|functional|integration|all)
            TEST_CATEGORY=$1
            shift
            ;;
        *)
            TEST_FILTER=$1
            shift
            ;;
    esac
done

# Default to all tests if no category specified
[ -z "$TEST_CATEGORY" ] && [ -z "$TEST_FILTER" ] && TEST_CATEGORY="all"

# Main execution
print_header "llama-zen-turbo Test Suite"

# Check prerequisites
check_library

if [ $USE_VALGRIND -eq 1 ]; then
    if ! command -v valgrind &> /dev/null; then
        print_warn "valgrind not found, running without memory checking"
        USE_VALGRIND=0
    else
        print_info "Running with valgrind memory checking"
    fi
fi

# Check test binaries
if ! check_test_binaries; then
    print_warn "Some test binaries are missing"
fi

# Run tests based on category or filter
failed=0

if [ ! -z "$TEST_FILTER" ]; then
    # Run all tests matching filter
    run_unit_tests || failed=1
    run_functional_tests || failed=1
    run_integration_tests || failed=1
elif [ "$TEST_CATEGORY" == "unit" ]; then
    run_unit_tests || failed=1
elif [ "$TEST_CATEGORY" == "functional" ]; then
    run_functional_tests || failed=1
elif [ "$TEST_CATEGORY" == "integration" ]; then
    run_integration_tests || failed=1
else
    # Run all tests
    run_unit_tests || failed=1
    run_functional_tests || failed=1
    run_integration_tests || failed=1
fi

# Print summary and exit
print_summary
exit $?