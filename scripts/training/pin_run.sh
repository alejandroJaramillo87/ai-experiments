#!/bin/bash
#
# pin_run.sh — run a command pinned to one CCD of the 9950X (Phase 0 D9).
#
# Host-side taskset wrapper for jobs that cannot host training/runtime/affinity.py
# (arbitrary commands, container invocations). This is CCD/L3 pinning, not
# NUMA pinning: the 9950X is one NUMA node with two CCDs (physical cores 0-7
# and 8-15; SMT siblings 16-23 and 24-31 respectively).
#
# Usage: pin_run.sh <cpu-list> <command> [args...]
# Example: pin_run.sh 0-7,16-23 docker compose -f docker-compose.training.yaml run --rm training ...

set -euo pipefail

RED='\033[0;31m' CYAN='\033[0;36m' NC='\033[0m'
if [[ -n "${NO_COLOR:-}" ]]; then
    RED='' CYAN='' NC=''
fi

print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }

if [[ $# -lt 2 ]]; then
    print_error "Usage: $0 <cpu-list> <command> [args...]"
    echo "  Example: $0 0-7,16-23 python -m training train --recipe ..."
    exit 2
fi

CPUS="$1"
shift

if ! command -v taskset >/dev/null; then
    print_error "taskset not found (util-linux)"
    echo "  Solution: install util-linux or run inside the training container"
    exit 1
fi

print_info "CPU affinity: $CPUS"
exec taskset -c "$CPUS" "$@"
