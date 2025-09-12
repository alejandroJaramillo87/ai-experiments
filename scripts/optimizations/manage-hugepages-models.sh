#!/bin/bash
# Manage models in hugetlbfs mount for CPU containers
# This script runs on the host to copy models to/from hugepages mount

set -euo pipefail

# Configuration
HUGEPAGES_MOUNT="/mnt/models-hugepages"
MODELS_SOURCE="/mnt/ai-data/models"
LOCK_FILE="/var/run/hugepages-model.lock"
STATE_FILE="/var/run/hugepages-model.state"

# ANSI color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
        exit 1
    fi
}

# Acquire lock to prevent concurrent operations
acquire_lock() {
    exec 200>"${LOCK_FILE}"
    if ! flock -n 200; then
        error "Another instance is already running"
        exit 1
    fi
}

# Release lock
release_lock() {
    flock -u 200 2>/dev/null || true
    rm -f "${LOCK_FILE}"
}

# Trap to ensure lock is released
trap release_lock EXIT

# Check if hugepages mount exists and is mounted
check_hugepages_mount() {
    if ! mountpoint -q "${HUGEPAGES_MOUNT}"; then
        error "Hugepages mount ${HUGEPAGES_MOUNT} is not mounted"
        error "Run: mount -t hugetlbfs -o pagesize=2M,size=30G none ${HUGEPAGES_MOUNT}"
        exit 1
    fi
    
    # Check if it's actually hugetlbfs
    if ! findmnt -t hugetlbfs "${HUGEPAGES_MOUNT}" >/dev/null 2>&1; then
        error "${HUGEPAGES_MOUNT} is not a hugetlbfs filesystem"
        exit 1
    fi
}

# List available models in source directory
list_available_models() {
    echo "Available models in ${MODELS_SOURCE}:"
    echo "----------------------------------------"
    
    if [[ ! -d "${MODELS_SOURCE}" ]]; then
        error "Models source directory ${MODELS_SOURCE} does not exist"
        exit 1
    fi
    
    # Find all .gguf files
    local models_found=0
    while IFS= read -r -d '' model; do
        local rel_path="${model#${MODELS_SOURCE}/}"
        local size=$(du -h "$model" | cut -f1)
        echo "  ${rel_path} (${size})"
        ((models_found++))
    done < <(find "${MODELS_SOURCE}" -name "*.gguf" -type f -print0 | sort -z)
    
    if [[ $models_found -eq 0 ]]; then
        warning "No .gguf models found in ${MODELS_SOURCE}"
    fi
    echo
}

# List models currently in hugepages
list_loaded_models() {
    echo "Models loaded in hugepages (${HUGEPAGES_MOUNT}):"
    echo "----------------------------------------"
    
    local models_found=0
    while IFS= read -r -d '' model; do
        local rel_path="${model#${HUGEPAGES_MOUNT}/}"
        local size=$(du -h "$model" | cut -f1)
        echo "  ${rel_path} (${size})"
        ((models_found++))
    done < <(find "${HUGEPAGES_MOUNT}" -name "*.gguf" -type f -print0 2>/dev/null | sort -z)
    
    if [[ $models_found -eq 0 ]]; then
        echo "  (no models currently loaded)"
    fi
    
    # Show memory usage
    echo
    echo "Hugepages memory status:"
    grep -E "HugePages_(Total|Free|Rsvd|Surp):" /proc/meminfo | sed 's/^/  /'
    echo
}

# Get current model from state file
get_current_model() {
    if [[ -f "${STATE_FILE}" ]]; then
        cat "${STATE_FILE}"
    else
        echo "none"
    fi
}

# Set current model in state file
set_current_model() {
    echo "$1" > "${STATE_FILE}"
}

# Load a model into hugepages
load_model() {
    local model_path="$1"
    
    # Handle relative or absolute paths
    if [[ "${model_path}" = /* ]]; then
        # Absolute path
        local source_file="${model_path}"
    else
        # Relative to MODELS_SOURCE
        local source_file="${MODELS_SOURCE}/${model_path}"
    fi
    
    # Check if source exists
    if [[ ! -f "${source_file}" ]]; then
        error "Model not found: ${source_file}"
        echo "Use 'list' command to see available models"
        exit 1
    fi
    
    # Check if it's a GGUF file
    if [[ ! "${source_file}" =~ \.gguf$ ]]; then
        error "File is not a .gguf model: ${source_file}"
        exit 1
    fi
    
    # Get model size
    local model_size=$(stat -c%s "${source_file}")
    local model_size_gb=$((model_size / 1024 / 1024 / 1024))
    
    # Check available hugepages
    local free_hugepages=$(grep HugePages_Free /proc/meminfo | awk '{print $2}')
    local hugepage_size_kb=$(grep Hugepagesize /proc/meminfo | awk '{print $2}')
    local free_memory_gb=$((free_hugepages * hugepage_size_kb / 1024 / 1024))
    
    log "Model size: ${model_size_gb}GB"
    log "Available hugepages memory: ${free_memory_gb}GB"
    
    if [[ ${model_size_gb} -gt ${free_memory_gb} ]]; then
        error "Not enough free hugepages memory"
        error "Need ${model_size_gb}GB but only ${free_memory_gb}GB available"
        exit 1
    fi
    
    # Extract just the filename for destination
    local model_filename=$(basename "${source_file}")
    local dest_file="${HUGEPAGES_MOUNT}/${model_filename}"
    
    # Check if model already loaded
    if [[ -f "${dest_file}" ]]; then
        warning "Model ${model_filename} is already loaded"
        set_current_model "${model_filename}"
        echo "Set as current model"
        return 0
    fi
    
    # Clear any existing models first (optional - comment out to keep multiple)
    log "Clearing existing models from hugepages..."
    find "${HUGEPAGES_MOUNT}" -name "*.gguf" -type f -delete 2>/dev/null || true
    
    # Copy model to hugepages
    log "Loading model ${model_filename} into hugepages..."
    log "This may take a few minutes for large models..."
    
    # Use dd for better progress reporting
    dd if="${source_file}" of="${dest_file}" bs=1M status=progress 2>&1 | \
        while IFS= read -r line; do
            echo -ne "\r${line}"
        done
    echo # New line after dd progress
    
    # Verify copy
    if [[ ! -f "${dest_file}" ]]; then
        error "Failed to load model into hugepages"
        exit 1
    fi
    
    # Verify sizes match
    local dest_size=$(stat -c%s "${dest_file}")
    if [[ ${dest_size} -ne ${model_size} ]]; then
        error "Model copy verification failed - size mismatch"
        rm -f "${dest_file}"
        exit 1
    fi
    
    # Update state
    set_current_model "${model_filename}"
    
    log "Successfully loaded ${model_filename} into hugepages"
    echo "Model path for containers: /hugepages/${model_filename}"
}

# Clear all models from hugepages
clear_models() {
    log "Clearing all models from hugepages..."
    
    local count=0
    while IFS= read -r -d '' model; do
        rm -f "$model"
        ((count++))
    done < <(find "${HUGEPAGES_MOUNT}" -name "*.gguf" -type f -print0 2>/dev/null)
    
    set_current_model "none"
    
    if [[ $count -gt 0 ]]; then
        log "Removed ${count} model(s) from hugepages"
    else
        log "No models to remove"
    fi
    
    # Show freed memory
    echo "Hugepages memory after clearing:"
    grep -E "HugePages_(Total|Free):" /proc/meminfo | sed 's/^/  /'
}

# Show current status
show_status() {
    echo "Hugepages Model Management Status"
    echo "=================================="
    echo
    
    # Current model
    local current=$(get_current_model)
    echo "Current model: ${current}"
    echo
    
    # Hugepages mount status
    if mountpoint -q "${HUGEPAGES_MOUNT}"; then
        echo "Hugepages mount: ${GREEN}MOUNTED${NC} at ${HUGEPAGES_MOUNT}"
        local mount_info=$(findmnt -t hugetlbfs "${HUGEPAGES_MOUNT}" -o OPTIONS -n)
        echo "Mount options: ${mount_info}"
    else
        echo "Hugepages mount: ${RED}NOT MOUNTED${NC}"
    fi
    echo
    
    # Memory status
    echo "System hugepages configuration:"
    grep -E "HugePages_(Total|Free|Rsvd|Surp):" /proc/meminfo | sed 's/^/  /'
    grep Hugepagesize /proc/meminfo | sed 's/^/  /'
    echo
    
    # Container status (check if any are using the model)
    echo "Container status:"
    for port in 8001 8002 8003; do
        local container="llama-cpu-$((port - 8001))"
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            echo "  ${container}: ${GREEN}RUNNING${NC} (port ${port})"
        else
            echo "  ${container}: STOPPED"
        fi
    done
}

# Print usage
usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
    list                    List available models in ${MODELS_SOURCE}
    loaded                  List models currently loaded in hugepages
    load <model_path>       Load a model into hugepages
                           (path relative to ${MODELS_SOURCE} or absolute)
    clear                   Clear all models from hugepages
    status                  Show current status
    current                 Show current model name only
    help                    Show this help message

Examples:
    $0 list                                                    # List available models
    $0 load gguf/Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf    # Load model
    $0 load /mnt/ai-data/models/llama-70b.gguf               # Load with absolute path
    $0 loaded                                                 # Show loaded models
    $0 clear                                                  # Clear all models
    $0 status                                                 # Show full status

Note: This script must be run as root to manage hugepages.
EOF
}

# Main function
main() {
    # Check prerequisites
    check_root
    acquire_lock
    
    # Parse command
    case "${1:-}" in
        list)
            list_available_models
            ;;
        loaded)
            check_hugepages_mount
            list_loaded_models
            ;;
        load)
            if [[ -z "${2:-}" ]]; then
                error "Model path required"
                echo "Usage: $0 load <model_path>"
                exit 1
            fi
            check_hugepages_mount
            load_model "$2"
            ;;
        clear)
            check_hugepages_mount
            clear_models
            ;;
        status)
            show_status
            ;;
        current)
            get_current_model
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            error "Unknown command: ${1:-}"
            usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"