#!/bin/bash

# storage_check.sh
# Comprehensive storage analysis for AI workstation with dual SSDs
# Tracks models, datasets, Docker, caches, and identifies problem areas

echo "==================================================="
echo " Storage Analysis Report"
echo "==================================================="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Helper function for human-readable sizes
human_size() {
    numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1}B"
}

# Function to get directory size safely
get_size() {
    local path="$1"
    if [ -d "$path" ] || [ -f "$path" ]; then
        du -sb "$path" 2>/dev/null | awk '{print $1}'
    else
        echo "0"
    fi
}

# Initialize warning counters
WARNING_COUNT=0
CRITICAL_COUNT=0

# --- 1. Overall Storage Summary ---
echo "--- Overall Storage Summary ---"
echo

# Check all mounted filesystems
df -h | grep -E "^/dev/" | while read -r line; do
    DEVICE=$(echo "$line" | awk '{print $1}')
    SIZE=$(echo "$line" | awk '{print $2}')
    USED=$(echo "$line" | awk '{print $3}')
    AVAIL=$(echo "$line" | awk '{print $4}')
    USE_PERCENT=$(echo "$line" | awk '{print $5}' | sed 's/%//')
    MOUNT=$(echo "$line" | awk '{print $6}')

    # Determine drive type
    if [ "$MOUNT" = "/" ]; then
        LABEL="OS SSD"
    elif [ "$MOUNT" = "/mnt/ai-data" ]; then
        LABEL="Data SSD"
    else
        LABEL="Other"
    fi

    # Status indicator
    if [ "$USE_PERCENT" -ge 90 ]; then
        STATUS="✗"
        ((CRITICAL_COUNT++))
    elif [ "$USE_PERCENT" -ge 80 ]; then
        STATUS="⚠"
        ((WARNING_COUNT++))
    else
        STATUS="✓"
    fi

    echo "$STATUS $LABEL ($MOUNT): $USED / $SIZE ($USE_PERCENT% used, $AVAIL free)"
done

echo

# --- 2. AI Data SSD Breakdown ---
if [ -d "/mnt/ai-data" ]; then
    echo "--- AI Data SSD Breakdown (/mnt/ai-data) ---"
    echo

    # Models directory
    echo "Models:"
    for model_type in hf gguf onnx safetensors quantized ollama; do
        MODEL_PATH="/mnt/ai-data/models/$model_type"
        if [ -d "$MODEL_PATH" ]; then
            SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | awk '{print $1}')
            COUNT=$(find "$MODEL_PATH" -type f 2>/dev/null | wc -l)
            echo "  $model_type: $SIZE ($COUNT files)"
        fi
    done

    echo
    echo "Datasets:"
    DATASET_TOTAL=$(du -sh /mnt/ai-data/datasets 2>/dev/null | awk '{print $1}')
    echo "  Total: $DATASET_TOTAL"
    for dataset_type in commoncrawl audio vision parquet mmap preprocessed; do
        DATASET_PATH="/mnt/ai-data/datasets/$dataset_type"
        if [ -d "$DATASET_PATH" ] && [ "$(ls -A "$DATASET_PATH" 2>/dev/null)" ]; then
            SIZE=$(du -sh "$DATASET_PATH" 2>/dev/null | awk '{print $1}')
            echo "    $dataset_type: $SIZE"
        fi
    done

    echo
    echo "Caches:"
    for cache_type in huggingface torch pip webui; do
        CACHE_PATH="/mnt/ai-data/cache/$cache_type"
        if [ -d "$CACHE_PATH" ]; then
            SIZE=$(du -sh "$CACHE_PATH" 2>/dev/null | awk '{print $1}')
            echo "  $cache_type: $SIZE"
        fi
    done

    echo
    echo "Other Directories:"
    for dir in embeddings workspace logs; do
        DIR_PATH="/mnt/ai-data/$dir"
        if [ -d "$DIR_PATH" ]; then
            SIZE=$(du -sh "$DIR_PATH" 2>/dev/null | awk '{print $1}')
            echo "  $dir: $SIZE"
        fi
    done

    echo
    echo "Model Size Distribution:"
    HUGE_COUNT=$(find /mnt/ai-data/models -type f -size +20G 2>/dev/null | wc -l)
    LARGE_COUNT=$(find /mnt/ai-data/models -type f -size +10G -size -20G 2>/dev/null | wc -l)
    MEDIUM_COUNT=$(find /mnt/ai-data/models -type f -size +1G -size -10G 2>/dev/null | wc -l)
    echo "  >20GB: $HUGE_COUNT models"
    echo "  10-20GB: $LARGE_COUNT models"
    echo "  1-10GB: $MEDIUM_COUNT models"

    echo
    echo "Top 10 Largest Files in /mnt/ai-data:"
    find /mnt/ai-data -type f -size +100M -exec du -h {} \; 2>/dev/null | sort -rh | head -10 | while read -r size file; do
        BASENAME=$(basename "$file")
        # Add last access time
        LAST_ACCESS=$(stat -c %x "$file" 2>/dev/null | cut -d' ' -f1)
        echo "  $size $BASENAME (accessed: $LAST_ACCESS)"
    done

    echo
    echo "Checking for potential duplicate models (same size):"
    # Find files >1GB and group by size
    DUPLICATES_FOUND=false
    find /mnt/ai-data/models -type f -size +1G -exec stat -c "%s %n" {} \; 2>/dev/null | \
        sort -n | awk '{size=$1; $1=""; files[size]=files[size] $0 "\n"} END {for (s in files) if (split(files[s], a, "\n") > 2) print s, files[s]}' | \
        while read -r line; do
            if [ -n "$line" ]; then
                DUPLICATES_FOUND=true
                SIZE_BYTES=$(echo "$line" | awk '{print $1}')
                SIZE_HUMAN=$(numfmt --to=iec-i --suffix=B "$SIZE_BYTES" 2>/dev/null || echo "${SIZE_BYTES}B")
                echo "  ⚠ Multiple files with size $SIZE_HUMAN - possible duplicates"
                break
            fi
        done

    if [ "$DUPLICATES_FOUND" = false ]; then
        echo "  ✓ No obvious duplicates found"
    fi
else
    echo "⚠ /mnt/ai-data not found"
    ((WARNING_COUNT++))
fi

echo

# --- 3. Docker Storage Analysis ---
echo "--- Docker Storage Analysis ---"
echo

if command -v docker &> /dev/null; then
    # Docker system df provides a nice summary
    if docker system df 2>/dev/null | grep -q "Images"; then
        docker system df 2>/dev/null | grep -E "Images|Containers|Local Volumes|Build Cache" | while read -r line; do
            TYPE=$(echo "$line" | awk '{print $1}')
            TOTAL=$(echo "$line" | awk '{print $2}')
            ACTIVE=$(echo "$line" | awk '{print $3}')
            SIZE=$(echo "$line" | awk '{print $4 " " $5}')
            RECLAIMABLE=$(echo "$line" | awk '{print $6 " " $7}')

            echo "$TYPE: $SIZE (Reclaimable: ${RECLAIMABLE:-0B})"
        done

        # Check for dangling images
        DANGLING_COUNT=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l)
        if [ "$DANGLING_COUNT" -gt 0 ]; then
            echo "⚠ Dangling images: $DANGLING_COUNT"
            ((WARNING_COUNT++))
        fi

        # Docker root directory size
        DOCKER_ROOT="/var/lib/docker"
        if [ -d "$DOCKER_ROOT" ]; then
            DOCKER_SIZE=$(sudo du -sh "$DOCKER_ROOT" 2>/dev/null | awk '{print $1}')
            echo "Docker root directory: $DOCKER_SIZE"
        fi
    else
        echo "Unable to get Docker storage info (may need sudo)"
    fi
else
    echo "Docker not installed"
fi

echo

# --- 4. System Caches ---
echo "--- System Caches & Temporary Files ---"
echo

# APT cache
APT_CACHE="/var/cache/apt"
if [ -d "$APT_CACHE" ]; then
    APT_SIZE=$(sudo du -sh "$APT_CACHE" 2>/dev/null | awk '{print $1}' || echo "N/A")
    echo "APT cache: $APT_SIZE"
fi

# Poetry cache
POETRY_CACHE="$HOME/.cache/pypoetry"
if [ -d "$POETRY_CACHE" ]; then
    POETRY_SIZE=$(du -sh "$POETRY_CACHE" 2>/dev/null | awk '{print $1}')
    echo "Poetry cache: $POETRY_SIZE"
fi

# Pip cache (check if it's symlinked)
PIP_CACHE="$HOME/.cache/pip"
if [ -L "$PIP_CACHE" ]; then
    LINK_TARGET=$(readlink -f "$PIP_CACHE")
    PIP_SIZE=$(du -sh "$LINK_TARGET" 2>/dev/null | awk '{print $1}')
    echo "Pip cache: $PIP_SIZE (→ $LINK_TARGET)"
elif [ -d "$PIP_CACHE" ]; then
    PIP_SIZE=$(du -sh "$PIP_CACHE" 2>/dev/null | awk '{print $1}')
    echo "Pip cache: $PIP_SIZE"
fi

# System temp directories
TMP_SIZE=$(du -sh /tmp 2>/dev/null | awk '{print $1}')
echo "/tmp: $TMP_SIZE"

VAR_TMP_SIZE=$(du -sh /var/tmp 2>/dev/null | awk '{print $1}')
echo "/var/tmp: $VAR_TMP_SIZE"

# Snap packages
if [ -d "/var/lib/snapd" ]; then
    SNAP_SIZE=$(sudo du -sh /var/lib/snapd 2>/dev/null | awk '{print $1}' || echo "N/A")
    echo "Snap packages: $SNAP_SIZE"
fi

echo

# --- 5. System Logs ---
echo "--- System Logs & Journal ---"
echo

# Systemd journal
JOURNAL_SIZE=$(journalctl --disk-usage 2>/dev/null | grep -oP '\d+\.\d+[KMGT]' || echo "N/A")
echo "Systemd journal: $JOURNAL_SIZE"

# Check for journal persistence
JOURNAL_MAX=$(grep -oP 'SystemMaxUse=\K[^\s]+' /etc/systemd/journald.conf 2>/dev/null || echo "default")
echo "  Max size: $JOURNAL_MAX"

# /var/log size
VAR_LOG_SIZE=$(sudo du -sh /var/log 2>/dev/null | awk '{print $1}' || echo "N/A")
echo "/var/log total: $VAR_LOG_SIZE"

# Find large log files
echo "Large log files (>100MB):"
sudo find /var/log -type f -size +100M -exec du -h {} \; 2>/dev/null | sort -rh | head -5 | while read -r size file; do
    echo "  ⚠ $size $(basename "$file")"
    ((WARNING_COUNT++))
done

# Old kernels
OLD_KERNELS=$(dpkg -l 'linux-*' | sed '/^ii/!d;/'"$(uname -r | sed "s/\(.*\)-\([^0-9]\+\)/\1/")"'/d;s/^[^ ]* [^ ]* \([^ ]*\).*/\1/;/[0-9]/!d' | wc -l)
if [ "$OLD_KERNELS" -gt 2 ]; then
    echo "⚠ Old kernels installed: $OLD_KERNELS (can be removed)"
    ((WARNING_COUNT++))
fi

echo

# --- 6. Home Directory Analysis ---
echo "--- Home Directory Analysis ---"
echo

echo "Top 10 directories in $HOME (excluding symlinks):"
find "$HOME" -maxdepth 1 -type d ! -name "." -exec du -sh {} \; 2>/dev/null | sort -rh | head -10 | while read -r size dir; do
    DIR_NAME=$(basename "$dir")
    # Check if it's a cache that should be on data SSD
    if [[ "$DIR_NAME" == ".cache" ]]; then
        if [ ! -L "$HOME/.cache" ]; then
            echo "  ⚠ $size $DIR_NAME (should be symlinked to /mnt/ai-data/cache)"
            ((WARNING_COUNT++))
        else
            LINK_TARGET=$(readlink -f "$HOME/.cache")
            echo "  $size $DIR_NAME → $LINK_TARGET"
        fi
    else
        echo "  $size $DIR_NAME"
    fi
done

echo
echo "Python Environments:"
# Check for pyenv
if [ -d "$HOME/.pyenv" ]; then
    PYENV_SIZE=$(du -sh "$HOME/.pyenv" 2>/dev/null | awk '{print $1}')
    PYENV_VERSIONS=$(ls "$HOME/.pyenv/versions" 2>/dev/null | wc -l)
    echo "  .pyenv: $PYENV_SIZE ($PYENV_VERSIONS versions)"
fi
# Check for Poetry environments
if [ -d "$HOME/.cache/pypoetry/virtualenvs" ]; then
    POETRY_ENV_SIZE=$(du -sh "$HOME/.cache/pypoetry/virtualenvs" 2>/dev/null | awk '{print $1}')
    POETRY_ENV_COUNT=$(ls "$HOME/.cache/pypoetry/virtualenvs" 2>/dev/null | wc -l)
    echo "  Poetry envs: $POETRY_ENV_SIZE ($POETRY_ENV_COUNT environments)"
fi
# Check for conda
if [ -d "$HOME/miniconda3" ] || [ -d "$HOME/anaconda3" ]; then
    CONDA_DIR=$([ -d "$HOME/miniconda3" ] && echo "$HOME/miniconda3" || echo "$HOME/anaconda3")
    CONDA_SIZE=$(du -sh "$CONDA_DIR" 2>/dev/null | awk '{print $1}')
    echo "  Conda: $CONDA_SIZE"
fi

echo
echo "Hidden directories >1GB:"
find "$HOME" -maxdepth 1 -type d -name ".*" -exec du -sb {} \; 2>/dev/null | while read -r bytes dir; do
    if [ "$bytes" -gt 1073741824 ]; then
        SIZE=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
        DIR_NAME=$(basename "$dir")
        # Skip if already shown in Python environments
        if [[ "$DIR_NAME" != ".pyenv" ]]; then
            echo "  $DIR_NAME: $SIZE"
        fi
    fi
done

echo

# --- 7. Cleanup Suggestions ---
echo "--- Cleanup Suggestions ---"
echo

SUGGESTIONS=0

# Docker cleanup
if [ "$DANGLING_COUNT" -gt 0 ] 2>/dev/null; then
    echo "• Remove Docker dangling images:"
    echo "  docker image prune"
    ((SUGGESTIONS++))
fi

# Check Docker build cache
if command -v docker &> /dev/null; then
    BUILD_CACHE_SIZE=$(docker system df 2>/dev/null | grep "Build Cache" | awk '{print $6 " " $7}')
    if [ -n "$BUILD_CACHE_SIZE" ] && [ "$BUILD_CACHE_SIZE" != "0B" ]; then
        echo "• Clear Docker build cache ($BUILD_CACHE_SIZE reclaimable):"
        echo "  docker builder prune"
        ((SUGGESTIONS++))
    fi
fi

# APT cache cleanup
APT_BYTES=$(get_size "/var/cache/apt")
if [ "$APT_BYTES" -gt 1073741824 ]; then  # >1GB
    echo "• Clean APT cache:"
    echo "  sudo apt autoclean && sudo apt autoremove"
    ((SUGGESTIONS++))
fi

# Journal cleanup
JOURNAL_BYTES=$(journalctl --disk-usage 2>/dev/null | grep -oP '\d+' | head -1)
if [ -n "$JOURNAL_BYTES" ] && [ "$JOURNAL_BYTES" -gt 1073741824 ] 2>/dev/null; then
    echo "• Reduce journal size:"
    echo "  sudo journalctl --vacuum-time=7d"
    ((SUGGESTIONS++))
fi

# Poetry cache
POETRY_BYTES=$(get_size "$HOME/.cache/pypoetry")
if [ "$POETRY_BYTES" -gt 5368709120 ]; then  # >5GB
    echo "• Clear Poetry cache:"
    echo "  poetry cache clear --all . -n"
    ((SUGGESTIONS++))
fi

# Old kernels
if [ "$OLD_KERNELS" -gt 2 ] 2>/dev/null; then
    echo "• Remove old kernels:"
    echo "  sudo apt autoremove --purge"
    ((SUGGESTIONS++))
fi

if [ $SUGGESTIONS -eq 0 ]; then
    echo "✓ No significant cleanup opportunities found"
fi

echo

# --- 8. Summary ---
echo "==================================================="
echo " Storage Summary"
echo "==================================================="

# Count status
if [ $CRITICAL_COUNT -gt 0 ]; then
    echo "✗ Critical: $CRITICAL_COUNT disk(s) >90% full"
fi
if [ $WARNING_COUNT -gt 0 ]; then
    echo "⚠ Warnings: $WARNING_COUNT potential issues found"
fi
if [ $CRITICAL_COUNT -eq 0 ] && [ $WARNING_COUNT -eq 0 ]; then
    echo "✓ All storage checks passed"
fi

echo
echo "Report complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "==================================================="