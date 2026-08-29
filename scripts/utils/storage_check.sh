#!/bin/bash

# storage_check.sh
# Comprehensive storage analysis for AI workstation with dual SSDs
# Tracks models, the training and interp tiers, Docker, caches, and problem areas

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Disable colors if NO_COLOR is set
[[ -n "${NO_COLOR:-}" ]] && RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''

# Print helper functions
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_skip()  { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }

echo "=== Storage Analysis Report ==="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Function to get directory size safely
get_size() {
    local path="$1"
    if [ -L "$path" ]; then
        # A cache path may be a symlink onto the data drive. -L follows it;
        # plain du would measure the link and report a few bytes. Applied only
        # to the symlink case so directory walks do not chase links inside.
        du -sbL "$path" 2>/dev/null | awk '{print $1}'
    elif [ -d "$path" ] || [ -f "$path" ]; then
        du -sb "$path" 2>/dev/null | awk '{print $1}'
    else
        echo "0"
    fi
}

# Initialize warning counters
WARNING_COUNT=0
CRITICAL_COUNT=0

# Overall Storage Summary
echo "Overall storage summary:"
echo

# Check all mounted filesystems
# Use process substitution (not a pipe) so WARNING_COUNT/CRITICAL_COUNT increments
# survive: a `... | while` loop runs in a subshell and its counter changes are lost.
while read -r line; do
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
        print_error "$LABEL ($MOUNT): $USED / $SIZE ($USE_PERCENT% used, $AVAIL free)"
        ((CRITICAL_COUNT++))
    elif [ "$USE_PERCENT" -ge 80 ]; then
        print_warn "$LABEL ($MOUNT): $USED / $SIZE ($USE_PERCENT% used, $AVAIL free)"
        ((WARNING_COUNT++))
    else
        print_ok "$LABEL ($MOUNT): $USED / $SIZE ($USE_PERCENT% used, $AVAIL free)"
    fi
done < <(df -h | grep -E "^/dev/")

echo

# AI Data SSD Breakdown
if [ -d "/mnt/ai-data" ]; then
    echo "AI Data SSD breakdown (/mnt/ai-data):"
    echo

    # Models directory
    echo "Models:"
    for model_type in hf gguf trained checkpoints; do
        MODEL_PATH="/mnt/ai-data/models/$model_type"
        if [ -d "$MODEL_PATH" ]; then
            SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | awk '{print $1}')
            COUNT=$(find "$MODEL_PATH" -type f 2>/dev/null | wc -l)
            echo "  $model_type: $SIZE ($COUNT files)"
        fi
    done

    # The tiers that carry current work. training/ is identity-mounted rw by
    # docker-compose.training.yaml (and again by the interp compose); interp/ is
    # created by `make interp-dirs`.
    echo
    echo "Training tier:"
    for sub in datasets shards tokenizers runs verify eval staging; do
        SUB_PATH="/mnt/ai-data/training/$sub"
        if [ -d "$SUB_PATH" ]; then
            SIZE=$(du -sh "$SUB_PATH" 2>/dev/null | awk '{print $1}')
            echo "  $sub: $SIZE"
        fi
    done
    if [ -f "/mnt/ai-data/training/.train.lock" ]; then
        print_info "  Run lock present (a training run may be active)"
    fi

    echo
    echo "Interpretability tier:"
    for sub in activations saes probes monitor vectors organisms hf-cache logs; do
        SUB_PATH="/mnt/ai-data/interp/$sub"
        if [ -d "$SUB_PATH" ]; then
            SIZE=$(du -sh "$SUB_PATH" 2>/dev/null | awk '{print $1}')
            echo "  $sub: $SIZE"
        fi
    done

    # Structured obs records (OBS_LOG_DIR). Identity-mounted into the training,
    # export, and interp containers so host and container agree on the path.
    echo
    echo "Observability logs:"
    for sub in training interp inference crash_dumps; do
        SUB_PATH="/mnt/ai-data/logs/$sub"
        if [ -d "$SUB_PATH" ]; then
            SIZE=$(du -sh "$SUB_PATH" 2>/dev/null | awk '{print $1}')
            echo "  $sub: $SIZE"
        fi
    done

    echo
    # huggingface is bound rw into the prepare service as HF_HOME; the rest are
    # host-side only.
    echo "Caches (huggingface shared with the prepare service; rest host-side only):"
    for cache_type in huggingface torch pip uv; do
        CACHE_PATH="/mnt/ai-data/cache/$cache_type"
        if [ -d "$CACHE_PATH" ]; then
            SIZE=$(du -sh "$CACHE_PATH" 2>/dev/null | awk '{print $1}')
            echo "  $cache_type: $SIZE"
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
        # Last access time; approximate under relatime mounts (atime only updated
        # once per day or when mtime changes), so treat as a coarse hint.
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
                print_warn "  Multiple files with size $SIZE_HUMAN - possible duplicates"
                break
            fi
        done

    if [ "$DUPLICATES_FOUND" = false ]; then
        print_ok "  No obvious duplicates found"
    fi
else
    print_warn "/mnt/ai-data not found"
    ((WARNING_COUNT++))
fi

echo

# Docker Storage Analysis
echo "Docker storage analysis:"
echo

if command -v docker &> /dev/null; then
    # Docker system df provides a nice summary
    if docker system df 2>/dev/null | grep -q "Images"; then
        docker system df 2>/dev/null | grep -E "Images|Containers|Local Volumes|Build Cache" | while read -r line; do
            TYPE=$(echo "$line" | awk '{print $1}')
            SIZE=$(echo "$line" | awk '{print $4 " " $5}')
            RECLAIMABLE=$(echo "$line" | awk '{print $6 " " $7}')

            echo "$TYPE: $SIZE (Reclaimable: ${RECLAIMABLE:-0B})"
        done

        # Check for dangling images
        DANGLING_COUNT=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l)
        if [ "$DANGLING_COUNT" -gt 0 ]; then
            print_warn "Dangling images: $DANGLING_COUNT"
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

# System Caches and Temporary Files
echo "System caches and temporary files:"
echo

# APT cache
APT_CACHE="/var/cache/apt"
if [ -d "$APT_CACHE" ]; then
    APT_SIZE=$(sudo du -sh "$APT_CACHE" 2>/dev/null | awk '{print $1}' || echo "N/A")
    echo "APT cache: $APT_SIZE"
fi

# uv cache. -L follows the symlink to the data drive; without it du measures the
# link itself and reports 0.
UV_CACHE="$HOME/.cache/uv"
if [ -d "$UV_CACHE" ]; then
    UV_CACHE_SIZE=$(du -shL "$UV_CACHE" 2>/dev/null | awk '{print $1}')
    echo "uv cache: $UV_CACHE_SIZE"
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

# System Logs and Journal
echo "System logs and journal:"
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

# Find large log files (process substitution so WARNING_COUNT survives the loop)
echo "Large log files (>100MB):"
while read -r size file; do
    [ -z "$size" ] && continue
    print_warn "  $size $(basename "$file")"
    ((WARNING_COUNT++))
done < <(sudo find /var/log -type f -size +100M -exec du -h {} \; 2>/dev/null | sort -rh | head -5)

# Old kernels
OLD_KERNELS=$(dpkg -l 'linux-*' | sed '/^ii/!d;/'"$(uname -r | sed "s/\(.*\)-\([^0-9]\+\)/\1/")"'/d;s/^[^ ]* [^ ]* \([^ ]*\).*/\1/;/[0-9]/!d' | wc -l)
if [ "$OLD_KERNELS" -gt 2 ]; then
    print_warn "Old kernels installed: $OLD_KERNELS (can be removed)"
    ((WARNING_COUNT++))
fi

echo

# Home Directory Analysis
echo "Home directory analysis:"
echo

echo "Top 10 directories in $HOME (excluding symlinks):"
while read -r size dir; do
    echo "  $size $(basename "$dir")"
done < <(find "$HOME" -maxdepth 1 -type d ! -name "." -exec du -sh {} \; 2>/dev/null | sort -rh | head -10)

echo
echo "AI cache symlinks:"
# Only the AI caches belong on the data drive. Redirecting all of ~/.cache pulls
# desktop caches (browser, GNOME, thumbnails) onto it as well, which defeats the
# drive's purpose of isolating AI I/O from system I/O.
if [ -L "$HOME/.cache" ]; then
    print_warn "  ~/.cache is a symlink to $(readlink "$HOME/.cache")"
    echo "    Expected: a real directory holding per-cache symlinks"
    echo "    Solution: rm ~/.cache && mkdir -p ~/.cache, then re-link the AI caches"
    echo "    Command: ./scripts/setup/setup_data_ssd.sh --skip-format"
    ((WARNING_COUNT++))
else
    for cache_type in huggingface torch pip uv; do
        LINK="$HOME/.cache/$cache_type"
        TARGET="/mnt/ai-data/cache/$cache_type"
        if [ -L "$LINK" ] && [ "$(readlink -f "$LINK")" = "$TARGET" ]; then
            SIZE=$(du -sh "$TARGET" 2>/dev/null | awk '{print $1}')
            print_ok "  $cache_type: ${SIZE:-unknown} -> $TARGET"
        elif [ -d "$LINK" ]; then
            SIZE=$(du -sh "$LINK" 2>/dev/null | awk '{print $1}')
            print_warn "  $cache_type: $SIZE on the OS drive (not linked to the data drive)"
            ((WARNING_COUNT++))
        else
            print_skip "  $cache_type: absent"
        fi
    done
fi

echo
echo "Python Environments:"
# Check for uv-managed interpreters and cache
if [ -d "$HOME/.local/share/uv" ]; then
    UV_SIZE=$(du -sh "$HOME/.local/share/uv" 2>/dev/null | awk '{print $1}')
    UV_PYTHONS=$(ls "$HOME/.local/share/uv/python" 2>/dev/null | wc -l)
    echo "  uv (interpreters/tools): $UV_SIZE ($UV_PYTHONS interpreters)"
fi
if [ -d "$HOME/.cache/uv" ]; then
    UV_CACHE_SIZE=$(du -shL "$HOME/.cache/uv" 2>/dev/null | awk '{print $1}')
    echo "  uv cache: $UV_CACHE_SIZE (prune with: uv cache prune)"
fi
# Leftovers from the retired pyenv/Poetry toolchain (removal documented in
# docs/os/python/setup_python.md)
if [ -d "$HOME/.pyenv" ]; then
    PYENV_SIZE=$(du -sh "$HOME/.pyenv" 2>/dev/null | awk '{print $1}')
    print_warn "  .pyenv: $PYENV_SIZE (retired; remove per docs/os/python/setup_python.md)"
fi
if [ -d "$HOME/.cache/pypoetry" ]; then
    POETRY_SIZE=$(du -sh "$HOME/.cache/pypoetry" 2>/dev/null | awk '{print $1}')
    print_warn "  Poetry cache: $POETRY_SIZE (retired; remove per docs/os/python/setup_python.md)"
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

# Cleanup Suggestions
echo "Cleanup suggestions:"
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

# uv cache
UV_BYTES=$(get_size "$HOME/.cache/uv")
if [ "$UV_BYTES" -gt 5368709120 ]; then  # >5GB
    # Size alone cannot distinguish reclaimable from live entries: a cache that
    # is entirely in use still trips this threshold. prune removes only unused
    # entries, so it is safe to suggest unconditionally, but say what it does.
    echo "• Prune unused uv cache entries (a large cache may be entirely in use):"
    echo "  uv cache prune"
    ((SUGGESTIONS++))
fi

# Retired Poetry cache (toolchain removed; see docs/os/python/setup_python.md)
POETRY_BYTES=$(get_size "$HOME/.cache/pypoetry")
if [ "$POETRY_BYTES" -gt 0 ]; then
    echo "• Remove retired Poetry cache:"
    echo "  rm -rf ~/.cache/pypoetry"
    ((SUGGESTIONS++))
fi

# Old kernels
if [ "$OLD_KERNELS" -gt 2 ] 2>/dev/null; then
    echo "• Remove old kernels:"
    echo "  sudo apt autoremove --purge"
    ((SUGGESTIONS++))
fi

if [ $SUGGESTIONS -eq 0 ]; then
    print_ok "No significant cleanup opportunities found"
fi

echo

# Storage Summary
echo "=== Storage Summary ==="

# Count status
if [ $CRITICAL_COUNT -gt 0 ]; then
    print_error "Critical: $CRITICAL_COUNT disk(s) >90% full"
fi
if [ $WARNING_COUNT -gt 0 ]; then
    print_warn "$WARNING_COUNT potential issues found"
fi
if [ $CRITICAL_COUNT -eq 0 ] && [ $WARNING_COUNT -eq 0 ]; then
    print_ok "All storage checks passed"
fi

echo
echo "Report complete: $(date '+%Y-%m-%d %H:%M:%S')"