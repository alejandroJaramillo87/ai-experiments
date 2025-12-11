# Bash Output Style Guide

This guide defines output formatting standards for all bash scripts in the repository. All script output must follow these conventions to maintain professional quality and Unix compatibility.

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Status Indicators](#status-indicators)
- [Color Output](#color-output)
- [Section Formatting](#section-formatting)
- [Indentation and Spacing](#indentation-and-spacing)
- [Script Types and Formatting Levels](#script-types-and-formatting-levels)
- [Error Handling](#error-handling)
- [Progress Reporting](#progress-reporting)
- [Examples](#examples)
- [Prohibited Elements](#prohibited-elements)
- [Implementation Checklist](#implementation-checklist)

## Core Philosophy

### Unix Principles

All script output follows traditional Unix conventions:

- **Parseable** - Output can be processed by standard tools (grep, awk, sed)
- **Minimal** - Include only essential information
- **Consistent** - Predictable format across all scripts
- **Professional** - No decorative elements or emojis
- **Actionable** - Clear indication of required user actions

### Output Hierarchy

```
SCRIPT_NAME: Brief description of what script does
Section: Major operation being performed
  Subsection: Specific task within operation
    Detail: Individual step or check result
```

## Status Indicators

All status indicators use bracketed format with ANSI colors:

| Indicator | Color | Usage |
|-----------|-------|-------|
| `[OK]` | Green | Operation completed successfully |
| `[WARN]` | Yellow | Non-critical issue requiring attention |
| `[ERROR]` | Red | Critical failure requiring action |
| `[INFO]` | Cyan | Informational message |
| `[RUN]` | Blue | Operation in progress |
| `[SKIP]` | Blue | Operation skipped |

### Status Format

```bash
# Correct format - bracketed indicators with color
print_ok "Disk space" "15% used, 85% free"
print_warn "Memory" "92% used - consider closing applications"
print_error "Network" "unable to connect to package repository"

# Output:
# [OK] Disk space: 15% used, 85% free
# [WARN] Memory: 92% used - consider closing applications
# [ERROR] Network: unable to connect to package repository
```

## Color Output

### Color Definitions

Use standard ANSI escape codes:

```bash
# Color definitions (add to script header)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'  # No Color
```

### Color Disable Support

Scripts must support disabling colors for log file compatibility:

```bash
# Check for NO_COLOR environment variable or --no-color flag
if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
fi
```

### Print Helper Functions

Use these helper functions for consistent output:

```bash
print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }
print_skip()  { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }
```

### Color Usage Rules

- Always reset color after colored text using `${NC}`
- Use `printf` instead of `echo -e` for portability
- Test output both with and without colors
- Colors are for terminal display only, not semantic meaning

## Section Formatting

### Section Headers

Use single-line separators with consistent width:

```bash
# Major sections (60 characters)
echo "============================================================"
echo "System Update"
echo "============================================================"

# Subsections (40 characters)
echo "----------------------------------------"
echo "Package Installation"
echo "----------------------------------------"

# Minor sections (20 characters)
echo "--------------------"
echo "Cleanup"
echo "--------------------"
```

### Preferred Format

```bash
# Simple header format (recommended for most cases)
echo "=== System Configuration ==="
echo
```

## Indentation and Spacing

### Hierarchy Levels

```bash
echo "Main operation"
echo "  Sub-operation"
echo "    Detail level"
echo "      Fine detail"
```

### Spacing Rules

- **2 spaces** per indentation level
- **1 blank line** after major sections
- **No blank lines** between related items
- **1 blank line** before script completion

## Script Types and Formatting Levels

### Level 1: Utility Scripts (Minimal)

Simple, focused scripts with minimal output:

```bash
print_info "Checking dependencies..."
print_ok "Python: 3.12.0"
print_error "Docker: not installed"
echo "Check complete"
```

### Level 2: Update Scripts (Structured)

More complex operations with clear progress:

```bash
echo "=== Ubuntu System Update ==="
echo
echo "Pre-update checks:"
print_ok "Disk space: 15% used"
print_ok "Network: connected"
echo
echo "Package updates:"
print_run "Updating package lists..."
print_run "Installing 42 package updates..."
print_ok "Update complete"
echo
echo "System restart required: YES"
```

### Level 3: Diagnostic Scripts (Detailed)

Comprehensive reports with structured output:

```bash
echo "=== System Diagnostic Report ==="
echo "Generated: 2025-09-24 14:30:15"
echo
echo "Hardware Configuration:"
echo "  CPU: AMD Ryzen 9950X (32 threads)"
echo "  GPU: RTX 5090 (32GB VRAM)"
echo "  Memory: 128GB DDR5"
echo
echo "Optimization Status:"
print_ok "CPU Governor: performance mode"
print_ok "Huge Pages: 46080 allocated"
print_ok "Swap: disabled"
echo
echo "Summary: 15 checks passed, 2 warnings"
```

## Error Handling

### Error Message Format

```bash
# Error with context and solution
print_error "Model file not found: $MODEL_PATH"
echo "  Solution: Verify model exists at specified path"
echo "  Command: ls -la \"$(dirname "$MODEL_PATH")\""
```

### Warning Message Format

```bash
# Warning with explanation
print_warn "Disk usage at 85% (recommend cleanup)"
echo "  Current usage: 1.7TB / 2.0TB"
echo "  Cleanup script: ./scripts/utils/storage_check.sh"
```

## Progress Reporting

### Long-Running Operations

```bash
echo "Installing NVIDIA drivers..."
# During operation - use simple dots or status updates
echo "  Downloading packages..."
echo "  Configuring modules..."
echo "  Rebuilding initramfs..."
echo "Installation complete"
```

### Batch Operations

```bash
echo "Updating 5 Docker images:"
print_ok "nvidia/cuda:13.0-base"
print_ok "ubuntu:24.04"
print_error "python:3.12-slim (network timeout)"
print_ok "pytorch/pytorch:latest"
print_ok "huggingface/transformers"
echo "Summary: 4 updated, 1 failed"
```

## Examples

### Good Output

```bash
#!/bin/bash
# Color and helper setup
RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'
[[ -n "${NO_COLOR:-}" ]] && RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

echo "=== GPU Optimization Setup ==="
echo
echo "System checks:"
print_ok "NVIDIA driver: 580.65.06"
print_ok "CUDA toolkit: 13.0.88"
print_ok "Docker runtime: nvidia-docker2"
echo
echo "Applying optimizations:"
print_run "Setting power limit: 600W"
print_run "Enabling persistence mode"
print_ok "Memory clocks: unlocked (optimal)"
echo
print_ok "Optimization complete"
echo "Restart required: NO"
```

### Bad Output (Avoid)

```bash
#!/bin/bash
echo "🚀🚀🚀 GPU OPTIMIZATION SETUP 🚀🚀🚀"
echo "========================================="
echo "||          System checks:            ||"
echo "========================================="
echo "✅ NVIDIA driver: 580.65.06 (AWESOME!)"
echo "✅ CUDA toolkit: 13.0.88 (PERFECT!)"
echo "✅ Docker runtime: nvidia-docker2 (GREAT!)"
echo "========================================="
echo "🔧🔧🔧 Applying optimizations... 🔧🔧🔧"
echo "⚡ Power limit: 600W"
echo "💾 Persistence mode: enabled"
echo "🔓 Memory clocks: unlocked (optimal)"
echo "🎉🎉🎉 OPTIMIZATION COMPLETE!!! 🎉🎉🎉"
```

## Prohibited Elements

### Never Use

- Emojis or Unicode symbols (✓, ⚠, ✗, etc.)
- Excessive decorative borders or boxes
- Marketing language ("awesome", "perfect", "amazing")
- Exclamation marks (except in error messages)
- All-caps text (except for status indicators)
- Unnecessary ASCII art or decorations
- Colors without NO_COLOR/--no-color support
- echo -e for colors (use printf instead)

### Text Alternatives

| Prohibited | Use Instead |
|-----------|-------------|
| ✓ Success | `[OK]` (green) |
| ✗ Failure | `[ERROR]` (red) |
| ⚠ Warning | `[WARN]` (yellow) |
| ℹ Information | `[INFO]` (cyan) |
| → In progress | `[RUN]` (blue) |

## Implementation Checklist

Before committing script changes:

### Color Setup
- [ ] Color variables defined at script start
- [ ] NO_COLOR environment variable respected
- [ ] --no-color flag supported (if applicable)
- [ ] printf used instead of echo -e

### Content Review
- [ ] All emojis replaced with bracketed indicators
- [ ] Status messages use [OK]/[WARN]/[ERROR] format
- [ ] Section headers use single-line separators
- [ ] Indentation follows 2-space hierarchy

### Format Review
- [ ] No decorative elements or ASCII art
- [ ] Consistent spacing and blank lines
- [ ] Error messages include context and solutions
- [ ] Output is parseable by standard tools (with NO_COLOR)

### Script Type Review
- [ ] Formatting level appropriate for script complexity
- [ ] Progress reporting clear and minimal
- [ ] Summary information provided where needed

### Final Review
- [ ] Output aligns with Unix philosophy
- [ ] Professional appearance maintained
- [ ] No marketing language or superlatives
- [ ] Consistent with other scripts in repository
- [ ] Colors tested both enabled and disabled

---

*Last Updated: 2025-12-10*