# Bash Output Style Guide

Output formatting for all bash scripts. Follow Unix conventions: parseable, minimal,
consistent, no decoration or emojis, clear about required user actions.

## Status Indicators

Bracketed format with ANSI colors:

| Indicator | Color | Usage |
|-----------|-------|-------|
| `[OK]` | Green | Operation completed successfully |
| `[WARN]` | Yellow | Non-critical issue requiring attention |
| `[ERROR]` | Red | Critical failure requiring action |
| `[INFO]` | Cyan | Informational message |
| `[RUN]` | Blue | Operation in progress |
| `[SKIP]` | Blue | Operation skipped |

Format is `[STATUS] Subject: detail`, e.g. `[OK] Disk space: 15% used, 85% free`.

## Color Setup

Use standard ANSI codes, support disabling for log files, and use `printf` (not `echo -e`):

```bash
RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' BOLD='\033[1m' NC='\033[0m'

# Respect NO_COLOR env var or --no-color flag
if [[ -n "${NO_COLOR:-}" ]] || [[ "${1:-}" == "--no-color" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' BOLD='' NC=''
fi

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*"; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }
print_skip()  { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }
```

Always reset color with `${NC}`. Colors are display-only; output must stay parseable with
`NO_COLOR`.

## Structure

Simple text headers, 2-space indentation per hierarchy level:

```bash
echo "=== System Configuration ==="   # preferred header format
echo
echo "Main operation"
echo "  Sub-operation"
echo "    Detail level"
```

Spacing: 1 blank line after major sections, no blank lines between related items.

For heavier separators use single-line rules sized to importance (60 chars major, 40 sub,
20 minor). Do not use boxes or ASCII art.

## Errors and Warnings

Include context and a solution:

```bash
print_error "Model file not found: $MODEL_PATH"
echo "  Solution: Verify model exists at specified path"
echo "  Command: ls -la \"$(dirname "$MODEL_PATH")\""

print_warn "Disk usage at 85% (recommend cleanup)"
echo "  Current usage: 1.7TB / 2.0TB"
```

## Batch and Progress

```bash
echo "Updating 5 Docker images:"
print_ok "ubuntu:24.04"
print_error "python:3.12-slim (network timeout)"
echo "Summary: 4 updated, 1 failed"
```

## Example

```bash
echo "=== GPU Optimization Setup ==="
echo
echo "System checks:"
print_ok "NVIDIA driver: 580.65.06"
print_ok "CUDA toolkit: 13.0.88"
echo
echo "Applying optimizations:"
print_run "Setting power limit: 600W"
print_ok "Memory clocks: unlocked (optimal)"
echo
print_ok "Optimization complete"
echo "Restart required: NO"
```

## Prohibited

- Emojis or Unicode symbols (`✓ ⚠ ✗`) — use `[OK]`/`[WARN]`/`[ERROR]` instead
- Decorative borders, boxes, or ASCII art
- Marketing language ("awesome", "perfect") and superlatives
- Exclamation marks (except in error messages) and all-caps (except status indicators)
- Colors without `NO_COLOR`/`--no-color` support; `echo -e` for colors (use `printf`)
