# Bash Output Style Guide

This guide defines output formatting standards for all bash scripts in the repository. All script output must follow these conventions to maintain professional quality and Unix compatibility.

## Table of Contents

- [Core Philosophy](#core-philosophy)
- [Status Indicators](#status-indicators)
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

### Text-Based Status (No Emojis)

| Status | Format | Usage |
|--------|--------|-------|
| Success | `OK` | Operation completed successfully |
| Warning | `WARN` | Non-critical issue requiring attention |
| Error | `ERROR` | Critical failure requiring action |
| Info | `INFO` | Informational message |
| Check | `CHECK` | Verification or validation step |

### Status Format

```bash
# Correct format
echo "Disk space: OK (15% used, 85% free)"
echo "Memory: WARN (92% used - consider closing applications)"
echo "Network: ERROR (unable to connect to package repository)"
```

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
echo "Checking dependencies..."
echo "Python: OK (3.12.0)"
echo "Docker: ERROR (not installed)"
echo "Check complete"
```

### Level 2: Update Scripts (Structured)

More complex operations with clear progress:

```bash
echo "=== Ubuntu System Update ==="
echo
echo "Pre-update checks:"
echo "  Disk space: OK (15% used)"
echo "  Network: OK (connected)"
echo
echo "Package updates:"
echo "  Updating package lists..."
echo "  Installing 42 package updates..."
echo "  Update complete"
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
echo "  CPU Governor: OK (performance mode)"
echo "  Huge Pages: OK (46080 allocated)"
echo "  Swap: OK (disabled)"
echo
echo "Summary: 15 checks passed, 2 warnings"
```

## Error Handling

### Error Message Format

```bash
# Error with context and solution
echo "ERROR: Model file not found: $MODEL_PATH"
echo "  Solution: Verify model exists at specified path"
echo "  Command: ls -la \"$(dirname "$MODEL_PATH")\""
```

### Warning Message Format

```bash
# Warning with explanation
echo "WARN: Disk usage at 85% (recommend cleanup)"
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
echo "  nvidia/cuda:13.0-base: OK"
echo "  ubuntu:24.04: OK"
echo "  python:3.12-slim: ERROR (network timeout)"
echo "  pytorch/pytorch:latest: OK"
echo "  huggingface/transformers: OK"
echo "Summary: 4 updated, 1 failed"
```

## Examples

### Good Output

```bash
#!/bin/bash
echo "=== GPU Optimization Setup ==="
echo
echo "System checks:"
echo "  NVIDIA driver: OK (580.65.06)"
echo "  CUDA toolkit: OK (13.0.88)"
echo "  Docker runtime: OK (nvidia-docker2)"
echo
echo "Applying optimizations:"
echo "  Power limit: 600W"
echo "  Persistence mode: enabled"
echo "  Memory clocks: unlocked (optimal)"
echo
echo "Optimization complete"
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

- ❌ Emojis or Unicode symbols (✓, ⚠, ✗, 🚀, etc.)
- ❌ Excessive decorative borders or boxes
- ❌ Marketing language ("awesome", "perfect", "amazing")
- ❌ Exclamation marks (except in error messages)
- ❌ All-caps text (except for specific technical terms)
- ❌ Colored output (incompatible with log files)
- ❌ Unnecessary ASCII art or decorations

### Text Alternatives

| Prohibited | Use Instead |
|-----------|-------------|
| ✓ Success | `OK` or `PASS` |
| ✗ Failure | `ERROR` or `FAIL` |
| ⚠ Warning | `WARN` or `WARNING` |
| ❌ Error | `ERROR` |
| ℹ Information | `INFO` |

## Implementation Checklist

Before committing script changes:

### Content Review
- [ ] All emojis replaced with text indicators
- [ ] Status messages use OK/WARN/ERROR format
- [ ] Section headers use single-line separators
- [ ] Indentation follows 2-space hierarchy

### Format Review
- [ ] No decorative elements or ASCII art
- [ ] Consistent spacing and blank lines
- [ ] Error messages include context and solutions
- [ ] Output is parseable by standard tools

### Script Type Review
- [ ] Formatting level appropriate for script complexity
- [ ] Progress reporting clear and minimal
- [ ] Summary information provided where needed

### Final Review
- [ ] Output aligns with Unix philosophy
- [ ] Professional appearance maintained
- [ ] No marketing language or superlatives
- [ ] Consistent with other scripts in repository

---

*Last Updated: 2025-09-24*