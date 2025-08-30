I'll create a canvas with the markdown file for easy copying:I'll create this as a canvas artifact that you can easily copy from:I don't have access to canvas creation tools in this environment. Let me provide the markdown in a different format that should be easier to copy. You can select and copy this entire block:


---

# AI Workstation Test Execution Guide

## Overview
This guide provides standardized commands and procedures for running comprehensive test suites on the AI workstation benchmark system, specifically designed for use with Claude Code.

## Context
- **Repository:** AI Workstation Benchmark System
- **Test Location:** `tests/` directory  
- **Target:** Run all tests except functional tests to assess system health
- **Expected Coverage:** ~55+ test files across unit, integration, and analysis test suites

## Test Execution Commands

### Primary Test Command
```bash
python -m pytest tests/ --ignore=tests/functional/ -v --tb=short --durations=10
```

### Enhanced Test Command with Reporting
```bash
python -m pytest tests/ \
    --ignore=tests/functional/ \
    --verbose \
    --tb=short \
    --durations=10 \
    --junit-xml=test-results.xml \
    --html=test-report.html --self-contained-html \
    | tee test-execution.log
```

### Test Discovery (Pre-execution Check)
```bash
python -m pytest --collect-only tests/ --ignore=tests/functional/
```

## Test Categories Included

| Category | Path | Purpose |
|----------|------|---------|
| **Unit Tests** | `tests/unit/` | Core functionality validation |
| **Integration Tests** | `tests/integration/` | Cross-system coordination |
| **Analysis Tests** | `tests/analysis/` | Advanced algorithm validation |
| **Evaluator Tests** | Various locations | Recently improved evaluator components |

## Test Categories Excluded
- **Functional Tests** (`tests/functional/`) - Excluded via `--ignore` flag

## Expected Outcomes

### Success Metrics
- [ ] All unit tests pass
- [ ] Integration tests demonstrate proper cross-system coordination  
- [ ] Analysis tests validate advanced algorithms
- [ ] Recent evaluator improvements show positive impact

### Failure Analysis
- Identify failing test categories
- Determine root causes of failures
- Assess impact on overall system stability
- Prioritize fixes based on test importance

## Claude Code Instructions

When executing this test plan:

1. **Navigate to repository root** before running commands
2. **Check dependencies** are installed (`pytest`, `pytest-html`, etc.)  
3. **Run test discovery** first to verify test collection
4. **Execute primary test command** and capture full output
5. **Analyze results** and categorize by test type and status
6. **Generate summary report** with:
   - Total tests run vs expected (~55+)
   - Pass/fail breakdown by category  
   - Slowest tests (from `--durations=10`)
   - Critical failures requiring immediate attention

## Output Files Generated
- `test-results.xml` - Machine-readable test results
- `test-report.html` - Human-readable HTML report  
- `test-execution.log` - Complete console output

## Quick Category-Specific Commands

```bash
# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests only  
python -m pytest tests/integration/ -v

# Analysis tests only
python -m pytest tests/analysis/ -v

# Specific test file
python -m pytest tests/path/to/specific_test.py -v
```

## Success Criteria
- 100% of expected tests execute successfully
- No critical system failures in core components
- Integration tests demonstrate proper cross-component communication
- Recent evaluator fixes show improved test results compared to previous runs

## Usage with Claude Code

Save this file as `test-execution-plan.md` and reference it when asking Claude Code to:

```bash
claude code "Following the test-execution-plan.md, run the complete test suite and provide a comprehensive analysis of results including any failures and recommended next steps"
```

This will enable consistent, comprehensive test execution and analysis across all development cycles.

---

