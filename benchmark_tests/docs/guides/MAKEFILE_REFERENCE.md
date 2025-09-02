# Makefile Reference Guide

**AI Workstation Benchmark Tests - Complete Makefile Documentation**

This comprehensive guide covers all commands available in the benchmark tests Makefile. The Makefile provides a unified interface for testing, debugging, calibration, and maintenance operations across all 30+ domains.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Testing Commands](#core-testing-commands)
3. [Quality Assurance](#quality-assurance)
4. [Systematic Calibration](#systematic-calibration)
5. [Domain Coverage Operations](#domain-coverage-operations)
6. [Debug & Troubleshooting](#debug--troubleshooting)
7. [Maintenance Commands](#maintenance-commands)
8. [Advanced Usage Patterns](#advanced-usage-patterns)
9. [Environment Variables](#environment-variables)
10. [Command Reference Index](#command-reference-index)

---

## Quick Start

### Essential Commands for New Contributors

```bash
# Get help and see all available commands
make help

# Check if your environment is ready
make check-env

# Run core regression tests (most important)
make test-regression

# Quick safety check for core modules
make test-core-safety

# Clean up artifacts
make clean
```

### Most Common Usage Patterns

```bash
# Fast unit tests only
make test MODE=quick SUITE=unit

# All tests with coverage report
make test MODE=coverage

# Run tests matching a pattern
make test ARGS='-k evaluator'

# Test specific file
make test-specific FILE=tests/unit/test_foo.py
```

---

## Core Testing Commands

### Main Test Command: `make test`

The primary testing interface with flexible options:

```bash
# Basic usage
make test [MODE=quick|verbose|coverage|standard] [SUITE=unit|integration|all]
```

**Modes:**
- `quick` - Fast execution with minimal output
- `verbose` - Detailed output with `-v -s` flags
- `coverage` - Full coverage reporting (80% minimum)
- `standard` - Default balanced mode

**Suites:**
- `unit` - Unit tests only (`tests/unit/`)
- `integration` - Integration tests only (`tests/integration/`)
- `all` - Both unit and integration tests

**Examples:**
```bash
make test MODE=quick SUITE=unit           # Fast unit tests
make test MODE=coverage                   # All tests with coverage
make test ARGS='-k pattern'              # Filter by pattern
```

### Critical Test Commands

#### `make test-regression` ⚠️ **CRITICAL**
Runs all tests except functional/ and calibration/ to ensure no functionality is broken.
- **When to use:** Before any major changes, after reorganization
- **Performance:** Uses pytest-xdist for concurrent execution
- **Excludes:** Tests requiring live server setup

```bash
make test-regression
```

#### `make test-core-safety`
Quick safety check for core modules and critical functionality.
- **When to use:** After core module changes
- **Scope:** `tests/unit/test_core_modules/` and `tests/unit/test_scripts_organization/`

```bash
make test-core-safety
```

### Legacy Test Aliases

```bash
make test-unit          # Equivalent to: make test SUITE=unit
make test-integration   # Equivalent to: make test SUITE=integration
make test-modular       # Modular test suite
```

### Specialized Test Commands

#### `make test-specific`
Run a specific test file or method:

```bash
make test-specific FILE=tests/unit/test_evaluator.py
make test-specific FILE=tests/unit/test_evaluator.py::TestClass::test_method
```

#### `make test-analysis`
Run analysis scripts and validation:

```bash
make test-analysis
```

#### `make test-watch`
Watch for file changes and auto-run tests:

```bash
make test-watch
```

---

## Quality Assurance

### Environment and Prerequisites

#### `make check-prerequisites` 
**Master command** for comprehensive system readiness:

```bash
make check-prerequisites
```

**Checks:**
1. Docker status and container health
2. Server connectivity (localhost:8004)
3. Model loading and API responses
4. Calibration framework readiness
5. System resources (GPU, memory)

#### `make check-env`
Basic environment validation:

```bash
make check-env
```

**Validates:**
- Python 3 installation
- pytest availability and version
- Working directory and paths
- Test timeout configuration

### API and Server Testing

#### `make test-api-suite`
Complete API connectivity test suite:

```bash
make test-api-suite
```

**Tests:**
1. Server health endpoint
2. Basic completion requests
3. Haiku completion (creative test)

#### `make docker-logs`
Inspect model server logs and configuration:

```bash
make docker-logs
```

### Component Testing

#### `make test-domain-loading`
Test domain file loading capabilities:

```bash
make test-domain-loading
```

#### `make test-enhanced-evaluator`
Test enhanced evaluator import and instantiation:

```bash
make test-enhanced-evaluator
```

#### `make test-semantic-analyzer`
Test semantic analyzer logging fixes:

```bash
make test-semantic-analyzer
```

---

## Systematic Calibration

**The calibration system supports progressive difficulty testing across all 30+ domains.**

### Primary Calibration Commands

#### `make systematic-base-calibration` 🎯
**Main calibration command** - Progressive base model calibration:

```bash
make systematic-base-calibration
```

**Process:**
- Easy → Medium → Hard progression
- Halt-on-failure methodology
- Enhanced evaluation with sophisticated evaluators
- Statistical validation (3 samples per test)
- Comprehensive reporting

**Target Score Ranges:**
- Easy: 70-85 points
- Medium: 60-80 points  
- Hard: 50-75 points

**Requirements:** Running LLM server at http://localhost:8004

#### `make test-systematic-calibration`
Test the systematic calibration framework (single domain):

```bash
make test-systematic-calibration
```

**Use case:** Validate calibration framework before full run

### Calibration Support Commands

#### `make calibration-validate`
Live calibration validation with actual LLM server:

```bash
make calibration-validate
```

#### `make calibration-demo`
Run calibration framework demonstration:

```bash
make calibration-demo
```

#### `make calibration-status`
Check calibration system status (alias for check-prerequisites):

```bash
make calibration-status
```

#### `make test-calibration`
Run calibration validation framework tests:

```bash
make test-calibration
```

---

## Domain Coverage Operations

**Commands for expanding test coverage across all 30 domains.**

### Test Conversion Commands

#### `make convert-creativity-tests`
Convert creativity domain from base to instruct models:

```bash
make convert-creativity-tests
```

#### `make convert-core-domains`
Convert all core domains (language, integration, knowledge, social) to instruct format:

```bash
make convert-core-domains
```

### Conversion Testing

#### `make test-creativity-conversion`
Test enhanced evaluator with converted creativity tests:

```bash
make test-creativity-conversion
```

#### `make test-core-domains-conversion`
Test enhanced evaluator with converted core domain tests:

```bash
make test-core-domains-conversion
```

### Domain Analysis

#### `make domain-audit`
Comprehensive domain coverage analysis:

```bash
make domain-audit
```

**Provides:**
- 30 domains analyzed across 3 sophistication tiers
- Production-ready domain identification
- Base/Instruct imbalance reporting
- Next steps recommendations

---

## Debug & Troubleshooting

### Consolidated Debug Commands

#### `make debug-test-components`
**New consolidated debug command** - Project structure and module status:

```bash
make debug-test-components
```

**Checks:**
- Project directory structure
- Core module file existence
- Domain test file count
- All essential components

#### `make debug-help`
Show all available debug utilities:

```bash
make debug-help
```

### Enhanced Evaluator Debugging

#### `make debug-enhanced-system`
Comprehensive enhanced universal evaluator system debug:

```bash
make debug-enhanced-system
```

#### `make debug-enhanced-evaluator`
Phase 1 enhanced universal evaluator debugging:

```bash
make debug-enhanced-evaluator
```

#### `make debug-scoring-calibration`
Debug Phase 1 scoring calibration fixes:

```bash
make debug-scoring-calibration
```

### Cultural Evaluation Debugging

#### `make test-cultural-calibration`
Test cultural reasoning evaluation on specific tests:

```bash
make test-cultural-calibration
```

**Tests:**
- Arabic Quranic verse patterns (basic_03)
- Native American creation stories (basic_04)

#### `make debug-cultural-task-detection`
Check if cultural tests are being detected properly:

```bash
make debug-cultural-task-detection
```

#### `make debug-batch-task-detection`
Check cultural task detection for test batches (basic_05-08):

```bash
make debug-batch-task-detection
```

### Reasoning Test Debugging

#### `make test-reasoning-batch`
Test reasoning batch (basic_09-15) systematically:

```bash
make test-reasoning-batch
```

#### `make test-reasoning-next-batch`
Test next reasoning batch (basic_16-22):

```bash
make test-reasoning-next-batch
```

#### `make debug-basic12-analysis`
Analyze what makes basic_12 score higher than others:

```bash
make debug-basic12-analysis
```

### Calibration Framework Debugging

#### `make debug-calibration-framework`
Step-by-step calibration framework debug:

```bash
make debug-calibration-framework
```

**Steps:**
1. Domain file loading
2. Enhanced evaluator
3. Benchmark runner
4. API connectivity

### Individual Component Tests

#### `make test-scripts-organization-only`
Test scripts organization changes only:

```bash
make test-scripts-organization-only
```

#### `make test-domain-loading-only`
Test domain loading functionality only:

```bash
make test-domain-loading-only
```

#### `make test-core-modules-debug`
Test core modules with detailed debug info:

```bash
make test-core-modules-debug
```

---

## Maintenance Commands

### Cleanup Operations

#### `make clean`
Basic cleanup of test artifacts:

```bash
make clean
```

**Removes:**
- `__pycache__` directories
- `.pyc` files
- `.pytest_cache` directories
- `.coverage` files

#### `make clean-all`
Deep cleaning including coverage reports:

```bash
make clean-all
```

**Additional cleanup:**
- HTML coverage reports
- Coverage XML files
- Egg info directories
- MyPy cache

### Setup and Dependencies

#### `make setup`
Install test dependencies:

```bash
make setup
```

**Installs:**
- pytest
- pytest-cov
- pytest-mock
- pytest-asyncio

### Examples and Demos

#### `make examples`
Run example scripts and demonstrations:

```bash
make examples
```

#### `make calibration-demo`
Run calibration framework demo:

```bash
make calibration-demo
```

### Special Commands

#### `make test-phase1-quality`
Test Phase 1 quality fixes:

```bash
make test-phase1-quality
```

---

## Advanced Usage Patterns

### Multi-Domain Testing Strategy

```bash
# Step 1: Ensure environment is ready
make check-prerequisites

# Step 2: Run regression tests to ensure baseline
make test-regression

# Step 3: Test specific domain conversion
make convert-creativity-tests
make test-creativity-conversion

# Step 4: Run systematic calibration
make systematic-base-calibration

# Step 5: Analyze results
make domain-audit
```

### Development Workflow

```bash
# During development
make test-core-safety              # Quick safety check
make test MODE=quick SUITE=unit    # Fast unit tests

# Before committing
make test-regression               # Full regression test
make clean                        # Cleanup artifacts

# For debugging issues
make debug-test-components         # Check project structure
make debug-help                   # See all debug options
```

### Performance Testing

```bash
# Coverage analysis
make test MODE=coverage

# Concurrent execution
make test-regression              # Uses pytest-xdist automatically

# Watch mode for continuous testing
make test-watch
```

---

## Environment Variables

### Core Variables

- **`MODE`**: `quick|verbose|coverage|standard`
  - Controls test execution mode
  - Default: `standard`

- **`SUITE`**: `unit|integration|modular|all`
  - Selects test suite to run
  - Default: `all`

- **`ARGS`**: Additional pytest arguments
  - Example: `ARGS='-k pattern'`
  - Example: `ARGS='--tb=long'`

- **`FILE`**: Specific test file for `test-specific`
  - Required for `make test-specific`
  - Example: `FILE=tests/unit/test_evaluator.py`

### Internal Configuration

```makefile
PYTHON := python3
PYTEST := python -m pytest
TEST_TIMEOUT := 300
COVERAGE_MIN := 80
```

### Usage Examples

```bash
# Combine multiple variables
make test MODE=verbose SUITE=unit ARGS='-k evaluator --tb=long'

# Environment-specific testing
MODE=coverage SUITE=integration make test

# File-specific debugging
make test-specific FILE=tests/unit/test_evaluator.py::TestClass::test_method
```

---

## Command Reference Index

### Core Testing
- `make test` - Main test runner
- `make test-regression` - **Critical regression tests**
- `make test-core-safety` - Core module safety check
- `make test-specific` - Specific file/method testing
- `make test-analysis` - Analysis scripts
- `make test-watch` - Watch mode

### Quality Assurance  
- `make check-prerequisites` - **Master system check**
- `make check-env` - Environment validation
- `make test-api-suite` - API connectivity tests
- `make docker-logs` - Server log inspection

### Systematic Calibration
- `make systematic-base-calibration` - **Main calibration**
- `make test-systematic-calibration` - Framework testing
- `make calibration-validate` - Live validation
- `make calibration-demo` - Demo mode
- `make test-calibration` - Framework tests

### Domain Coverage
- `make convert-creativity-tests` - Convert creativity domain
- `make convert-core-domains` - Convert core domains
- `make test-creativity-conversion` - Test creativity conversion
- `make test-core-domains-conversion` - Test core conversion
- `make domain-audit` - Coverage analysis

### Debug & Troubleshooting
- `make debug-test-components` - **Consolidated debug**
- `make debug-help` - Debug utilities list
- `make debug-enhanced-system` - Enhanced evaluator debug
- `make debug-calibration-framework` - Calibration debug
- `make test-cultural-calibration` - Cultural tests
- `make test-reasoning-batch` - Reasoning batch tests

### Maintenance
- `make clean` - Basic cleanup
- `make clean-all` - Deep cleanup
- `make setup` - Install dependencies
- `make examples` - Run examples

### Legacy/Aliases
- `make test-unit` - Unit tests only
- `make test-integration` - Integration tests only
- `make calibration-status` - Check calibration status

---

## Future Expansion Notes

As the project grows to support all 30+ domains, the Makefile will likely expand with:

### Planned Additions
- **Domain-specific calibration commands** for each of the 30 domains
- **Batch processing commands** for multiple domains simultaneously  
- **Cross-domain consistency testing** commands
- **Performance benchmarking** across different hardware configurations
- **Automated reporting** and dashboard generation commands
- **Production deployment** validation commands

### Scalability Considerations
- The current structure supports adding domain-specific targets
- Pattern-based commands (like `test-*-batch`) can be templated
- Environmental variables can be expanded for domain selection
- The prerequisite checking system can be extended for domain-specific requirements

### Development Workflow Evolution
As domains are added, the typical workflow will expand to:
1. Domain-specific conversion: `make convert-DOMAIN-tests`
2. Domain-specific calibration: `make calibrate-DOMAIN`
3. Cross-domain validation: `make test-cross-domain-consistency`
4. Production readiness: `make validate-production-ready DOMAIN=all`

---

**This Makefile serves as the central command interface for the entire benchmark testing system. As we expand to cover all 30 domains, it will continue to evolve while maintaining backward compatibility and consistent patterns.**

For questions or suggestions about Makefile commands, refer to:
- `make help` - Always up-to-date command list
- `make debug-help` - Debug utilities
- This documentation - Comprehensive usage guide