# Test Cleanup Guide

This guide explains how to manage test artifacts and keep your benchmark_tests directory clean.

## Overview

The benchmark test suite creates several types of artifacts during execution:

1. **Empty test directories**: `test_flags/`, `custom_flags/`, `test_integration_flags/`
2. **JSON result files**: `validation_results.json`, `scoring_pattern_analysis.json`, `edge_case_test_results.json`
3. **Python cache**: `__pycache__/` directories throughout the project

## Automatic Cleanup (Recommended)

### Using Make Commands
The easiest way to run tests with automatic cleanup:

```bash
# Run all tests with automatic cleanup
make test

# Run specific test categories
make test-unit           # Unit tests only
make test-integration    # Integration tests only
make test-analysis       # Analysis scripts
make test-quick          # Quick subset

# Clean without running tests
make clean
```

### Using Python Test Runner
Run tests with the automated cleanup runner:

```bash
# Default: run unit + integration tests with cleanup
python run_tests_with_cleanup.py

# Run specific tests
python run_tests_with_cleanup.py tests/unit/
python run_tests_with_cleanup.py tests/unit/test_evaluator.py -v

# Any pytest arguments work
python run_tests_with_cleanup.py -k "test_entropy" --tb=short
```

## Manual Cleanup

If you need to clean up artifacts manually:

```bash
# Run the standalone cleanup script
python cleanup_test_artifacts.py
```

This will remove:
- ✅ Empty test flag directories
- ✅ Known test result JSON files  
- ✅ Python cache directories
- ❌ Important project files (preserved)

## What Gets Cleaned Up

### Empty Directories
- `test_flags/` and subdirectories (if empty)
- `custom_flags/` and subdirectories (if empty)  
- `test_integration_flags/` and subdirectories (if empty)

### Result Files
- `scoring_pattern_analysis.json`
- `validation_results.json`
- `edge_case_test_results.json`
- Other JSON files matching test result patterns

### Cache Directories
- `__pycache__/` directories (all locations)
- `.pytest_cache/` directories

## What's Preserved

The cleanup system is designed to be safe and only removes known test artifacts:

✅ **Always Preserved:**
- Source code files (`.py`)
- Test definitions (`domains/`)
- Configuration files
- Documentation (`.md`)
- Data files (`data/`)
- Important JSON files (not test results)

## Cleanup Features

### Smart Detection
The cleanup script uses intelligent detection to identify test artifacts:

- **Pattern matching**: Recognizes test result file naming patterns
- **Empty directory detection**: Only removes completely empty directory trees
- **Safe fallback**: Won't remove files unless it's confident they're test artifacts

### Error Handling
- **Graceful failures**: Reports errors but continues cleaning other items
- **Permission handling**: Handles file permission issues appropriately
- **Dry-run capable**: Easy to modify for preview mode

## Integration Examples

### CI/CD Integration
```bash
# In your CI pipeline
make test || (echo "Tests failed but cleaning up..." && make clean && exit 1)
```

### Development Workflow
```bash
# Clean development cycle
make test-quick    # Quick test + cleanup
# ... make changes ...
make test-unit     # Unit tests + cleanup  
# ... more changes ...
make test          # Full test + cleanup
```

### Watch Mode (Future Enhancement)
```bash
# Automatically re-run tests on file changes with cleanup
make test-watch
```

## Troubleshooting

### Permission Issues
If cleanup fails due to permissions:
```bash
# Make cleanup script executable
chmod +x cleanup_test_artifacts.py

# Or run with explicit python
python cleanup_test_artifacts.py
```

### Partial Cleanup
If some artifacts remain:
- Check the cleanup output for error messages
- Manually remove stubborn directories: `rm -rf test_flags/`
- Verify no processes are using the files

### Custom Result Files
To clean additional result files, modify `cleanup_test_artifacts.py`:

```python
# Add to known_test_files list
known_test_files = [
    "your_custom_results.json",
    # ... existing files
]
```

## Best Practices

1. **Always use automated cleanup**: Use `make test` or `run_tests_with_cleanup.py`
2. **Regular cleaning**: Run `make clean` periodically during development
3. **Check before committing**: Ensure no test artifacts are committed to git
4. **CI integration**: Include cleanup in your CI/CD pipeline

This keeps your workspace clean and prevents test artifact accumulation!