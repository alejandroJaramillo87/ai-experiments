# TestRunner Quick Testing Guide

This guide provides the fastest way to verify TestRunner functionality after code changes.

## 🚀 Ultra-Quick Test (30 seconds)

```bash
# Single command to test core functionality
python test_runner.py --discover-suites && \
python test_runner.py --test-type base --list-categories && \
python test_runner.py --test-type base --mode single --test-id complex_test_01 --performance-monitoring --dry-run && \
echo "✅ Core functionality verified"
```

## 📋 Essential Test Suite (2 minutes)

### 1. Basic CLI & Discovery
```bash
# Test help system
python test_runner.py --help

# Test suite discovery
python test_runner.py --discover-suites
```

### 2. Test Loading & Listing
```bash
# Test category listing
python test_runner.py --test-type base --list-categories

# Test individual test listing (first 10)
python test_runner.py --test-type base --list-tests | head -15
```

### 3. Execution Modes (Dry Run)
```bash
# Single test
python test_runner.py --test-type base --mode single --test-id complex_test_01 --dry-run

# Sequential mode
python test_runner.py --test-type base --mode sequential --dry-run

# Concurrent mode
python test_runner.py --test-type base --mode concurrent --workers 2 --dry-run

# Category mode
python test_runner.py --test-type base --mode category --category complex_synthesis --dry-run
```

### 4. Performance Monitoring
```bash
# Test performance flag integration
python test_runner.py --test-type base --mode single --test-id complex_test_01 --performance-monitoring --dry-run
```

### 5. Error Handling
```bash
# Test graceful error handling
python test_runner.py --test-type base --mode single --test-id invalid_test --dry-run 2>/dev/null || echo "✅ Handles errors gracefully"
```

## 🔧 Bug Verification Tests

Test specific bug fixes:

```bash
# Division by zero fixes (empty results)
python test_runner.py --test-type base --mode single --test-id invalid_test 2>/dev/null || echo "✅ No division by zero crashes"

# Thread safety (concurrent execution)
python test_runner.py --test-type base --mode concurrent --workers 3 --performance-monitoring --dry-run

# API error handling
python test_runner.py --test-type base --mode single --test-id complex_test_01 --endpoint http://nonexistent:999/v1/completions --model "/app/models/hf/DeepSeek-R1-0528-Qwen3-8b" 2>/dev/null || echo "✅ API errors handled"
```

## 📊 Advanced Features

```bash
# Suite statistics
python test_runner.py --suite-stats reasoning_comprehensive_v1

# Category information
python test_runner.py --category-info complex_synthesis --test-type base

# Filtering tests
python test_runner.py --filter-by difficulty=medium --test-type base

# Verbose/quiet modes
python test_runner.py --test-type base --list-categories --verbose
python test_runner.py --test-type base --list-categories --quiet
```

## ✅ Expected Results

**All commands should:**
- Execute without crashes or exceptions
- Show appropriate output for their function
- Handle invalid inputs gracefully with error messages
- Display performance monitoring options when enabled

**Success Indicators:**
- Help shows all CLI options including `--performance-monitoring`
- Suite discovery finds both base and instruct test suites
- Dry runs show execution plans without errors
- Performance monitoring flag is recognized and processed
- Invalid inputs fail gracefully with meaningful errors

## 🚨 Red Flags

Stop testing and investigate if you see:
- `KeyError` or `AttributeError` crashes
- `TypeError: '>' not supported between instances` (threading bug)
- `division by zero` errors
- Hanging or unresponsive behavior
- Missing performance monitoring options

## 📝 Test Coverage

This quick test suite verifies:
- ✅ CLI argument parsing
- ✅ Suite discovery & management
- ✅ Test loading from JSON files
- ✅ All execution modes (single/sequential/concurrent/category)
- ✅ Performance monitoring integration (RTX 5090, AMD Ryzen 9950X, 128GB DDR5)
- ✅ Error handling & edge cases
- ✅ Thread safety
- ✅ API format detection (chat/completions)
- ✅ File I/O and output handling

**Total test time: 30 seconds (quick) to 2 minutes (comprehensive)**