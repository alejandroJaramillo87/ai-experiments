# Test Output Style Guide

This guide defines the canonical output format for all test programs and test runners in llama-zen-turbo and related projects.

## Core Principles

### Unix Philosophy
- Tests do one thing well
- Output is simple, direct, and parseable
- Exit codes are the primary success indicator (0=success, non-zero=failure)
- Tests are composable and can be piped
- No unnecessary complexity or abstractions

### Information Hierarchy
1. Status comes first (PASS/FAIL)
2. Details follow status
3. Summary comes last
4. Progressive disclosure through verbosity levels

## Status Indicators

All status indicators use ANSI color codes and consistent formatting:

| Indicator | Color | Usage |
|-----------|-------|-------|
| **[OK]** | Green | Test or assertion passed |
| **[FAIL]** | Red | Test or assertion failed |
| **[WARN]** | Yellow | Warning or non-critical issue |
| **[RUN]** | Blue | Test execution started |
| **[SKIP]** | Blue | Test was skipped |
| **[INFO]** | Cyan | Informational message |
| **[TEST]** | Bold | Test name/description |

## Output Structure

### Basic Test Output
```
[TEST] Test description
[RUN] Running specific operation
[OK] Operation succeeded
```

### Test Runner Output
```
===========================================
Test Suite Name
===========================================

===========================================
Category Name
===========================================

[RUN] test_name
[OK] test_name completed (0.001s)

===========================================
Test Summary
===========================================

Results:
  Passed:  5/6
  Failed:  1/6

Duration: 1s

ALL TESTS PASSED
```

## Formatting Rules

### Headers
- Use 45 equal signs for major headers
- Headers are bold when color is available
- Empty line before and after header blocks

### Spacing
- Single empty line between tests
- No empty lines within a test unless separating logical sections
- No trailing newlines in messages

### Indentation
- Sub-tests indent with 2 spaces
- Multi-line output indents continuation lines
- Hierarchical information uses consistent indentation

### Colors
- Must be disable-able with --no-color flag
- Use standard ANSI escape codes
- Always reset color after colored text

## Verbosity Levels

### Default Output
Show only essential information:
```
[RUN] test_memory_boundaries
[OK] test_memory_boundaries completed (.001s)
```

### Verbose Output (--verbose)
Show detailed progress and intermediate results:
```
[RUN] test_memory_boundaries
[TEST] Memory allocation boundaries
[INFO] Threshold: 1073741824 bytes (1.000 GB)

[RUN] Testing: Exactly 1GB
  File size: 1073741824 bytes (1.000 GB)
  Expected: Use hugepages
  Result: OK

[OK] test_memory_boundaries completed (.001s)
```

## Implementation

### C++ Tests
Use the common test_colors.h header:

```cpp
#include "../include/test_colors.h"

int main() {
    PRINT_TEST("Test description");
    PRINT_RUN("Running operation");

    if (success) {
        PRINT_OK("Operation succeeded");
        return 0;
    } else {
        PRINT_FAIL("Operation failed: %s", error);
        return 1;
    }
}
```

### Shell Scripts
Use printf for portability (not echo -e):

```bash
# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

print_ok() {
    printf "${GREEN}[OK]${NC} %s\n" "$1"
}

print_fail() {
    printf "${RED}[FAIL]${NC} %s\n" "$1"
}
```

## Test Categories

### Unit Tests
- Test individual components
- Execute in <1 second
- Minimal output by default

### Functional Tests
- Test complete features
- May show progress for longer operations
- Include boundary and edge cases

### Integration Tests
- Test end-to-end workflows
- Show major steps in progress
- May include system interaction details

## Error Reporting

### Failure Output
- Show failure immediately when it occurs
- Include actionable error message
- Preserve context (what was being tested)

Example:
```
[RUN] test_cpu
[FAIL] CPU is not AMD Zen 5
  Expected: Family 0x1A
  Found: Family 0x19
  This optimizer requires AMD Zen 5 processors
```

### Warning Output
- Use for non-fatal issues
- Include explanation of impact
- Continue test execution

Example:
```
[WARN] valgrind not found, skipping memory leak check
```

## Summary Sections

### Test Summary Format
```
Results:
  Passed:  10/12
  Failed:  2/12
  Skipped: 1/13  (only if any were skipped)

Failed Tests:
  ✗ test_name1
  ✗ test_name2

Duration: 2s

TESTS FAILED  (or ALL TESTS PASSED)
```

## Best Practices

### DO:
- Keep messages concise and direct
- Use consistent terminology
- Include actionable error messages
- Provide enough context to debug failures
- Make output parseable by scripts
- Use exit codes correctly

### DON'T:
- Add decorative elements or ASCII art
- Use emojis or Unicode symbols (except ✗ in summaries)
- Include unnecessary verbose output by default
- Create deeply nested output structures
- Mix stdout and stderr inappropriately
- Over-engineer test infrastructure

## Examples

### Simple Test Success
```
[TEST] CPU detection
[OK] AMD Zen 5 detected
```

### Test with Sub-tests
```
[TEST] Memory boundaries
  Testing 1GB threshold...OK
  Testing 2GB allocation...OK
  Testing partial mapping...OK
[OK] All boundary tests passed
```

### Integration Test Output
```
Step 1: Checking library build...
[OK] Library found: ../build/libzen5_optimizer.so

Step 2: Creating test file...
[OK] Test file created: /tmp/test.dat

Step 3: Running with LD_PRELOAD...
[OK] Library loaded successfully
[OK] Hugepage allocation detected
```

## Compliance

All tests must:
1. Follow this style guide consistently
2. Return appropriate exit codes
3. Support --verbose and --no-color flags where applicable
4. Include clear, actionable error messages
5. Complete in reasonable time (<1s for unit, <5s for integration)

---

This style guide is inspired by Unix testing traditions and modern developer experience best practices. When in doubt, choose simplicity and clarity over cleverness.