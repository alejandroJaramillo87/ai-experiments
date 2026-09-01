# Test Output Style Guide

Canonical output format for test programs and runners in llama-zen-turbo and related projects.
Unix philosophy: simple, parseable output; exit codes are the primary success signal
(0 = success). Status first, details next, summary last.

## Status Indicators

ANSI-colored, disable-able with `--no-color`:

| Indicator | Color | Usage |
|-----------|-------|-------|
| `[OK]` | Green | Test or assertion passed |
| `[FAIL]` | Red | Test or assertion failed |
| `[WARN]` | Yellow | Warning or non-critical issue |
| `[RUN]` | Blue | Test execution started |
| `[SKIP]` | Blue | Test was skipped |
| `[INFO]` | Cyan | Informational message |
| `[TEST]` | Bold | Test name/description |

## Structure

Basic test and runner output:

```
[TEST] Test description
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

Rules: 45 equal signs for major headers (bold when color available), blank line before and
after header blocks, single blank line between tests, 2-space indent for sub-tests. Always
reset color after colored text.

## Verbosity

Default shows only essentials; `--verbose` shows progress and intermediate results:

```
# default
[RUN] test_memory_boundaries
[OK] test_memory_boundaries completed (.001s)

# --verbose
[RUN] test_memory_boundaries
[TEST] Memory allocation boundaries
[INFO] Threshold: 1073741824 bytes (1.000 GB)
[RUN] Testing: Exactly 1GB
  Expected: Use hugepages
  Result: OK
[OK] test_memory_boundaries completed (.001s)
```

## Implementation

```cpp
// C++: use the common test_colors.h header
#include "../include/test_colors.h"
PRINT_TEST("Test description");
PRINT_RUN("Running operation");
if (success) { PRINT_OK("Operation succeeded"); return 0; }
else         { PRINT_FAIL("Operation failed: %s", error); return 1; }
```

```bash
# Shell: printf for portability (not echo -e)
GREEN='\033[0;32m' RED='\033[0;31m' NC='\033[0m'
print_ok()   { printf "${GREEN}[OK]${NC} %s\n" "$1"; }
print_fail() { printf "${RED}[FAIL]${NC} %s\n" "$1"; }
```

## Error Reporting

Show failure immediately with actionable detail and preserved context:

```
[RUN] test_cpu
[FAIL] CPU is not AMD Zen 5
  Expected: Family 0x1A
  Found: Family 0x19
  This optimizer requires AMD Zen 5 processors

[WARN] valgrind not found, skipping memory leak check
```

Failure summary lists failed tests and ends with `TESTS FAILED` or `ALL TESTS PASSED`.

## Guidelines

- DO: keep messages concise, use consistent terminology and correct exit codes, provide enough
  context to debug, make output parseable.
- DON'T: decorative elements or ASCII art, emojis or Unicode (except `✗` in summaries),
  verbose output by default, deeply nested structures, over-engineered infrastructure.
- Timing: <1s for unit tests, <5s for integration.
