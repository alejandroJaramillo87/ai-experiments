# Testing investigations

The frozen audit of the sandbox test suite: what the training and interp packages were
covered by, and what was deliberately left untested, on the day it was written.

The record's body is never edited, and drift against the current suite is expected and
correct (`.styles/DOCUMENTATION_STYLE.md`). Read it for the reasoning behind the gaps,
not for the current state, which [../test-suite.md](../test-suite.md) owns.

## Records

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [test_coverage_investigation.md](test_coverage_investigation.md) | Audit of the sandbox test suite, its coverage of the training and interp packages, dead and duplicated code, and readiness for the forward model policy (Gemma 4 primary, Qwen3 fallback). |

<!-- inventory:pages:end -->

## Related documentation

- [../README.md](../README.md) — how the repository is tested now
- [../../../tests/README.md](../../../tests/README.md) — the suite's layout, generated
  from its directories
