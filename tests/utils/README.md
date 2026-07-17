# Utility script tests

Tests for the host-environment reporters in `scripts/utils/`. Unlike the fixture-driven
suite in `tests/agents/`, these scripts have no fixture modes — they read live host
state (`nvidia-smi`, `sysctl`, uv, `.venv`, `/mnt/ai-data`), so the tests assert the
stable contract rather than specific values: exit codes, `NO_COLOR` handling, the
style-guide status vocabulary, and section headers.

## Test index

| Test file | Covers | Approach |
|-----------|--------|----------|
| `test_bash_output_contract.py` | `find_pythons.sh`, `dependency_check.sh`, `storage_check.sh` | run each script, assert exit 0, no ANSI under `NO_COLOR`, only `[OK]/[WARN]/[ERROR]/[INFO]/[SKIP]` tags, expected headers |
| `test_check_py_deps.py` | `check_py_deps_install.py` | unit tests for `Printer`, `_pkg_version`, `quiet_import`; end-to-end `--json` (pure JSON stdout) and human-mode contract (status on stderr) |

## Notes

- `storage_check.sh` invokes `sudo`; the test stubs `sudo` on `PATH` (runs the command
  unprivileged) so the suite can never block on a password prompt. A 120-second
  timeout backstops any other hang.
- The end-to-end checks and the bash scripts' PyTorch sections import torch through
  the uv `.venv` environment, which dominates suite runtime (roughly 30–45 seconds
  total). They run in plain `make test` by decision — one command runs everything.
- The end-to-end assertions accept exit codes 0 and 1 because
  `check_py_deps_install.py` reports host GPU state; the contract under test is the
  output format, not the host's health.
- Python human status writes to stderr (`common/output.py` per
  `docs/observability/roadmap.md` OBS-3), so tests capture status lines from stderr
  and reserve stdout assertions for `--json` payloads. Bash scripts keep their
  stdout output; `test_bash_output_contract.py` is unaffected.
