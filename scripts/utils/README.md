# scripts/utils

Host-environment reporters: which Pythons are installed, whether the AI dependencies and
the repository's optimizations are present, how the two SSDs are used, and whether the
PyTorch GPU stack imports. All run in a sandbox, where they report what is absent rather
than failing; `tests/utils/` drives two of them.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `check_py_deps_install.py` | PyTorch GPU environment validation utility. |
| `dependency_check.sh` | Checks and reports on AI engineering dependencies and repository-defined optimizations. |
| `find_pythons.sh` | Project-specific Python environment checker for AI experiments. |
| `storage_check.sh` | Comprehensive storage analysis for AI workstation with dual SSDs. |

<!-- inventory:scripts:end -->

## Where the real docs live

- [../../docs/os/python/setup_python.md](../../docs/os/python/setup_python.md) — the Python layout `find_pythons.sh` checks
- [../../docs/os/storage-configuration.md](../../docs/os/storage-configuration.md) — the SSD layout `storage_check.sh` reports on
