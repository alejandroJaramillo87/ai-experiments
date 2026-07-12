"""Output-contract tests for the scripts/utils/ bash reporters.

These scripts report live host state and have no fixture modes, so the tests
assert the stable contract instead of specific values: exit 0, NO_COLOR is
honored, only style-guide status tags appear, and the section headers exist.
storage_check.sh runs with a stubbed sudo so it can never block on a password
prompt.
"""

import re

ANSI_RE = re.compile(r"\x1b\[")
STATUS_RE = re.compile(r"^\s*\[([A-Z]+)\]", re.MULTILINE)
ALLOWED_TAGS = {"OK", "WARN", "ERROR", "INFO", "SKIP"}


def assert_output_contract(result):
    assert result.returncode == 0
    assert not ANSI_RE.search(result.stdout)
    tags = set(STATUS_RE.findall(result.stdout))
    assert tags <= ALLOWED_TAGS, f"unexpected status tags: {tags - ALLOWED_TAGS}"


def test_find_pythons(run_script):
    result = run_script("utils/find_pythons.sh")
    assert_output_contract(result)
    assert "=== AI Experiments Python Environment Check ===" in result.stdout
    assert "Expected Python version check:" in result.stdout
    assert "Poetry configuration check:" in result.stdout
    # Documented workstation state (docs/os/python/setup_python.md).
    assert "[OK] Python 3.12.11: installed via pyenv" in result.stdout


def test_dependency_check(run_script):
    result = run_script("utils/dependency_check.sh", timeout=120)
    assert_output_contract(result)
    assert "=== AI Engineering Dependency & Optimization Report ===" in result.stdout
    assert "=== Optimization Summary ===" in result.stdout
    assert "Report complete:" in result.stdout


def test_storage_check(run_script, fake_bin):
    # Stub sudo so the script cannot block on a password prompt; the stubbed
    # commands run unprivileged and the script tolerates their failures.
    fake_bin.stub("sudo")
    result = run_script("utils/storage_check.sh", timeout=120)
    assert_output_contract(result)
    assert "=== Storage Analysis Report ===" in result.stdout
    assert "=== Storage Summary ===" in result.stdout
    assert "Report complete:" in result.stdout
