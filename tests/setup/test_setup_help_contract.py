"""Help-text contract tests for the provisioning scripts in scripts/setup/.

Nothing else in scripts/setup/ is testable here: every install path needs sudo,
the host filesystem, or the GPU, and is verified by review and by the human
running it (scripts/README.md). The --help path is the exception -- it touches
nothing -- so it is what the suite holds to the style guide.

The assertions are the same contract tests/utils/test_bash_output_contract.py
applies to the reporters: exit 0, no ANSI under NO_COLOR, and no status tag
outside the vocabulary in .styles/BASH_OUTPUT_STYLE.md.
"""

import re

import pytest

ANSI_RE = re.compile(r"\x1b\[")
STATUS_RE = re.compile(r"^\s*\[([A-Z-]+)\]", re.MULTILINE)
ALLOWED_TAGS = {"OK", "WARN", "ERROR", "INFO", "RUN", "SKIP", "DRY-RUN"}

SETUP_SCRIPTS = [
    "setup_cuda.sh",
    "setup_cudnn.sh",
    "setup_data_ssd.sh",
    "setup_docker.sh",
    "setup_nvidia.sh",
    "setup_security.sh",
    "setup_sudo_user.sh",
    "setup_terminal.sh",
]


@pytest.mark.parametrize("script", SETUP_SCRIPTS)
def test_help_contract(run_script, script):
    result = run_script(f"setup/{script}", "--help")
    assert result.returncode == 0, result.stderr
    assert not ANSI_RE.search(result.stdout)
    assert result.stdout.startswith(f"Usage: {script}")
    assert "--help" in result.stdout
    assert "--no-color" in result.stdout

    tags = set(STATUS_RE.findall(result.stdout))
    assert tags <= ALLOWED_TAGS, f"unexpected status tags: {tags - ALLOWED_TAGS}"


@pytest.mark.parametrize("script", SETUP_SCRIPTS)
def test_unknown_option_is_rejected(run_script, script):
    """An unrecognized flag must fail loudly rather than fall through to apt."""
    result = run_script(f"setup/{script}", "--not-a-real-option")
    assert result.returncode == 2
    assert "Unknown option" in result.stderr
