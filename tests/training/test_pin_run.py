"""scripts/training/pin_run.sh: the host-side taskset wrapper (D9).

The script has no fixture mode, so the tests drive it as a subprocess with a
stub taskset on a scratch PATH that records its arguments instead of pinning
anything. Covers the usage error, the missing-taskset refusal, and the exec
passthrough of cpu list, command, arguments, and exit code."""

from __future__ import annotations

import os
import stat
import subprocess

import pytest
from support import REPO_ROOT

SCRIPT = os.path.join(REPO_ROOT, "scripts", "training", "pin_run.sh")

TASKSET_STUB = """#!/bin/sh
# Record the arguments, then run the wrapped command so its exit code flows back.
printf 'TASKSET_ARGS:%s\\n' "$*"
shift 2
exec "$@"
"""


def run(args: list[str], path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", SCRIPT, *args],
        capture_output=True,
        text=True,
        env={"PATH": path, "NO_COLOR": "1"},
        timeout=30,
    )


@pytest.fixture
def stub_path(tmp_path) -> str:
    """A PATH with sh, bash and coreutils but only the stub taskset."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "taskset"
    stub.write_text(TASKSET_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return f"{bin_dir}:/usr/bin:/bin"


def test_usage_error_exits_2_without_pinning(stub_path):
    result = run(["0-7"], stub_path)
    assert result.returncode == 2
    assert "[ERROR] Usage:" in result.stdout
    assert "TASKSET_ARGS" not in result.stdout


def test_exec_passes_cpu_list_command_and_exit_code(stub_path):
    result = run(["0-7,16-23", "sh", "-c", "echo hello; exit 7"], stub_path)
    assert result.returncode == 7
    assert "[INFO] CPU affinity: 0-7,16-23" in result.stdout
    assert "TASKSET_ARGS:-c 0-7,16-23 sh -c echo hello; exit 7" in result.stdout
    assert "hello" in result.stdout


def test_missing_taskset_exits_1_naming_util_linux(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    # bash itself must still resolve, so keep the system dirs but shadow
    # nothing: taskset is only absent when util-linux is not installed, which
    # is not this host, so hide it with a directory that lacks it and a PATH
    # that does not reach /usr/bin.
    result = subprocess.run(
        ["/bin/bash", SCRIPT, "0-7", "true"],
        capture_output=True,
        text=True,
        env={"PATH": str(empty), "NO_COLOR": "1"},
        timeout=30,
    )
    assert result.returncode == 1
    assert "[ERROR] taskset not found (util-linux)" in result.stdout
    assert "Solution:" in result.stdout


def test_no_color_output_uses_only_status_tags(stub_path):
    result = run(["0-7", "true"], stub_path)
    assert result.returncode == 0
    assert "\x1b[" not in result.stdout
