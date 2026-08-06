import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = REPO_ROOT / "tests" / "agents"

# Where pytest puts tmp_path when the default base is unusable. Git-ignored.
SANDBOX_TMP = REPO_ROOT / ".pytest-tmp"


def _tmpdir_is_noexec() -> bool:
    """Whether a file written under the default temp base can be executed.

    In a contained agent sandbox /tmp is a tmpfs mounted noexec while /work is
    not, so any test that writes an executable stub into tmp_path -- a fake
    `docker`, a fake `llama-quantize` -- dies with `Permission denied` rather
    than exercising anything. That is an environment limitation, not a defect in
    the test, and it silently reads as a red suite. Probe once and relocate.
    """
    base = Path(os.environ.get("TMPDIR", "/tmp"))
    try:
        base.mkdir(parents=True, exist_ok=True)
        probe = base / f".exec-probe-{os.getpid()}"
        probe.write_text("#!/bin/sh\nexit 0\n")
        probe.chmod(probe.stat().st_mode | stat.S_IXUSR)
        try:
            return subprocess.run([str(probe)], capture_output=True).returncode != 0
        finally:
            probe.unlink(missing_ok=True)
    except OSError:
        return True


def pytest_configure(config):
    """Move the temp base onto an executable filesystem when the default is not.

    Set before tmp_path_factory is first used, so every tmp_path in the session
    lands somewhere a stub can actually run. On the workstation the probe passes
    and nothing changes.
    """
    if config.option.basetemp is None and _tmpdir_is_noexec():
        SANDBOX_TMP.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(SANDBOX_TMP)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def fake_bin(tmp_path, monkeypatch) -> SimpleNamespace:
    """Temp dir prepended to PATH for stubbing commands (e.g. sudo)."""
    bin_dir = tmp_path / "fake_bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    def stub(name: str, body: str = 'exec "$@"') -> Path:
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return path

    return SimpleNamespace(dir=bin_dir, stub=stub)


@pytest.fixture
def run_script():
    """Run a script under scripts/ (path relative to it) with captured output.

    NO_COLOR is set so output stays parseable; a timeout guards against a
    script blocking on stdin (e.g. an unexpected sudo password prompt).
    """

    def _run(
        script: str,
        *args: str,
        stdin: str = "",
        env: dict | None = None,
        timeout: int = 60,
    ):
        merged = {**os.environ, "NO_COLOR": "1", **(env or {})}
        return subprocess.run(
            [str(SCRIPTS_DIR / script), *args],
            input=stdin,
            capture_output=True,
            text=True,
            env=merged,
            cwd=REPO_ROOT,
            timeout=timeout,
        )

    return _run
