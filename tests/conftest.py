import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIXTURES_DIR = REPO_ROOT / "tests" / "agents"


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
