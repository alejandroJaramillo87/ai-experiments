"""Fixtures for the observability core tests (OBS-1).

Runs CPU-only, no network or services. A new tests/common/ dir has no path
bootstrap of its own (the root tests/conftest.py does not insert REPO_ROOT), so
this file makes `from common.obs import ...` resolvable, matching the interp
conftest pattern.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(autouse=True)
def _reset_obs(monkeypatch, tmp_path):
    """Isolate obs logging state per test.

    Points the log directory at a per-test tmp_path (the sandbox has no
    /mnt/ai-data) and clears AIX_LOG_LEVEL. On teardown, removes and closes the
    obs handlers, resets the package logger level, and clears the run-context
    ContextVar so no handler, file descriptor, or correlation id leaks forward.
    """
    monkeypatch.setenv("AIX_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("AIX_LOG_LEVEL", raising=False)
    yield
    from common.obs import core

    for handler in list(core._pkg_logger.handlers):
        if isinstance(handler, core._ObsStreamHandler | core._ObsFileHandler):
            handler.close()
            core._pkg_logger.removeHandler(handler)
    core._pkg_logger.setLevel(logging.NOTSET)
    core._run_context.set(core._RunContext())
