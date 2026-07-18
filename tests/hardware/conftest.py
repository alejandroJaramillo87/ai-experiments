"""Shared plumbing for the workstation hardware suites (WI-3).

Both tiers are deselected by default (pyproject addopts); the human runs
them on the workstation:

    uv run pytest -m hw_smoke tests/hardware/    # session-start gate
    uv run pytest -m hw_full tests/hardware/     # per verification phase

Tests FAIL (never skip) when the hardware they exist to prove is absent —
except real-model loads, which skip cleanly when weights are not present
(gated hub access is out of scope by decision). Modules import without
torch so sandbox collection stays clean; heavy imports live inside tests.
"""

from __future__ import annotations

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AI_DATA_ROOT = "/mnt/ai-data"

# Make `from tests.hardware.conftest import ...` resolvable at collection
# time (the tests/interp conftest pattern); without this, module-level
# imports in this directory abort collection of the whole suite.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse the synthetic-shard builder the training suite uses (tests/training
# is path-imported, matching the existing `from support import ...` pattern).
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "training"))


@pytest.fixture()
def torch():
    import torch

    return torch


@pytest.fixture()
def cuda(torch):
    """torch with a CUDA device, failing loudly when absent: an hw test
    running on a GPU-less host is a misconfiguration, not a skip."""
    assert torch.cuda.is_available(), (
        "CUDA unavailable; hardware tests run on the workstation (RTX 5090) only"
    )
    return torch


@pytest.fixture()
def train_lock(tmp_path, monkeypatch):
    """Point the D9 run lock at tmp so hardware runs cannot collide with a
    stale real lock file (one GPU consumer at a time still holds: pytest
    runs these serially)."""
    monkeypatch.setenv("TRAIN_LOCK_PATH", str(tmp_path / ".train.lock"))
