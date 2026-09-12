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

import json
import os
import sys

import pytest

from common.obs.core import DEFAULT_LOG_DIR

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AI_DATA_ROOT = "/mnt/ai-data"

# The release harness the run-record suites spawn, and the obs root it
# resolves when OBS_LOG_DIR is unset. The root is spelled from the Python
# side's own constant rather than through catalog_path, which would honour an
# OBS_LOG_DIR in the test's shell that the child never sees.
HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
HARNESS = os.path.join(HARNESS_DIR, "target", "release", "harness")
LOG_ROOT = os.path.join(DEFAULT_LOG_DIR, "harness")

# The repo root is on sys.path already (tests/conftest.py), which is what makes
# `from tests.hardware.conftest import ...` resolvable at collection time.
# Reuse the synthetic-shard builder the training suite uses (tests/training
# is path-imported, matching the existing `from support import ...` pattern).
sys.path.insert(0, os.path.join(REPO_ROOT, "tests", "training"))


def e4b_checkpoint() -> str:
    """The E4B tier names the checkpoint through HARNESS_E4B_MODEL; outside
    that tier fall back to the same root the sibling weight test reads."""
    named = os.environ.get("HARNESS_E4B_MODEL")
    if named:
        return named
    root = os.environ.get("GEMMA4_CHECKPOINT_ROOT", f"{AI_DATA_ROOT}/models/hf")
    return os.path.join(root, "gemma-4-E4B")


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built by `make
    harness-build` on the same host, so its absence is a misconfigured run
    and not the unavailable hardware the module rule exempts."""
    assert os.path.isfile(HARNESS), (
        f"built binary absent: {HARNESS}; run `make harness-build` in {HARNESS_DIR}"
    )


def default_root_env() -> dict[str, str]:
    """The child's environment with OBS_LOG_DIR removed, so the binary resolves
    the default root. That resolution is a fact the record suites check;
    inheriting an override from the shell would test the override instead."""
    env = {k: v for k, v in os.environ.items() if k != "OBS_LOG_DIR"}
    env["NO_COLOR"] = "1"
    return env


def event_file(run_id: str) -> str:
    return os.path.join(LOG_ROOT, f"{run_id}.jsonl")


def events(run_id: str) -> list[dict]:
    """Every record of one run's event file under the default root."""
    path = event_file(run_id)
    assert os.path.isfile(path), f"no event file for {run_id} at {path}"
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
