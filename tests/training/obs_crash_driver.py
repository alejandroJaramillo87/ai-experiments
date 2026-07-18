"""Subprocess driver for the SIGKILL durability test (roadmap OBS-10).

Not a test module (no test_ prefix): the durability test and the host
verification runbook both spawn it. Binds a run identity, opens a jsonl
Tracker, prints READY, then logs steps forever — the parent SIGKILLs it
mid-write and asserts every completed line in metrics.jsonl still parses.

Usage: python tests/training/obs_crash_driver.py <run_dir>
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.obs import core  # noqa: E402
from training.recipes.schema import TrackingConfig  # noqa: E402
from training.runtime.tracking import Tracker  # noqa: E402

RUN_ID = "R-20260718-000000-c8a54d"


def main() -> int:
    run_dir = sys.argv[1]
    # bind_run alone provides the header identity; configure is skipped so
    # the driver writes nothing outside run_dir
    core.bind_run("training", "crash-driver", RUN_ID)
    tracker = Tracker(
        TrackingConfig(tensorboard=False),
        "crash",
        run_dir,
        header={"tier": "sft", "model_id": "test/model", "recipe_hash": "dead01"},
    )
    tracker.log({"loss": 1.0}, 0)
    print("READY", flush=True)
    step = 1
    while True:  # runs until the parent kills the process
        tracker.log({"loss": 1.0 / (step + 1), "tokens_per_s": step * 10.0}, step)
        step += 1


if __name__ == "__main__":
    sys.exit(main())
