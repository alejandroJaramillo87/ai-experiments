"""Training-side serving policy plus re-exports of the shared client.

The registry, health poll, model swap, and chat client live in
common/serving.py (shared with the interp steering scorecard); this module
adds the training-idle preflight, which reads the D9 lock, and re-exports the
rest so training-side callers keep one import path.
"""

from __future__ import annotations

import os

from common.hashing import sha256_file  # noqa: F401  (re-exported)
from common.serving import (  # noqa: F401  (re-exported)
    CONTAINER_MODELS,
    HOST_MODELS,
    INFER_COMPOSE,
    NON_COMPOSE_SERVICES,
    REPO_ROOT,
    SERVICES,
    TRAIN_COMPOSE,
    ServiceSpec,
    ServingError,
    ServingTransientError,
    chat_completion,
    compose,
    model_paths,
    quant_from_name,
    swap_model,
    wait_healthy,
)
from training.runtime.lock import lock_path


def ensure_training_idle() -> None:
    """Refuse when the D9 lock is held.

    Serving-side consumers read the lock and refuse; they never acquire it
    (the lock names the training job as the GPU holder, which a serving tool
    is not). Running containers are not probed: the operator manages
    containers manually and validates GPU exclusivity themselves.
    """
    lock = lock_path()
    if os.path.exists(lock):
        raise ServingError(
            f"Training lock present: {lock}\n"
            "  Solution: wait for the training run to finish (one GPU consumer "
            "at a time)"
        )
