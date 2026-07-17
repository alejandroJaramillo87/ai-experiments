"""Structured logging for the training and interp CLIs (see roadmap.md, OBS)."""

from common.obs.core import (
    StructuredLogger,
    bind_run,
    configure,
    current_run_id,
    get_logger,
    new_run_id,
)

__all__ = [
    "StructuredLogger",
    "bind_run",
    "configure",
    "current_run_id",
    "get_logger",
    "new_run_id",
]
