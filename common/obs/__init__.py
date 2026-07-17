"""Observability package: structured logging keyed by an ambient run id.

Specified by docs/observability/roadmap.md. The package grows one module
per roadmap item; ``core`` (OBS-1) holds the logging core.
"""

from common.obs.core import (
    StructuredLogger,
    bind_run,
    configure,
    current_run_dir,
    current_run_id,
    emit,
    get_logger,
    new_run_id,
    set_run_dir,
)

__all__ = [
    "StructuredLogger",
    "bind_run",
    "configure",
    "current_run_dir",
    "current_run_id",
    "emit",
    "get_logger",
    "new_run_id",
    "set_run_dir",
]
