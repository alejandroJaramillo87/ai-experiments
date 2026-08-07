"""Observability package: structured logging keyed by an ambient run id.

Specified by docs/observability/design-and-roadmap.md. The package grows one module
per roadmap item; ``core`` (OBS-1) holds the logging core and ``dispatch``
(OBS-4) the traced CLI dispatch.
"""

from common.obs.core import (
    StructuredLogger,
    bind_run,
    configure,
    current_cli,
    current_command,
    current_run_dir,
    current_run_id,
    emit,
    get_logger,
    new_run_id,
    phase,
    set_run_dir,
)

__all__ = [
    "StructuredLogger",
    "bind_run",
    "configure",
    "current_cli",
    "current_command",
    "current_run_dir",
    "current_run_id",
    "emit",
    "get_logger",
    "new_run_id",
    "phase",
    "set_run_dir",
]
