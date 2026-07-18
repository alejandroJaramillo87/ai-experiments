"""Event-file retention for the observability log directory (roadmap OBS-8).

configure() calls prune() opportunistically at startup, so the log
directory stays bounded without a separate scheduler. Only files matching
the run-id pattern (R-*.jsonl) are candidates; the catalog (runs.jsonl)
and any foreign files are never touched.
"""

from __future__ import annotations

import time
from pathlib import Path

SECONDS_PER_DAY = 86400


def prune(log_dir: Path, retention_days: int) -> None:
    """Remove event files older than the retention window, best-effort.

    Age is file mtime, the run's last write. A non-positive window
    disables pruning. An unreadable directory or unremovable file is
    skipped silently: retention must never break the command it rides on.
    """
    if retention_days <= 0:
        return
    cutoff = time.time() - retention_days * SECONDS_PER_DAY
    try:
        candidates = list(log_dir.glob("R-*.jsonl"))
    except OSError:
        return
    for path in candidates:
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue
