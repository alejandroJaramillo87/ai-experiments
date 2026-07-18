"""Event-file retention for the observability log directory (roadmap OBS-8).

configure() calls prune() opportunistically at startup, so the log
directory stays bounded without a separate scheduler. Only files matching
the run-id pattern (R-*.jsonl) are candidates; the catalog (runs.jsonl)
and any foreign files are never touched.
"""

from __future__ import annotations

import fcntl
import time
from pathlib import Path

SECONDS_PER_DAY = 86400


def _is_live(path: Path) -> bool:
    """True when a writer still holds the shared lock on its event file.

    configure() takes LOCK_SH on the file it writes; a long-idle run's
    mtime can age past any retention window while the file is still open,
    and unlinking it would send the rest of the run to a deleted inode.
    """
    try:
        with open(path, "rb") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(probe, fcntl.LOCK_UN)
    except OSError:
        return True  # unreadable: leave it to the per-file skip
    return False


def prune(log_dir: Path, retention_days: int) -> None:
    """Remove event files older than the retention window, best-effort.

    Age is file mtime, the run's last write. A non-positive window
    disables pruning. A file whose writer is still alive is skipped
    regardless of age. An unreadable directory or unremovable file is
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
            if path.stat().st_mtime < cutoff and not _is_live(path):
                path.unlink()
        except OSError:
            continue
