"""List recent runs from the observability catalog (roadmap OBS-9).

Reads ${OBS_LOG_DIR}/training/runs.jsonl. The command surface lives in
common/obs/runs.py, shared with interp and the rust harness; this module binds
it to this CLI's name.
"""

from __future__ import annotations

from common.obs import runs


def configure(sub) -> None:
    runs.configure(sub, "training")
