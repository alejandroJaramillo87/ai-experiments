"""Repository provenance shared by every manifest writer.

One repo-root-pinned git_sha() replaces the eight per-module copies whose
failure contracts had drifted ('unknown' vs None vs raise; process-cwd vs
repo-root). Default contract: the HEAD commit of this checkout, 'unknown' on
any failure. strict=True re-raises instead, for tools whose provenance must
be real (scorecard promotion).
"""

from __future__ import annotations

import os
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git_sha(repo_root: str | None = None, strict: bool = False) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root or REPO_ROOT,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except Exception:
        if strict:
            raise
        return "unknown"
