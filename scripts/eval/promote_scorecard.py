#!/usr/bin/env python3
"""Promote an F7 eval run's scorecard into docs/training/scorecards/ (F8).

Copies scorecard.json and scorecard.md verbatim from an eval run directory
(/mnt/ai-data/training/eval/<name>_<UTC>/) into the committed reference
location as <card>.scorecard.json / <card>.scorecard.md, and writes a
<card>.provenance.json stamping the source run directory, the repository
git SHA, the promotion time, and the sha256 of each copied file. Committed
scorecards are never edited by hand; re-validation re-runs this tool and
replaces the pair wholesale.

This is a HOST tool: it reads /mnt/ai-data and depends only on the Python
standard library.

Exit codes: 0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DEST = os.path.join(REPO_ROOT, "docs", "training", "scorecards")
SCORECARD_FILES = ("scorecard.json", "scorecard.md")


class Printer:
    """Styled status output (see .styles/PYTHON_OUTPUT_STYLE.md)."""

    def __init__(self) -> None:
        use_color = not os.environ.get("NO_COLOR") and sys.stdout.isatty()
        if use_color:
            self._c = {"OK": "\033[0;32m", "ERROR": "\033[0;31m",
                       "INFO": "\033[0;36m"}
            self._nc = "\033[0m"
        else:
            self._c = {}
            self._nc = ""

    def _status(self, tag: str, message: str) -> None:
        color = self._c.get(tag, "")
        print(f"{color}[{tag}]{self._nc} {message}")

    def ok(self, m: str) -> None:
        self._status("OK", m)

    def error(self, m: str) -> None:
        self._status("ERROR", m)

    def info(self, m: str) -> None:
        self._status("INFO", m)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True).stdout.strip()


def promote(run_dir: str, card: str, dest: str, out: Printer) -> int:
    sources = [os.path.join(run_dir, name) for name in SCORECARD_FILES]
    missing = [p for p in sources if not os.path.isfile(p)]
    if missing:
        out.error(f"Scorecard not found: {', '.join(missing)}")
        print("  Solution: point --run-dir at a completed eval-ft run "
              "directory containing scorecard.json and scorecard.md")
        return 1

    os.makedirs(dest, exist_ok=True)
    copied: dict[str, str] = {}
    for src in sources:
        ext = os.path.splitext(src)[1]
        target = os.path.join(dest, f"{card}.scorecard{ext}")
        shutil.copyfile(src, target)
        copied[os.path.basename(target)] = sha256_file(target)
        out.ok(f"Promoted: {os.path.relpath(target, REPO_ROOT)}")

    provenance = {
        "card": card,
        "source_run_dir": os.path.abspath(run_dir),
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "sha256": copied,
    }
    prov_path = os.path.join(dest, f"{card}.provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")
    out.ok(f"Provenance: {os.path.relpath(prov_path, REPO_ROOT)}")
    out.info("Commit the promoted files; committed scorecards are never "
             "edited by hand")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote an eval run's scorecard into "
                    "docs/training/scorecards/")
    parser.add_argument("--run-dir", required=True,
                        help="eval-ft run directory containing scorecard.json "
                             "and scorecard.md")
    parser.add_argument("--card", required=True,
                        help="recipe card name the scorecard validates, e.g. "
                             "sft-qlora-gemma-e4b")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"destination directory (default: {DEFAULT_DEST})")
    args = parser.parse_args(argv)

    out = Printer()
    if not os.path.isdir(args.run_dir):
        out.error(f"Run directory not found: {args.run_dir}")
        return 1
    return promote(args.run_dir, args.card, args.dest, out)


if __name__ == "__main__":
    sys.exit(main())
