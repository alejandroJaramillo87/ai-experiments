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
import json
import os
import shutil
import sys
from datetime import UTC, datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.hashing import sha256_file  # noqa: E402
from common.output import Printer  # noqa: E402
from common.provenance import git_sha as provenance_git_sha  # noqa: E402

DEFAULT_DEST = os.path.join(REPO_ROOT, "docs", "training", "scorecards")
SCORECARD_FILES = ("scorecard.json", "scorecard.md")


def git_sha() -> str:
    # strict: promotion provenance must be real, so failure raises rather
    # than recording 'unknown'
    return provenance_git_sha(REPO_ROOT, strict=True)


def promote(run_dir: str, card: str, dest: str, out: Printer) -> int:
    sources = [os.path.join(run_dir, name) for name in SCORECARD_FILES]
    missing = [p for p in sources if not os.path.isfile(p)]
    if missing:
        out.error(f"Scorecard not found: {', '.join(missing)}")
        print(
            "  Solution: point --run-dir at a completed eval-ft run "
            "directory containing scorecard.json and scorecard.md"
        )
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
        "promoted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "sha256": copied,
    }
    prov_path = os.path.join(dest, f"{card}.provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
        f.write("\n")
    out.ok(f"Provenance: {os.path.relpath(prov_path, REPO_ROOT)}")
    out.info("Commit the promoted files; committed scorecards are never edited by hand")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Promote an eval run's scorecard into docs/training/scorecards/"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="eval-ft run directory containing scorecard.json and scorecard.md",
    )
    parser.add_argument(
        "--card",
        required=True,
        help="recipe card name the scorecard validates, e.g. sft-qlora-gemma-e4b",
    )
    parser.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    out = Printer()
    if not os.path.isdir(args.run_dir):
        out.error(f"Run directory not found: {args.run_dir}")
        return 1
    return promote(args.run_dir, args.card, args.dest, out)


if __name__ == "__main__":
    sys.exit(main())
