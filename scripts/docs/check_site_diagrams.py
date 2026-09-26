#!/usr/bin/env python3
"""Fail the quality gate when a figura diagram copied into the site differs from its export.

site/static/img/diagrams/ holds copies of the static diagrams the figura
repository exports as SVG. figura is the source: a diagram is edited there,
exported there, and only then copied here. Nothing held the copies to that, so
on 2026-09-18 commit 02a546b edited tier-loop.svg in place to change one label,
and the copy stopped being any export figura had ever produced. The next
re-copy would have reverted it silently.

The record beside the copies, figura.json, names the figura commit the copies
came from and, per file, its export path in figura and its sha256. This gate
compares the directory against it: a copy the record does not name, a recorded
file that is absent, and a copy whose bytes hash differently all fail. It reads
no figura checkout and touches no network, so it runs in the sandbox; whether
the recorded hashes are the ones figura committed at the recorded commit is
checked once, by whoever updates the record, with the figura repository at hand.

The same record carries the snapshot of figura's design tokens that
check_site_tokens.py compares site/src/css/custom.css against. This gate reads
only the file half.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

DIAGRAMS_DIR = Path("site") / "static" / "img" / "diagrams"
RECORD_NAME = "figura.json"

FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def copies(directory: Path) -> list[str]:
    """Every file under the diagrams directory except the record, as relative paths."""
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != RECORD_NAME
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    directory = root / DIAGRAMS_DIR
    record_path = DIAGRAMS_DIR / RECORD_NAME
    try:
        record = json.loads((root / record_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        out.error(f"site-diagrams: cannot read {record_path}: {exc}")
        out.line("  The record of which figura export each copy came from is required.")
        return 1

    files = record.get("files") if isinstance(record, dict) else None
    # The empty-set guard. A record listing nothing would report every copy as
    # unrecorded, and an empty directory beside it would pass having checked nothing.
    if not isinstance(files, dict) or not files:
        out.error(f"site-diagrams: {record_path} records no files")
        out.line("  Expected a `files` object: copy name -> {export, sha256}.")
        return 1

    findings: list[str] = []
    commit = record.get("commit", "")
    if not isinstance(commit, str) or not FULL_COMMIT.match(commit):
        findings.append(
            f"{record_path}: commit {commit!r} is not a full 40-character figura hash"
        )

    present = copies(directory) if directory.is_dir() else []
    mismatched = False
    for name in present:
        if name not in files:
            findings.append(f"{DIAGRAMS_DIR / name}: not in the record")
    for name, entry in sorted(files.items()):
        path = DIAGRAMS_DIR / name
        export = entry.get("export", "") if isinstance(entry, dict) else ""
        recorded = entry.get("sha256", "") if isinstance(entry, dict) else ""
        if not isinstance(export, str) or PurePosixPath(export).name != name:
            findings.append(
                f"{path}: export path {export!r} does not name the same file"
            )
        if not isinstance(recorded, str) or not SHA256.match(recorded):
            findings.append(f"{path}: recorded sha256 {recorded!r} is malformed")
            continue
        if name not in present:
            findings.append(f"{path}: recorded but absent")
            continue
        actual = sha256(root / path)
        if actual != recorded:
            mismatched = True
            findings.append(
                f"{path}: sha256 {actual[:12]}, recorded {recorded[:12]} "
                f"(figura {export} at {str(commit)[:12]})"
            )

    out.info(
        f"site-diagrams: {len(files)} recorded files, {len(present)} copies "
        f"under {DIAGRAMS_DIR}"
    )
    if not findings:
        out.ok("site-diagrams: every copy is the figura export its record names")
        return 0

    noun = "finding" if len(findings) == 1 else "findings"
    out.error(f"site-diagrams: {len(findings)} {noun}")
    for finding in findings:
        out.line(f"  {finding}")
    if mismatched:
        out.line("  A copy whose sha256 differs was either edited here or copied from")
        out.line("  an export other than the one recorded.")
    out.line("  Solution: do not hand-edit; fix the diagram in figura, re-export,")
    out.line("  re-copy, and update the record: the figura commit, and each file's")
    out.line("  export path and sha256.")
    out.line(f"  Command: sha256sum {DIAGRAMS_DIR}/*.svg")
    return 1


if __name__ == "__main__":
    sys.exit(main())
