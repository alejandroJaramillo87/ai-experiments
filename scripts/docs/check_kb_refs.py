#!/usr/bin/env python3
"""Fail the quality gate when a KB identifier does not resolve to a ledger row.

The known-bugs ledger is cited from far outside itself: thirty-one KB ids sit in
permanent source comments explaining why a workaround is there, and
tests/interp/test_harvest.py asserts on one of them as a literal string. That
makes a KB id a cross-reference between code and documentation with nothing
holding the two ends together, and the reference breaks in the direction that is
hardest to notice -- the code keeps compiling, the comment keeps reading
sensibly, and only the id has stopped meaning anything.

This check was written by hand during the 2026-08-08 ledger reset and earned its
place immediately: it caught a row deleted as an unverified claim when the
runbook recorded it as observed on the workstation. The deletion was mine and I
would not have found it by re-reading, which is the argument for a gate rather
than a careful pass.

Resolution is deliberately one-directional. Every id that appears anywhere must
be defined; an id defined but never cited is fine, because a ledger row is
allowed to be the only record of its own bug.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

# Same prune set as the sibling gates.
PRUNE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".ruff_cache",
    ".pytest-tmp",
}

# The two documents that may define an id: the live ledger and the frozen archive
# holding the closed history moved out of it.
LEDGERS = (
    "docs/verification/bugs/known-bugs.md",
    "docs/verification/archive/ledger-archive.md",
)

# Files scanned for citations. Wider than the doc gates because the point of the
# check is that code cites these ids too.
SCAN_SUFFIXES = (".md", ".py", ".sh", ".yaml", ".yml", ".toml", ".mk", ".rs")

# A ledger row defines its id in the first table column.
DEFINITION = re.compile(r"^\|\s*(KB-\d+)\s*\|")
CITATION = re.compile(r"\bKB-(\d+)\b")

# The ledger reserves the next identifier in prose ("The next bug filed is KB-67"),
# which is by construction not yet defined. Matching the sentence rather than
# allowlisting the number keeps this self-maintaining: an allowlist would need
# editing every time a bug is filed, which is the moment nobody is thinking about
# this gate.
NEXT_ID = re.compile(r"next bug filed is\s+KB-\d+", re.IGNORECASE)

# Exempt by path component, matching the sibling gates. A frozen record quotes the
# tree as it stood on one day, and two ids it cites were retired in the 2026-08-08
# reset. Editing the record to fix them would falsify what it recorded, which is
# the one thing a frozen record exists not to do.
EXEMPT_PARTS = ("investigations", "archive", "vendor")

# Exempt by name, each for a stated reason.
EXEMPT_FILES = {
    # This file and its test spell ids to demonstrate the patterns.
    "scripts/docs/check_kb_refs.py",
    "tests/docs/test_kb_refs.py",
}


def is_exempt(rel_path: str) -> bool:
    """Whether a file is outside this gate's scope."""
    return (
        any(p in EXEMPT_PARTS for p in Path(rel_path).parts) or rel_path in EXEMPT_FILES
    )


def collect_defined(root: Path) -> set[str]:
    """Every id defined by a row in either ledger."""
    defined: set[str] = set()
    for rel in LEDGERS:
        path = root / rel
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = DEFINITION.match(line)
            if match:
                defined.add(match.group(1))
    return defined


def extract_citations(text: str) -> list[tuple[int, str]]:
    """Every (line number, id) pair citing a KB identifier."""
    found = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if NEXT_ID.search(line):
            continue
        for number in CITATION.findall(line):
            found.append((lineno, f"KB-{number}"))
    return found


def iter_files(root: Path) -> list[Path]:
    """Every in-scope file. Enumerated without git; see check_doc_refs.py."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in filenames:
            if not name.endswith(SCAN_SUFFIXES):
                continue
            path = Path(dirpath) / name
            if not is_exempt(str(path.relative_to(root))):
                found.append(path)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    defined = collect_defined(root)
    if not defined:
        out.error("kb-refs: the ledgers define no identifiers")
        out.line(f"  Expected rows in {', '.join(LEDGERS)}.")
        out.line("  Every citation in the tree would be reported as dangling.")
        return 1

    files = iter_files(root)
    if not files:
        out.error(f"kb-refs: enumerated zero files under {root}")
        out.line("  The gate would pass without checking anything.")
        return 1

    dangling: list[tuple[str, int, str]] = []
    cited: set[str] = set()
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Unreadable files are skipped rather than fatal, matching
            # check_retired_names.py: this gate scans binaries' neighbours.
            continue
        rel = str(path.relative_to(root))
        for lineno, kb_id in extract_citations(text):
            cited.add(kb_id)
            if kb_id not in defined:
                dangling.append((rel, lineno, kb_id))

    out.info(
        f"kb-refs: {len(files)} files, {len(cited)} identifiers cited, "
        f"{len(defined)} defined"
    )
    if not dangling:
        out.ok("kb-refs: every cited identifier resolves to a ledger row")
        return 0

    out.error(f"kb-refs: {len(dangling)} citations name no ledger row")
    for rel, lineno, kb_id in dangling:
        out.line(f"  {rel}:{lineno}: {kb_id}")
    out.line("  Solution: file the row, correct the id, or -- if the row was")
    out.line("  deleted -- add it to the deleted table so the id stays retired.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
