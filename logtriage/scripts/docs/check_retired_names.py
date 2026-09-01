#!/usr/bin/env python3
"""Fail the gate when a retired name is still in use anywhere in this project.

check_doc_refs.py catches a document naming a path that does not exist. It
cannot catch a *renamed* file, because its classifier requires a slash: a bare
`generator.py` in prose is not a path claim and is never checked. Loosening that
rule was measured in the parent repository at a 67% false-positive rate, because
prose legitimately says `config.json` and `main.rs` in good faith.

So this takes the opposite approach. Rather than infer which tokens are claims,
it holds an explicit table of names that have been retired and fails on any
occurrence of one. That has no false positives by construction, and it names the
replacement in the error, so the fix is mechanical rather than archaeological.

The table is the point. Adding an entry when you rename something is what makes
the next rename self-policing; a rename that skips this file is a rename whose
stragglers a human finds months later.

Unlike check_doc_refs.py this scans code as well as documentation, because a
retired filename hides in a recipe, a Python import, or a test fixture just as
readily as in prose -- and there it is a broken command, not a stale sentence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logtriage.output import Printer  # noqa: E402

# Retired name -> (replacement, why it was retired). Both halves are shown in
# the error, so whoever hits it can fix it without going digging.
#
# Empty is the correct state for a project that has not renamed anything yet.
# It is not a reason to skip the file on the next rename: an empty table that
# is never added to is the same as no table at all, and the whole value here is
# that the rule is written down before it is needed.
#
# Example of the shape an entry takes:
#
#   "old_module.py": ("new_module.py", "split when the parser grew a resolver"),
RETIRED: dict[str, tuple[str, str]] = {}

# Scanned: everything textual this project owns. Pruned: caches, the fetched
# third-party corpus, and .styles/ -- byte-identical copies of the parent's
# guides that a parent test forbids editing, so a hit there could not be fixed.
PRUNE_DIRS = {
    ".git",
    ".venv",
    ".kb",
    ".styles",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}

SUFFIXES = (".py", ".md", ".toml", ".yaml", ".yml", ".json", ".cfg", ".sh", ".txt")

# Frozen records name what things were called on their date, which is the point
# of the class. Same exemption as check_doc_refs.py, for the same reason.
EXEMPT_PARTS = {"investigations"}
EXEMPT_SUFFIXES = ("_investigation.md",)

# This file necessarily contains every retired name.
SELF = Path(__file__).name


def scanned(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        parts = set(path.parts)
        if PRUNE_DIRS & parts or EXEMPT_PARTS & parts:
            continue
        if path.name.endswith(EXEMPT_SUFFIXES) or path.name == SELF:
            continue
        # Never committed, and its content is not ours.
        if path.name == "uv.lock":
            continue
        found.append(path)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    out = Printer()
    root = Path(args.root).resolve()
    files = scanned(root)

    if not files:
        out.error("check_retired_names: enumerated zero files")
        out.line("  The gate would pass having checked nothing.")
        return 1

    if not RETIRED:
        out.ok(
            f"check_retired_names: no names retired yet ({len(files)} files scanned)"
        )
        return 0

    hits: list[tuple[Path, int, str]] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        # Cheap rejection first: most files contain none of the names, and
        # splitting every file into lines to find that out is the slow way.
        if not any(name in text for name in RETIRED):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for name in RETIRED:
                if name in line:
                    hits.append((path.relative_to(root), number, name))

    if hits:
        out.error(f"check_retired_names: {len(hits)} use(s) of a retired name")
        for path, number, name in hits:
            replacement, reason = RETIRED[name]
            out.line(f"  {path}:{number}: {name} -> {replacement} ({reason})")
        return 1

    out.ok(
        f"check_retired_names: {len(RETIRED)} retired name(s) absent "
        f"from {len(files)} files"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
