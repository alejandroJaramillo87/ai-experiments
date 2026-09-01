#!/usr/bin/env python3
"""Fail the gate when a document's stated count disagrees with itself.

A heading like `## Entries (3)` asserts a fact about the rows beneath it, and the
two are edited at different moments: a row is added, and the number in the
heading is the thing nobody scrolls back up to change. The failure is quiet and
it compounds, because the count is what a reader trusts instead of counting.

This is the narrowest possible self-consistency check. It compares a document
against itself, needs no knowledge of the tree, and cannot be wrong about what
the right answer is. It says nothing about whether the rows are correct -- only
that the document agrees with itself about how many there are.

What counts as a row is deliberately narrow: a table row, a list item, or an H3
between this heading and the next heading at the same level or higher. Anything
looser would start counting prose.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logtriage.output import Printer  # noqa: E402

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

# `## Entries (0)` or `### Open (12)`. The count must be the last thing on the
# line, so `## Gemma 4 (E4B, 12B)` is not read as a claim about rows.
COUNTED_HEADING_RE = re.compile(
    r"^(?P<hashes>#{2,4})\s+(?P<title>.+?)\s+\((?P<count>\d+)\)\s*$"
)
HEADING_RE = re.compile(r"^(#{1,6})\s")

# A markdown table row that is not the header separator.
TABLE_ROW_RE = re.compile(r"^\|.+\|\s*$")
SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|\s*$")
LIST_ITEM_RE = re.compile(r"^\s*[-*]\s+\S")
SUBHEADING_RE = re.compile(r"^###\s+\S")


def documents(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*.md"):
        if PRUNE_DIRS & set(path.parts):
            continue
        found.append(path)
    return sorted(found)


def count_rows(lines: list[str], start: int, level: int) -> int:
    """Rows under a counted heading, to the next heading at or above its level.

    A table's header row and its separator are not rows: they describe the
    columns, and counting them would make every table two too many.
    """
    rows = 0
    in_table = False
    table_header_seen = False
    for line in lines[start + 1 :]:
        heading = HEADING_RE.match(line)
        if heading and len(heading.group(1)) <= level:
            break
        if SEPARATOR_RE.match(line):
            # The separator marks the line above as the header, not a row.
            in_table = True
            table_header_seen = True
            rows -= 1 if rows else 0
            continue
        if TABLE_ROW_RE.match(line):
            rows += 1
            continue
        if in_table and not TABLE_ROW_RE.match(line):
            in_table = False
        if LIST_ITEM_RE.match(line) or SUBHEADING_RE.match(line):
            rows += 1
    return rows if table_header_seen or rows else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    out = Printer()
    root = Path(args.root).resolve()
    found = documents(root)

    if not found:
        out.error("check_doc_counts: enumerated zero documents")
        return 1

    wrong: list[tuple[Path, str, int, int]] = []
    claims = 0
    for document in found:
        lines = document.read_text(encoding="utf-8", errors="replace").splitlines()
        for index, line in enumerate(lines):
            match = COUNTED_HEADING_RE.match(line)
            if not match:
                continue
            claims += 1
            stated = int(match.group("count"))
            actual = count_rows(lines, index, len(match.group("hashes")))
            if stated != actual:
                wrong.append(
                    (document.relative_to(root), match.group("title"), stated, actual)
                )

    if wrong:
        out.error(f"check_doc_counts: {len(wrong)} heading(s) disagree with their rows")
        for document, title, stated, actual in wrong:
            out.line(f"  {document}: '{title}' says {stated}, counted {actual}")
        return 1

    out.ok(f"check_doc_counts: {claims} counted heading(s) agree with their rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
