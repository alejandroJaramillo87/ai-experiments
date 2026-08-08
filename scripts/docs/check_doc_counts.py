#!/usr/bin/env python3
"""Fail the quality gate when a heading's stated count disagrees with its own section.

A heading like `## Open (3)` asserts a fact about the twelve lines beneath it,
and the two are edited at different moments: a row is moved, and the number in
the heading is the thing nobody scrolls back up to change. The failure is quiet
and it compounds, because the count is what a reader trusts instead of counting.
docs/verification/README.md drifted to "56 defects" against an actual 66 and sat
that way through several sessions; the heading counts in the ledger beside it
were hand-maintained through five separate edits in one afternoon.

This is the narrowest possible version of a self-consistency check: it compares a
document against itself, needs no knowledge of the tree, and cannot be wrong about
what the right answer is. It says nothing about whether the rows are correct, only
that the document agrees with itself about how many there are.
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

# A heading whose text ends in a parenthesised integer. H1 is excluded: a document
# title with a number in brackets is a title, not a section header over a table.
COUNT_HEADING = re.compile(r"^(#{2,6})\s+(.*?)\s*\((\d+)\)\s*$")
ANY_HEADING = re.compile(r"^(#{1,6})\s")
TABLE_ROW = re.compile(r"^\s*\|")
TABLE_RULE = re.compile(r"^\s*\|[\s|:-]+\|\s*$")
LIST_ITEM = re.compile(r"^([-*+])\s+\S")
FENCE = re.compile(r"^(?:\s*)(?:```|~~~)")

# A parenthesised number in this range is a year, not a count. "### Speculative
# Decoding (2025)" heads a prose section and counts nothing.
YEAR_MIN, YEAR_MAX = 1900, 2100


def is_year(value: int) -> bool:
    """True when a parenthesised number is a date rather than a quantity."""
    return YEAR_MIN <= value <= YEAR_MAX


def count_body(lines: list[str]) -> tuple[str, int] | None:
    """The countable items in one section body, as (kind, count).

    A section is counted as a table if it contains one, else as a list. Tables win
    because a section that introduces a table with a bulleted preamble is stating
    the size of the table; the reverse arrangement does not occur here.

    Returns None when the section holds neither, which is not a failure: a
    parenthesised number over prose is a qualifier, not a claim about rows.
    """
    in_fence = False
    rows = 0
    items = 0
    for line in lines:
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if TABLE_ROW.match(line):
            if not TABLE_RULE.match(line):
                rows += 1
        elif LIST_ITEM.match(line):
            items += 1
    if rows:
        # Every table has a header row, which is not data.
        return ("table rows", rows - 1)
    if items:
        return ("list items", items)
    return None


def find_sections(text: str) -> list[tuple[int, str, int, list[str]]]:
    """Every counted section as (line number, heading text, claimed count, body).

    A section body ends at the next heading of equal or shallower depth, so a
    subsection's own table is counted toward its parent, which is what a reader
    counting by hand would do.
    """
    lines = text.splitlines()
    sections = []
    for index, line in enumerate(lines):
        match = COUNT_HEADING.match(line)
        if not match:
            continue
        depth = len(match.group(1))
        end = len(lines)
        for offset, later in enumerate(lines[index + 1 :], start=index + 1):
            heading = ANY_HEADING.match(later)
            if heading and len(heading.group(1)) <= depth:
                end = offset
                break
        sections.append(
            (index + 1, match.group(2), int(match.group(3)), lines[index + 1 : end])
        )
    return sections


def iter_docs(root: Path) -> list[Path]:
    """Every markdown file in the tree. Enumerated without git; see check_doc_refs.py.

    Nothing is exempt. A frozen record must still agree with itself: its counts
    describe rows it carries, not a tree that has moved on since.
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        found.extend(Path(dirpath) / n for n in filenames if n.endswith(".md"))
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    docs = iter_docs(root)
    if not docs:
        out.error(f"doc-counts: enumerated zero documents under {root}")
        out.line("  The gate would pass without checking anything.")
        return 1

    wrong: list[tuple[str, int, str, int, int, str]] = []
    checked = 0
    for doc in docs:
        try:
            text = doc.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            out.error(f"doc-counts: {doc.relative_to(root)} is unreadable: {exc}")
            return 1
        rel_doc = str(doc.relative_to(root))
        for lineno, heading, claimed, body in find_sections(text):
            if is_year(claimed):
                continue
            counted = count_body(body)
            if counted is None:
                continue
            kind, actual = counted
            checked += 1
            if actual != claimed:
                wrong.append((rel_doc, lineno, heading, claimed, actual, kind))

    # The guard that matters for this check. Every count heading in the tree could
    # be deleted or reworded past the pattern, and the gate would keep reporting
    # success over an empty set.
    if not checked:
        out.error("doc-counts: found no counted sections to check")
        out.line("  The gate would pass without checking anything.")
        return 1

    out.info(f"doc-counts: {len(docs)} documents, {checked} counted sections")
    if not wrong:
        out.ok("doc-counts: every stated count matches its section")
        return 0

    noun = "heading disagrees" if len(wrong) == 1 else "headings disagree"
    out.error(f"doc-counts: {len(wrong)} {noun} with their own section")
    for rel_doc, lineno, heading, claimed, actual, kind in wrong:
        out.line(f"  {rel_doc}:{lineno}: {heading} says {claimed}, has {actual} {kind}")
    out.line("  Solution: correct whichever is wrong. The count in the heading is")
    out.line("  the one a reader trusts instead of counting, so it is usually it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
