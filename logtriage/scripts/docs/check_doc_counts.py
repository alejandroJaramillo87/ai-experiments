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

# A verification runbook makes the same kind of claim about itself, in the tally
# under its front matter. It is updated by hand in the edit that closes an entry
# -- the edit whose attention is on the entry, not on the header -- and against
# it stands one `Status:` line per entry, so the document already carries the
# answer and only needs someone to add it up.
TALLY_RE = re.compile(
    r"^\*\*Entries:\s*(?P<verified>\d+)\s+verified,\s*(?P<blocked>\d+)\s+blocked,"
    r"\s*(?P<pending>\d+)\s+pending,\s*of\s*(?P<total>\d+)\.\*\*"
)

# Column zero: a bolded `**Status: ...**` inside a paragraph is prose about an
# entry, and counting it would inflate the tally by one per mention.
STATUS_RE = re.compile(r"^Status:\s*([A-Za-z]+)")

# A fenced block opener or closer, in either spelling.
FENCE_RE = re.compile(r"^\s*(?:```|~~~)")


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


def outside_fences(lines: list[str]) -> list[str]:
    """The lines that are prose, with fenced blocks blanked out.

    A tally or a status line inside a fenced block is an example of the format,
    not a claim about this document -- README.md shows both, and counting them
    made it disagree with itself about a runbook it does not contain.
    """
    kept: list[str] = []
    in_fence = False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
            kept.append("")
            continue
        kept.append("" if in_fence else line)
    return kept


def check_tally(lines: list[str]) -> tuple[str, str] | None:
    """Compare a runbook's tally against its own Status lines.

    Returns (stated, counted) when they disagree, None when they agree or when
    the document carries no tally. Like the heading check, this cannot be wrong
    about the right answer: both halves are in the same file.
    """
    prose = outside_fences(lines)

    tally = None
    for line in prose:
        match = TALLY_RE.match(line)
        if match:
            tally = match
            break
    if tally is None:
        return None

    counted = {"verified": 0, "blocked": 0, "pending": 0}
    for line in prose:
        status = STATUS_RE.match(line)
        if status:
            kind = status.group(1).lower()
            if kind in counted:
                counted[kind] += 1

    stated = {key: int(tally.group(key)) for key in counted}
    total_stated = int(tally.group("total"))
    total_counted = sum(counted.values())

    if stated == counted and total_stated == total_counted:
        return None
    return (
        f"{stated['verified']} verified, {stated['blocked']} blocked, "
        f"{stated['pending']} pending, of {total_stated}",
        f"{counted['verified']} verified, {counted['blocked']} blocked, "
        f"{counted['pending']} pending, of {total_counted}",
    )


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
    tallies: list[tuple[Path, str, str]] = []
    claims = 0
    tally_claims = 0
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

        disagreement = check_tally(lines)
        if disagreement is not None:
            tally_claims += 1
            tallies.append((document.relative_to(root), *disagreement))
        elif any(TALLY_RE.match(line) for line in outside_fences(lines)):
            tally_claims += 1

    if wrong or tallies:
        out.error(
            f"check_doc_counts: {len(wrong)} heading(s) and {len(tallies)} tally "
            "disagree with their own rows"
        )
        for document, title, stated, actual in wrong:
            out.line(f"  {document}: '{title}' says {stated}, counted {actual}")
        for document, stated, counted in tallies:
            out.line(f"  {document}: tally says {stated}; Status lines say {counted}")
        return 1

    out.ok(
        f"check_doc_counts: {claims} counted heading(s) and {tally_claims} runbook "
        "tally/tallies agree with their rows"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
