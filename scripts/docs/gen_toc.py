#!/usr/bin/env python3
"""Render a document's table of contents from its own headings.

A long document is read by someone looking for one section, and the headings
are the index they scan. A table of contents copies that index to the top so it
can be scanned without scrolling, and for a session reading the file in pieces
it is the only view of the whole. Copied by hand it is true on the day it is
written: the next heading renamed or section added leaves a row pointing at an
anchor that no longer exists, and nothing notices, because the table is not the
part of the document anyone edits. So this script copies it on demand instead,
the way scripts/docs/gen_inventory.py copies a README's tables from the tree.

A document declares the table with one marker pair under its heading:

    ## Table of Contents

    <!-- toc:start -->
    <!-- toc:end -->

The list rendered between the markers is one numbered row per H2 and one
indented numbered row per H3, in document order, each linking the heading's
GitHub anchor. H1 is the title and is not listed; H4 and below are not listed
because the documentation style guide says not to write them. A heading inside
a fenced block is output, not structure, and is skipped. The heading text is
copied verbatim, code spans included, so the runbook rows that
scripts/docs/check_runbook_ids.py orders by entry id keep the id first.

Which documents must carry one is decided here too, in `qualifies`, so the gate
and its tests share the definition: over 500 words and four or more sections.
A shorter document may carry the markers anyway and is rendered like any other.
Frozen records are never touched. A README is a router and never needs one.

Run on a document with no marker pair, the script adds one: an existing
`## Table of Contents` section has its hand-written list replaced by the
markers, and a document without the section gains it in front of its first H2.
That is what makes the gate's remedy one command. With no document argument the
script finds every document in the tree that carries a marker pair, frozen
records excepted. --check regenerates in memory and diffs against what is
committed, and never edits, which is what scripts/docs/check_toc.py runs in the
quality gate.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

START = "<!-- toc:start -->"
END = "<!-- toc:end -->"
MARKER_LINE = re.compile(r"^\s*<!-- toc:(start|end) -->\s*$")

# The threshold. Under 500 words the document fits one screen or two, and under
# four sections the list would be shorter than the headings it copies.
MIN_WORDS = 500
MIN_SECTIONS = 4

# The heading the markers sit under. The case-insensitive match accepts the
# lower-case `contents` a few older pages spell; the inserted heading is
# title-cased, which is what check_runbook_ids.py looks for as well.
TOC_HEADING = re.compile(r"^##\s+Table of Contents\s*$", re.IGNORECASE)
TOC_TITLE = "## Table of Contents"
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(?:```|~~~)")

# A markdown link inside a heading: GitHub anchors its text alone, and a link
# nested in the row's own link would not render, so both use the text.
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")

# Directories a walk never enters, the same set the sibling gates prune.
PRUNE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".ruff_cache",
    ".pytest-tmp",
    # Docusaurus's cache under site/, gitignored and rebuilt by every site build;
    # it carries a JSON copy of each page, including pages since deleted.
    ".docusaurus",
}

# A frozen record, by the same paths the other gates exempt: the documentation
# style guide makes these the definition. A frozen record is never edited, so
# it is never given a marker pair, and one that has none is never asked for one.
FROZEN_PARTS = ("investigations", "archive", "vendor")
FROZEN_SUFFIX = "_investigation.md"

# Trees the requirement does not reach, though a marker there is still
# rendered: the agent and tool definitions under .claude/ and its .opencode/
# mirror are read by the tools, not scanned by a reader, and the site's pages
# get their table from Docusaurus.
UNRULED_ROOTS = (".claude", ".opencode", "site")


def is_frozen(rel: Path) -> bool:
    """A frozen record, which no gate edits or requires anything of."""
    return any(part in FROZEN_PARTS for part in rel.parts) or rel.name.endswith(
        FROZEN_SUFFIX
    )


def is_ruled(rel: Path) -> bool:
    """A document the threshold applies to: living, not a README, in a read tree."""
    if is_frozen(rel) or rel.name == "README.md":
        return False
    return not rel.parts or rel.parts[0] not in UNRULED_ROOTS


def headings(text: str) -> list[tuple[int, str]]:
    """Every H2 and H3 outside a fence as (level, text), the ToC heading excepted."""
    found: list[tuple[int, str]] = []
    fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            fence = not fence
            continue
        if fence:
            continue
        match = HEADING.match(line)
        if not match:
            continue
        level = len(match.group(1))
        if level not in (2, 3) or TOC_HEADING.match(line):
            continue
        found.append((level, LINK.sub(r"\1", match.group(2))))
    return found


def slug(text: str) -> str:
    """The GitHub anchor for a heading: lower-cased, punctuation dropped, spaces
    to hyphens. Nothing else is collapsed, so `python -m` anchors as `python--m`,
    which is what the runbooks' hand-written rows already link."""
    lowered = text.lower()
    kept = re.sub(r"[^\w\- ]", "", lowered)
    return kept.replace(" ", "-")


def render(entries: list[tuple[int, str]]) -> str:
    """The numbered list: H2 rows at the margin, H3 rows indented under them.

    Anchors repeat the way GitHub disambiguates them, `-1`, `-2` and so on, so a
    document with two `### Observed` sections links each to its own.
    """
    seen: dict[str, int] = {}
    counts = {2: 0, 3: 0}
    rows: list[str] = []
    for level, text in entries:
        anchor = slug(text)
        n = seen.get(anchor, 0)
        seen[anchor] = n + 1
        if n:
            anchor = f"{anchor}-{n}"
        if level == 2:
            counts[3] = 0
        counts[level] += 1
        indent = "" if level == 2 else "   "
        rows.append(f"{indent}{counts[level]}. [{text}](#{anchor})")
    return "\n".join(rows)


def word_count(text: str) -> int:
    return len(text.split())


def qualifies(text: str) -> bool:
    """Whether the threshold asks this text for a table of contents."""
    return word_count(text) > MIN_WORDS and len(headings(text)) >= MIN_SECTIONS


def marker_lines(text: str) -> list[tuple[int, str]]:
    """Every marker standing on its own line outside a fence, as (line, edge).

    A marker quoted in a code span, or shown inside a fenced example the way the
    style guide shows it, is prose about the marker and not one.
    """
    found: list[tuple[int, str]] = []
    fence = False
    for i, line in enumerate(text.splitlines()):
        if FENCE.match(line):
            fence = not fence
            continue
        if fence:
            continue
        match = MARKER_LINE.match(line)
        if match:
            found.append((i, match.group(1)))
    return found


def carries_markers(text: str) -> bool:
    return bool(marker_lines(text))


def find_markers(text: str, label: Path) -> bool:
    """Whether the text carries the one marker pair, under the ToC heading.

    Raises ValueError naming the document on a second pair, a start with no end,
    an end with no start, or a pair whose nearest heading above is not the ToC's.
    """
    found = marker_lines(text)
    if not found:
        return False
    edges = [edge for _, edge in found]
    if edges != ["start", "end"]:
        if edges[0] == "end":
            raise ValueError(f"{label}: toc:end without its start")
        if len(edges) == 1:
            raise ValueError(f"{label}: toc:start never closes")
        raise ValueError(f"{label}: more than one toc marker pair")
    above = [ln for ln in text.splitlines()[: found[0][0]] if HEADING.match(ln)]
    if not above or not TOC_HEADING.match(above[-1]):
        raise ValueError(f"{label}: toc:start is not under a `{TOC_TITLE}` heading")
    return True


def bounds(text: str) -> tuple[int, int]:
    """The line indexes of the start and end markers of a validated pair."""
    (start, _), (end, _) = marker_lines(text)
    return start, end


def committed(text: str) -> str:
    """The list between the markers as it stands in the text."""
    start, end = bounds(text)
    return "\n".join(text.splitlines()[start + 1 : end]).strip("\n")


def splice(text: str, table: str) -> str:
    """The text with `table` between the markers, replacing what was there."""
    lines = text.splitlines(keepends=True)
    start, end = bounds(text)
    head = "".join(lines[:start])
    tail = "".join(lines[end + 1 :])
    return f"{head}{START}\n\n{table}\n\n{END}\n{tail}"


def adopt(text: str, label: Path) -> str:
    """The text with a marker pair added: replacing the hand-written list under an
    existing ToC heading, or as a new section in front of the first H2.

    Raises ValueError when the document has no H2 to put the section before.
    """
    lines = text.splitlines(keepends=True)
    fence = False
    toc_at: int | None = None
    first_h2: int | None = None
    section_end: int | None = None
    for i, line in enumerate(lines):
        if FENCE.match(line):
            fence = not fence
            continue
        if fence:
            continue
        match = HEADING.match(line)
        if not match:
            continue
        if toc_at is not None and section_end is None:
            section_end = i
        if TOC_HEADING.match(line):
            toc_at = i
        elif first_h2 is None and len(match.group(1)) == 2:
            first_h2 = i
    block = f"{TOC_TITLE}\n\n{START}\n{END}\n\n"
    if toc_at is not None:
        end = len(lines) if section_end is None else section_end
        return "".join(lines[:toc_at]) + block + "".join(lines[end:])
    if first_h2 is None:
        raise ValueError(f"{label}: no H2 heading to put a table of contents before")
    return "".join(lines[:first_h2]) + block + "".join(lines[first_h2:])


def discover(root: Path) -> list[Path]:
    """Every living markdown file under root that carries a marker."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in filenames:
            if not name.endswith(".md"):
                continue
            path = Path(dirpath) / name
            if is_frozen(path.relative_to(root)):
                continue
            if carries_markers(path.read_text(encoding="utf-8", errors="replace")):
                found.append(path)
    return sorted(found)


def process(doc: Path, root: Path, check: bool, out: Printer) -> int:
    """Render (or, with check, verify) one document's table of contents."""
    label = doc.relative_to(root)
    text = doc.read_text(encoding="utf-8")
    try:
        present = find_markers(text, label)
        if not present:
            if check:
                out.error(f"toc: {label} carries no toc markers")
                out.line("  Solution: add the table of contents")
                out.line(f"  Command: python scripts/docs/gen_toc.py {label}")
                return 1
            text = adopt(text, label)
    except ValueError as exc:
        out.error(f"toc: {exc}")
        return 1

    entries = headings(text)
    # The empty-set guard: a document with no sections would render an empty
    # list, and --check would then hold the document to it.
    if not entries:
        out.error(f"toc: {label}: no H2 or H3 headings to list")
        return 1
    table = render(entries)
    if not check:
        doc.write_text(splice(text, table), encoding="utf-8")
        out.ok(f"{label}: {len(entries)} rows")
        return 0
    current = committed(text)
    if current == table:
        out.ok(f"toc: {label}: {len(entries)} rows")
        return 0
    out.error(f"toc: {label} is stale")
    diff = difflib.unified_diff(
        current.splitlines(),
        table.splitlines(),
        fromfile="committed",
        tofile="generated",
        lineterm="",
    )
    for line in diff:
        out.line(f"  {line}")
    out.line("  Solution: regenerate the table of contents")
    out.line(f"  Command: python scripts/docs/gen_toc.py {label}")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "docs",
        nargs="*",
        help="markdown files to render (default: every document carrying a marker)",
    )
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate in memory and fail when the committed list differs",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    if args.docs:
        docs = [
            (p if p.is_absolute() else root / p).resolve() for p in map(Path, args.docs)
        ]
        for doc in docs:
            if not doc.is_file():
                out.error(f"toc: no such file {doc}")
                return 1
    else:
        docs = discover(root)
        if not docs:
            out.error(f"toc: no document under {root} carries a marker")
            return 1

    status = 0
    for doc in docs:
        status |= process(doc, root, args.check, out)
    return status


if __name__ == "__main__":
    sys.exit(main())
