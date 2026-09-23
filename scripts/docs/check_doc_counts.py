#!/usr/bin/env python3
"""Fail the quality gate when a document's stated count disagrees with itself.

Two patterns are checked, both self-referential.

A heading like `## Open (3)` asserts a fact about the twelve lines beneath it,
and the two are edited at different moments: a row is moved, and the number in
the heading is the thing nobody scrolls back up to change. The failure is quiet
and it compounds, because the count is what a reader trusts instead of counting.
docs/verification/README.md drifted to "56 defects" against an actual 66 and sat
that way through several sessions; the heading counts in the ledger beside it
were hand-maintained through five separate edits in one afternoon.

A verification runbook states the same kind of claim about itself, in the tally
under its front matter: `**Entries: N verified, M blocked, K pending, of T.**`
It is updated by hand in the edit that closes an entry, which is the edit whose
attention is on the entry. Against it stands one `Status: ` line per entry, so
the runbook already carries the answer and only needs someone to add it up.

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
    # Docusaurus's cache under site/, gitignored and rebuilt by every site build;
    # it carries a JSON copy of each page, including pages since deleted.
    ".docusaurus",
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

# A runbook's progress line, per docs/verification/README.md. Trailing prose after
# the closing `**` is allowed: rust-harness.md continues the sentence past its own
# tally to say which entries are blocked.
TALLY = re.compile(
    r"^\*\*Entries:\s*(\d+)\s+verified,\s*(\d+)\s+blocked,\s*(\d+)\s+pending,"
    r"\s*of\s*(\d+)\.\*\*"
)

# An entry's own status. Column zero is the whole distinction: a bolded
# `**Status: ...**` inside a paragraph is prose about an entry, not the entry's
# status line, and counting it would inflate the tally by one per mention.
STATUS_LINE = re.compile(r"^Status:\s*(\S*)")

# The kind is the leading run of letters, so the date and any parenthetical after
# it are ignored -- `Status: PENDING (verified 2026-08-14; reopened by U7)` and
# `Status: VERIFIED 2026-08-14` each yield their kind and nothing else.
KIND_WORD = re.compile(r"[A-Za-z]+")

# The three the tally has a term for. docs/verification/README.md also allows
# FAILED, which the tally cannot represent; see the failure text in main().
STATUS_KINDS = ("VERIFIED", "BLOCKED", "PENDING")


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


def find_tallies(text: str) -> list[tuple[int, dict[str, int], int]]:
    """Every runbook tally as (line number, claimed count per kind, claimed total).

    A runbook carries one, under its front matter. All of them are checked
    against the same status lines, because a second tally in the same document
    is a claim about the same entries and has to agree with them too.
    """
    found = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = TALLY.match(line)
        if not match:
            continue
        claimed = {
            "VERIFIED": int(match.group(1)),
            "BLOCKED": int(match.group(2)),
            "PENDING": int(match.group(3)),
        }
        found.append((lineno, claimed, int(match.group(4))))
    return found


def count_statuses(text: str) -> tuple[dict[str, int], list[tuple[int, str]]]:
    """Status lines by kind, and every (line number, word) naming no known kind.

    Status lines are the only countable thing here. Entry numbering is not: it is
    neither contiguous nor one-per-entry, so interp.md reaches its 13 entries via
    an interstitial I.harvest-crash-recovery with no I.workbench-build. Headings are not either, because a parent
    entry may carry no status of its own -- rust-harness.md RH.9 heads RH.9a
    through RH.9d, each of which has one.

    An unrecognized word is returned rather than dropped. Ignoring it would make
    a typo (`Status: VERIFED`) read as an entry with no status at all, which is
    the tally drifting by one with nothing left to show for it.
    """
    counts = dict.fromkeys(STATUS_KINDS, 0)
    unknown: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = STATUS_LINE.match(line)
        if not match:
            continue
        word = KIND_WORD.match(match.group(1))
        kind = word.group(0) if word else match.group(1)
        if kind in counts:
            counts[kind] += 1
        else:
            unknown.append((lineno, kind))
    return counts, unknown


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
    tally_wrong: list[tuple[str, int, str]] = []
    unknown_status: list[tuple[str, int, str]] = []
    checked = 0
    tallied = 0
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

        # Only a document stating a tally is in scope for the status check. A line
        # beginning `Status: ` is ordinary prose elsewhere in the tree -- see
        # docs/inference/engines/vllm.md -- and the three-word vocabulary is a
        # runbook convention, not a rule about markdown.
        tallies = find_tallies(text)
        if not tallies:
            continue
        counts, unknown = count_statuses(text)
        unknown_status.extend((rel_doc, lineno, word) for lineno, word in unknown)
        for lineno, claimed_by_kind, total in tallies:
            tallied += 1
            for kind in STATUS_KINDS:
                if counts[kind] != claimed_by_kind[kind]:
                    tally_wrong.append(
                        (
                            rel_doc,
                            lineno,
                            f"says {claimed_by_kind[kind]} {kind.lower()}, "
                            f"has {counts[kind]}",
                        )
                    )
            parts = sum(claimed_by_kind.values())
            if parts != total:
                tally_wrong.append(
                    (rel_doc, lineno, f"parts sum to {parts}, not the stated {total}")
                )

    # The guards that matter for this check. Every count heading in the tree could
    # be deleted or reworded past the pattern, and the gate would keep reporting
    # success over an empty set.
    if not checked:
        out.error("doc-counts: found no counted sections to check")
        out.line("  The gate would pass without checking anything.")
        return 1
    if not tallied:
        out.error("doc-counts: found no runbook tallies to check")
        out.line("  The gate would pass without checking anything.")
        out.line("  Expected a bolded `Entries: N verified, ...` line per runbook.")
        return 1

    out.info(
        f"doc-counts: {len(docs)} documents, {checked} counted sections, "
        f"{tallied} runbook tallies"
    )

    failed = False
    if wrong:
        failed = True
        noun = "heading disagrees" if len(wrong) == 1 else "headings disagree"
        out.error(f"doc-counts: {len(wrong)} {noun} with their own section")
        for rel_doc, lineno, heading, claimed, actual, kind in wrong:
            out.line(
                f"  {rel_doc}:{lineno}: {heading} says {claimed}, has {actual} {kind}"
            )
        out.line("  Solution: correct whichever is wrong. The count in the heading is")
        out.line("  the one a reader trusts instead of counting, so it is usually it.")
    else:
        out.ok("doc-counts: every stated count matches its section")

    if unknown_status:
        failed = True
        noun = "line names" if len(unknown_status) == 1 else "lines name"
        out.error(f"doc-counts: {len(unknown_status)} Status {noun} no counted kind")
        kinds = ", ".join(STATUS_KINDS)
        for rel_doc, lineno, word in unknown_status:
            out.line(f"  {rel_doc}:{lineno}: `Status: {word}` is none of {kinds}")
        out.line("  Solution: use a kind the tally has a term for. FAILED is a legal")
        out.line("  status per docs/verification/README.md but the tally cannot hold")
        out.line("  it, so a failed entry is resolved rather than counted.")

    if tally_wrong:
        failed = True
        noun = "disagreement" if len(tally_wrong) == 1 else "disagreements"
        out.error(f"doc-counts: {len(tally_wrong)} tally {noun} with the Status lines")
        for rel_doc, lineno, detail in tally_wrong:
            out.line(f"  {rel_doc}:{lineno}: tally {detail}")
        out.line("  Solution: recount the `Status: ` lines and correct the tally. It")
        out.line("  is updated by hand in the edit that closes an entry, which is the")
        out.line("  edit whose attention is on the entry.")
    elif not unknown_status:
        out.ok("doc-counts: every runbook tally matches its Status lines")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
