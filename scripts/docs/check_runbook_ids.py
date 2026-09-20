#!/usr/bin/env python3
"""Fail the quality gate when a runbook's entry ids and section order disagree.

docs/verification/protocol.md ("Entry identifiers") separates two things an entry
id used to do at once. The id names the entry: it is a prefix, a dot and a slug
(FT.calibrate-qlora), assigned once, kept when the title changes, never reused.
The Table of Contents orders the entries: it is a numbered list whose rows are
the sections in the order they run. The numbers the ids carried before
2026-09-19 were handed out in writing order and read as an execution order they
were not, once sections had been reordered by dependency; nothing could see the
two drift apart, because the same token stated both.

This gate is what keeps them apart now. For every live runbook it checks that
the ids are one shape (slug or legacy, never a mix), that a slug is a slug,
that no id heads two entries or reappears after retirement, that the Table of
Contents lists the entries in the order the headings have them, and that an
entry's stated dependencies come earlier in the file. Legacy runbooks (numeric
ids: RH.24, OBS-V10) pass the order checks unchanged and are not asked for a
Retired line; a runbook converts whole or not at all.

None of this says an entry is in the right place. It says the two statements of
where it is agree, which is the whole of what a reader can be promised.
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

# Frozen records keep the ids they were written with, for the reason
# docs/verification/README.md gives, and are outside every gate here.
EXEMPT_DIRS = {"archive", "investigations"}

HEADING = re.compile(r"^(#{2,3})\s+(\S+)\s*(.*)$")
TOC_HEADING = re.compile(r"^##\s+Table of Contents\s*$", re.IGNORECASE)
SECTION_HEADING = re.compile(r"^##\s")

# The two id shapes. Legacy: any token carrying a digit (RH.24, OBS-V10,
# P0.1, HS.3, 0.1, RH.2a). Slug: an uppercase prefix, a dot, a lowercase hyphenated
# slug. The prefix is uppercase on purpose: a heading that opens with
# `config.json` would otherwise read as an entry of a runbook named `config`.
LEGACY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*\d[A-Za-z0-9.\-]*$")
SLUG_ID = re.compile(r"^[A-Z][A-Z0-9-]*\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
SLUG_MAX_WORDS = 4

# A ToC row: a list marker, then a link whose text opens with the id.
TOC_ROW = re.compile(r"^\s*(\d+\.|[-*])\s+\[(\S+)[^\]]*\]\(#[^)]*\)")
NUMBERED = re.compile(r"^\s*\d+\.\s")

RETIRED_LINE = re.compile(r"^Retired:\s*(.*)$")
# An id token in prose: the runbook's own prefix, so a fine-tuning entry that
# says "I.probe-loop uses the same directory" does not become a dependency
# on an entry this file does not have.
DEPENDS_ON = re.compile(r"\bdepends on\b(.*?)(?:\.\s|\.$|;|$)", re.IGNORECASE)


def iter_runbooks(root: Path) -> list[Path]:
    """Every live runbook. The frozen subtrees are skipped, not scanned and passed."""
    base = root / "docs" / "verification"
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXEMPT_DIRS]
        found.extend(Path(dirpath) / n for n in filenames if n.endswith(".md"))
    return sorted(found)


def is_slug(ident: str) -> bool:
    return SLUG_ID.match(ident) is not None


def is_legacy(ident: str) -> bool:
    return LEGACY_ID.match(ident) is not None and not is_slug(ident)


def is_entry_id(ident: str) -> bool:
    return is_slug(ident) or is_legacy(ident)


def entry_headings(text: str) -> list[tuple[int, str]]:
    """Every entry heading as (line number, id), in document order.

    Parents count (RH.9 heads RH.9a onward, I.harvest heads its captures):
    the ToC lists them, so the order check must see them too. Fenced blocks
    are skipped, since a pasted `## ` inside one is output, not a heading.
    """
    found: list[tuple[int, str]] = []
    fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if re.match(r"^\s*(?:```|~~~)", line):
            fence = not fence
            continue
        if fence:
            continue
        match = HEADING.match(line)
        if match and is_entry_id(match.group(2)):
            found.append((lineno, match.group(2)))
    return found


def toc_rows(text: str) -> list[tuple[int, str, bool]]:
    """Every entry row of the Table of Contents as (line, id, numbered).

    Rows naming a section that is not an entry ("Unentried surface") are
    dropped, so a runbook may list its closing sections without the gate
    asking the headings to carry them in the same place.
    """
    lines = text.splitlines()
    rows: list[tuple[int, str, bool]] = []
    inside = False
    for lineno, line in enumerate(lines, start=1):
        if TOC_HEADING.match(line):
            inside = True
            continue
        if inside and SECTION_HEADING.match(line):
            break
        if not inside:
            continue
        match = TOC_ROW.match(line)
        if match and is_entry_id(match.group(2)):
            rows.append((lineno, match.group(2), NUMBERED.match(line) is not None))
    return rows


def has_toc(text: str) -> bool:
    return any(TOC_HEADING.match(line) for line in text.splitlines())


def retired_ids(text: str) -> list[str] | None:
    """The ids on the `Retired:` line, or None when the runbook has no such line.

    The line may wrap: the paragraph runs to the next blank line.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = RETIRED_LINE.match(line)
        if not match:
            continue
        body = [match.group(1)]
        for later in lines[index + 1 :]:
            if not later.strip():
                break
            body.append(later)
        joined = " ".join(body)
        # One id per `;`-separated item, leading the item: the parenthetical
        # after it may name the entry that absorbed it, which is not retired.
        found = []
        for item in joined.split(";"):
            leading = item.strip().split(" ", 1)[0].rstrip(".,")
            if is_entry_id(leading):
                found.append(leading)
        return found
    return None


def first_paragraph(lines: list[str], start: int) -> str:
    """The prose paragraph after a heading, joined on spaces."""
    body: list[str] = []
    seen_text = False
    for line in lines[start:]:
        if HEADING.match(line):
            break
        if not line.strip():
            if seen_text:
                break
            continue
        seen_text = True
        body.append(line.strip())
    return " ".join(body)


def dependencies(text: str, prefix: str) -> list[tuple[int, str, str]]:
    """Every (line, entry id, cited id) from a `depends on` clause, own prefix only."""
    lines = text.splitlines()
    found: list[tuple[int, str, str]] = []
    token = re.compile(rf"\b{re.escape(prefix)}\.[\w-]+")
    for lineno, ident in entry_headings(text):
        paragraph = first_paragraph(lines, lineno)
        for clause in DEPENDS_ON.findall(paragraph):
            for cited in token.findall(clause):
                if is_entry_id(cited) and cited != ident:
                    found.append((lineno, ident, cited))
    return found


def prefix_of(ident: str) -> str:
    return ident.split(".", 1)[0]


def check_runbook(rel: str, text: str) -> list[str]:
    """Every finding for one runbook, as `path:line: message` strings."""
    findings: list[str] = []
    headings = entry_headings(text)
    if not headings:
        return findings
    ids = [ident for _, ident in headings]
    slugs = [i for i in ids if is_slug(i)]
    slug_book = bool(slugs)

    if slugs and len(slugs) != len(ids):
        legacy = [i for i in ids if not is_slug(i)]
        findings.append(
            f"{rel}: mixes slug and legacy ids ({', '.join(legacy[:3])}"
            f"{', ...' if len(legacy) > 3 else ''} beside {slugs[0]})"
        )

    for lineno, ident in headings:
        if is_slug(ident) and ident.count("-") + 1 > SLUG_MAX_WORDS:
            findings.append(f"{rel}:{lineno}: {ident} runs past {SLUG_MAX_WORDS} words")

    seen: dict[str, int] = {}
    for lineno, ident in headings:
        if ident in seen:
            findings.append(
                f"{rel}:{lineno}: {ident} already heads the entry at line {seen[ident]}"
            )
        seen.setdefault(ident, lineno)

    retired = retired_ids(text)
    if slug_book and retired is None:
        findings.append(f"{rel}: no `Retired:` line under the tally")
    for ident in retired or []:
        if ident in seen:
            findings.append(
                f"{rel}:{seen[ident]}: {ident} is on the Retired line and heads an entry"
            )

    if has_toc(text):
        rows = toc_rows(text)
        toc_ids = [ident for _, ident, _ in rows]
        if toc_ids != ids:
            for position, (want, have) in enumerate(
                zip(ids, toc_ids, strict=False), start=1
            ):
                if want != have:
                    findings.append(
                        f"{rel}: Table of Contents position {position} is {have}, "
                        f"the section there is {want}"
                    )
                    break
            else:
                findings.append(
                    f"{rel}: Table of Contents lists {len(toc_ids)} entries, "
                    f"the headings have {len(ids)}"
                )
        if slug_book:
            for lineno, ident, numbered in rows:
                if not numbered:
                    findings.append(
                        f"{rel}:{lineno}: Table of Contents row {ident} is not numbered"
                    )
                    break
    elif slug_book:
        findings.append(f"{rel}: no Table of Contents to hold the order")

    order = {ident: index for index, ident in enumerate(ids)}
    for lineno, ident, cited in dependencies(text, prefix_of(ids[0])):
        if cited not in order:
            findings.append(
                f"{rel}:{lineno}: {ident} depends on {cited}, which has no entry"
            )
        elif order[cited] > order[ident]:
            findings.append(
                f"{rel}:{lineno}: {ident} depends on {cited}, which comes later"
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    runbooks = iter_runbooks(root)
    findings: list[str] = []
    entries = 0
    books = 0
    for doc in runbooks:
        try:
            text = doc.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            out.error(f"runbook-ids: {doc.relative_to(root)} is unreadable: {exc}")
            return 1
        headings = entry_headings(text)
        if not headings:
            continue
        books += 1
        entries += len(headings)
        findings.extend(check_runbook(str(doc.relative_to(root)), text))

    # The same guard the sibling gates carry: a pattern that stops matching
    # leaves a gate reporting success over an empty set.
    if not entries:
        out.error("runbook-ids: found no entry headings to check")
        out.line("  The gate would pass without checking anything.")
        return 1

    out.info(f"runbook-ids: {books} runbooks, {entries} entry headings")
    if findings:
        noun = "finding" if len(findings) == 1 else "findings"
        out.error(f"runbook-ids: {len(findings)} {noun}")
        for finding in findings:
            out.line(f"  {finding}")
        out.line("  Solution: the id names the entry and the Table of Contents orders")
        out.line("  it (docs/verification/protocol.md, Entry identifiers). Move the ToC")
        out.line("  row or the section so they agree; never renumber or reuse an id,")
        out.line("  and list an entry that leaves on the Retired line.")
        return 1
    out.ok("runbook-ids: every id names one entry and every ToC states its order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
