#!/usr/bin/env python3
"""Fail the gate when a verification entry is not well-formed.

Two checks over docs/verification/, both aimed at one failure class the sibling
gates cannot see: an entry that looks verified and is not.

The first is the load-bearing one. An entry may carry its evidence in either of
two forms -- output pasted from the machine that produced it, or the name of a
test that asserts the signal -- and the second form is a reference into the tree.
References rot. Renaming a test is a one-line edit in one file, and the runbook
citing it is exactly the thing nobody thinks to update; the entry then claims
coverage that no longer exists while every instrument stays green.

The second is cruder and puts a floor under the other failure mode. No checker
can judge whether pasted output demonstrates the signal it claims -- that needs a
reader. It can insist there is some. An entry marked VERIFIED with an empty
Observed block and no named check is not verified, it is an entry somebody
intended to come back to.

Neither says an entry is correct. They say it is not obviously hollow, which is
a lower bar than it sounds.

The status vocabulary is closed, because the tally in check_doc_counts.py counts
these words and a typo would make a runbook silently disagree with itself in the
direction of looking more finished.
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

RUNBOOK_DIR = Path("docs") / "verification"

# Frozen records are exempt everywhere else and here too: their paths and names
# are accurate to the day they were written.
EXEMPT_PARTS = {"investigations"}

# An entry heading leads with an identifier carrying a digit -- D1, D14, E3. A
# section heading does not, so "Deployment" and "Related" are not read as
# entries and are not required to carry a status.
HEADING_RE = re.compile(r"^(#{2,4})\s+(?P<id>\S+)")
ENTRY_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]*\d[A-Za-z0-9.\-]*$")

# Column zero, deliberately. A bolded `**Status: ...**` inside a paragraph is
# prose about an entry, not the entry's own status line.
STATUS_RE = re.compile(r"^Status:\s*(\S+)")
STATUSES = {"PENDING", "VERIFIED", "BLOCKED"}

# `Observed:` and the dated form `Observed (2026-09-01):`, both at column zero.
OBSERVED_RE = re.compile(r"^\s*(?:\*\*)?Observed\b")

# A named check: `tests/test_kb.py::test_something`. The file and the symbol are
# both resolved -- a citation naming a real file and a function that is not in it
# is the more common half of this rot.
CITATION_RE = re.compile(r"`([\w./-]+\.py)::(\w+)`")

# Where a symbol may be defined.
DEFINITION_TEMPLATES = (r"def {name}\b", r"class {name}\b")

# An Observed block that is present but says nothing. `_(pending)_` is the
# placeholder a new runbook is written with, and it is not evidence.
PLACEHOLDERS = {"", "_(pending)_", "(pending)", "_(not yet run)_", "TBD"}


def runbooks(root: Path) -> list[Path]:
    base = root / RUNBOOK_DIR
    if not base.is_dir():
        return []
    return sorted(
        path for path in base.rglob("*.md") if not (EXEMPT_PARTS & set(path.parts))
    )


def entries(lines: list[str]) -> list[tuple[str, int, int]]:
    """Every entry as (id, start line, end line), by its heading."""
    found: list[tuple[str, int, int]] = []
    open_entry: tuple[str, int] | None = None
    for index, line in enumerate(lines):
        heading = HEADING_RE.match(line)
        if not heading:
            continue
        if open_entry:
            found.append((open_entry[0], open_entry[1], index))
            open_entry = None
        name = heading.group("id")
        if ENTRY_ID_RE.match(name):
            open_entry = (name, index)
    if open_entry:
        found.append((open_entry[0], open_entry[1], len(lines)))
    return found


def has_evidence(body: list[str]) -> bool:
    """Whether an entry offers evidence of either legal kind."""
    text = "\n".join(body)
    if CITATION_RE.search(text):
        return True
    for index, line in enumerate(body):
        if not OBSERVED_RE.match(line):
            continue
        rest = "\n".join(body[index + 1 :]).strip()
        # A fenced block, or any prose that is not the placeholder.
        first = rest.splitlines()[0].strip() if rest else ""
        if first.startswith("```") or (first and first not in PLACEHOLDERS):
            return True
    return False


def resolve(root: Path, path: str, symbol: str) -> bool:
    target = root / path
    if not target.is_file():
        return False
    source = target.read_text(encoding="utf-8", errors="replace")
    return any(
        re.search(template.format(name=re.escape(symbol)), source)
        for template in DEFINITION_TEMPLATES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    out = Printer()
    root = Path(args.root).resolve()
    found = runbooks(root)

    if not found:
        out.skip(f"check_runbook_entries: no runbooks under {RUNBOOK_DIR}")
        return 0

    problems: list[str] = []
    counted = 0
    citations = 0

    for document in found:
        relative = document.relative_to(root)
        lines = document.read_text(encoding="utf-8", errors="replace").splitlines()

        for name, start, end in entries(lines):
            counted += 1
            body = lines[start:end]
            statuses = [
                STATUS_RE.match(line).group(1)  # type: ignore[union-attr]
                for line in body
                if STATUS_RE.match(line)
            ]
            if not statuses:
                problems.append(f"{relative}: {name} has no `Status:` line")
                continue
            if len(statuses) > 1:
                problems.append(f"{relative}: {name} has {len(statuses)} status lines")
            status = statuses[0].upper().rstrip(".")
            if status not in STATUSES:
                problems.append(
                    f"{relative}: {name} status {statuses[0]!r} is not one of "
                    f"{sorted(STATUSES)}"
                )
            elif status == "VERIFIED" and not has_evidence(body):
                problems.append(
                    f"{relative}: {name} is VERIFIED with neither pasted output "
                    "nor a named check"
                )

        for number, line in enumerate(lines, start=1):
            for path, symbol in CITATION_RE.findall(line):
                citations += 1
                if not resolve(root, path, symbol):
                    problems.append(
                        f"{relative}:{number}: {path}::{symbol} does not resolve"
                    )

    if problems:
        out.error(f"check_runbook_entries: {len(problems)} problem(s)")
        for problem in problems:
            out.line(f"  {problem}")
        return 1

    out.ok(
        f"check_runbook_entries: {counted} entries in {len(found)} runbooks, "
        f"{citations} named check(s) resolve"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
