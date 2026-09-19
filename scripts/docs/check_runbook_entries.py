#!/usr/bin/env python3
"""Fail the quality gate when a runbook entry is not well-formed.

Two checks, both over docs/verification/, and both aimed at one failure class the
sibling gates cannot see: an entry that looks verified and is not.

The first is the load-bearing one. A gate-backed entry does not carry a pasted
command and its output; it names the test function that asserts its signal, and
its whole honesty rests on that name. The observability runbook's roster states
the rule outright -- "an element whose evidence is 'the gate was green'
without a named check is not verified". A named check is a reference into the
tree, and references rot: renaming a test is a one-line edit in one file, and the
runbook citing it is exactly the thing nobody thinks to update. The entry then
claims coverage that no longer exists, and every instrument in the repository
stays green while it does.

The second is cruder and puts a floor under the other failure mode this campaign
found. Five observability entries were marked VERIFIED against Observed blocks
that did not demonstrate the signal they claimed; one clause was supported by
nothing at all. No checker can judge whether pasted evidence demonstrates a
signal -- that needs a reader. It can insist there is some.

Neither check says an entry is correct. They say it is not obviously hollow,
which is a lower bar than it sounds and was still not being met.
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

# Frozen records, exempt from both gates for the reason docs/verification/README.md
# gives: their paths and line numbers are accurate to the day they were written.
EXEMPT_DIRS = {"archive", "investigations"}

# A citation of a check inside a source file: a path, `::`, and a symbol. Both
# spellings in the tree are covered -- `tests/common/test_obs_core.py::test_x` and
# `rust-harness/runtime/src/model/cpu.rs::par_rows_is_bit_identical_at_every_width`.
QUALIFIED = re.compile(r"`([\w./-]+\.(?:rs|py))::(\w+)`")

# A continuation naming a second check in the file just cited: the roster writes
# `path::first`, and then `::second` rather than repeating the path.
CONTINUATION = re.compile(r"`::(\w+)`")

# The last column of a gate-backed entry's signal table is a bare function name,
# because the file is named once in the `Asserted by` line above the table.
ASSERTED_BY = re.compile(r"Asserted by `([\w./-]+\.(?:rs|py))`")
BARE_NAME = re.compile(r"^`(\w+)`$")

# A bare name inside an `Asserted by` paragraph, where interp.md lists several
# functions from one file without repeating the path. Deliberately narrow: a
# lowercase identifier carrying an underscore. Widening it to any backticked word
# would swallow `CASES` and every prose mention of a struct or a config key that
# happens to sit in the same paragraph, and a gate that invents references is
# worse than one that misses a few.
PARAGRAPH_NAME = re.compile(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`")

TABLE_ROW = re.compile(r"^\s*\|")
TABLE_RULE = re.compile(r"^\s*\|[\s|:-]+\|\s*$")
SIGNAL_TABLE_HEADER = re.compile(r"^\s*\|\s*Expected signal\s*\|\s*Test function\s*\|")

HEADING = re.compile(r"^(#{2,6})\s+(\S+)\s*(.*)$")
STATUS_LINE = re.compile(r"^Status:\s*([A-Za-z]+)")

# `Observed:`, `Observed (2026-08-08):`, and a parenthetical that runs past the
# end of its own line before reaching the colon -- all three are in the tree, and
# the dated forms are the point: an entry re-run after a code change carries more
# than one block. Anchored at column zero rather than closed with a colon, since
# the colon is not always on this line.
OBSERVED_MARKER = re.compile(r"^\s*(?:\*\*)?Observed\b")
FENCE = re.compile(r"^\s*(?:```|~~~)")

# An entry heading leads with its identifier, in one of two shapes. The legacy
# shape carries a digit: RH.24, OBS-V10, P0.1, HS.3, 0.1. The slug shape (the
# standard since 2026-09-19, docs/verification/README.md "Entry identifiers") is
# a prefix, a dot and a lowercase hyphenated slug: FT.calibrate-qlora,
# I.harvest-crash-recovery. A section heading is neither -- "Coverage roster"
# and "Preconditions" also end in a `Status:` line in one case, and must not be
# read as entries.
LEGACY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*\d[A-Za-z0-9.\-]*$")
SLUG_ID = re.compile(r"^[A-Za-z][A-Za-z0-9-]*\.[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
ENTRY_ID = re.compile(f"(?:{LEGACY_ID.pattern})|(?:{SLUG_ID.pattern})")

# Where a symbol may be defined. Rust test functions and Python test functions
# are the whole population today; the pattern is per-language because `fn` and
# `def` are the only two spellings that matter.
# Where a symbol may be defined. `class` is in the Python list because the bug
# ledger cites test classes -- `test_saelens_contract.py::TestOutputPathContract`
# is a real and resolvable reference, and a gate that called it missing would be
# reporting its own blind spot as a documentation defect.
DEFINITION = {
    ".rs": (r"fn {name}\b",),
    ".py": (r"def {name}\b", r"class {name}\b"),
}


def iter_runbooks(root: Path) -> list[Path]:
    """Every live runbook. The frozen subtrees are skipped, not scanned and passed."""
    base = root / "docs" / "verification"
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXEMPT_DIRS]
        found.extend(Path(dirpath) / n for n in filenames if n.endswith(".md"))
    return sorted(found)


def find_citations(text: str) -> list[tuple[int, str, str]]:
    """Every named check as (line number, file path, symbol).

    Three spellings, and the order of the branches is the reading order: a line
    may carry a qualified citation and then continuations of it, and a bare name
    in a signal table resolves against the `Asserted by` line above.
    """
    citations: list[tuple[int, str, str]] = []
    asserted_by: str | None = None
    in_signal_table = False
    in_asserted_paragraph = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        by = ASSERTED_BY.search(line)
        if by:
            asserted_by = by.group(1)
            in_asserted_paragraph = True
        elif not line.strip():
            in_asserted_paragraph = False

        if SIGNAL_TABLE_HEADER.match(line):
            in_signal_table = True
            continue
        if in_signal_table and not TABLE_ROW.match(line):
            in_signal_table = False

        qualified = QUALIFIED.findall(line)
        for path, symbol in qualified:
            citations.append((lineno, path, symbol))
        if qualified:
            # A continuation inherits the last path named on its own line, which
            # is how every multi-check roster row in the tree is written.
            asserted_by = qualified[-1][0]
            for symbol in CONTINUATION.findall(line):
                citations.append((lineno, qualified[-1][0], symbol))
        elif asserted_by:
            for symbol in CONTINUATION.findall(line):
                citations.append((lineno, asserted_by, symbol))

        if in_asserted_paragraph and asserted_by:
            for symbol in PARAGRAPH_NAME.findall(line):
                citations.append((lineno, asserted_by, symbol))

        if in_signal_table and not TABLE_RULE.match(line) and asserted_by:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2:
                bare = BARE_NAME.match(cells[-1])
                if bare:
                    citations.append((lineno, asserted_by, bare.group(1)))
    return citations


def defines(path: Path, symbol: str) -> bool:
    """True when the file defines the symbol in one of its language's spellings."""
    patterns = DEFINITION.get(path.suffix)
    if patterns is None:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    name = re.escape(symbol)
    return any(re.search(p.format(name=name), text) for p in patterns)


def find_entries(text: str) -> list[tuple[int, str, str, list[str]]]:
    """Every entry as (line number, identifier, status kind, body).

    The body stops at the NEXT HEADING OF ANY DEPTH, not at the next one of equal
    or shallower depth. A parent entry heads several children -- rust-harness.md
    RH.2 heads RH.2a through RH.2c, each with its own status -- and reading to
    the end of the section would attribute the first child's status to the
    parent, which carries none. That misreading makes a parent look like a
    verified entry with no evidence, and the parent is not an entry at all.
    """
    lines = text.splitlines()
    entries = []
    for index, line in enumerate(lines):
        match = HEADING.match(line)
        if not match or not ENTRY_ID.match(match.group(2)):
            continue
        end = len(lines)
        for offset, later in enumerate(lines[index + 1 :], start=index + 1):
            if HEADING.match(later):
                end = offset
                break
        body = lines[index + 1 : end]
        status = next(
            (
                m.group(1).upper()
                for m in (STATUS_LINE.match(b) for b in body)
                if m is not None
            ),
            None,
        )
        if status is None:
            continue
        entries.append((index + 1, match.group(2), status, body))
    return entries


def has_evidence(body: list[str]) -> bool:
    """True when a verified entry offers evidence of either legal kind.

    Two shapes, per the campaign that built this directory. A full entry pastes a
    command's output into an `Observed:` block. A gate-backed entry has no output
    to paste -- it names the check that asserts its signal, and running the gate
    is the verification. Both are legal; an entry offering neither is a status
    line and nothing else.

    A named check counts here only because the other half of this gate resolves
    it against the tree. Accepting the citation without that would make
    "verified" mean "cites something", which is the failure this file exists to
    catch, one level up.
    """
    if find_citations("\n".join(body)):
        return True
    try:
        start = next(i for i, line in enumerate(body) if OBSERVED_MARKER.match(line))
    except StopIteration:
        return False
    for line in body[start + 1 :]:
        if STATUS_LINE.match(line):
            break
        if FENCE.match(line) or not line.strip():
            continue
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    runbooks = iter_runbooks(root)
    if not runbooks:
        out.error(f"runbook-entries: enumerated zero runbooks under {root}")
        out.line("  The gate would pass without checking anything.")
        return 1

    unresolved: list[tuple[str, int, str, str, str]] = []
    hollow: list[tuple[str, int, str]] = []
    cited = 0
    verified = 0
    for doc in runbooks:
        try:
            text = doc.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            out.error(f"runbook-entries: {doc.relative_to(root)} is unreadable: {exc}")
            return 1
        rel_doc = str(doc.relative_to(root))

        for lineno, path, symbol in find_citations(text):
            cited += 1
            target = root / path
            if not target.is_file():
                unresolved.append((rel_doc, lineno, path, symbol, "no such file"))
            elif not defines(target, symbol):
                unresolved.append((rel_doc, lineno, path, symbol, "not defined there"))

        for lineno, entry_id, status, body in find_entries(text):
            if status != "VERIFIED":
                continue
            verified += 1
            if not has_evidence(body):
                hollow.append((rel_doc, lineno, entry_id))

    # Both guards are the same one the sibling gates carry: a pattern that stops
    # matching leaves a gate reporting success over an empty set, which is worse
    # than no gate because it reads as evidence.
    if not cited:
        out.error("runbook-entries: found no named checks to resolve")
        out.line("  The gate would pass without checking anything.")
        out.line("  Expected `path::symbol` citations, or a signal table under an")
        out.line("  `Asserted by` line.")
        return 1
    if not verified:
        out.error("runbook-entries: found no VERIFIED entries to check")
        out.line("  The gate would pass without checking anything.")
        return 1

    out.info(
        f"runbook-entries: {len(runbooks)} runbooks, {cited} named checks, "
        f"{verified} verified entries"
    )

    failed = False
    if unresolved:
        failed = True
        noun = "check does" if len(unresolved) == 1 else "checks do"
        out.error(f"runbook-entries: {len(unresolved)} named {noun} not resolve")
        for rel_doc, lineno, path, symbol, why in unresolved:
            out.line(f"  {rel_doc}:{lineno}: {path}::{symbol} -- {why}")
        out.line("  Solution: an entry naming a check that does not exist claims")
        out.line("  coverage nothing provides. Point it at the check that replaced")
        out.line("  the renamed one, or reopen the entry.")
    else:
        out.ok("runbook-entries: every named check resolves to a definition")

    if hollow:
        failed = True
        noun = "entry is" if len(hollow) == 1 else "entries are"
        out.error(f"runbook-entries: {len(hollow)} verified {noun} evidence-free")
        for rel_doc, lineno, entry_id in hollow:
            out.line(f"  {rel_doc}:{lineno}: {entry_id} is VERIFIED with no Observed")
        out.line("  Solution: paste the command and its output, or set the entry back")
        out.line("  to PENDING. Verified without evidence is the state this directory")
        out.line("  exists to prevent.")
    else:
        out.ok("runbook-entries: every verified entry carries observed evidence")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
