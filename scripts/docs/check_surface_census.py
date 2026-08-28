#!/usr/bin/env python3
"""Fail the quality gate when a shipped rust-harness surface appears in no runbook.

The sibling gates all start from the documentation and ask whether it is true.
This one runs the other direction: it enumerates what the code ships, then
subtracts what the runbooks mention, and reports the remainder. That difference
matters, because the two questions have different blind spots and only this one
can see a feature nobody thought to verify.

It exists because that blind spot was real. `attention-capture` and `layer-diff`
shipped as binaries and were named exactly once across every runbook, in a list
of catalog values -- prose, not an entry. `long-context-golden` and `tiny-model`
had no mention anywhere. Three surface inventories run by hand found them, and a
hand inventory is true on the day it is run.

Scope is deliberately the rust-harness command line and nothing else. The
enumeration has to be exact to be worth running, and this is where the tree makes
that cheap: every flag of every binary is a `Flag { name: "--x", .. }` literal in
one of eight files, by the `cli` module's own design, so the surface is a set of
string literals rather than something inferred from argparse at import time. A
census over `make` targets was considered and dropped: most targets are docker
and training plumbing that no verification runbook should mention, so the check
would have shipped with an allowlist longer than its findings.

A mention is a mention. This says nothing about whether the mention verifies
anything -- check_runbook_entries.py asks that of entries, and a reader asks it of
evidence. Here the bar is only that the element is visible to someone auditing
coverage, which is the bar the four binaries above failed.
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

# Frozen records, for the reason docs/verification/README.md gives. A mention in
# one of them is history, not coverage, so they are excluded from the subtrahend
# as well as from the checks.
EXEMPT_DIRS = {"archive", "investigations"}

# The `Flag` literal, which every binary's surface is built from. `name:` is the
# spelling accepted on the command line, leading dashes included.
FLAG_LITERAL = re.compile(r'name:\s*"(--[a-z0-9-]+)"')

# Universal and uninteresting: present on every binary by construction, pinned by
# the cli-surface tests, and a runbook entry per binary saying "--help prints
# help" would be noise standing in for coverage.
UNIVERSAL = {"--help"}


def surface(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """The shipped command line as (binaries, flags), each mapped to its sources.

    The binary set is the file names under `runtime/src/bin/`, plus `harness`
    itself, which is the workspace's one non-`bin/` program.
    """
    bins_dir = root / "rust-harness" / "runtime" / "src" / "bin"
    harness_cli = root / "rust-harness" / "harness" / "src" / "cli.rs"

    binaries: dict[str, list[Path]] = {}
    flags: dict[str, list[Path]] = {}
    sources = sorted(bins_dir.glob("*.rs")) + [harness_cli]
    for source in sources:
        if not source.is_file():
            continue
        if source.parent == bins_dir:
            binaries.setdefault(source.stem, []).append(source)
        for flag in FLAG_LITERAL.findall(source.read_text(encoding="utf-8")):
            if flag in UNIVERSAL:
                continue
            flags.setdefault(flag, []).append(source)
    binaries.setdefault("harness", [harness_cli])
    return binaries, flags


def mentioned(root: Path) -> str:
    """Every live runbook's text, concatenated, as the thing to subtract from."""
    base = root / "docs" / "verification"
    chunks = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXEMPT_DIRS]
        for name in filenames:
            if name.endswith(".md"):
                chunks.append(
                    (Path(dirpath) / name).read_text(encoding="utf-8", errors="replace")
                )
    return "\n".join(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    binaries, flags = surface(root)
    text = mentioned(root)

    # The empty-set guard the sibling gates all carry. A refactor that moves the
    # `Flag` literals would otherwise leave this reporting success over nothing,
    # which reads as a census rather than as the absence of one.
    if not binaries or not flags:
        out.error("surface-census: enumerated no rust-harness command line")
        out.line("  The gate would pass without checking anything.")
        out.line(f'  Expected `Flag {{ name: "--x" }}` literals under {root}')
        return 1
    if not text:
        out.error(f"surface-census: read no runbooks under {root}/docs/verification")
        out.line("  The gate would pass by finding every element unmentioned.")
        return 1

    # A binary is matched as a whole word so `layer-diff` is not satisfied by a
    # sentence about layers; a flag carries its own dashes and needs no anchor.
    missing_bins = [
        name
        for name in sorted(binaries)
        if not re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text)
    ]
    missing_flags = [flag for flag in sorted(flags) if flag not in text]

    out.info(
        f"surface-census: {len(binaries)} binaries, {len(flags)} flags, "
        f"against docs/verification/"
    )

    if not missing_bins and not missing_flags:
        out.ok("surface-census: every shipped binary and flag appears in a runbook")
        return 0

    total = len(missing_bins) + len(missing_flags)
    noun = "element is" if total == 1 else "elements are"
    out.error(f"surface-census: {total} shipped {noun} named in no runbook")
    for name in missing_bins:
        where = ", ".join(str(p.relative_to(root)) for p in binaries[name])
        out.line(f"  binary {name} ({where})")
    for flag in missing_flags:
        where = ", ".join(str(p.relative_to(root)) for p in flags[flag])
        out.line(f"  flag {flag} ({where})")
    out.line("  Solution: give it a full entry, a gate-backed entry naming the check")
    out.line("  that asserts it, or a coverage-roster row naming the test that pins")
    out.line("  it. Shipped and unmentioned is the state this census exists to find.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
