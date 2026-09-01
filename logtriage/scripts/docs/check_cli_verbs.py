#!/usr/bin/env python3
"""Fail the gate when a document names a `python -m logtriage` verb that does not exist.

`check_make_targets.py` covers `make foo`. It cannot see the other half of this
project's command surface, and that half is the one documents actually name:
almost every command block in `docs/` is a `python -m logtriage` line.

This gate was written after the gap it closes was found the expensive way. The
first draft of the Phase D runbook named five commands, and three of them did
not exist -- `ingest`, `eval-retrieval`, `bench` were all written up before they
were written. Nothing in the tree could tell: the paths resolved, the make
targets resolved, and the invocation was prose to every checker here.

Discovery is lexical, from the `add_parser("verb"` literals in commands/, rather
than by importing the CLI. Importing would execute the package, which drags in
pymongo and drain3 on a machine that may not have them, to learn a set of string
literals that a regex reads directly.

Flags are deliberately not checked. A verb is a name that either exists or does
not; a flag set changes far more often, and a gate that failed on a renamed
`--out` would be edited into silence within a month.
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

EXEMPT_PARTS = {"investigations"}
EXEMPT_SUFFIXES = ("_investigation.md",)

# `sub.add_parser("triage",` -- the one place a verb is named.
VERB_RE = re.compile(r'add_parser\(\s*["\']([a-z][a-z0-9-]*)["\']')

# An invocation, in any of the three spellings the documentation uses:
# `python -m logtriage x`, `uv run python -m logtriage x`, and
# `uv run --no-sync python -m logtriage x`.
INVOCATION_RE = re.compile(r"python\s+-m\s+logtriage\s+([a-z][a-z0-9-]*)")

# Read only from fenced blocks and code spans, for the reason the sibling gate
# does: prose says "the triage command" and that is not an invocation.
FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")

# Words that follow the module name but are not verbs.
NOT_VERBS = {"--help", "-h"}


def verbs(root: Path) -> set[str]:
    found: set[str] = set()
    for module in sorted((root / "logtriage" / "commands").glob("*.py")):
        found.update(VERB_RE.findall(module.read_text(encoding="utf-8")))
    return found


def documents(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*.md"):
        parts = set(path.parts)
        if PRUNE_DIRS & parts or EXEMPT_PARTS & parts:
            continue
        if path.name.endswith(EXEMPT_SUFFIXES):
            continue
        found.append(path)
    return sorted(found)


def invocations(text: str) -> set[str]:
    """Verbs named in a fenced block or a code span, and nowhere else."""
    fragments = FENCE_RE.findall(text) + CODE_SPAN_RE.findall(text)
    found: set[str] = set()
    for fragment in fragments:
        found.update(INVOCATION_RE.findall(fragment))
    return found - NOT_VERBS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    out = Printer()
    root = Path(args.root).resolve()

    known = verbs(root)
    if not known:
        out.error("check_cli_verbs: found no verbs in logtriage/commands/")
        out.line("  An empty verb set would fail every invocation rather than none.")
        return 1

    found = documents(root)
    if not found:
        out.error("check_cli_verbs: enumerated zero documents")
        return 1

    missing: list[tuple[Path, str]] = []
    checked = 0
    for document in found:
        text = document.read_text(encoding="utf-8", errors="replace")
        for verb in sorted(invocations(text)):
            checked += 1
            if verb not in known:
                missing.append((document.relative_to(root), verb))

    if missing:
        out.error(f"check_cli_verbs: {len(missing)} documented verb(s) do not exist")
        for document, verb in missing:
            out.line(f"  {document}: python -m logtriage {verb}")
        out.line()
        out.line(f"  Known verbs: {', '.join(sorted(known))}")
        out.line("  Solution: fix the name, or write the command before documenting it")
        return 1

    out.ok(
        f"check_cli_verbs: {checked} invocation(s) in {len(found)} documents "
        f"all resolve ({len(known)} verbs)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
