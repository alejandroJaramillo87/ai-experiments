#!/usr/bin/env python3
"""Fail the gate when a document names a `make` target that does not exist.

The two sibling gates check paths and retired names. Neither can see a command,
and a documented command is the assertion a reader acts on first: a wrong path
costs a grep, a wrong target costs the afternoon spent believing the tool is
broken. Renaming a target is cheap and touches one line of one file, so the
documentation naming it is exactly what nobody thinks to update.

Target discovery is lexical -- a regex over the Makefiles rather than shelling
out to `make help`, which would pull in the whole file's variable expansion for
a set of literal rule heads.

Two decisions are recorded here.

Reference extraction reads only fenced blocks and code spans, because `make` is
also an English verb. "make sure the service is up" is not a command, and a
classifier that treated it as one would fail the gate on good prose.

PARENT_TARGETS is what makes this work in an extracted copy. This project's
documentation legitimately names `make logtriage-fetch-kb`, which lives in the
surrounding repository's make/logtriage.mk and does not exist standalone. When
that file is present its targets are read from it; when it is absent they
resolve through the table below, each with the reason it is elsewhere.
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

# Targets defined in the surrounding ai-experiments repository, in
# make/logtriage.mk. They drive this project from outside it -- provisioning its
# virtualenv, running its gate, starting the served stack -- so its own
# documentation names them, and a standalone copy legitimately has none of them.
PARENT_TARGETS = {
    "logtriage-install": "provisions this project's virtualenv, from the parent",
    "logtriage-quality": "runs this project's gate, from the parent",
    "logtriage-test": "runs this project's tests, from the parent",
    "logtriage-fetch-kb": "fetches the corpus, from the parent",
    "logtriage-index-kb": "builds the index, from the parent",
    "logtriage-secrets": "generates the served stack's credentials",
    "logtriage-up": "starts the served stack",
    "logtriage-down": "stops the served stack",
    "logtriage-status": "shows the served stack",
    "logtriage-logs": "tails the served stack",
    "logtriage-bootstrap": "initiates the replica set and creates the mongot user",
    "logtriage-shell-mongo": "opens mongosh against the deployment",
}

# A rule head: a target name at the start of a line, followed by a colon that is
# not `:=`. Pattern rules stand for a family rather than a name and are skipped.
TARGET_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*:(?!=)", re.MULTILINE)

# `make foo`, but only inside a fenced block or a code span.
INVOCATION_RE = re.compile(r"\bmake\s+(?:-C\s+\S+\s+)?([a-z][a-z0-9-]*)")
FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")

# Words that follow "make" in an invocation but are arguments, not targets.
NOT_TARGETS = {"sure", "it", "the", "a", "an", "this", "that", "them"}


def collect_targets(root: Path) -> tuple[set[str], list[Path]]:
    """Every rule head in this project's Makefile, and the parent's if present."""
    makefiles = [root / "Makefile"]
    parent = root.parent / "make" / "logtriage.mk"
    if parent.is_file():
        makefiles.append(parent)

    targets: set[str] = set()
    read: list[Path] = []
    for path in makefiles:
        if not path.is_file():
            continue
        targets.update(TARGET_RE.findall(path.read_text(encoding="utf-8")))
        read.append(path)
    return targets, read


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
    """Targets named in a fenced block or a code span, and nowhere else."""
    fragments = FENCE_RE.findall(text) + CODE_SPAN_RE.findall(text)
    found: set[str] = set()
    for fragment in fragments:
        found.update(INVOCATION_RE.findall(fragment))
    return found - NOT_TARGETS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    out = Printer()
    root = Path(args.root).resolve()

    targets, makefiles = collect_targets(root)
    if not targets:
        out.error("check_make_targets: found no targets in any Makefile")
        out.line("  An empty target set would fail every reference rather than none.")
        return 1

    known = targets | set(PARENT_TARGETS)
    found = documents(root)
    if not found:
        out.error("check_make_targets: enumerated zero documents")
        return 1

    missing: list[tuple[Path, str]] = []
    checked = 0
    for document in found:
        text = document.read_text(encoding="utf-8", errors="replace")
        for target in sorted(invocations(text)):
            checked += 1
            if target not in known:
                missing.append((document.relative_to(root), target))

    if missing:
        out.error(
            f"check_make_targets: {len(missing)} documented target(s) do not exist"
        )
        for document, target in missing:
            out.line(f"  {document}: make {target}")
        out.line()
        out.line("  Solution: fix the name, add the target, or -- if it lives in the")
        out.line("            surrounding repository -- add it to PARENT_TARGETS")
        return 1

    where = ", ".join(str(path.name) for path in makefiles)
    out.ok(
        f"check_make_targets: {checked} invocation(s) in {len(found)} documents "
        f"all resolve ({len(targets)} local from {where}, "
        f"{len(PARENT_TARGETS)} from the parent)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
