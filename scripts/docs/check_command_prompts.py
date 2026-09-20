#!/usr/bin/env python3
"""Fail the quality gate when a command block carries a shell prompt marker.

A command block is pasted, not read. A line written `$ make quality` fails in a
shell as a command named `$`, and a line written `# apt-get update` is a comment
that runs nothing, so a block written that way cannot be run as written and the
reader finds out by pasting it. .styles/DOCUMENTATION_STYLE.md therefore requires
the command exactly as the shell must receive it, with `sudo` spelled for root and
output kept in a separate block.

What is checked is narrow on purpose. Only fenced blocks tagged as a shell
language (`bash`, `sh`, `shell`, `zsh`) are read, because a `$` at the start of
a line in a Python or Makefile block is not a prompt. A `console` block is not
read either: that tag is the one the style guide reserves for a transcript that
shows a command beside its output, where the prompt is what tells the two apart
and nothing is meant to be pasted. Inside a shell block, a
prompt is a line whose first non-blank characters are `$ ` followed by a word.
A literal `$` that is part of the command (`$HOME`, `$(...)`, `"$MODEL"`) never
starts a line with `$ `, so it is never matched. The `# ` root prompt the guide
also forbids is not checked: a commented-out command (`# nvidia-smi -lgc ...`)
starts the same way and is legitimate, the tree carried no real `# ` prompt when
this gate was written, and a rule that cannot tell the two apart would report
the comments.

Two exemptions. A verification runbook's Observed block is a transcript recorded
as typed, prompt included, and is evidence rather than instructions; nothing
under an `Observed` heading up to the next entry heading is checked. The
runbook README asks that those blocks are never edited, so the gate must not
demand it. And a frozen record (`_investigation.md`, or anything under an
`investigations/` directory) is never edited by rule, so it is skipped whole,
with the same EXEMPT_PARTS and EXEMPT_SUFFIXES the path gate uses.
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

# Frozen records, never edited by rule (.styles/DOCUMENTATION_STYLE.md); the
# same two tables as check_doc_refs.py.
EXEMPT_PARTS = ("investigations", "vendor", "archive")
EXEMPT_SUFFIXES = ("_investigation.md",)

# Fence openers whose contents a shell would receive. `console` is deliberately
# absent: it is the transcript tag.
SHELL_FENCE = re.compile(r"^\s*(?:```|~~~)\s*(bash|sh|shell|zsh)\b", re.I)
ANY_FENCE = re.compile(r"^\s*(?:```|~~~)")

# `$ ` then a command word. The word requirement keeps a bare `$` line, which
# no document writes, from mattering either way.
USER_PROMPT = re.compile(r"^\s*\$ +\S")

# Runbook evidence: from an Observed marker to the next entry heading.
OBSERVED = re.compile(r"^\s*(?:\*\*)?Observed\b")
HEADING = re.compile(r"^#{2,6}\s")


def find_prompts(text: str) -> list[tuple[int, str]]:
    """Every (line number, line) inside a shell block that starts with a prompt."""
    found: list[tuple[int, str]] = []
    in_fence = False
    shell = False
    observed = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not in_fence:
            if HEADING.match(line):
                observed = False
            elif OBSERVED.match(line):
                observed = True
        if ANY_FENCE.match(line):
            if in_fence:
                in_fence = False
                shell = False
            else:
                in_fence = True
                shell = bool(SHELL_FENCE.match(line))
            continue
        if not (in_fence and shell) or observed:
            continue
        if USER_PROMPT.match(line):
            found.append((lineno, line.rstrip()))
    return found


def is_exempt(rel: Path) -> bool:
    """True for a frozen record, which the style guide says is never edited."""
    return any(part in EXEMPT_PARTS for part in rel.parts) or rel.name.endswith(
        EXEMPT_SUFFIXES
    )


def iter_docs(root: Path) -> list[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in PRUNE_DIRS)
        for name in filenames:
            path = Path(dirpath) / name
            if name.endswith(".md") and not is_exempt(path.relative_to(root)):
                found.append(path)
    return sorted(found)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    out = Printer(quiet=args.quiet)

    docs = iter_docs(args.root)
    hits: list[tuple[Path, int, str]] = []
    for path in docs:
        for lineno, line in find_prompts(path.read_text(encoding="utf-8")):
            hits.append((path.relative_to(args.root), lineno, line))

    out.info(f"command-prompts: {len(docs)} documents")
    if not hits:
        out.ok("command-prompts: no command block carries a prompt marker")
        return 0
    out.error(f"command-prompts: {len(hits)} command lines carry a prompt marker")
    for path, lineno, line in hits:
        out.line(f"  {path}:{lineno}: {line.strip()}")
    out.line("  Solution: drop the marker so the block pastes and runs as written.")
    out.line("  A transcript that shows output beside the command is tagged console,")
    out.line("  or sits under an Observed heading; this gate reads neither.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
