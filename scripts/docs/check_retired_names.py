#!/usr/bin/env python3
"""Fail the quality gate when a retired name is still in use anywhere in the tree.

check_doc_refs.py catches a document naming a path that does not exist. It cannot
catch a *renamed* file, because the classifier requires a slash: a bare
`docker-compose.yaml` in prose is not a path claim and is never checked.
Loosening that rule was measured and rejected -- accepting extension-bearing
tokens with no slash produced 361 new failures, because prose legitimately says
`config.json`, `main.rs`, and `manifest.json`.

So this takes the opposite approach. Rather than infer which tokens are claims,
it holds an explicit table of names that have been retired, and fails on any
occurrence of one. That has no false positives by construction, and it names the
replacement in the error, so the fix is mechanical rather than archaeological.

The table is the point. Adding an entry when you rename something is what makes
the next rename self-policing; a rename that skips this file is a rename whose
stragglers will be found by a human months later, which is the failure mode the
whole documentation campaign exists to end.

Unlike check_doc_refs.py this scans code as well as documentation, because a
retired filename hides in shell recipes, Python, Dockerfiles, and templates just
as readily as in prose -- and there it is a broken command, not a stale sentence.
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

# Retired name -> (replacement, why it was retired). Both halves are shown in the
# failure, so the reason should read as an explanation to whoever hits it.
RETIRED: dict[str, tuple[str, str]] = {
    "docker-compose.yaml": (
        "docker-compose.inference.yaml",
        "the inference compose file gained a domain suffix, matching its four "
        "siblings (agents, interp, training, rust-harness); a bare `docker "
        "compose` with no -f no longer finds it",
    ),
}

# Directories holding no text of ours.
PRUNE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".ruff_cache",
    # pytest's tmp_path base when the default temp dir is noexec (the agent
    # sandbox; see tests/conftest.py). Generated fixtures, not our text.
    ".pytest-tmp",
}

# Extensions worth reading. Anything textual that can name a file.
SCAN_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".mk",
    ".rs",
    ".json",
    ".cfg",
    ".ini",
    ".example",
    ".txt",
}

# Extensionless or oddly-named files that still carry commands.
SCAN_NAMES = {"Makefile", "Dockerfile"}
SCAN_NAME_PREFIXES = ("Dockerfile.",)

# Frozen by policy: investigation and verification write-ups record the tree as it
# was on the day they were written, so naming the old file is accurate history.
# Same exemption set as check_doc_refs.py, for the same reason.
EXEMPT_PARTS = ("investigations", "vendor")
EXEMPT_PREFIXES = ("docs/verification/",)
EXEMPT_SUFFIXES = ("_investigation.md",)

# Exempt by name, each for a stated reason.
EXEMPT_FILES = {
    # This file: it defines the table, so it necessarily spells every retired name.
    "scripts/docs/check_retired_names.py",
    # The unit tests, which assert on the table and so quote its keys.
    "tests/utils/test_retired_names.py",
}


def is_exempt(rel_path: str) -> bool:
    """Whether a file is outside this gate's scope."""
    parts = Path(rel_path).parts
    return (
        any(p in EXEMPT_PARTS for p in parts)
        or rel_path.startswith(EXEMPT_PREFIXES)
        or rel_path.endswith(EXEMPT_SUFFIXES)
        or rel_path in EXEMPT_FILES
    )


def wants_scan(name: str) -> bool:
    """Whether a filename looks like text that could name a file."""
    return (
        Path(name).suffix in SCAN_SUFFIXES
        or name in SCAN_NAMES
        or name.startswith(SCAN_NAME_PREFIXES)
    )


def iter_files(root: Path) -> list[Path]:
    """Every in-scope text file under root."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in filenames:
            if not wants_scan(name):
                continue
            path = Path(dirpath) / name
            if not is_exempt(str(path.relative_to(root))):
                found.append(path)
    return sorted(found)


def pattern_for(old: str) -> re.Pattern[str]:
    """Match `old` as a whole name rather than as a fragment of a longer one.

    The trailing guard deliberately permits a following `.` so a sentence ending
    in the name still matches; the cost is that a hypothetical `<old>.bak` would
    also match, which is the rarer and more tolerable error.
    """
    return re.compile(rf"(?<![\w.-]){re.escape(old)}(?![\w-])")


def find_retired(
    text: str, patterns: dict[str, re.Pattern[str]]
) -> list[tuple[int, str]]:
    """Every (line number, retired name) occurrence in text."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for old, pattern in patterns.items():
            if pattern.search(line):
                hits.append((lineno, old))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    if not RETIRED:
        out.ok("retired-names: no names are currently retired")
        return 0

    files = iter_files(root)
    # The shellcheck step's empty-set guard, for the same reason: a gate that
    # silently checked nothing is indistinguishable from one that passed.
    if not files:
        out.error(f"retired-names: enumerated zero files under {root}")
        out.line("  The gate would pass without checking anything.")
        return 1

    patterns = {old: pattern_for(old) for old in RETIRED}
    hits: list[tuple[Path, int, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Not text after all; nothing to check.
            continue
        for lineno, old in find_retired(text, patterns):
            hits.append((path, lineno, old))

    out.info(f"retired-names: {len(files)} files, {len(RETIRED)} retired names")
    if not hits:
        out.ok("retired-names: no retired name is still in use")
        return 0

    out.error(f"retired-names: {len(hits)} references use a retired name")
    for path, lineno, old in hits:
        replacement, _ = RETIRED[old]
        out.line(f"  {path.relative_to(root)}:{lineno}: {old} -> {replacement}")
    for old, (replacement, reason) in RETIRED.items():
        out.line(f"  {old} was replaced by {replacement}: {reason}")
    out.line("  Solution: use the replacement name.")
    out.line("  A frozen dated record may instead belong under an exemption.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
