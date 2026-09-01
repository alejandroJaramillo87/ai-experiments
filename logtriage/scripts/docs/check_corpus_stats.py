#!/usr/bin/env python3
"""Fail the gate when the documented corpus statistics disagree with the index.

This one has no analogue in the parent repository, and it guards the failure
this project is most exposed to.

docs/corpus/statistics.md states counts measured at a named corpus commit. After
a `fetch-kb --refresh` the manual moves and every one of those numbers is quietly
wrong -- with no symptom, because a stale statistic reads exactly like a fresh
one. Nobody re-derives 15,025 by hand to check, and a reader who did would have
to rebuild the index to do it.

So the numbers are held against their two sources: kb-manifest.json, which is
committed and describes the corpus, and .kb/index/index.json, which is generated
and describes what was built from it.

Absent either source, the checks that need it skip **by name**, saying which
numbers went unchecked. A silent skip is indistinguishable from a pass, and in a
bare checkout -- where there is no corpus at all -- that is the normal case
rather than an exception.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logtriage.output import Printer  # noqa: E402

STATISTICS_DOC = Path("docs/corpus/statistics.md")
MANIFEST = Path("kb-manifest.json")
INDEX_META = Path(".kb/index/index.json")

# `| Chunks | 15,025 |` -> ("Chunks", "15,025")
ROW_RE = re.compile(r"^\|\s*(?P<label>[^|]+?)\s*\|\s*(?P<value>[^|]+?)\s*\|\s*$")

# `8,394 of 8,394` -> both halves matter, and they are different claims.
OF_RE = re.compile(r"^([\d,]+)\s+of\s+([\d,]+)$")
NUMBER_RE = re.compile(r"^[\d,]+$")

# The commit is stated in prose, not in a table.
COMMIT_RE = re.compile(r"corpus commit `([0-9a-f]{7,40})`")


def number(text: str) -> int | None:
    return int(text.replace(",", "")) if NUMBER_RE.match(text) else None


def rows(document: Path) -> dict[str, str]:
    """Every `| label | value |` row, by label.

    Later rows win on a duplicate label. There are none today; if there ever
    are, two rows claiming different values for one label is a defect this
    check cannot see, and the duplicate-labelled table is the thing to fix.
    """
    found: dict[str, str] = {}
    for line in document.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line)
        if match:
            # Backticks are removed rather than stripped from the ends: a
            # label like "Over `MAX_TOKENS`" would otherwise lose only its
            # trailing one and never match anything.
            label = match.group("label").strip().replace("`", "")
            found[label] = match.group("value").strip()
    return found


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    out = Printer()
    root = Path(args.root).resolve()
    document = root / STATISTICS_DOC

    if not document.is_file():
        out.error(f"check_corpus_stats: {STATISTICS_DOC} is missing")
        out.line("  The statistics are the only visible signal on 15,000 chunks.")
        return 1

    text = document.read_text(encoding="utf-8")
    stated = rows(document)
    manifest = load(root / MANIFEST)
    meta = load(root / INDEX_META)
    index_stats = (meta or {}).get("stats", {})

    mismatches: list[str] = []
    skipped: list[str] = []
    checked = 0

    def compare(label: str, actual: Any, source: str) -> None:
        nonlocal checked
        if label not in stated:
            mismatches.append(f"{label}: not stated in {STATISTICS_DOC}")
            return
        checked += 1
        claimed = number(stated[label])
        if claimed is None:
            mismatches.append(f"{label}: '{stated[label]}' is not a number")
        elif claimed != actual:
            mismatches.append(
                f"{label}: document says {claimed:,}, {source} says {actual:,}"
            )

    # The corpus, from the committed manifest.
    if manifest is None:
        skipped.append("files, bytes, and the corpus commit (kb-manifest.json absent)")
    else:
        compare("Files", int(manifest["file_count"]), "the manifest")
        compare("Bytes", int(manifest["total_bytes"]), "the manifest")
        checked += 1
        commit = COMMIT_RE.search(text)
        if not commit:
            mismatches.append("no corpus commit stated in the document")
        elif not str(manifest["commit"]).startswith(commit.group(1)):
            mismatches.append(
                f"corpus commit: document says {commit.group(1)}, "
                f"the manifest says {manifest['commit'][:12]}"
            )

    # What was built from it, from the index sidecar.
    if not index_stats:
        skipped.append(
            "pages, chunks, code-block integrity and include resolution "
            "(.kb/index/index.json absent; run index-kb)"
        )
    else:
        compare("Pages (.txt)", int(index_stats["documents"]), "the index")
        compare("Chunks", int(index_stats["chunks"]), "the index")
        compare("Chunks carrying code", int(index_stats["with_code"]), "the index")
        compare("Over MAX_TOKENS", int(index_stats["over_max_tokens"]), "the index")
        compare("Under MIN_TOKENS", int(index_stats["under_min_tokens"]), "the index")
        compare(
            "Unexpanded {+x+} substitutions",
            int(index_stats["unexpanded_substitutions"]),
            "the index",
        )
        compare("Residual role syntax", int(index_stats["residual_roles"]), "the index")

        # Two "N of M" rows, where both halves are separate claims: an index
        # that split a code block would move only the first number.
        for label, (got, total) in {
            "Code blocks intact": (
                int(index_stats["code_blocks_intact"]),
                int(index_stats["code_blocks"]),
            ),
            "Include directives resolved": (
                int(index_stats["includes_resolved"]),
                int(index_stats["includes_resolved"])
                + int(index_stats["includes_missing"]),
            ),
        }.items():
            if label not in stated:
                mismatches.append(f"{label}: not stated in {STATISTICS_DOC}")
                continue
            checked += 1
            match = OF_RE.match(stated[label])
            if not match:
                mismatches.append(f"{label}: '{stated[label]}' is not 'N of M'")
            elif (number(match.group(1)), number(match.group(2))) != (got, total):
                mismatches.append(
                    f"{label}: document says {stated[label]}, "
                    f"the index says {got:,} of {total:,}"
                )

    if mismatches:
        out.error(f"check_corpus_stats: {len(mismatches)} statistic(s) disagree")
        for line in mismatches:
            out.line(f"  {line}")
        out.line()
        out.line("  Solution: re-measure and update the document")
        out.line("  Command: python -m logtriage index-kb --json")
        return 1

    for line in skipped:
        out.skip(f"check_corpus_stats: {line}")
    out.ok(f"check_corpus_stats: {checked} statistic(s) match their source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
