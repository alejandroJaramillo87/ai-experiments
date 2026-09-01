#!/usr/bin/env python3
"""Fail the gate when something this project ships is named in no document.

The sibling gates all start from the documentation and ask whether it is true.
This one runs the other direction: it enumerates what the code ships, subtracts
what the documentation mentions, and reports the remainder.

That difference matters, because the two questions have different blind spots
and only this one can see a feature nobody thought to document. A gate reading
outwards from prose is silent about a command that exists and is written up
nowhere -- there is no false claim to catch, only an absence, and an absence
leaves no trace in any document.

A mention is a mention. This says nothing about whether the mention explains
anything; that is a reader's question. The bar is only that the surface is
visible to someone auditing coverage, which is a lower bar than it sounds and is
exactly the one an undocumented feature fails.

Scope is the surface a user or an agent meets: the CLI verbs, the implementation
of each port, the pytest markers, and the make targets. Deliberately not every
public function -- a census with an allowlist longer than its findings is a
census nobody runs.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logtriage.output import Printer  # noqa: E402

# Where a mention counts. The documentation tree plus the two files at the root
# that are read before it: an entry point named only in README.md is documented.
DOC_PATHS = ("docs", "README.md", "AGENTS.md")

PRUNE_DIRS = {".git", ".venv", ".kb", ".styles", "__pycache__"}

# `sub.add_parser("triage",` in a commands/ module.
VERB_RE = re.compile(r'add_parser\(\s*["\']([a-z][a-z0-9-]*)["\']')

# A class implementing one of the four ports. Both halves of every pair are
# enumerated, not just the served ones: an offline implementation nobody
# documents is how a reader comes to believe a score means something it does not.
PORT_CLASS_RE = re.compile(
    r"^class\s+((?:InMemory|Hashing|Scripted|Mongo|Llama)[A-Za-z]*)\b", re.MULTILINE
)

MARKER_RE = re.compile(r"^([a-z_]+):")
TARGET_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*:(?!=)", re.MULTILINE)

# Make targets that are plumbing rather than surface. `preflight` is an internal
# dependency of `quality` and is never invoked directly; the four gate steps are
# named by `quality` itself.
INTERNAL_TARGETS = {
    "preflight",
    "fmt",
    "lint",
    "types",
    "docs",
    "test",
    "quality",
    "lock",
}


def documentation(root: Path) -> str:
    """Every document a mention could live in, concatenated."""
    chunks: list[str] = []
    for entry in DOC_PATHS:
        path = root / entry
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
        elif path.is_dir():
            for document in sorted(path.rglob("*.md")):
                if PRUNE_DIRS & set(document.parts):
                    continue
                chunks.append(document.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def verbs(root: Path) -> set[str]:
    found: set[str] = set()
    for module in sorted((root / "logtriage" / "commands").glob("*.py")):
        found.update(VERB_RE.findall(module.read_text(encoding="utf-8")))
    return found


def port_classes(root: Path) -> set[str]:
    found: set[str] = set()
    for package in ("store", "embed", "llm"):
        for module in sorted((root / "logtriage" / package).glob("*.py")):
            found.update(PORT_CLASS_RE.findall(module.read_text(encoding="utf-8")))
    return found


def markers(root: Path) -> set[str]:
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    declared = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    found: set[str] = set()
    for entry in declared.get("markers", []):
        match = MARKER_RE.match(entry)
        if match:
            found.add(match.group(1))
    return found


def targets(root: Path) -> set[str]:
    """The user-facing make targets, which are the parent's.

    This project's own Makefile is entirely gate plumbing -- `quality` and the
    steps it depends on -- and none of it is a surface anyone invokes by name.
    What a user actually types is `make logtriage-up`, defined in the
    surrounding repository's make/logtriage.mk. So the census reads that file
    when it is there, and reports the surface as unchecked when it is not,
    rather than censusing an empty set and calling it complete.
    """
    parent = root.parent / "make" / "logtriage.mk"
    if not parent.is_file():
        return set()
    return set(TARGET_RE.findall(parent.read_text(encoding="utf-8"))) - INTERNAL_TARGETS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    out = Printer()
    root = Path(args.root).resolve()
    text = documentation(root)

    if not text:
        out.error("check_surface_census: found no documentation to search")
        return 1

    surfaces = {
        "CLI verb": verbs(root),
        "port implementation": port_classes(root),
        "pytest marker": markers(root),
        "make target": targets(root),
    }

    # The make targets are the one surface that can legitimately be absent: in
    # an extracted copy there is no parent to define them. Every other empty set
    # means the enumerator broke, and an enumerator that finds nothing reports
    # nothing missing rather than reporting none.
    if not surfaces["make target"]:
        del surfaces["make target"]
        out.skip("check_surface_census: make targets (no parent make/logtriage.mk)")

    for kind, names in surfaces.items():
        if not names:
            out.error(f"check_surface_census: enumerated zero of: {kind}")
            out.line(
                "  An empty surface set would report nothing missing rather than none."
            )
            return 1

    undocumented: list[tuple[str, str]] = []
    counted = 0
    for kind, names in surfaces.items():
        for name in sorted(names):
            counted += 1
            if name not in text:
                undocumented.append((kind, name))

    if undocumented:
        out.error(
            f"check_surface_census: {len(undocumented)} shipped surface(s) named in no document"
        )
        for kind, name in undocumented:
            out.line(f"  {kind}: {name}")
        out.line()
        out.line("  Solution: name it in docs/, README.md, or AGENTS.md")
        out.line(
            "  A mention is enough; whether it explains anything is a reader's question."
        )
        return 1

    summary = ", ".join(f"{len(names)} {kind}s" for kind, names in surfaces.items())
    out.ok(f"check_surface_census: {counted} surfaces all documented ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
