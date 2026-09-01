#!/usr/bin/env python3
"""Fail the gate when a document cites a path in this project that does not exist.

Documents assert facts; nothing checked the assertions; the tree moved. The class
of defect this catches is the dull, mechanical majority: a file renamed or moved
while four documents kept naming the old path. A finding in a report rots; a
finding in a gate fails the build.

What this deliberately does not check: whether a stated port is wrong, whether
prose contradicts a design, whether a number is stale. Those need a reader, or
one of the sibling gates. This converts only the "names something that is not
there" category into an exit code.

Four decisions are recorded here rather than left to whoever next reads the
output.

Frozen records are exempt, because their class obligation is that their bodies
are never edited, and a gate demanding their historical paths still resolve would
force exactly the edits the class forbids.

`.styles/` is pruned. Those files are byte-identical copies of the parent
repository's guides, policed by a parent test that fails on a single changed
byte, and their references describe the parent's tree. They cannot be fixed here,
so reporting them would be noise that never goes away.

A reference is resolved against several roots, not one. Prose in this project
writes `kb/parse.py` meaning the package directory and `tests/test_kb.py` meaning
the project directory, and both are correct English. Demanding a single
interpretation would either reject good prose or force link syntax on every
mention.

EXTERNAL_PATHS names what lives in somebody else's tree: the surrounding
ai-experiments repository, which a standalone copy legitimately does not have,
and the MongoDB documentation corpus, which is fetched rather than committed.
Each is listed with its reason rather than silently skipped.
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

# Directories holding no documents that describe this tree: caches, the
# third-party corpus, and the copied style guides (see the module docstring).
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

# Frozen records: bodies are never edited, so drift in them is correct. Matched
# exactly the way .styles/DOCUMENTATION_STYLE.md says one must be findable.
EXEMPT_PARTS = {"investigations"}
EXEMPT_SUFFIXES = ("_investigation.md",)

# Paths in somebody else's tree, each with why it is legitimately absent.
EXTERNAL_PATHS = {
    # The surrounding ai-experiments repository. This project is extractable, so
    # a reference to one of these describes where it came from rather than
    # claiming something about its own tree.
    "/common/output.py": "the canonical Printer this project deliberately copies",
    "/tests/logtriage/test_conformance.py": "the parent test policing that copy",
    "/.styles/PYTHON_OUTPUT_STYLE.md": "the parent guide the copy is the exception to",
    "common/output.py": "the canonical Printer this project deliberately copies",
    "common/": "the parent package this project deliberately does not import",
    "rust-harness/": "the parent project this one is shaped after",
    "tests/logtriage/test_conformance.py": "the parent test policing the copies",
    "docker-compose.logtriage.yaml": "the served stack lives with the parent's compose files",
    "docker/logtriage/mongod.conf": "stack configuration, in the parent repository",
    "docker/logtriage/mongot.conf": "stack configuration, in the parent repository",
    "docker/logtriage/*.conf": "stack configuration, in the parent repository",
    "docker/llama-gpu/": "the build context the two inference services share",
    "scripts/logtriage/bootstrap.sh": "host-side stack tooling, in the parent repository",
    "scripts/logtriage/generate-secrets.sh": "host-side stack tooling, in the parent repository",
    "env.logtriage.example": "the stack's template, beside its compose file",
    ".secrets/logtriage/stack.env": "generated at run time and never committed",
    ".env.logtriage": "generated from the template and never committed",
    # The MongoDB documentation monorepo, fetched by `logtriage fetch-kb` and
    # never committed. These are paths inside that corpus.
    "content/manual/manual": "a path inside the fetched MongoDB docs monorepo",
    "reference/log-messages.txt": "a page of the fetched manual",
    "includes/extracts-*.yaml": "fragments of the fetched manual",
    "includes/search-in-community/docker-deploy-procedure.rst": "a page of the fetched manual",
    ".styles/BASH_OUTPUT_STYLE.md": "not copied: this project ships no shell scripts of its own",
    ".secrets/logtriage/": "generated at run time and never committed",
    "scripts/logtriage/": "host-side stack tooling, in the parent repository",
    "/mnt/ai-data": "the workstation's model storage, not a path in any repository",
    # Docker Hub references. They are `org/name`, which is path-shaped and is
    # not a path.
    "mongodb/mongodb-community-server": "a Docker Hub image, not a path",
    "mongodb/mongodb-community-search": "a Docker Hub image, not a path",
}

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#\s]+)")
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")

# Not path claims, however many slashes they carry.
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#", "$", "mongodb://")
SKIP_SUBSTRINGS = ("*", "?", "<", ">", "://", "{", "}")

# A leading segment shaped like a hostname. `github.com/mongodb/docs` is a URL
# written without its scheme, not a path in this tree.
DOMAIN_RE = re.compile(r"^[a-z0-9-]+\.(com|org|net|io|dev)/")

# An HTTP API route, written bare because the host varies: `/v1/embeddings`.
# Path-shaped, root-relative, and never a file.
ENDPOINT_RE = re.compile(r"^/v\d+/")

# `store/knowledge.py:fuse` names a symbol inside a file. The file half is a
# path claim and rots like any other; the symbol half is checked by a reader.
SYMBOL_RE = re.compile(r"^(?P<path>[^:]+\.(py|md|toml|yaml|yml|json)):[A-Za-z_]")


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


def candidates(text: str) -> set[str]:
    """Path-shaped tokens from links and code spans.

    Links are unambiguous claims. Code spans are included because much of this
    project's prose names a file as `logtriage/store/base.py` rather than as a
    link, and those references rot exactly as readily.
    """
    found = set(LINK_RE.findall(text))
    for span in CODE_SPAN_RE.findall(text):
        token = span.strip()
        if " " in token or "/" not in token:
            continue
        found.add(token)
    return found


def is_path_claim(reference: str) -> bool:
    if not reference or reference.startswith(SKIP_PREFIXES):
        return False
    if any(bad in reference for bad in SKIP_SUBSTRINGS):
        return False
    if DOMAIN_RE.match(reference) or ENDPOINT_RE.match(reference):
        return False
    return "/" in reference


def path_part(reference: str) -> str:
    """The file a reference names, dropping any `:symbol` suffix."""
    match = SYMBOL_RE.match(reference)
    return match.group("path") if match else reference


def roots_for(document: Path, root: Path) -> list[Path]:
    """Where a relative reference may legitimately be rooted.

    Three, in the order a reader would try them: beside the document, at the
    project root, and inside the package -- because prose says `kb/parse.py`
    meaning `logtriage/kb/parse.py` and that is correct English.
    """
    return [document.parent, root, root / "logtriage"]


def resolves(reference: str, document: Path, root: Path) -> bool:
    if reference.startswith("/"):
        # Repository-root-relative, as .styles/DOCUMENTATION_STYLE.md writes them.
        return (root / reference.lstrip("/")).exists()
    return any((base / reference).exists() for base in roots_for(document, root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    out = Printer()
    root = Path(args.root).resolve()
    found = documents(root)

    if not found:
        out.error("check_doc_refs: enumerated zero documents")
        out.line("  The gate would pass having checked nothing.")
        return 1

    external = {path.rstrip("/") for path in EXTERNAL_PATHS}
    broken: list[tuple[Path, str]] = []
    checked = 0

    for document in found:
        text = document.read_text(encoding="utf-8", errors="replace")
        for reference in sorted(candidates(text)):
            if not is_path_claim(reference) or reference.rstrip("/") in external:
                continue
            target = path_part(reference)
            if target.rstrip("/") in external:
                continue
            checked += 1
            if not resolves(target, document, root):
                broken.append((document.relative_to(root), reference))

    if broken:
        out.error(f"check_doc_refs: {len(broken)} reference(s) do not resolve")
        for document, reference in broken:
            out.line(f"  {document}: {reference}")
        out.line()
        out.line(
            "  Solution: fix the path, or add it to EXTERNAL_PATHS with its reason"
        )
        out.line("  Command: python scripts/docs/check_doc_refs.py")
        return 1

    out.ok(
        f"check_doc_refs: {checked} references in {len(found)} documents all resolve"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
