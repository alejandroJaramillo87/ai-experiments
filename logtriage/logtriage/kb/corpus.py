"""Reading the MongoDB Database Manual from a fetched checkout.

The manual is not a directory of self-contained pages. Of its 1,611 pages, 1,281
pull content in with `.. include::` -- 7,350 directives in total, resolved
against a tree of 2,649 fragments that are never pages themselves. A reader that
treats a page file as the page therefore sees roughly a fifth of the manual and
a great many dangling references, which is why this module exists between the
filesystem and the parser rather than the parser reading files directly.

Two path conventions, both Snooty's:

  /includes/fact-log-redaction.rst    absolute, resolved from source/
  fact-log-redaction.rst              relative, resolved from the including file

The fragments are deliberately not indexed as documents of their own. They have
no headings of their own to speak of and would retrieve as context-free
sentences, and once they are spliced into the pages that include them their
content is already in the index -- twice over would only let a fragment outrank
the page that explains it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from logtriage.kb.constants import expand, load_constants

# Page files. The manual uses .txt for pages and .rst for included fragments;
# the extension is the only thing that distinguishes them, since both are rST.
PAGE_SUFFIX = ".txt"

# How deep an include chain may nest before it is abandoned. Fragments include
# other fragments (1,135 times in this corpus), so a bound is needed; this one is
# far above the deepest real chain and exists to stop a cycle, not to shape
# output.
MAX_INCLUDE_DEPTH = 8


@dataclass(slots=True)
class ResolutionStats:
    """What resolving a document's references cost, kept for the index report.

    Counted rather than logged. A missing include silently degrades a chunk into
    something that reads fine and says less, which is the kind of corpus defect
    that is never noticed unless a number moves.
    """

    includes_resolved: int = 0
    includes_missing: int = 0
    missing_targets: set[str] = field(default_factory=set)
    unknown_constants: set[str] = field(default_factory=set)

    def merge(self, other: ResolutionStats) -> None:
        self.includes_resolved += other.includes_resolved
        self.includes_missing += other.includes_missing
        self.missing_targets |= other.missing_targets
        self.unknown_constants |= other.unknown_constants


class Corpus:
    """A fetched manual: its pages, its fragments, and its constants."""

    def __init__(self, root: Path) -> None:
        """`root` is the manual project directory -- the one holding snooty.toml
        and source/."""
        self.root = Path(root)
        self.source = self.root / "source"
        if not self.source.is_dir():
            raise FileNotFoundError(
                f"No source/ directory under {self.root}. Expected a manual "
                f"checkout produced by `logtriage fetch-kb`."
            )
        snooty = self.root / "snooty.toml"
        self.constants = load_constants(snooty) if snooty.is_file() else {}
        self.stats = ResolutionStats()

    @property
    def version(self) -> str:
        """The manual's own version constant, for the index sidecar."""
        return self.constants.get("version", "unknown")

    def pages(self) -> list[Path]:
        """Every page file, sorted, so an index build is order-stable."""
        return sorted(self.source.rglob(f"*{PAGE_SUFFIX}"))

    def document_name(self, path: Path) -> str:
        """The path a citation carries: relative to source/, posix separators."""
        return path.relative_to(self.source).as_posix()

    def read(self, path: Path) -> str:
        """Read a file and expand its Snooty constants.

        Expansion happens here rather than in the parser so that it applies
        uniformly to pages and to included fragments, and so that it happens
        before any role is stripped -- several constants expand into roles.
        """
        text = path.read_text(encoding="utf-8", errors="replace")
        expanded, missing = expand(text, self.constants)
        self.stats.unknown_constants.update(missing)
        return expanded

    def resolve_include(
        self, target: str, options: Mapping[str, str], parent: str
    ) -> str | None:
        """Return the text an `.. include::` refers to, or None if it is absent.

        Missing targets are normal and expected: 844 `literalinclude` directives
        point into a code-examples tree that the documentation toolchain
        generates at build time and that a source checkout does not contain.
        Returning None lets the parser leave a legible marker instead of
        inventing content.
        """
        path = self._locate(target, parent)
        if path is None or not path.is_file():
            self.stats.includes_missing += 1
            self.stats.missing_targets.add(target)
            return None

        text = self.read(path)
        text = _slice(text, options.get("start-after"), options.get("end-before"))
        self.stats.includes_resolved += 1
        return text

    def _locate(self, target: str, parent: str) -> Path | None:
        if not target:
            return None
        if target.startswith("/"):
            candidate = self.source / target.lstrip("/")
        else:
            candidate = (self.source / parent).parent / target
        if not candidate.is_file():
            # Snooty allows an include target to omit its extension, and the
            # manual uses that form for whole families of fragments -- names
            # like /includes/command/unshardCollection or
            # /includes/8.0-write-concern-majority-reads. Treating those as
            # missing silently drops real prose from the pages that include it.
            for suffix in (".rst", ".txt"):
                extended = candidate.with_name(candidate.name + suffix)
                if extended.is_file():
                    candidate = extended
                    break
        try:
            # Refuse a target that escapes the corpus. These paths come from
            # documentation source rather than a user, but a resolver that will
            # read any path given a few `../` segments is not one to keep in a
            # tool that also reads production logs.
            candidate = candidate.resolve()
            candidate.relative_to(self.source.resolve())
        except (ValueError, OSError):
            return None
        return candidate

    def iter_documents(self) -> Iterator[tuple[str, str]]:
        """Every page as (document name, expanded text)."""
        for path in self.pages():
            yield self.document_name(path), self.read(path)


def _slice(text: str, start_after: str | None, end_before: str | None) -> str:
    """Apply an include's :start-after:/:end-before: bounds.

    A bound that does not appear is ignored rather than treated as an empty
    match. Sphinx errors here; for a retrieval corpus the whole fragment is a
    better answer than none, and the alternative silently drops content.
    """
    if start_after:
        index = text.find(start_after)
        if index != -1:
            text = text[index + len(start_after) :]
    if end_before:
        index = text.find(end_before)
        if index != -1:
            text = text[:index]
    return text.strip("\n")


def file_digest(path: Path) -> str:
    """sha256 of a file, for the corpus manifest."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
