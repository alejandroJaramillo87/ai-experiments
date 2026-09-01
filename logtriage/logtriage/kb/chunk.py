"""Turning parsed blocks into retrievable chunks.

A chunk is the unit that gets embedded, retrieved, and shown to the model, so
its boundaries decide whether retrieval is useful. Two rules do most of the
work.

Every chunk carries its heading path. A retrieved fragment arrives with no
surrounding document, so three sentences about "the threshold" are worthless
without "Slow Query Log > Configuring the Threshold" above them -- and the
breadcrumb also gives the embedding the topic words that the prose often omits
precisely because the heading already said them.

A code block is never split. It is usually the actionable half of a page -- the
command that fixes the thing being diagnosed -- and half a command is worse than
none, because it looks runnable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from logtriage.kb.parse import Block, parse_rst
from logtriage.tools.budget import estimate_tokens

# Target chunk size. Small enough that several fit a retrieval budget alongside
# the corpus profile and the findings, large enough to hold a section's
# explanation plus its example.
TARGET_TOKENS = 400

# A chunk is closed once adding the next block would exceed this. The gap above
# TARGET_TOKENS is deliberate: closing exactly at the target would split a
# paragraph off from the code block that illustrates it about half the time.
MAX_TOKENS = 600

# Below this a chunk is merged into its successor instead of standing alone. A
# heading with one sentence under it retrieves badly -- there is not enough text
# for a similarity score to mean anything.
MIN_TOKENS = 20


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable passage of documentation."""

    chunk_id: str
    document: str
    heading_path: tuple[str, ...]
    text: str
    token_estimate: int
    has_code: bool = False
    anchors: tuple[str, ...] = field(default_factory=tuple)
    references: tuple[str, ...] = field(default_factory=tuple)
    line_no: int = 0

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path)

    def render(self) -> str:
        """The chunk as the model sees it, breadcrumb included."""
        header = (
            f"[{self.document}] {self.breadcrumb}"
            if self.heading_path
            else f"[{self.document}]"
        )
        return f"{header}\n{self.text}"


def _block_text(block: Block) -> str:
    if block.kind == "code":
        language = f" ({block.language})" if block.language else ""
        return f"code{language}:\n{block.text}"
    if block.kind == "include":
        # Named rather than dropped: a chunk that says a fragment was included
        # here is more honest than one that silently omits a definition.
        return f"(includes {block.text})"
    return block.text


class _Accumulator:
    """Collects blocks under one heading path into chunks."""

    def __init__(self, document: str) -> None:
        self.document = document
        self.chunks: list[Chunk] = []
        self._reset()

    def _reset(self) -> None:
        self._parts: list[str] = []
        self._tokens = 0
        self._has_code = False
        self._anchors: list[str] = []
        self._references: list[str] = []
        self._line_no = 0

    def add(self, block: Block, heading_path: tuple[str, ...]) -> None:
        text = _block_text(block)
        cost = estimate_tokens(text)

        # An oversized code block is emitted whole rather than split. It breaks
        # the budget, and that is the correct trade: a truncated command reads
        # as runnable and is not.
        if self._parts and self._tokens + cost > MAX_TOKENS:
            self.flush(heading_path)

        if not self._parts:
            self._line_no = block.line_no
        self._parts.append(text)
        self._tokens += cost
        self._has_code = self._has_code or block.kind == "code"
        self._anchors.extend(block.anchors)
        self._references.extend(block.references)

        if self._tokens >= TARGET_TOKENS and not block.kind == "code":
            self.flush(heading_path)

    def flush(self, heading_path: tuple[str, ...]) -> None:
        if not self._parts:
            return
        text = "\n\n".join(self._parts)
        # A too-small trailing fragment is folded back into the previous chunk
        # of the same section rather than emitted: a lone sentence has too
        # little text for a similarity score to mean anything.
        if (
            self._tokens < MIN_TOKENS
            and self.chunks
            and self.chunks[-1].heading_path == heading_path
        ):
            previous = self.chunks.pop()
            text = f"{previous.text}\n\n{text}"
            self._has_code = self._has_code or previous.has_code
            self._anchors = list(previous.anchors) + self._anchors
            self._references = list(previous.references) + self._references
            self._line_no = previous.line_no

        self.chunks.append(
            Chunk(
                chunk_id=f"{self.document}#{len(self.chunks)}",
                document=self.document,
                heading_path=heading_path,
                text=text,
                token_estimate=estimate_tokens(text),
                has_code=self._has_code,
                anchors=tuple(dict.fromkeys(self._anchors)),
                references=tuple(dict.fromkeys(self._references)),
                line_no=self._line_no,
            )
        )
        self._reset()


def chunk_blocks(blocks: list[Block], document: str) -> list[Chunk]:
    """Group blocks into chunks, one heading section at a time."""
    accumulator = _Accumulator(document)
    path: list[str] = []

    for block in blocks:
        if block.kind == "heading":
            # A heading closes the current chunk. Carrying prose across a section
            # boundary produces a chunk whose breadcrumb is wrong for half its
            # content, which is worse than a short chunk.
            accumulator.flush(tuple(path))
            del path[block.level - 1 :]
            path.append(block.text)
            continue
        accumulator.add(block, tuple(path))

    accumulator.flush(tuple(path))
    return accumulator.chunks


def chunk_document(text: str, document: str) -> list[Chunk]:
    """Parse and chunk one reStructuredText document."""
    return chunk_blocks(parse_rst(text, document), document)
