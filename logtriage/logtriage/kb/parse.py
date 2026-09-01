"""Parsing the MongoDB manual's reStructuredText into blocks.

The manual is Snooty-flavoured rST, not plain rST, and the difference matters
for retrieval. Three constructs carry most of the value in a log-diagnosis
knowledge base and all three are directives rather than prose:

  .. code-block::   the command that fixes the thing being diagnosed
  .. include::      the shared fragment where a definition actually lives
  .. tabs::         parallel instructions, only one of which applies

A naive line splitter destroys the first, silently drops the third's structure,
and leaves the second as a dangling path. It also loses heading context, which
is what makes a retrieved fragment interpretable at all -- three sentences about
"the threshold" are useless without the section that says which threshold.

So this is a real block parser rather than a regex over lines. It is not a
complete rST implementation and does not try to be: it recovers structure,
keeps code verbatim, and flattens the rest to readable text.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

# rST section adornment characters. Levels are not fixed by the character --
# a document establishes its own hierarchy by the order it first uses each one,
# which is why the level is assigned during the parse rather than looked up.
ADORNMENT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

# An adornment line must be at least this long to count. Guards against a line
# of dashes inside a table or a horizontal rule being read as a heading.
MIN_ADORNMENT = 3

# `.. name:: argument`
DIRECTIVE_RE = re.compile(r"^\.\.\s+([a-zA-Z0-9_-]+)::\s*(.*)$")

# `.. _some-target:` -- an anchor, which is what :ref: links resolve to and
# therefore worth keeping as metadata even though it renders as nothing.
ANCHOR_RE = re.compile(r"^\.\.\s+_([A-Za-z0-9_.-]+):\s*$")

# `:option: value` inside a directive body.
OPTION_RE = re.compile(r"^:([a-zA-Z0-9_-]+):\s*(.*)$")

# Inline roles: :ref:`text <target>`, :doc:`/path`, :manual:`...`, ``literal``.
ROLE_WITH_TARGET_RE = re.compile(r":([a-zA-Z0-9_+:.-]+):`([^<`]*?)\s*<([^>]*)>`")
ROLE_RE = re.compile(r":([a-zA-Z0-9_+:.-]+):`([^`]*)`")
LITERAL_RE = re.compile(r"``([^`]*)``")

# Directives whose body is prose worth keeping, flattened into the surrounding
# text with a label. Everything else with a body is kept as its own block.
PROSE_DIRECTIVES = frozenset(
    {"note", "warning", "important", "tip", "seealso", "example", "admonition"}
)

# Directives that render nothing and carry no retrievable content.
IGNORED_DIRECTIVES = frozenset(
    {
        "contents",
        "default-domain",
        "meta",
        "raw",
        "toctree",
        "class",
        "highlight",
        "index",
    }
)


@dataclass(frozen=True, slots=True)
class Block:
    """One parsed unit of a document."""

    kind: str
    text: str
    level: int = 0
    language: str | None = None
    directive: str | None = None
    line_no: int = 0
    anchors: tuple[str, ...] = field(default_factory=tuple)
    references: tuple[str, ...] = field(default_factory=tuple)


def strip_roles(text: str) -> tuple[str, tuple[str, ...]]:
    """Flatten inline roles to their display text, keeping the targets.

    `:ref:`slow queries <slow-query-log>`` becomes "slow queries" with
    "slow-query-log" recorded. Where a role has no display text the target is
    used, since `:doc:`/reference/log-messages`` should read as something rather
    than vanish.
    """
    references: list[str] = []

    def with_target(match: re.Match[str]) -> str:
        references.append(match.group(3))
        return match.group(2)

    def bare(match: re.Match[str]) -> str:
        target = match.group(2)
        references.append(target)
        # A leading slash is a document path; its last segment reads better than
        # the whole path in running text.
        return target.rsplit("/", 1)[-1] if target.startswith("/") else target

    text = ROLE_WITH_TARGET_RE.sub(with_target, text)
    text = ROLE_RE.sub(bare, text)
    text = LITERAL_RE.sub(r"\1", text)
    return text, tuple(references)


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_adornment(line: str) -> bool:
    stripped = line.rstrip()
    return (
        len(stripped) >= MIN_ADORNMENT
        and set(stripped) <= ADORNMENT
        and len(set(stripped)) == 1
    )


class _Parser:
    def __init__(self, lines: list[str], document: str) -> None:
        self.lines = lines
        self.document = document
        self.index = 0
        # Adornment character to heading level, in order of first appearance --
        # the rST rule. A document that uses = then - then ~ has those as levels
        # 1, 2, 3 whatever any other document does.
        self.levels: dict[str, int] = {}
        self.pending_anchors: list[str] = []

    def _take_indented_body(self, base_indent: int) -> list[str]:
        """Consume the indented block following a directive.

        Blank lines inside the body are kept: they separate paragraphs, and in a
        code block they are part of the code.
        """
        body: list[str] = []
        while self.index < len(self.lines):
            line = self.lines[self.index]
            if not line.strip():
                body.append("")
                self.index += 1
                continue
            if _indent_of(line) <= base_indent:
                break
            body.append(line)
            self.index += 1
        # Blank lines bounding the body are the directive's own formatting, not
        # content. A leading one would make every code block start with an empty
        # line, which shows up verbatim in anything that renders it.
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        return body

    def _dedent(self, body: list[str]) -> str:
        indents = [_indent_of(line) for line in body if line.strip()]
        common = min(indents) if indents else 0
        return "\n".join(line[common:] if line.strip() else "" for line in body)

    def _heading(self, level: int, text: str, line_no: int) -> Block:
        clean, references = strip_roles(text.strip())
        block = Block(
            kind="heading",
            text=clean,
            level=level,
            line_no=line_no,
            anchors=tuple(self.pending_anchors),
            references=references,
        )
        self.pending_anchors = []
        return block

    def parse(self) -> Iterator[Block]:
        while self.index < len(self.lines):
            line = self.lines[self.index]
            line_no = self.index + 1

            if not line.strip():
                self.index += 1
                continue

            anchor = ANCHOR_RE.match(line)
            if anchor:
                # Held rather than emitted: an anchor labels whatever follows it,
                # and attaching it to the next heading is what lets a chunk say
                # which :ref: target reaches it.
                self.pending_anchors.append(anchor.group(1))
                self.index += 1
                continue

            # Overline form: adornment, title, matching adornment.
            if (
                _is_adornment(line)
                and self.index + 2 < len(self.lines)
                and self.lines[self.index + 1].strip()
                and _is_adornment(self.lines[self.index + 2])
            ):
                char = line.strip()[0]
                level = self.levels.setdefault(char, len(self.levels) + 1)
                title = self.lines[self.index + 1]
                self.index += 3
                yield self._heading(level, title, line_no)
                continue

            # Underline form: title followed by adornment at least as long.
            if (
                self.index + 1 < len(self.lines)
                and _is_adornment(self.lines[self.index + 1])
                and len(self.lines[self.index + 1].rstrip()) >= len(line.rstrip())
                and not line.startswith("..")
            ):
                char = self.lines[self.index + 1].strip()[0]
                level = self.levels.setdefault(char, len(self.levels) + 1)
                self.index += 2
                yield self._heading(level, line, line_no)
                continue

            directive = DIRECTIVE_RE.match(line)
            if directive:
                yield from self._parse_directive(directive, line, line_no)
                continue

            if line.startswith(".."):
                # A comment. Skipped along with its indented body.
                self.index += 1
                self._take_indented_body(_indent_of(line))
                continue

            yield from self._parse_paragraph(line_no)

    def _parse_directive(
        self, match: re.Match[str], line: str, line_no: int
    ) -> Iterator[Block]:
        name = match.group(1).lower()
        argument = match.group(2).strip()
        base_indent = _indent_of(line)
        self.index += 1
        body = self._take_indented_body(base_indent)

        if name in IGNORED_DIRECTIVES:
            return

        if name == "include":
            # The body of an include lives in another file. Recording the
            # reference rather than resolving it keeps the parser a parser; a
            # later pass can follow them, and a chunk that mentions an
            # unresolved include is more honest than one that silently omits it.
            yield Block(
                kind="include",
                text=argument,
                line_no=line_no,
                references=(argument,),
                anchors=tuple(self.pending_anchors),
            )
            self.pending_anchors = []
            return

        if name in {"code-block", "code", "literalinclude", "io-code-block"}:
            yield Block(
                kind="code",
                text=self._dedent(body),
                language=argument or None,
                directive=name,
                line_no=line_no,
            )
            return

        # Options (`:tabid: atlas`) are stripped from the head of the body; the
        # tab label is worth keeping, the rest is rendering metadata.
        content: list[str] = []
        options: dict[str, str] = {}
        seen_content = False
        for entry in body:
            option = OPTION_RE.match(entry.strip())
            if option and not seen_content:
                options[option.group(1)] = option.group(2)
                continue
            if entry.strip():
                seen_content = True
            content.append(entry)

        label = argument or options.get("tabid") or ""
        if name in PROSE_DIRECTIVES:
            prefix = f"{name.upper()}: "
        elif name in {"tab", "tabs"}:
            prefix = f"[{label or name}] " if label else ""
        else:
            prefix = f"[{name}{' ' + label if label else ''}] "

        nested = _Parser(self._dedent(content).splitlines(), self.document)
        nested.levels = self.levels
        children = list(nested.parse())
        if not children:
            return

        # The label is attached to the first block of the body rather than
        # emitted alone, so a chunk boundary cannot separate "[Atlas]" from the
        # instructions it qualifies.
        first = children[0]
        yield Block(
            kind=first.kind,
            text=f"{prefix}{first.text}" if first.kind != "code" else first.text,
            level=first.level,
            language=first.language,
            directive=name,
            line_no=line_no,
            anchors=first.anchors,
            references=first.references,
        )
        yield from children[1:]

    def _parse_paragraph(self, line_no: int) -> Iterator[Block]:
        collected: list[str] = []
        base_indent = _indent_of(self.lines[self.index])
        is_table = self.lines[self.index].lstrip().startswith(("+--", "|"))
        while self.index < len(self.lines):
            line = self.lines[self.index]
            if not line.strip():
                break
            if line.lstrip().startswith(".."):
                break
            if self.index + 1 < len(self.lines) and _is_adornment(
                self.lines[self.index + 1]
            ):
                break
            collected.append(
                line[base_indent:] if line[:base_indent].isspace() else line.strip()
            )
            self.index += 1

        if not collected:
            self.index += 1
            return

        raw = "\n".join(collected)
        if is_table:
            # Kept verbatim and atomic. A table split across two chunks is two
            # fragments of unreadable punctuation in both of them.
            yield Block(kind="table", text=raw, line_no=line_no)
            return

        text, references = strip_roles(" ".join(c.strip() for c in collected))
        yield Block(
            kind="paragraph",
            text=text,
            line_no=line_no,
            references=references,
            anchors=tuple(self.pending_anchors),
        )
        self.pending_anchors = []


def parse_rst(text: str, document: str = "<string>") -> list[Block]:
    """Parse a Snooty-flavoured reStructuredText document into blocks."""
    return list(_Parser(text.splitlines(), document).parse())
