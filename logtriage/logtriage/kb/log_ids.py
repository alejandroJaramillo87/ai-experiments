"""Log identifiers the manual actually documents.

Every mongod line carries a numeric id naming the call site that wrote it, which
would make an id the ideal thing for a detector to match on -- if there were a
published catalogue of them. There is not. The manual documents ids only as they
appear inside worked examples on reference/log-messages.txt: fourteen of them,
against a server that emits thousands.

So this extracts what the corpus does say, for one narrow purpose: proving that
the handful of ids this tool relies on are the ids MongoDB documents, and
catching it if that ever stops being true. It is not a lookup table for
diagnostics, and using it as one would key detection on a sample of examples.

One thing the extraction shows is worth carrying forward. The examples for the
connection lifecycle (22943, 22944) still carry the 4.4-era templated messages,
`connection accepted from {session_remote} ...`, which 5.0 replaced with the
plain "Connection accepted" plus attributes. The published examples are older
than the format they illustrate, which is the second reason detectors here match
on component and message rather than on anything read out of the manual.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from logtriage.kb.parse import parse_rst

# The page that carries the worked examples. The only page that does.
LOG_MESSAGES_PAGE = "reference/log-messages.txt"

# The manual's own stand-in id, used in examples that illustrate the envelope
# rather than a real call site. Excluded so it cannot be mistaken for one.
PLACEHOLDER_ID = 12345


@dataclass(frozen=True, slots=True)
class DocumentedId:
    """A log identifier as one of the manual's examples shows it."""

    log_id: int
    component: str
    severity: str
    msg: str


def _objects(text: str) -> list[dict]:
    """Every JSON log object in a page, from either example style.

    The page shows both a single line per record and a pretty-printed form of
    the same record, so a code block is tried whole and then line by line.
    """
    found: list[dict] = []
    for block in parse_rst(text, LOG_MESSAGES_PAGE):
        if block.kind != "code":
            continue
        candidates = [block.text, *block.text.splitlines()]
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate.startswith("{"):
                continue
            try:
                document = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(document, dict) and "id" in document and "msg" in document:
                found.append(document)
    return found


def documented_ids(page_text: str) -> dict[int, DocumentedId]:
    """Map every documented log id to how the manual shows it."""
    table: dict[int, DocumentedId] = {}
    for document in _objects(page_text):
        log_id = document.get("id")
        if not isinstance(log_id, int) or log_id == PLACEHOLDER_ID:
            continue
        table[log_id] = DocumentedId(
            log_id=log_id,
            component=str(document.get("c", "")),
            severity=str(document.get("s", "")),
            msg=str(document.get("msg", "")),
        )
    return table


def documented_ids_from_corpus(source_root: Path) -> dict[int, DocumentedId]:
    """Read the log message reference out of a fetched corpus."""
    page = source_root / LOG_MESSAGES_PAGE
    return documented_ids(page.read_text(encoding="utf-8", errors="replace"))
