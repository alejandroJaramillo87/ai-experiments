"""Snooty substitution constants, expanded before anything else reads the text.

The manual is written against a table of constants declared in snooty.toml and
referenced as `{+version+}`, `{+atlas+}`, `{+mongosh+}`. Left alone they reach a
chunk verbatim, and a passage reading "install MongoDB {+version+}" is worse
than one reading "install MongoDB 8.3": it is visibly broken to a reader and it
denies the retriever the version number, which is often the whole question.

Expansion happens before role stripping, not after, because some constants
expand *into* roles -- snooty.toml defines

    mongosh = ":binary:`~bin.mongosh`"

so substituting late would leave role syntax behind in the output of the pass
whose job was to remove it.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

# `{+name+}` -- the Snooty substitution form. Hyphens are common in the names
# (`latest-lts-version`), digits less so, but both appear.
SUBSTITUTION_RE = re.compile(r"\{\+([a-zA-Z0-9._-]+)\+\}")

# Constants may refer to other constants. Two rounds covers everything the
# manual actually uses; the bound exists so a circular definition upstream
# cannot hang the indexer.
MAX_PASSES = 4


def load_constants(snooty_toml: Path) -> dict[str, str]:
    """Read the `[constants]` table from a snooty.toml.

    Non-string values are dropped rather than coerced. A constant that is a
    number in the toml is still spliced into prose as text, but anything else
    (a list, a table) has no sensible textual form and guessing one would put
    invented text in the corpus.
    """
    with open(snooty_toml, "rb") as handle:
        document = tomllib.load(handle)
    table = document.get("constants", {})
    return {
        key: value if isinstance(value, str) else str(value)
        for key, value in table.items()
        if isinstance(value, str | int | float)
    }


def expand(text: str, constants: dict[str, str]) -> tuple[str, tuple[str, ...]]:
    """Substitute `{+name+}` references, returning the text and what was missed.

    An unknown constant is left exactly as written rather than blanked. A blank
    is unrecoverable -- nobody reading the chunk later can tell that a word was
    removed -- whereas `{+some-name+}` in the output is a legible defect that
    the index statistics count.
    """
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        value = constants.get(name)
        if value is None:
            missing.append(name)
            return match.group(0)
        return value

    for _ in range(MAX_PASSES):
        expanded = SUBSTITUTION_RE.sub(replace, text)
        if expanded == text:
            break
        text = expanded

    return text, tuple(dict.fromkeys(missing))
