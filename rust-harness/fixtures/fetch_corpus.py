"""Fetch and normalize the public-domain corpus L3 scores.

Run once; the result is committed. This exists so the corpus is reproducible
rather than a blob somebody pasted in: it names its source, strips the
packaging deterministically, and prints the sha256 the artifacts will record.

Why a committed text and not WikiText-2
---------------------------------------

L3 compares our per-token NLL against the reference implementation's on the
same token ids from the same checkpoint. It is a differential test, so the
corpus is arbitrary -- what matters is that both sides score identical tokens.
WikiText-2's one advantage is comparability with published perplexity figures,
which nothing here uses, and it costs a download outside the egress allowlist,
preprocessing conventions that must match a published pipeline exactly to mean
anything, and a run no fresh clone can reproduce.

Why this text
-------------

Ordinary narrative English. That is a numerical argument rather than a
literary one: off-distribution input measurably widens the gap between two
implementations of one model, and the same prompt captured without a BOS gave
rel_L2 3.3e-5 where the in-distribution version gave 5.0e-6. Technical
documentation, markup, or archaic verse would all push the tolerance the wrong
way for no benefit.

The work is public domain. This strips Project Gutenberg's own packaging --
their trademark license covers their file, not the text -- so what is committed
is the work itself.

Needs network access, which a contained agent sandbox does not have. Run it on
the workstation.

Usage:

    .venv/bin/python3 rust-harness/fixtures/fetch_corpus.py
"""

import hashlib
import re
import sys
import urllib.request
from pathlib import Path

# The Adventures of Sherlock Holmes, Arthur Conan Doyle, 1892. Plain narrative
# prose with dialogue, which is the register the tolerance argument above
# wants. Pinned by ebook number so a rerun fetches the same work.
SOURCE = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
TITLE = "The Adventures of Sherlock Holmes, Arthur Conan Doyle (1892)"

# Enough for deep's 8192 tokens with margin. English runs roughly four
# characters to a token, so 60000 characters is about 15000 tokens.
CHARS = 60000

START = re.compile(r"\*\*\* *START OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
END = re.compile(r"\*\*\* *END OF TH[EIS]+ PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)

OUT = Path(__file__).resolve().parent / "reference" / "corpus.txt"


def body(text):
    """The work itself, with the packaging removed."""
    start = START.search(text)
    end = END.search(text)
    if not start or not end:
        print(
            "[ERROR] Boilerplate: could not find both Gutenberg markers",
            file=sys.stderr,
        )
        print(
            "  Solution: the file layout changed; update START and END", file=sys.stderr
        )
        return None
    return text[start.end() : end.start()]


def normalize(text):
    """CRLF to LF, collapse runs of blank lines, strip trailing spaces.

    Deterministic so that a rerun produces identical bytes: the corpus hash is
    recorded in every L3 artifact, and a whitespace difference would invalidate
    stored references for no reason.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clip(text, chars):
    """The first `chars` characters, ending at a paragraph boundary.

    Cut on a blank line rather than mid-sentence: the last window of the corpus
    is scored like every other, and a truncated clause is off-distribution text
    at exactly the place a reader would not think to look for it.
    """
    if len(text) <= chars:
        return text
    cut = text.rfind("\n\n", 0, chars)
    return text[: cut if cut > 0 else chars].strip() + "\n"


def main():
    print(f"[RUN] Fetch: {SOURCE}", file=sys.stderr)
    try:
        with urllib.request.urlopen(SOURCE, timeout=60) as r:
            raw = r.read().decode("utf-8")
    except OSError as e:
        print(f"[ERROR] Fetch: {e}", file=sys.stderr)
        print(
            "  Solution: run on the workstation; the sandbox has no egress here",
            file=sys.stderr,
        )
        return 1

    stripped = body(raw)
    if stripped is None:
        return 1
    corpus = clip(normalize(stripped), CHARS)

    OUT.write_text(corpus)
    digest = hashlib.sha256(corpus.encode("utf-8")).hexdigest()
    print(f"[OK] Wrote {OUT}: {len(corpus)} chars", file=sys.stderr)
    print(f"[OK] Source: {TITLE}", file=sys.stderr)
    print(f"[OK] sha256: {digest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
