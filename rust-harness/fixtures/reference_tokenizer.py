"""Capture the reference tokenizer's ids for L0, the ladder's first rung.

L0 is the rung every rung above it depends on. L1 and L2 compare two forward
passes on the assumption that both sides were fed the same tokens; if the
token ids differ, their agreement is a coincidence rather than evidence. Yet
the only tokenizer test this workspace had was a golden the same author wrote
from the same understanding as the tokenizer -- the exact shape of gate that
stayed green for nine work packages while the forward pass produced noise.

The rendered-turn case is the one that matters most and the one that has
already been wrong: the harness carried Gemma 3's <start_of_turn> literals
against a Gemma 4 vocabulary that spells them <|turn>, so every turn boundary
segmented as a byte run and output stayed fluent throughout. That defect is
invisible to any test that does not encode a real rendered prompt.

Two conventions, mapped onto the reference deliberately
-------------------------------------------------------

`Tokenizer::encode_prompt` takes a per-call allowlist of special literals: the
harness names the delimiters it emits, and every other special-looking run in
the text stays data. The reference has no such control -- HF exposes one
global `split_special_tokens` switch -- so only the two extremes can be posed
as a question it can answer:

    allowed == []      ->  split_special_tokens=True   (nothing resolves)
    allowed == all in text ->  split_special_tokens=False  (all of them resolve)

A partial allowlist has no reference formulation, so no case uses one, and the
generator refuses to write a case whose non-empty allowlist omits a special its
text carries, rather than emitting one whose two sides mean different things.
An empty allowlist is not partial: it is the first extreme above, and the case
that carries specials it does not allow is exactly the one worth measuring.

`add_special_tokens=False` throughout: the harness owns its own BOS through
the chat template, and letting the reference inject one would compare two
different questions.

Workstation only -- it needs transformers and the real tokenizer.

Usage:

    .venv/bin/python3 rust-harness/fixtures/reference_tokenizer.py \\
        --model /mnt/ai-data/models/hf/gemma-4-E4B \\
        --out rust-harness/fixtures/reference/e4b/quick/tokenizer_parity.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from reference_common import ensure_destination, provenance

# The literals the harness itself emits into a prompt, and the only ones it
# permits to resolve. Mirrors harness/src/generator.rs PROMPT_SPECIALS, which
# builds the same set from template.rs; the chat case is worth nothing if it
# is encoded under a more permissive allowlist than production uses.
PROMPT_SPECIALS = ["<bos>", "<|turn>", "<turn|>"]

# A conversation with the shapes that break tokenizers: a system turn, an
# apostrophe, a digit run, a non-ASCII character, and an interior newline.
# Short on purpose -- L0 compares exact id sequences, so a divergence is read
# by eye, and 400 ids of prose would bury the one that moved.
CONVERSATION = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What's 2+2 in Montréal?\nShow work."},
    {"role": "model", "content": "4"},
]


def chat_text():
    """The conversation rendered by the training tier's own renderer.

    Imported rather than transcribed, for the reason the whole ladder exists:
    a Python copy of the template written here would agree with a Rust copy
    written from the same reading. harness/tests/template_contract.rs already
    pins this renderer against the Rust one, so a case built from it measures
    our tokenizer against a string the harness genuinely produces.
    """
    from training.data.sft import render_segments
    from training.models.families import family_spec

    spec = family_spec("gemma4").chat_template
    return "".join(t for t, _ in render_segments(CONVERSATION, spec))


def cases(chat):
    """Every parity case, as (name, text, allowed)."""
    return [
        ("plain", "The capital of France is", []),
        # Unicode, digits, and punctuation in one line: byte-fallback and
        # digit splitting are per-vocabulary decisions no reader can predict.
        ("mixed", 'Montréal 2026 -- café, 3.14; "quoted".', []),
        # The safety property, as a measurement. The harness encodes untrusted
        # text with an empty allowlist precisely so a pasted turn marker
        # cannot open a real turn, and this case is what proves both sides
        # agree it stays bytes.
        ("specials_as_text", "<bos>hello<|turn>user", []),
        ("bos_resolved", "<bos>hello", ["<bos>"]),
        ("chat_turn", chat, PROMPT_SPECIALS),
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--out", required=True, help="destination JSON")
    return p.parse_args()


def main():
    args = parse_args()
    if not ensure_destination(args.out):
        return 1

    # The parent tree carries the chat template. Added here rather than
    # required in the environment so the command in the README is the whole
    # command.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # Imported here rather than at module scope so --help works without them.
    try:
        from transformers import AutoTokenizer

        from training.models.families import CHAT_TEMPLATE_VERSION
    except ImportError as e:
        print(f"[ERROR] Import: {e}", file=sys.stderr)
        print(
            "  Solution: run on the workstation venv with transformers installed",
            file=sys.stderr,
        )
        return 1

    print(f"[RUN] Load tokenizer: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    added = set(tokenizer.get_added_vocab())

    chat = chat_text()
    print(
        f"[OK] Rendered: chat turn v{CHAT_TEMPLATE_VERSION}, {len(chat)} chars",
        file=sys.stderr,
    )

    encoded = []
    for name, text, allowed in cases(chat):
        # A partial allowlist cannot be posed to the reference, whose switch is
        # global. Refused rather than written, because the artifact would look
        # ordinary and the two sides would be answering different questions.
        #
        # Only the middle is refused. An empty allowlist is the other extreme,
        # answered exactly by split_special_tokens=True, and a case whose text
        # carries specials it does not allow is the safety property this rung
        # exists to measure -- not a mistake to reject.
        stray = (
            sorted(s for s in added if s in text and s not in allowed)
            if allowed
            else []
        )
        if stray:
            print(
                f"[ERROR] Case {name}: carries specials outside its allowlist: {stray}",
                file=sys.stderr,
            )
            print(
                "  Solution: allow them, or remove them from the case text",
                file=sys.stderr,
            )
            return 1
        ids = tokenizer(
            text,
            add_special_tokens=False,
            # The whole mapping, in one argument: an empty allowlist means
            # nothing in the text resolves to a special id.
            split_special_tokens=not allowed,
        )["input_ids"]
        encoded.append({"name": name, "text": text, "allowed": allowed, "ids": ids})
        print(f"[OK] Case {name}: {len(ids)} tokens", file=sys.stderr)

    payload = {
        "chat_template_version": CHAT_TEMPLATE_VERSION,
        "cases": encoded,
        **provenance(args.model),
    }
    # The tokenizer is what this artifact measures, so it is identified
    # directly rather than through the config hash the other artifacts use.
    tokenizer_json = Path(args.model) / "tokenizer.json"
    if tokenizer_json.is_file():
        payload["tokenizer_sha256"] = hashlib.sha256(
            tokenizer_json.read_bytes()
        ).hexdigest()

    Path(args.out).write_text(json.dumps(payload))
    print(f"[OK] Wrote {args.out}: {len(encoded)} cases", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
