"""Capture transformers' greedy continuations for L4, the ladder's top rung.

L4 asks the question no teacher-forced rung can: left to decode on its own,
does the engine stay on the reference's trajectory? L2 feeds the reference's
tokens back in at every position, so a small per-step error never compounds
there. Here it does -- greedy decoding is autoregressive, and one wrong token
sends the rest of the continuation somewhere else entirely.

Two properties this shares with the rung that reads it. `do_sample=False`
matches the harness's temperature-zero path, and ties resolve to the first
index on both sides: torch.argmax does, and runtime/src/sampler.rs was changed
to match precisely so a tie cannot make the two disagree about arithmetic they
agree on. The prompts are read in file order and a size takes a prefix, so a
deep artifact is a valid quick one.

Separate from reference_capture.py rather than folded into its --generate
flag. That flag is a diagnostic on one prompt and stays one; this writes a
different artifact with a different shape, and one script with two output
formats and two meanings for --out is a script nobody can read.

Workstation only: this loads the full checkpoint.

Usage:

    .venv/bin/python3 rust-harness/fixtures/reference_greedy.py \\
        --model /mnt/ai-data/models/hf/gemma-4-E4B \\
        --size quick \\
        --out rust-harness/fixtures/reference/e4b/quick/greedy.json
"""

import argparse
import json
import sys
from pathlib import Path

from reference_common import ensure_destination, provenance

# How many prompts each size decodes. Mirrors Size::prompts() in
# runtime/src/ladder.rs; the loader rejects an artifact holding fewer than the
# size it is read at, so a drift here is caught rather than silently trusted.
PROMPTS = {"quick": 2, "standard": 8, "deep": 32}

# How far each prompt runs. Mirrors ladder::GREEDY_TOKENS. Recorded in the
# artifact as well, so a stored reference states its own length.
TOKENS = 64

PROMPT_FILE = Path(__file__).resolve().parent / "reference" / "prompts.txt"


def read_prompts(path, want):
    """The first `want` prompts, skipping comments and blank lines."""
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) < want:
        print(
            f"[ERROR] Prompts: {path} holds {len(lines)} and this size needs {want}",
            file=sys.stderr,
        )
        print("  Solution: append prompts to the end of the file", file=sys.stderr)
        return None
    return lines[:want]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--out", required=True, help="destination JSON")
    p.add_argument(
        "--size",
        default="quick",
        choices=sorted(PROMPTS),
        help="how many prompts to decode (default quick)",
    )
    p.add_argument(
        "--tokens",
        type=int,
        default=TOKENS,
        help=f"continuation length (default {TOKENS}, matching ladder::GREEDY_TOKENS)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not ensure_destination(args.out):
        return 1

    prompts = read_prompts(PROMPT_FILE, PROMPTS[args.size])
    if prompts is None:
        return 1

    # Imported here rather than at module scope so --help works on a machine
    # without torch, which is every sandbox.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"[ERROR] Import: {e}", file=sys.stderr)
        print(
            "  Solution: run on the workstation venv with torch installed",
            file=sys.stderr,
        )
        return 1

    print(f"[RUN] Load tokenizer: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(
        "[RUN] Load model: float32 on cpu, this reads the whole file", file=sys.stderr
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="cpu",
        # Eager, so the comparison is against the reference formulation of
        # attention rather than against a fused kernel's rounding.
        attn_implementation="eager",
    )
    model.eval()

    # BOS on every prompt, for the reason reference_capture.py records it:
    # without one the model treats the sequence as already underway, and a
    # measured capture put its highest probability on repeating the prompt's
    # own last token.
    rows = []
    for i, text in enumerate(prompts):
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        with torch.no_grad():
            grown = model.generate(
                torch.tensor([ids]),
                max_new_tokens=args.tokens,
                do_sample=False,
            )
        continuation = grown[0, len(ids) :].tolist()
        rows.append(
            {
                "text": text,
                "ids": ids,
                "continuation_ids": continuation,
                "continuation": tokenizer.decode(continuation),
            }
        )
        print(
            f"[OK] Prompt {i}: {len(ids)} + {len(continuation)} tokens "
            f"{rows[-1]['continuation']!r}",
            file=sys.stderr,
        )

    payload = {
        "tokens": args.tokens,
        "size": args.size,
        "add_bos": True,
        "prompts": rows,
        **provenance(args.model),
    }
    Path(args.out).write_text(json.dumps(payload))
    print(
        f"[OK] Wrote {args.out}: {len(rows)} prompts x {args.tokens} tokens",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
