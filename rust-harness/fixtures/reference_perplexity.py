"""Capture the reference implementation's per-token NLL for L3.

Perplexity is the regression scalar for engine changes: a wrong RoPE base, a
transposed weight, or a bad quant kernel moves it sharply while spot checks
stay plausible. L3 compares ours against transformers' on the same token ids
from the same checkpoint, so the number's absolute value carries no claim --
what is under test is the difference.

The windowing is runtime/src/eval.rs's, reproduced exactly
-----------------------------------------------------------

Perplexity computed in different windows is a different quantity, so an
artifact that does not state its windowing cannot be compared to anything. The
convention here is llama.cpp's, and matching it is not optional:

    chunks of CHUNK tokens, the trailing partial window dropped
    each window prefilled from an empty cache
    warmup = CHUNK // 2, and positions warmup..CHUNK-1 score the next token
    the scored token's corpus index is w * CHUNK + p + 1

The scored positions are written out explicitly rather than implied, and the
rung errors -- rather than fails -- when the two sides disagree about which
tokens were scored. That distinction matters: a windowing mismatch is a broken
artifact, and reporting it as a red rung would send a reader looking for a
numerics defect that is not there.

No BOS. The corpus is scored as a continuous text in fixed windows, and each
window after the first begins mid-document by construction; prepending a
sequence-start marker to window zero alone would make one window a different
measurement from the rest.

Workstation only: this loads the full checkpoint and runs a forward pass per
window.

Usage:

    .venv/bin/python3 rust-harness/fixtures/reference_perplexity.py \\
        --model /mnt/ai-data/models/hf/gemma-4-E4B \\
        --size quick \\
        --out rust-harness/fixtures/reference/e4b/quick/ppl.json
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from reference_common import ensure_destination, provenance

# Mirrors ladder::CHUNK and Size::tokens() in runtime/src/ladder.rs;
# runtime/tests/ladder_reference_contract.rs pins them together.
CHUNK = 256
TOKENS = {"quick": 512, "standard": 2048, "deep": 8192}

CORPUS = Path(__file__).resolve().parent / "reference" / "corpus.txt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--out", required=True, help="destination JSON")
    p.add_argument(
        "--size",
        default="quick",
        choices=sorted(TOKENS),
        help="how many corpus tokens to score (default quick)",
    )
    p.add_argument(
        "--corpus",
        default=str(CORPUS),
        help="corpus text (default the committed one)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not ensure_destination(args.out):
        return 1

    corpus_path = Path(args.corpus)
    if not corpus_path.is_file():
        print(f"[ERROR] Corpus: {corpus_path} does not exist", file=sys.stderr)
        print("  Solution: fetch and commit it once", file=sys.stderr)
        print(
            "  Command: python3 rust-harness/fixtures/fetch_corpus.py", file=sys.stderr
        )
        return 1
    corpus_bytes = corpus_path.read_bytes()

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

    want = TOKENS[args.size]

    print(f"[RUN] Load tokenizer: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_ids = tokenizer(corpus_bytes.decode("utf-8"), add_special_tokens=False)[
        "input_ids"
    ]
    if len(all_ids) < want:
        print(
            f"[ERROR] Corpus: tokenizes to {len(all_ids)} tokens and {args.size} needs {want}",
            file=sys.stderr,
        )
        print("  Solution: raise CHARS in fetch_corpus.py and refetch", file=sys.stderr)
        return 1
    # A prefix, so the sizes are nested: a deep run reproduces a quick run's
    # first scored tokens exactly rather than being a different measurement.
    ids = all_ids[:want]
    print(
        f"[OK] Tokenized: {len(ids)} of {len(all_ids)} corpus tokens", file=sys.stderr
    )

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

    warmup = CHUNK // 2
    windows = len(ids) // CHUNK
    scored = []
    for w in range(windows):
        window = ids[w * CHUNK : (w + 1) * CHUNK]
        with torch.no_grad():
            logits = model(torch.tensor([window])).logits[0]
        for p in range(warmup, CHUNK - 1):
            target = window[p + 1]
            # logsumexp minus the target logit, in float64, which is eval.rs's
            # `nll` written in torch. Row by row rather than over the whole
            # window: a 262144-wide float64 copy of 256 positions is half a
            # gigabyte, and only half the rows are scored.
            row = logits[p].to(torch.float64)
            nll = (torch.logsumexp(row, dim=-1) - row[target]).item()
            scored.append([w * CHUNK + p + 1, nll])
        print(
            f"[OK] Window {w + 1}/{windows}: {len(scored)} tokens scored",
            file=sys.stderr,
        )

    mean_nll = sum(n for _, n in scored) / len(scored)
    payload = {
        "chunk": CHUNK,
        "size": args.size,
        "corpus_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "token_ids": ids,
        "scored": scored,
        **provenance(args.model),
    }
    Path(args.out).write_text(json.dumps(payload))
    print(
        f"[OK] Wrote {args.out}: {len(scored)} scored tokens, ppl {math.exp(mean_nll):.4f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
