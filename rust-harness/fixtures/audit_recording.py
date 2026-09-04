"""Audit a flight recording against transformers, token by token.

The ladder compares the CPU backend with transformers on prompts the ladder
chose: one distribution at the end of a short prompt (L2) and greedy
continuations (L4). Neither says anything about a reply the harness actually
sampled -- a run at temperature 0.8 draws tokens no greedy reference visits,
and a reader looking at an odd token in such a reply has had no way to ask
whether the model behind transformers would have put weight on it too.

A recording (`harness --record`) holds every forward call's logits and every
token the sampler chose. That is enough to ask the question exactly: feed the
recorded token sequence to transformers once, teacher-forced, and compare its
logits at every recorded position with the row the harness recorded there.
The distributions must agree to the ladder's L2 tolerance at every position,
not only the first -- which is the strongest statement this workspace can
make about a specific reply. For each token the sampler chose, the report
also gives its rank and probability under transformers at that position, so
"was that token a plausible draw" has a number rather than an opinion.

Two verdicts, deliberately separate. The distributions agreeing is asserted
and decides the exit code. The chosen tokens' ranks are reported and not
asserted: a sampler at temperature 0.8 legitimately picks rank 30, and a
threshold on rank would be a threshold on luck. What the rank column settles
is whether the token the harness drew is one the reference distribution
offered -- a decoder fault produces tokens the reference gives no weight to,
and that shows as a rank in the thousands, not a rank of 7.

Workstation only, like the generators: the checkpoint and torch. The forward
pass is one teacher-forced call over the whole sequence, so a 134-token
recording costs seconds, and the recording itself is the expensive input --
a 262144-wide row per decoded token, about 2 MB of JSON each.

Usage:

    uv run python rust-harness/fixtures/audit_recording.py \\
        --model /mnt/ai-data/models/hf/gemma-4-E4B \\
        --record /tmp/rh2/flight.jsonl \\
        --out /tmp/rh2/audit.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reference_common import ensure_destination, provenance  # noqa: E402

# The ladder's Tier E L2 tolerances (runtime/src/ladder.rs TIER_E), restated
# so this report and a ladder table are read against the same numbers.
MEAN_KL_MAX = 1e-3
WORST_KL_MAX = 5e-3
TOP_K = 50
OVERLAP_MIN = 0.98
ARGMAX_MIN = 0.98


def parse_recording(lines):
    """Read a recording's lines into (header, rows, selections, sequence).

    `rows` maps an absolute position to the logits the harness produced
    there, which predict the token at the next position. `selections` maps
    the position a sampled token occupies to that token. `sequence` is the
    token id sequence the run consumed, reconstructed from the forward
    records' `start` and `tokens`.

    A recording holds one branch per position (record.rs, `replay`): two
    records writing different tokens at the same position are refused here
    the same way, because a teacher-forced pass has one sequence to force.
    """
    header = None
    rows = {}
    selections = {}
    sequence = []
    for lineno, raw in enumerate(lines, start=1):
        raw = raw.strip()
        if not raw:
            continue
        line = json.loads(raw)
        if header is None:
            if "vocab_size" not in line:
                raise ValueError(f"line {lineno}: expected the recording header first")
            header = line
            continue
        if "selected" in line:
            sel = line["selected"]
            selections[sel["position"]] = sel["token"]
            continue
        start, tokens = line["start"], line["tokens"]
        if start > len(sequence):
            raise ValueError(
                f"line {lineno}: forward at {start} leaves a gap after {len(sequence)} tokens"
            )
        for offset, tok in enumerate(tokens):
            pos = start + offset
            if pos < len(sequence):
                if sequence[pos] != tok:
                    raise ValueError(
                        f"line {lineno}: position {pos} holds {sequence[pos]} and this "
                        f"forward writes {tok}; a recording with two branches cannot be "
                        "teacher-forced"
                    )
            else:
                sequence.append(tok)
        for row in line["rows"]:
            rows[row["position"]] = np.asarray(row["logits"], dtype=np.float32)
    if header is None:
        raise ValueError("empty recording")
    return header, rows, selections, sequence


def log_softmax(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x - m - math.log(np.exp(x - m).sum())


def compare_row(reference, ours, top_k=TOP_K):
    """L2's measures for one position: KL(reference || ours) in nats, whether
    the argmaxes agree, and the top-k overlap as a fraction."""
    lr, lo = log_softmax(reference), log_softmax(ours)
    kl = float(np.sum(np.exp(lr) * (lr - lo)))
    argmax = bool(int(np.argmax(reference)) == int(np.argmax(ours)))
    top_ref = set(np.argpartition(-np.asarray(reference), top_k)[:top_k].tolist())
    top_ours = set(np.argpartition(-np.asarray(ours), top_k)[:top_k].tolist())
    overlap = len(top_ref & top_ours) / top_k
    return kl, argmax, overlap


def rank_of(logits, token):
    """1-based rank of `token` under `logits`, and its probability."""
    logits = np.asarray(logits, dtype=np.float64)
    rank = int(np.sum(logits > logits[token])) + 1
    prob = float(np.exp(log_softmax(logits)[token]))
    return rank, prob


def verdict(per_position):
    """The L2 verdict over every compared position, as (ok, measures)."""
    n = len(per_position)
    if n == 0:
        return False, {}
    kls = [p["kl"] for p in per_position]
    measures = {
        "positions": n,
        "mean_kl": sum(kls) / n,
        "worst_kl": max(kls),
        "worst_kl_position": per_position[int(np.argmax(kls))]["position"],
        "argmax_agreement": sum(p["argmax"] for p in per_position) / n,
        "worst_top50_overlap": min(p["overlap"] for p in per_position),
    }
    ok = (
        measures["mean_kl"] <= MEAN_KL_MAX
        and measures["worst_kl"] <= WORST_KL_MAX
        and measures["argmax_agreement"] >= ARGMAX_MIN
        and measures["worst_top50_overlap"] >= OVERLAP_MIN
    )
    return ok, measures


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--record", required=True, help="a harness --record file")
    p.add_argument("--out", help="destination JSON for the per-position report")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="reference dtype (default float32, the ladder's)",
    )
    p.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="also print the N chosen tokens of lowest reference rank, decoded",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.out and not ensure_destination(args.out):
        return 1
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

    print(f"[RUN] Read recording: {args.record}", file=sys.stderr)
    with open(args.record, encoding="utf-8") as f:
        header, rows, selections, sequence = parse_recording(f)
    sampling = header.get("sampling")
    print(
        f"[OK] Recording: {len(sequence)} tokens, {len(rows)} logit rows, "
        f"{len(selections)} selections, sampling {sampling}",
        file=sys.stderr,
    )
    for pos, tok in selections.items():
        if pos >= len(sequence) or sequence[pos] != tok:
            print(
                f"[ERROR] Selection at {pos} chose {tok} but the sequence holds "
                f"{sequence[pos] if pos < len(sequence) else 'nothing'}",
                file=sys.stderr,
            )
            return 1

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"[RUN] Load tokenizer: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(
        f"[RUN] Load model: {args.dtype} on cpu, this reads the whole file",
        file=sys.stderr,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cpu", attn_implementation="eager"
    )
    model.eval()
    print(
        f"[RUN] Forward pass: {len(sequence)} positions, teacher-forced",
        file=sys.stderr,
    )
    with torch.no_grad():
        logits = (
            model(input_ids=torch.tensor([sequence]))
            .logits[0]
            .to(torch.float32)
            .numpy()
        )

    per_position = []
    for pos in sorted(rows):
        kl, argmax, overlap = compare_row(logits[pos], rows[pos])
        per_position.append(
            {"position": pos, "kl": kl, "argmax": argmax, "overlap": overlap}
        )
    ok, measures = verdict(per_position)

    chosen = []
    for pos in sorted(selections):
        tok = selections[pos]
        rank, prob = rank_of(logits[pos - 1], tok)
        chosen.append(
            {
                "position": pos,
                "token": tok,
                "text": tokenizer.decode([tok]),
                "reference_rank": rank,
                "reference_prob": prob,
            }
        )

    status = "OK" if ok else "ERROR"
    print(
        f"[{status}] Distributions: {measures['positions']} positions, "
        f"mean kl {measures['mean_kl']:.6f} (<= {MEAN_KL_MAX}), "
        f"worst kl {measures['worst_kl']:.6f} (<= {WORST_KL_MAX}) at position "
        f"{measures['worst_kl_position']}, argmax agreement "
        f"{measures['argmax_agreement']:.4f} (>= {ARGMAX_MIN}), worst top-{TOP_K} overlap "
        f"{measures['worst_top50_overlap']:.4f} (>= {OVERLAP_MIN})",
        file=sys.stderr,
    )
    if chosen:
        ranks = [c["reference_rank"] for c in chosen]
        within = sum(r <= TOP_K for r in ranks)
        print(
            f"[OK] Chosen tokens: {len(chosen)}, {within} within the reference's top-{TOP_K}, "
            f"worst rank {max(ranks)} at position "
            f"{chosen[int(np.argmax(ranks))]['position']}, last token "
            f"{chosen[-1]['text']!r} rank {chosen[-1]['reference_rank']} "
            f"p={chosen[-1]['reference_prob']:.4f}",
            file=sys.stderr,
        )
        for c in sorted(chosen, key=lambda c: -c["reference_rank"])[: args.show]:
            print(
                f"  position {c['position']:4d} rank {c['reference_rank']:6d} "
                f"p={c['reference_prob']:.4f} {c['text']!r}",
                file=sys.stderr,
            )

    if args.out:
        payload = {
            "record": str(args.record),
            "dtype": args.dtype,
            "sampling": sampling,
            "sequence_len": len(sequence),
            "distributions": measures,
            "distributions_ok": ok,
            "per_position": per_position,
            "chosen": chosen,
        }
        payload.update(provenance(args.model))
        Path(args.out).write_text(json.dumps(payload, indent=1))
        print(f"[OK] Wrote {args.out}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
