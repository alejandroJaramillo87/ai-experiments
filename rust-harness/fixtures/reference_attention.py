"""Capture transformers' post-softmax attention rows past the window, for L6.

Every rung below this one compares what attention produces -- residuals,
logits, tokens -- and none compares attention itself. L5 shows the output
distribution agrees past the sliding window, which is evidence that the mask
is right, at one remove: a wrong span or a wrong key-value head that happens
not to move the argmax at six positions would pass it. L6 reads the
distribution over keys directly, at the same six positions, for the layers
where the two attention regimes and the shared-KV tail engage.

The reference computes its rows the masking way: every key is scored, the
forbidden ones are pushed to the floor before the softmax, and they come out
exactly zero. runtime/src/model/cpu.rs never scores a forbidden key at all
and its capture pads the row with zeros. Comparing the two rows key for key
is what makes the zeros a checked fact about the mask rather than a fact
about either side's encoding: a key wrongly allowed carries weight where the
reference has zero, and a key wrongly forbidden has zero where the reference
has weight.

Which rows, mirrored in runtime/src/ladder.rs
--------------------------------------------

Positions are L5's schedule, `reference_long_context.window_schedule`, so
the anchors sit either side of the window and at both crossings. Layers are
`attention_layers` below, derived from the checkpoint's `layer_types` rather
than from an index list, so the same rule picks the interesting layers on the
real checkpoint and on the tiny fixture: the first two layers, the sliding
layer nearest the middle of the stack, and the last global layer, which on
the released E-series sits in the shared-KV tail. Every query head is kept,
because the query-to-key-value head mapping is one of the things the rung
exists to check. runtime/tests/ladder_reference_contract.rs pins the layer
rule to the Rust one.

The weights are taken from a forward hook on each chosen layer's attention
module rather than from `output_attentions=True`, which keeps every layer's
full [heads, n, n] tensor resident until the pass returns. Under eager
attention the module returns `(attn_output, attn_weights)` whether or not
the weights were asked for (docs/models/gemma4-upstream.txt, line 1279 of
the modeling file), so a hook sees them and only the chosen layers are kept.

Workstation only: this loads the full checkpoint and prefills thousands of
positions in float32.

Usage:

    # Tier E, against the real checkpoint.
    .venv/bin/python3 rust-harness/fixtures/reference_attention.py \\
        --model /mnt/ai-data/models/hf/gemma-4-E4B \\
        --out rust-harness/fixtures/reference/e4b/quick/attention.json

    # Tier T, against the committed tiny fixture, on the same synthetic ids
    # as its long-context artifact.
    .venv/bin/python3 rust-harness/fixtures/reference_attention.py \\
        --model rust-harness/fixtures/tiny-hf-gemma4 \\
        --source synthetic --length 150 \\
        --out rust-harness/fixtures/tiny-hf-gemma4/attention.json
"""

import argparse
import json
import sys
from pathlib import Path

from reference_capture import find_text_model
from reference_common import ensure_destination, provenance
from reference_long_context import (
    CORPUS,
    corpus_ids,
    min_tokens,
    synthetic_ids,
    window_schedule,
)

SLIDING = "sliding_attention"
GLOBAL = "full_attention"


def attention_layers(layer_types):
    """The layers L6 compares, from the schedule the checkpoint declares.

    Mirrors ladder::attention_layers. The first two layers, where a wrong
    input shows up before anything can average it away; the sliding layer
    nearest the middle of the stack, ties to the lower index; and the last
    global layer, which on the released E-series is inside the shared-KV
    tail and so reads a donor layer's keys. Sorted and without repeats, so
    a four-layer fixture whose middle and end coincide with its first two
    layers yields each once.
    """
    n = len(layer_types)
    chosen = set(range(min(2, n)))
    sliding = [i for i, t in enumerate(layer_types) if t == SLIDING]
    if sliding:
        middle = n // 2
        chosen.add(min(sliding, key=lambda i: (abs(i - middle), i)))
    global_layers = [i for i, t in enumerate(layer_types) if t == GLOBAL]
    if global_layers:
        chosen.add(global_layers[-1])
    return sorted(chosen)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="checkpoint directory")
    p.add_argument("--out", required=True, help="destination JSON")
    p.add_argument(
        "--source",
        default="corpus",
        choices=["corpus", "synthetic"],
        help="where the prompt ids come from (default corpus, which needs the "
        "checkpoint's tokenizer)",
    )
    p.add_argument(
        "--corpus",
        default=str(CORPUS),
        help="corpus text for --source corpus (default the committed one)",
    )
    p.add_argument(
        "--length",
        type=int,
        default=0,
        help="prompt length in tokens (default 4*sliding_window + 4, the "
        "shortest the schedule fits in)",
    )
    return p.parse_args()


def capture_rows(model, ids, layers, positions):
    """Attention rows at `positions` for every head of `layers`.

    Returns (n_heads, rows), rows as the artifact records them: one dict per
    (layer, head, position) holding the dense reference row over keys
    0..=position, masked entries included as the zeros transformers wrote.
    """
    import torch

    path, text_model = find_text_model(model)
    stack = text_model.layers
    print(f"[OK] Decoder layers: {len(stack)} at {path}.layers", file=sys.stderr)
    for layer in layers:
        if layer >= len(stack):
            raise SystemExit(
                f"[ERROR] Layers: layer {layer} chosen from layer_types, model has "
                f"{len(stack)}"
            )

    captured = {}

    def hook(index):
        def fn(_module, _inputs, output):
            weights = output[1] if isinstance(output, tuple) else None
            if weights is None:
                raise SystemExit(
                    f"[ERROR] Attention: layer {index} returned no weights; the "
                    "model must be loaded with attn_implementation='eager'"
                )
            # [batch, heads, query, key]; only the scheduled query rows are
            # kept, so the chosen layers cost heads x positions x n floats
            # rather than heads x n x n.
            captured[index] = weights[0, :, positions, :].detach().to(torch.float32)

        return fn

    handles = [
        stack[layer].self_attn.register_forward_hook(hook(layer)) for layer in layers
    ]
    print(f"[RUN] Forward pass: {len(ids)} positions", file=sys.stderr)
    with torch.no_grad():
        model(torch.tensor([ids]))
    for h in handles:
        h.remove()

    missing = [layer for layer in layers if layer not in captured]
    if missing:
        raise SystemExit(f"[ERROR] Attention: layers {missing} produced no weights")

    n_heads = None
    rows = []
    for layer in layers:
        w = captured[layer]
        if n_heads is None:
            n_heads = w.shape[0]
        elif w.shape[0] != n_heads:
            raise SystemExit(
                f"[ERROR] Attention: layer {layer} has {w.shape[0]} heads, layer "
                f"{layers[0]} has {n_heads}"
            )
        for head in range(n_heads):
            for i, position in enumerate(positions):
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "position": position,
                        "weights": w[head, i, : position + 1].tolist(),
                    }
                )
    return n_heads, rows


def main():
    args = parse_args()
    if not ensure_destination(args.out):
        return 1

    # Imported here rather than at module scope so --help works on a machine
    # without torch, which is every sandbox. runtime/tests/ladder_reference_
    # contract.rs imports this module for attention_layers, and does so in the
    # sandbox, so nothing above this line may need torch either.
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

    config = json.loads((Path(args.model) / "config.json").read_text())
    text_config = config.get("text_config", config)
    window = text_config.get("sliding_window")
    layer_types = text_config.get("layer_types")
    if not window or not layer_types:
        print(
            f"[ERROR] Config: {args.model} declares no sliding_window or no "
            "layer_types, so L6 has no schedule to read",
            file=sys.stderr,
        )
        return 1
    length = args.length or min_tokens(window)
    try:
        positions = window_schedule(window, length)
    except ValueError as e:
        print(f"[ERROR] Schedule: {e}", file=sys.stderr)
        print(
            f"  Solution: pass --length {min_tokens(window)} or more", file=sys.stderr
        )
        return 1
    layers = attention_layers(layer_types)
    kinds = [layer_types[layer] for layer in layers]
    print(
        f"[OK] Window: {window} tokens, scoring {positions} of {length}",
        file=sys.stderr,
    )
    print(f"[OK] Layers: {layers} ({', '.join(kinds)})", file=sys.stderr)

    if args.source == "synthetic":
        ids = synthetic_ids(
            length, text_config["vocab_size"], text_config.get("bos_token_id", 2)
        )
    else:
        print(f"[RUN] Load tokenizer: {args.model}", file=sys.stderr)
        ids = corpus_ids(AutoTokenizer.from_pretrained(args.model), args.corpus, length)
    print(f"[OK] Prompt: {len(ids)} ids from {args.source}", file=sys.stderr)

    print(
        "[RUN] Load model: float32 on cpu, this reads the whole file",
        file=sys.stderr,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="cpu",
        # Eager, and not only for the reference formulation of attention: it
        # is the one implementation that materializes the weights at all. A
        # fused kernel never holds the post-softmax row, so there would be
        # nothing for the hook to read.
        attn_implementation="eager",
    )
    model.eval()

    n_heads, rows = capture_rows(model, ids, layers, positions)

    payload = {
        "sliding_window": window,
        "length": length,
        "layers": layers,
        "layer_types": kinds,
        "heads": n_heads,
        "positions": positions,
        "source": args.source,
        "add_bos": True,
        "dtype": "float32",
        "token_ids": list(ids),
        "rows": rows,
        **provenance(args.model),
    }
    Path(args.out).write_text(json.dumps(payload))
    print(
        f"[OK] Wrote {args.out}: {len(rows)} rows across {len(layers)} layers x "
        f"{n_heads} heads at {len(positions)} positions; window {window}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
