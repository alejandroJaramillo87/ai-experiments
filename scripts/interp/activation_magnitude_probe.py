"""Measure residual-stream magnitude to decide whether fp16 can store a capture.

The activation cache cannot be stored at bf16: Arrow has no bfloat16, so
harvest.dtype resolves through datasets.Array2D and fails (KB-57, see
docs/interp/harvest-precision-and-resume.md). fp32 works and is lossless but
does not fit -- the Gemma-2-2B calibration capture is 2.76TB at fp32 against a
1.97TB volume. That leaves fp16 as the only two-byte option needing no code.

fp16 carries 5 exponent bits to bf16's 8 and saturates to inf above 65504.
Residual streams in this model family are known for large-magnitude outlier
features, so whether fp16 is safe here is an empirical question about this
model, not one the format definitions can answer. This script answers it.

The measurement is the max absolute value any hookable decoder layer emits over
a corpus sample, per layer. What matters is the headroom between that and
65504, and whether the trend across layers suggests later layers or longer runs
would exceed it. A margin under roughly 8x should be read as fp16 being unsafe:
this samples a corpus slice, not the 300M-token capture, and the maximum of a
larger sample is not smaller.

Host-only: needs the GPU and the model. One GPU consumer at a time (AGENTS.md).

Usage:
  python scripts/interp/activation_magnitude_probe.py \
      --model google/gemma-4-E2B [--sequences 64] [--context-len 1024]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run directly (scripts/interp/...) rather than as a module, so the repo root is
# not on sys.path. Same idiom as scripts/docs/check_doc_refs.py.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FP16_MAX = 65504.0

# Below this ratio of FP16_MAX to observed peak, fp16 is not safe to store at.
# Not a tight bound -- it is the margin that makes a corpus-sample maximum
# stand in for a 300M-token one.
SAFE_HEADROOM = 8.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-E2B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--corpus", default="monology/pile-uncopyrighted")
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--json-out", default=None, help="write the report here")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from interp.harvest import hook_name, resolve_model_meta

    d_model, spec, depth = resolve_model_meta(args.model, args.revision)
    print(f"[INFO] {args.model}: d_model {d_model}, {depth} layers, {spec.model_type}")
    if not depth:
        print("[ERROR] config states no layer count; cannot enumerate hooks")
        return 1

    # bfloat16 for the forward pass: this measures what the model produces, and
    # the model's own dtype is what the capture would widen or narrow from.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peak: dict[int, float] = dict.fromkeys(range(depth), 0.0)
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            # float() before abs().max(): bf16 can represent the magnitude but
            # comparing in fp32 keeps the reported number exact.
            observed = tensor.detach().float().abs().max().item()
            if observed > peak[layer]:
                peak[layer] = observed

        return hook

    try:
        for layer in range(depth):
            module = model.get_submodule(hook_name(layer, spec))
            handles.append(module.register_forward_hook(make_hook(layer)))

        stream = load_dataset(args.corpus, split="train", streaming=True)
        batch: list[str] = []
        seen = 0
        for record in stream:
            if seen >= args.sequences:
                break
            batch.append(record["text"])
            if len(batch) < args.batch_size:
                continue
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=args.context_len,
            ).to(model.device)
            with torch.no_grad():
                model(**inputs)
            seen += len(batch)
            batch = []
            print(f"[RUN] {seen}/{args.sequences} sequences", end="\r", flush=True)
    finally:
        for handle in handles:
            handle.remove()

    print(f"[OK] sampled {seen} sequences at context_len {args.context_len}\n")

    overall = max(peak.values())
    headroom = FP16_MAX / overall if overall else float("inf")

    print(f"{'layer':>6}  {'max|act|':>12}  {'fp16 headroom':>14}")
    for layer in sorted(peak):
        ratio = FP16_MAX / peak[layer] if peak[layer] else float("inf")
        print(f"{layer:>6}  {peak[layer]:>12.2f}  {ratio:>13.1f}x")

    print(f"\n[INFO] peak across all layers: {overall:.2f}")
    print(f"[INFO] fp16 ceiling: {FP16_MAX:.0f} ({headroom:.1f}x headroom)")

    safe = headroom >= SAFE_HEADROOM
    if safe:
        print(
            f"[OK] fp16 is safe for this subject: {headroom:.1f}x headroom clears "
            f"the {SAFE_HEADROOM:.0f}x bar. Move the reference captures to fp16."
        )
    else:
        print(
            f"[WARN] fp16 is NOT safe for this subject: {headroom:.1f}x headroom "
            f"is under the {SAFE_HEADROOM:.0f}x bar. Values would saturate to inf, "
            f"silently. Use the uint16 bit-cast or an own writer instead."
        )

    if args.json_out:
        report = {
            "model_id": args.model,
            "model_revision": args.revision,
            "d_model": d_model,
            "num_layers": depth,
            "sequences": seen,
            "context_len": args.context_len,
            "peak_abs_by_layer": peak,
            "peak_abs": overall,
            "fp16_max": FP16_MAX,
            "fp16_headroom": headroom,
            "fp16_safe": safe,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
        print(f"[OK] report written to {args.json_out}")

    return 0 if safe else 2


if __name__ == "__main__":
    sys.exit(main())
