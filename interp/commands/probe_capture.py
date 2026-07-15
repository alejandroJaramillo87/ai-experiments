#!/usr/bin/env python3
"""Capture labeled activations for probe training (W2).

Workbench batch job (GPU, inference down). Exit codes: 0 success, 1 failure,
2 invalid usage.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from interp.output import Printer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-hash", default="unknown")
    parser.add_argument("--dataset", required=True, help='JSONL of {"text","label"}')
    parser.add_argument("--layers", required=True, help="comma-separated layer indices")
    parser.add_argument("--out", required=True, help="capture output directory")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    out = Printer()
    from interp.probes.capture import CaptureError, run_capture

    try:
        out.run(f"Capturing activations: {args.model}")
        manifest = run_capture(
            args.model,
            args.dataset,
            [int(layer) for layer in args.layers.split(",")],
            args.out,
            revision=args.revision,
            model_hash=args.model_hash,
            batch_size=args.batch_size,
        )
    except (CaptureError, OSError) as exc:
        out.error(f"Capture failed: {exc}")
        return 1
    out.ok(
        f"Capture complete: {manifest['count']} examples, "
        f"{manifest['shards']} shards, layers {manifest['layers']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
