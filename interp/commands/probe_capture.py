"""Capture labeled activations for probe training (W2).

Workbench batch job (GPU, inference down). Exit codes: 0 success, 1 failure,
2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.output import Printer


def configure(sub) -> None:
    p = sub.add_parser(
        "probe-capture",
        help="capture labeled activations for probe training (W2)",
        description=__doc__,
    )
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--model-hash", default="unknown")
    p.add_argument("--dataset", required=True, help='JSONL of {"text","label"}')
    p.add_argument("--layers", required=True, help="comma-separated layer indices")
    p.add_argument("--out", required=True, help="capture output directory")
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
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
