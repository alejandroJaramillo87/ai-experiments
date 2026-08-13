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
    p.add_argument(
        "--model-hash",
        default=None,
        help="identity of the weights; derived from --model when it names a "
        "local checkpoint directory",
    )
    p.add_argument("--dataset", required=True, help='JSONL of {"text","label"}')
    p.add_argument("--layers", required=True, help="comma-separated layer indices")
    p.add_argument("--out", required=True, help="capture output directory")
    p.add_argument("--batch-size", type=int, default=16)
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from interp.probes.capture import (
        UNKNOWN_MODEL_HASH,
        CaptureError,
        run_capture,
    )

    try:
        layers = [int(layer) for layer in args.layers.split(",")]
    except ValueError:
        out.error(f"Invalid --layers value: {args.layers!r}")
        out.line("  Solution: pass comma-separated integers, e.g. --layers 3,5")
        return 2

    try:
        out.run(f"Capturing activations: {args.model}")
        manifest = run_capture(
            args.model,
            args.dataset,
            layers,
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
    # The identity travels with the probe into the serving harness, which
    # compares it against the checkpoint it loaded. The sentinel is not a
    # comparison the harness loses -- it is one it skips, so say which of the
    # two happened rather than leaving it to be discovered at serve time.
    recorded = manifest["model_hash"]
    if recorded == UNKNOWN_MODEL_HASH:
        out.warn(f"model_hash {UNKNOWN_MODEL_HASH}: a probe from this capture")
        out.line("  will load into any model of the right width, unchecked.")
        out.line("  Solution: pass --model a local checkpoint directory, or")
        out.line("  --model-hash the sha256 of its model.safetensors")
    else:
        out.ok(f"model_hash {recorded}", indent=2)
    return 0
