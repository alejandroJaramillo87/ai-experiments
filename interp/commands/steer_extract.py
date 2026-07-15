#!/usr/bin/env python3
"""Extract a steering direction and export it as control-vector GGUF (W1).

Thin CLI over interp.steering.{extract,export}. Runs inside the workbench
container (bf16 substrate; GPU batch work). Exit codes: 0 success, 1 failure,
2 invalid usage.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")  # repo root is the working directory in the workbench

from interp.output import Printer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["repeng", "sae_row"], default="repeng")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model-hash", default="unknown")
    parser.add_argument("--pairs", help="contrastive pairs JSONL (method repeng)")
    parser.add_argument("--sae-release", help="SAELens release (method sae_row)")
    parser.add_argument("--sae-id", help="SAELens sae_id (method sae_row)")
    parser.add_argument("--feature", type=int, help="feature index (method sae_row)")
    parser.add_argument("--layer", type=int, help="target layer (method sae_row)")
    parser.add_argument("--out", required=True, help="output vector directory")
    args = parser.parse_args()

    out = Printer()
    from interp.steering import export, extract

    try:
        if args.method == "repeng":
            if not args.pairs:
                parser.error("--pairs is required with --method repeng")
            out.run(f"Extracting repeng vector: {args.model}")
            direction_set = extract.extract_repeng(
                args.model, args.pairs, revision=args.revision
            )
        else:
            missing = [
                f
                for f in ("sae_release", "sae_id", "feature", "layer")
                if getattr(args, f) is None
            ]
            if missing:
                parser.error(f"--method sae_row requires: {missing}")
            out.run(f"Reading SAE decoder row: {args.sae_release}/{args.sae_id}")
            direction_set = extract.extract_sae_row(
                args.model,
                args.sae_release,
                args.sae_id,
                args.feature,
                args.layer,
                revision=args.revision,
            )
        manifest = export.export_control_vector(
            direction_set, args.out, model_hash=args.model_hash
        )
    except (extract.ExtractionError, export.ExportError, OSError) as exc:
        out.error(f"Extraction failed: {exc}")
        return 1

    out.ok(f"Vector exported: {args.out}/vector.gguf")
    out.ok(f"Layers: {len(manifest['layers'])}, width: {manifest['width']}")
    out.ok(f"Vector xxhash: {manifest['vector_xxhash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
