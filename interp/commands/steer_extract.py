"""Extract a steering direction and export it as control-vector safetensors (W1).

Thin CLI over interp.steering.{extract,export}. diff_means reads a capture
directory and needs no model or GPU; sae_row reads a pretrained SAE. Exit
codes: 0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.output import Printer


def configure(sub) -> None:
    p = sub.add_parser(
        "steer-extract",
        help=(
            "extract a steering direction and export it as "
            "control-vector safetensors (W1)"
        ),
        description=__doc__,
    )
    p.add_argument("--method", choices=["diff_means", "sae_row"], default="diff_means")
    p.add_argument(
        "--capture",
        help="activation-capture directory (method diff_means); identity and "
        "model hash come from its manifest",
    )
    p.add_argument(
        "--layers",
        help="comma-separated layer subset (method diff_means); default every "
        "layer in the capture",
    )
    p.add_argument("--model", help="HF model id or local path (method sae_row)")
    p.add_argument("--revision", default="main")
    p.add_argument("--model-hash", help="checkpoint sha256 (method sae_row)")
    p.add_argument("--sae-release", help="SAELens release (method sae_row)")
    p.add_argument("--sae-id", help="SAELens sae_id (method sae_row)")
    p.add_argument("--feature", type=int, help="feature index (method sae_row)")
    p.add_argument("--layer", type=int, help="target layer (method sae_row)")
    p.add_argument("--out", required=True, help="output vector directory")
    p.set_defaults(func=cmd)


def parse_layers(text: str) -> list[int]:
    layers = []
    for token in text.split(","):
        token = token.strip()
        if not token.isdigit():
            raise ValueError(f"--layers expects comma-separated integers, got {text!r}")
        layers.append(int(token))
    return layers


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from interp.manifest import ManifestError
    from interp.probes.capture import CaptureError
    from interp.steering import export, extract

    try:
        if args.method == "diff_means":
            if not args.capture:
                out.error("--capture is required with --method diff_means")
                return 2
            if args.model_hash is not None:
                out.error(
                    "--model-hash is not accepted with --method diff_means: "
                    "the capture manifest records the hash"
                )
                return 2
            layers = None
            if args.layers:
                try:
                    layers = parse_layers(args.layers)
                except ValueError as exc:
                    out.error(str(exc))
                    return 2
            out.run(f"Extracting diff-means vector: {args.capture}")
            direction_set = extract.extract_diff_means(args.capture, layers=layers)
            src = direction_set.source
            out.info(
                f"Classes: {src.get('positive_count')} positive, "
                f"{src.get('negative_count')} negative; "
                f"model_hash {direction_set.model_hash}"
            )
            if direction_set.model_hash == extract.UNKNOWN_MODEL_HASH:
                out.warn(
                    "Model hash: unknown in the capture manifest; "
                    "the harness will skip its identity check"
                )
            model_hash = None
        else:
            missing = [
                f
                for f in ("model", "sae_release", "sae_id", "feature", "layer")
                if getattr(args, f) is None
            ]
            if missing:
                out.error(f"--method sae_row requires: {missing}")
                return 2
            out.run(f"Reading SAE decoder row: {args.sae_release}/{args.sae_id}")
            direction_set = extract.extract_sae_row(
                args.model,
                args.sae_release,
                args.sae_id,
                args.feature,
                args.layer,
                revision=args.revision,
            )
            model_hash = args.model_hash or extract.UNKNOWN_MODEL_HASH
        manifest = export.export_control_vector(
            direction_set, args.out, model_hash=model_hash
        )
    except (
        extract.ExtractionError,
        export.ExportError,
        CaptureError,
        ManifestError,
        OSError,
    ) as exc:
        out.error(f"Extraction failed: {exc}")
        return 1

    out.ok(f"Vector exported: {args.out}/{export.VECTOR_NAME}")
    out.ok(f"Layers: {len(manifest['layers'])}, width: {manifest['width']}")
    out.ok(f"Vector xxhash: {manifest['vector_xxhash']}")
    return 0
