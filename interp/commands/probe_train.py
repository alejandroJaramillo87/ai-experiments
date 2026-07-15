#!/usr/bin/env python3
"""Train per-layer probes from a capture and register the best (W2).

CPU work (scikit-learn over cached activations). Exit codes: 0 success,
1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from interp.output import Printer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, help="capture directory")
    parser.add_argument("--name", required=True, help="probe name")
    parser.add_argument("--out", default="/mnt/ai-data/interp/probes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Printer()
    from interp.manifest import ManifestError
    from interp.probes.train import TrainError, train_probes

    try:
        out.run(f"Training probes from {args.capture}")
        manifest = train_probes(args.capture, args.name, args.out, seed=args.seed)
    except (TrainError, ManifestError, OSError) as exc:
        out.error(f"Training failed: {exc}")
        return 1
    out.ok(
        f"Probe registered: {args.name} (layer {manifest['layer']}, "
        f"val AUROC {manifest['metrics']['val_auroc']}, "
        f"test AUROC {manifest['metrics']['test_auroc']})"
    )
    for layer, metrics in manifest["per_layer_metrics"].items():
        out.info(f"layer {layer}: val {metrics['val_auroc']}", indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
