#!/usr/bin/env python3
"""Evaluate a trained SAE against its capture (interp Phase 4 P4-D3).

Core metric set — explained variance, L0, and optionally delta-LM-loss —
written as eval_report.json in the SAE run directory. Delta-LM-loss runs
subject-model forward passes (GPU, inference down); the reconstruction
metrics need only the cached activations. Exit codes: 0 success, 1 failure,
2 invalid usage.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, ".")

from interp.output import Printer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sae-dir",
        required=True,
        help="SAE run directory (/mnt/ai-data/interp/saes/<name>)",
    )
    parser.add_argument(
        "--capture-dir",
        required=True,
        help="P4-D2 capture directory to evaluate against",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1_000_000,
        help="activation rows sampled for the reconstruction metrics",
    )
    parser.add_argument(
        "--delta-loss-texts",
        default="",
        help="JSONL of texts for delta-LM-loss; omit to skip the forward passes",
    )
    args = parser.parse_args()

    out = Printer()
    from interp.manifest import ManifestError
    from interp.sae.evaluate import SaeEvalError, delta_lm_loss, evaluate_sae
    from interp.sae.registry import SaeRegistryError, load_sae_run

    try:
        delta = None
        if args.delta_loss_texts:
            run = load_sae_run(args.sae_dir)
            with open(args.delta_loss_texts, encoding="utf-8") as f:
                texts = [json.loads(line)["text"] for line in f if line.strip()]
            out.run(
                f"Delta-LM-loss over {len(texts)} texts: {run.manifest['model_id']}"
            )
            from interp.sae.evaluate import _load_sae_fn

            delta = delta_lm_loss(
                run.manifest["model_id"],
                run.hook,
                _load_sae_fn(args.sae_dir),
                texts,
            )
        out.run(f"Core metrics: {args.sae_dir}")
        report = evaluate_sae(
            args.sae_dir,
            args.capture_dir,
            max_tokens=args.max_tokens,
            delta_loss=delta,
        )
    except (SaeEvalError, SaeRegistryError, ManifestError, OSError) as exc:
        out.error(f"SAE evaluation failed: {exc}")
        return 1
    core = report["core"]
    out.ok(
        f"Evaluation complete: explained variance {core['explained_variance']}, "
        f"L0 {core['l0']}, {core['tokens_evaluated']} tokens"
    )
    if report["delta_lm_loss"]:
        out.ok(
            f"Delta-LM-loss {report['delta_lm_loss']['delta_lm_loss']}, "
            f"loss recovered {report['delta_lm_loss']['loss_recovered']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
