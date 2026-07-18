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

from common.output import Printer


def configure(sub) -> None:
    p = sub.add_parser(
        "sae-eval",
        help="evaluate a trained SAE against its capture (P4-D3)",
        description=__doc__,
    )
    p.add_argument(
        "--sae-dir",
        required=True,
        help="SAE run directory (/mnt/ai-data/interp/saes/<name>)",
    )
    p.add_argument(
        "--capture-dir",
        required=True,
        help="P4-D2 capture directory to evaluate against",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=1_000_000,
        help="activation rows sampled for the reconstruction metrics",
    )
    p.add_argument(
        "--delta-loss-texts",
        default="",
        help="JSONL of texts for delta-LM-loss; omit to skip the forward passes",
    )
    p.set_defaults(func=cmd)


def _load_texts(path: str) -> list[str]:
    """Delta-LM-loss texts from a JSONL of {"text": ...} records.

    Malformed lines and an empty file are SaeEvalError refusals with the
    offending path:lineno — an empty text list would fabricate an all-zero
    delta report, and a raw KeyError names no line.
    """
    from interp.sae.evaluate import SaeEvalError

    texts = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SaeEvalError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict) or "text" not in record:
                raise SaeEvalError(f"{path}:{lineno}: needs a 'text' key")
            texts.append(record["text"])
    if not texts:
        raise SaeEvalError(f"{path}: no texts found")
    return texts


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from interp.manifest import ManifestError
    from interp.sae.evaluate import SaeEvalError, delta_lm_loss, evaluate_sae
    from interp.sae.registry import load_sae_run

    try:
        delta = None
        if args.delta_loss_texts:
            run = load_sae_run(args.sae_dir)
            texts = _load_texts(args.delta_loss_texts)
            out.run(
                f"Delta-LM-loss over {len(texts)} texts: {run.manifest['model_id']}"
            )
            from interp.sae.evaluate import _load_sae_fn

            delta = delta_lm_loss(
                run.manifest["model_id"],
                run.hook,
                _load_sae_fn(args.sae_dir),
                texts,
                revision=run.manifest.get("model_revision"),
            )
        out.run(f"Core metrics: {args.sae_dir}")
        report = evaluate_sae(
            args.sae_dir,
            args.capture_dir,
            max_tokens=args.max_tokens,
            delta_loss=delta,
        )
    except (SaeEvalError, ManifestError, OSError) as exc:
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
