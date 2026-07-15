"""Build the benchmark decontamination index (P2-D1).

Extracts n-grams from the P2-D6 benchmark list into an index directory whose
manifest hash is what shard manifests reference via decontaminated_against.
Downloads benchmark test sets from the Hugging Face hub, so this runs on the
workstation (or inside the training container), not in an agent sandbox.

  python -m training build-contamination \
      --benchmarks training/continual/benchmarks.yaml \
      --out /mnt/ai-data/training/datasets/_contamination/v1

Exit codes: 0 success, 1 build failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.output import Printer
from training.data import contamination


def configure(sub) -> None:
    parser = sub.add_parser(
        "build-contamination",
        help="build the benchmark n-gram contamination index (P2-D1)",
        description="Build the benchmark n-gram contamination index.")
    parser.add_argument("--benchmarks", default="training/continual/benchmarks.yaml",
                        help="Benchmark list YAML (default: training/continual/benchmarks.yaml)")
    parser.add_argument("--out", required=True,
                        help="Index output directory "
                             "(convention: /mnt/ai-data/training/datasets/_contamination/<ver>)")
    parser.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()

    out.line("=== Contamination index build (P2-D1) ===")
    out.line()
    out.info(f"Benchmarks: {args.benchmarks}")
    out.info(f"Output: {args.out}")
    out.run("Extracting benchmark n-grams")

    try:
        manifest = contamination.build_index(args.benchmarks, args.out)
    except (ValueError, KeyError, FileNotFoundError, OSError) as exc:
        out.error(f"Index build failed: {exc}")
        return 1

    out.line()
    for bench in manifest["benchmarks"]:
        out.ok(f"{bench['name']}: {bench['documents']:,} documents, "
               f"{bench['new_ngrams']:,} new n-grams", indent=2)
    out.ok(f"Total: {manifest['total_ngrams']:,} unique {manifest['ngram_size']}-grams")
    out.ok(f"decontaminated_against: {contamination.index_hash(args.out)}")
    return 0
