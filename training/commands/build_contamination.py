#!/usr/bin/env python3
"""Build the benchmark decontamination index (P2-D1).

Extracts n-grams from the P2-D6 benchmark list into an index directory whose
manifest hash is what shard manifests reference via decontaminated_against.
Downloads benchmark test sets from the Hugging Face hub, so this runs on the
workstation (or inside the training container), not in an agent sandbox.

  python scripts/training/build_contamination.py \
      --benchmarks training/continual/benchmarks.yaml \
      --out /mnt/ai-data/training/datasets/_contamination/v1

Exit codes: 0 success, 1 build failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.data import contamination


class Printer:
    """Styled status output (see .styles/PYTHON_OUTPUT_STYLE.md)."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        use_color = (
            not quiet
            and not os.environ.get("NO_COLOR")
            and sys.stdout.isatty()
        )
        if use_color:
            self._c = {
                "OK": "\033[0;32m", "WARN": "\033[1;33m", "ERROR": "\033[0;31m",
                "INFO": "\033[0;36m", "RUN": "\033[0;34m",
            }
            self._nc = "\033[0m"
        else:
            self._c = {}
            self._nc = ""

    def _status(self, tag: str, message: str, indent: int = 0) -> None:
        if self.quiet:
            return
        color = self._c.get(tag, "")
        pad = " " * indent
        print(f"{pad}{color}[{tag}]{self._nc} {message}")

    def ok(self, m, indent=0):    self._status("OK", m, indent)
    def error(self, m, indent=0): self._status("ERROR", m, indent)
    def info(self, m, indent=0):  self._status("INFO", m, indent)
    def run(self, m, indent=0):   self._status("RUN", m, indent)

    def line(self, message: str = "") -> None:
        if not self.quiet:
            print(message)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the benchmark n-gram contamination index.")
    parser.add_argument("--benchmarks", default="training/continual/benchmarks.yaml",
                        help="Benchmark list YAML (default: training/continual/benchmarks.yaml)")
    parser.add_argument("--out", required=True,
                        help="Index output directory "
                             "(convention: /mnt/ai-data/training/datasets/_contamination/<ver>)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
