#!/usr/bin/env python3
"""Full-scale corpus preparation (P2-D1).

Thin CLI over training/data/pipeline.py: language identification, quality
filtering, exact and MinHash-LSH deduplication, and benchmark decontamination
in front of the Phase 0 D2 tokenize-and-shard writer. prepare_data.py remains
the lite path; this is a superset ending in the same writer.

  python scripts/training/prepare_corpus.py \
      --source HuggingFaceFW/fineweb-edu --split train \
      --pipeline-config training/data/configs/default.yaml \
      --contamination /mnt/ai-data/training/datasets/_contamination/v1 \
      --tokenizer google/gemma-4-E2B \
      --work-dir /mnt/ai-data/training/datasets/fineweb-edu-2B \
      --out /mnt/ai-data/training/shards/fineweb-edu-2B \
      --max-tokens 2_000_000_000

Exit codes: 0 success, 1 pipeline failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.data import pipeline as pipeline_mod
from training.data import shards as shards_mod


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
    def warn(self, m, indent=0):  self._status("WARN", m, indent)
    def error(self, m, indent=0): self._status("ERROR", m, indent)
    def info(self, m, indent=0):  self._status("INFO", m, indent)
    def run(self, m, indent=0):   self._status("RUN", m, indent)

    def line(self, message: str = "") -> None:
        if not self.quiet:
            print(message)


def positive_int(value: str) -> int:
    n = int(value.replace("_", ""))
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return n


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter, deduplicate, decontaminate, and shard a corpus.")
    parser.add_argument("--source", required=True,
                        help="HF dataset id or path to a local .jsonl file")
    parser.add_argument("--out", required=True,
                        help="Output shard directory (convention: /mnt/ai-data/training/shards/<name>)")
    parser.add_argument("--work-dir", required=True,
                        help="Stage-intermediate directory "
                             "(convention: /mnt/ai-data/training/datasets/<name>)")
    parser.add_argument("--tokenizer", required=True,
                        help="Tokenizer directory, tokenizer.json path, or HF hub id")
    parser.add_argument("--pipeline-config", default=None,
                        help="Pipeline YAML (default: built-in defaults, "
                             "documented in training/data/configs/default.yaml)")
    parser.add_argument("--contamination", default=None,
                        help="Contamination index directory from build_contamination.py; "
                             "omitting it skips decontamination (recorded as null in the manifest)")
    parser.add_argument("--split", default="train", help="Dataset split for HF sources")
    parser.add_argument("--revision", default=None, help="Dataset revision for HF sources")
    parser.add_argument("--text-field", default="text", help="Text field name (default: text)")
    parser.add_argument("--max-tokens", type=positive_int, default=None,
                        help="Stop sharding after this many tokens (underscores allowed)")
    parser.add_argument("--max-documents", type=positive_int, default=None,
                        help="Ingest at most this many documents from an HF source")
    parser.add_argument("--num-proc", type=positive_int, default=None,
                        help="Pipeline worker count (default: pipeline config workers, 30)")
    parser.add_argument("--shard-tokens", type=positive_int,
                        default=shards_mod.DEFAULT_SHARD_TOKENS,
                        help="Tokens per shard (default: 512M)")
    parser.add_argument("--eos-token", default=None,
                        help="EOS token string when the tokenizer does not declare one")
    parser.add_argument("--license", default="unknown", dest="license_str",
                        help="License string recorded in the manifest (default: unknown)")
    parser.add_argument("--clean-intermediates", action="store_true",
                        help="Delete stage JSONL data (keeping stats.json) after shards validate")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = Printer()

    out.line("=== Corpus preparation (P2-D1) ===")
    out.line()
    out.info(f"Source: {args.source}")
    out.info(f"Tokenizer: {args.tokenizer}")
    out.info(f"Work dir: {args.work_dir}")
    out.info(f"Output: {args.out}")
    if args.contamination is None:
        out.warn("Decontamination: skipped (no --contamination index)")
    else:
        out.info(f"Contamination index: {args.contamination}")

    def progress(tokens: int, documents: int) -> None:
        out.info(f"Progress: {tokens:,} tokens, {documents:,} documents", indent=2)

    try:
        config = pipeline_mod.load_pipeline_config(args.pipeline_config)
        if args.num_proc is not None:
            config.workers = args.num_proc
        out.run(f"Running pipeline stages ({config.workers} workers)")
        manifest = pipeline_mod.prepare_corpus(
            args.source, args.out, args.tokenizer, args.work_dir,
            config=config, contamination_dir=args.contamination,
            text_field=args.text_field, split=args.split, revision=args.revision,
            max_tokens=args.max_tokens, max_documents=args.max_documents,
            shard_tokens=args.shard_tokens, eos_token=args.eos_token,
            license_str=args.license_str, clean=args.clean_intermediates,
            progress=progress)
    except FileExistsError as exc:
        out.error(str(exc))
        out.line("  Solution: choose a new output directory or delete the existing one")
        return 1
    except (ValueError, KeyError, FileNotFoundError, RuntimeError) as exc:
        out.error(f"Pipeline failed: {exc}")
        return 1

    out.line()
    for stage in manifest["pipeline"]["stage_funnel"]:
        dropped = stage["docs_in"] - stage["docs_out"]
        out.ok(f"{stage['stage']}: {stage['docs_in']:,} -> {stage['docs_out']:,} "
               f"({dropped:,} dropped)", indent=2)
    out.ok(f"Shards written: {len(manifest['shards'])} "
           f"({manifest['dtype']}, {manifest['shard_tokens']:,} tokens/shard)")
    out.ok(f"Total: {manifest['total_tokens']:,} tokens, "
           f"{manifest['total_documents']:,} documents")
    out.ok(f"decontaminated_against: {manifest['decontaminated_against']}")
    out.ok(f"Manifest: {os.path.join(args.out, shards_mod.MANIFEST_NAME)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
