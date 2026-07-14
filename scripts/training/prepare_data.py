#!/usr/bin/env python3
"""Tokenize a dataset into memory-mapped token shards (Phase 0 D2).

Thin CLI over training/data/prepare.py. Examples:

  python scripts/training/prepare_data.py \
      --source HuggingFaceFW/fineweb-edu --split train \
      --tokenizer /mnt/ai-data/training/tokenizers/base \
      --out /mnt/ai-data/training/shards/fineweb-edu-10B \
      --max-tokens 10_000_000_000 --num-proc 30

  python scripts/training/prepare_data.py \
      --source ./data.jsonl --text-field text \
      --tokenizer HuggingFaceTB/SmolLM2-135M \
      --out /mnt/ai-data/training/shards/local-test

Exit codes: 0 success, 1 pipeline failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.data import prepare as prepare_mod
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
                "INFO": "\033[0;36m", "RUN": "\033[0;34m", "SKIP": "\033[0;34m",
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
        description="Tokenize a HF dataset or local JSONL into token shards with a manifest.")
    parser.add_argument("--source", required=True,
                        help="HF dataset id or path to a local .jsonl file")
    parser.add_argument("--out", required=True,
                        help="Output shard directory (convention: /mnt/ai-data/training/shards/<name>)")
    parser.add_argument("--tokenizer", required=True,
                        help="Tokenizer directory, tokenizer.json path, or HF hub id")
    parser.add_argument("--split", default="train", help="Dataset split for HF sources")
    parser.add_argument("--revision", default=None, help="Dataset revision for HF sources")
    parser.add_argument("--text-field", default="text", help="Text field name (default: text)")
    parser.add_argument("--max-tokens", type=positive_int, default=None,
                        help="Stop after this many tokens (underscores allowed: 10_000_000_000)")
    parser.add_argument("--num-proc", type=positive_int, default=prepare_mod.DEFAULT_NUM_PROC,
                        help=f"Tokenizer worker processes for non-streaming HF sources "
                             f"(default: {prepare_mod.DEFAULT_NUM_PROC})")
    parser.add_argument("--shard-tokens", type=positive_int,
                        default=shards_mod.DEFAULT_SHARD_TOKENS,
                        help="Tokens per shard (default: 512M, ~1GB at uint16)")
    parser.add_argument("--streaming", action="store_true",
                        help="Stream the HF dataset (for corpora larger than RAM)")
    parser.add_argument("--eos-token", default=None,
                        help="EOS token string when the tokenizer does not declare one")
    parser.add_argument("--license", default="unknown", dest="license_str",
                        help="License string recorded in the manifest (default: unknown)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = Printer()

    out.line("=== Data preparation (Phase 0 D2) ===")
    out.line()
    out.info(f"Source: {args.source}")
    out.info(f"Tokenizer: {args.tokenizer}")
    out.info(f"Output: {args.out}")
    out.run("Tokenizing and writing shards")

    def progress(tokens: int, documents: int) -> None:
        out.info(f"Progress: {tokens:,} tokens, {documents:,} documents", indent=2)

    try:
        manifest = prepare_mod.prepare(
            args.source, args.out, args.tokenizer,
            text_field=args.text_field, split=args.split, revision=args.revision,
            streaming=args.streaming, max_tokens=args.max_tokens,
            num_proc=args.num_proc, shard_tokens=args.shard_tokens,
            eos_token=args.eos_token, license_str=args.license_str,
            progress=progress)
    except FileExistsError as exc:
        out.error(str(exc))
        out.line("  Solution: choose a new output directory or delete the existing one")
        return 1
    except (ValueError, KeyError, FileNotFoundError) as exc:
        out.error(f"Pipeline failed: {exc}")
        return 1

    out.line()
    out.ok(f"Shards written: {len(manifest['shards'])} "
           f"({manifest['dtype']}, {manifest['shard_tokens']:,} tokens/shard)")
    out.ok(f"Total: {manifest['total_tokens']:,} tokens, "
           f"{manifest['total_documents']:,} documents")
    out.ok(f"Manifest: {os.path.join(args.out, shards_mod.MANIFEST_NAME)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
