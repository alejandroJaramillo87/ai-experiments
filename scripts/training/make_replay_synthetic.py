#!/usr/bin/env python3
"""Synthetic replay shard set generator (P2-D4, resolved decision 3).

Writes a shard directory of random tokens through the P2-D1 ShardWriter and
manifest path, sized to exercise the ReplayBuffer's budget, subsampling,
RAM-floor, and copy-on-write behaviors for real. Used by the P2-D4 host-side
acceptance run instead of a downloaded corpus; the real FineWeb-Edu and code
replay sets are prepared later via scripts/training/prepare_corpus.py.

  python scripts/training/make_replay_synthetic.py \
      --out /mnt/ai-data/training/shards/replay-synth-4gb \
      --gb 4 --vocab 262144 --tokenizer-hash synthetic --seed 0

The tokenizer hash defaults to "synthetic"; the domain set mixed against it
must be generated with the same hash (MixtureDataset refuses a mismatch).

Exit codes: 0 success, 1 runtime failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from training.data.shards import ShardWriter, dtype_for_vocab, write_manifest


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


# Tokens generated per write call; bounds peak memory during generation.
CHUNK_TOKENS = 16 * 1024 * 1024


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic replay shard set for P2-D4 acceptance.")
    parser.add_argument("--out", required=True, help="output shard directory")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--gb", type=float, help="target size in GB on disk")
    size.add_argument("--tokens", type=int, help="exact token count")
    parser.add_argument("--vocab", type=int, default=262_144,
                        help="vocabulary size; selects uint16/uint32 (default "
                             "262144, the Gemma 4 class -> uint32)")
    parser.add_argument("--tokenizer-hash", default="synthetic",
                        help="manifest tokenizer hash; must match the domain "
                             "set mixed against (default: synthetic)")
    parser.add_argument("--eos-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard-tokens", type=int, default=512 * 1024 * 1024,
                        help="tokens per shard file (default 512Mi)")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    out = Printer()
    dtype = dtype_for_vocab(args.vocab)
    tokens = args.tokens if args.tokens is not None else int(
        args.gb * (1 << 30) / dtype.itemsize)
    if tokens <= 0:
        out.error(f"Token count: must be positive, got {tokens}")
        return 2

    out.line("=== Synthetic replay shard set ===")
    out.line()
    out.info(f"Output: {args.out}")
    out.info(f"Tokens: {tokens} ({tokens * dtype.itemsize / (1 << 30):.2f}GB "
             f"at {dtype.name})")
    out.run(f"Writing shards: seed {args.seed}")

    rng = np.random.default_rng(args.seed)
    writer = ShardWriter(args.out, dtype, shard_tokens=args.shard_tokens)
    remaining = tokens
    while remaining > 0:
        n = min(CHUNK_TOKENS, remaining)
        writer.write_tokens(rng.integers(0, args.vocab, size=n, dtype=np.int64))
        remaining -= n
    records = writer.close()
    write_manifest(args.out, {
        "dataset": "replay-synthetic", "split": None, "revision": None,
        "license": "synthetic", "tokenizer": "synthetic",
        "tokenizer_hash": args.tokenizer_hash, "dtype": dtype.name,
        "eos_id": args.eos_id, "shard_tokens": args.shard_tokens,
        "seed": args.seed, "shards": records, "total_tokens": tokens,
        "total_documents": 1,
    })

    out.ok(f"Shards written: {len(records)}")
    out.ok(f"Manifest: {os.path.join(args.out, 'manifest.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
