"""Synthetic replay shard set generator (P2-D4, resolved decision 3).

Writes a shard directory of random tokens through the P2-D1 ShardWriter and
manifest path, sized to exercise the ReplayBuffer's budget, subsampling,
RAM-floor, and copy-on-write behaviors for real. Used by the P2-D4 host-side
acceptance run instead of a downloaded corpus; the real FineWeb-Edu and code
replay sets are prepared later via `python -m training prepare-corpus`.

  python -m training make-replay-synthetic \
      --out /mnt/ai-data/training/shards/replay-synth-4gb \
      --gb 4 --vocab 262144 --tokenizer-hash synthetic --seed 0

The tokenizer hash defaults to "synthetic"; the domain set mixed against it
must be generated with the same hash (MixtureDataset refuses a mismatch).

Exit codes: 0 success, 1 runtime failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from common.output import Printer
from training.data.shards import ShardWriter, dtype_for_vocab, write_manifest

# Tokens generated per write call; bounds peak memory during generation.
CHUNK_TOKENS = 16 * 1024 * 1024


def configure(sub) -> None:
    parser = sub.add_parser(
        "make-replay-synthetic",
        help="generate a synthetic replay shard set for P2-D4 acceptance",
        description="Generate a synthetic replay shard set for P2-D4 acceptance.",
    )
    parser.add_argument("--out", required=True, help="output shard directory")
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--gb", type=float, help="target size in GB on disk")
    size.add_argument("--tokens", type=int, help="exact token count")
    parser.add_argument(
        "--vocab",
        type=int,
        default=262_144,
        help="vocabulary size; selects uint16/uint32 (default "
        "262144, the Gemma 4 class -> uint32)",
    )
    parser.add_argument(
        "--tokenizer-hash",
        default="synthetic",
        help="manifest tokenizer hash; must match the domain "
        "set mixed against (default: synthetic)",
    )
    parser.add_argument("--eos-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=512 * 1024 * 1024,
        help="tokens per shard file (default 512Mi)",
    )
    parser.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    dtype = dtype_for_vocab(args.vocab)
    tokens = (
        args.tokens
        if args.tokens is not None
        else int(args.gb * (1 << 30) / dtype.itemsize)
    )
    if tokens <= 0:
        out.error(f"Token count: must be positive, got {tokens}")
        return 2

    out.line("=== Synthetic replay shard set ===")
    out.line()
    out.info(f"Output: {args.out}")
    out.info(
        f"Tokens: {tokens} ({tokens * dtype.itemsize / (1 << 30):.2f}GB "
        f"at {dtype.name})"
    )
    out.run(f"Writing shards: seed {args.seed}")

    rng = np.random.default_rng(args.seed)
    writer = ShardWriter(args.out, dtype, shard_tokens=args.shard_tokens)
    remaining = tokens
    while remaining > 0:
        n = min(CHUNK_TOKENS, remaining)
        writer.write_tokens(rng.integers(0, args.vocab, size=n, dtype=np.int64))
        remaining -= n
    records = writer.close()
    write_manifest(
        args.out,
        {
            "dataset": "replay-synthetic",
            "split": None,
            "revision": None,
            "license": "synthetic",
            "tokenizer": "synthetic",
            "tokenizer_hash": args.tokenizer_hash,
            "dtype": dtype.name,
            "eos_id": args.eos_id,
            "shard_tokens": args.shard_tokens,
            "seed": args.seed,
            "shards": records,
            "total_tokens": tokens,
            "total_documents": 1,
        },
    )

    out.ok(f"Shards written: {len(records)}")
    out.ok(f"Manifest: {os.path.join(args.out, 'manifest.json')}")
    return 0
