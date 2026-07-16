"""Prepare a chat/instruction dataset into tokenized, masked SFT examples (Phase 1 F1).

Thin subcommand over training/data/sft.py. Examples:

  python -m training prepare-sft \
      --source HuggingFaceH4/ultrachat_200k --split train_sft \
      --tokenizer google/gemma-4-E2B --model-type gemma4 \
      --out /mnt/ai-data/training/datasets/sft/ultrachat-gemma4 \
      --max-seq-len 4096 --holdout 0.02

  python -m training prepare-sft \
      --source ./conversations.jsonl \
      --tokenizer /mnt/ai-data/training/tokenizers/gemma-4-e2b \
      --model-type gemma4 \
      --out /mnt/ai-data/training/datasets/sft/local-test

Exit codes: 0 success, 1 pipeline failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os

from common.output import Printer
from training.commands.args import positive_int
from training.data import sft as sft_mod
from training.families import UnknownModelFamilyError, supported_families


def holdout_fraction(value: str) -> float:
    f = float(value)
    if not 0.0 <= f < 1.0:
        raise argparse.ArgumentTypeError(f"must be in [0, 1), got {value}")
    return f


def configure(sub) -> None:
    parser = sub.add_parser(
        "prepare-sft",
        help="tokenize and loss-mask a chat/instruction dataset (F1)",
        description="Tokenize and loss-mask a chat/instruction dataset with a manifest.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="HF dataset id or path to a local .jsonl file "
        "(messages lists or prompt/completion pairs)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory (convention: /mnt/ai-data/training/datasets/sft/<name>)",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer directory, tokenizer.json path, or HF hub id",
    )
    parser.add_argument(
        "--model-type",
        required=True,
        help=f"HF config model_type selecting the chat template "
        f"(registered: {', '.join(supported_families())})",
    )
    parser.add_argument("--split", default="train", help="Dataset split for HF sources")
    parser.add_argument(
        "--revision", default=None, help="Dataset revision for HF sources"
    )
    parser.add_argument(
        "--max-seq-len",
        type=positive_int,
        default=4096,
        help="Truncate examples to this many tokens (default: 4096)",
    )
    parser.add_argument(
        "--holdout",
        type=holdout_fraction,
        default=sft_mod.DEFAULT_HOLDOUT,
        help=f"Held-out fraction for F7 held-out loss "
        f"(default: {sft_mod.DEFAULT_HOLDOUT})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Held-out split seed (default: 42)"
    )
    parser.add_argument(
        "--license",
        default="unknown",
        dest="license_str",
        help="License string recorded in the manifest (default: unknown)",
    )
    parser.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()

    out.line("=== SFT data preparation (Phase 1 F1) ===")
    out.line()
    out.info(f"Source: {args.source}")
    out.info(f"Tokenizer: {args.tokenizer}")
    out.info(f"Model family: {args.model_type}")
    out.info(f"Output: {args.out}")
    out.run("Rendering, tokenizing, and masking conversations")

    def progress(examples: int, tokens: int) -> None:
        out.info(f"Progress: {examples:,} examples, {tokens:,} tokens", indent=2)

    try:
        manifest = sft_mod.prepare_sft(
            args.source,
            args.out,
            args.tokenizer,
            args.model_type,
            split=args.split,
            revision=args.revision,
            max_seq_len=args.max_seq_len,
            holdout=args.holdout,
            seed=args.seed,
            license_str=args.license_str,
            progress=progress,
        )
    except FileExistsError as exc:
        out.error(str(exc))
        out.line("  Solution: choose a new output directory or delete the existing one")
        return 1
    except UnknownModelFamilyError as exc:
        out.error(str(exc))
        return 1
    except (ValueError, KeyError, FileNotFoundError) as exc:
        out.error(f"Pipeline failed: {exc}")
        return 1

    out.line()
    out.ok(f"Train examples: {manifest['train_examples']:,}")
    out.ok(
        f"Holdout examples: {manifest['holdout_examples']:,} "
        f"(fraction {manifest['holdout_fraction']}, seed {manifest['holdout_seed']})"
    )
    if manifest["dropped_examples"]:
        out.warn(
            f"Dropped examples: {manifest['dropped_examples']:,} "
            "(no supervised tokens after truncation)"
        )
    out.ok(
        f"Tokens: {manifest['total_tokens']:,} total, "
        f"{manifest['supervised_tokens']:,} supervised"
    )
    out.ok(f"Manifest: {os.path.join(args.out, sft_mod.MANIFEST_NAME)}")
    return 0
