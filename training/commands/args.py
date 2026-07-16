"""Shared argparse helpers for training subcommands."""

from __future__ import annotations

import argparse


def positive_int(value: str) -> int:
    """int argument that accepts 1_000_000-style underscores and must be > 0."""
    n = int(value.replace("_", ""))
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return n
