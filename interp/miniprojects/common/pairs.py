"""Contrastive prompt-pair utilities (interp M3).

Load pair sets from JSONL, split train/test with near-duplicate awareness, and compute a
difference-in-means direction over a chosen hook point. The dedup across the split boundary
exists because probe results on contaminated splits look real and are not.
"""

from __future__ import annotations

import json
import random
import re

Pair = dict[str, str]  # {"positive": ..., "negative": ...}


def load_pairs(path: str) -> list[Pair]:
    """Read a JSONL pair set; every line needs 'positive' and 'negative' fields."""
    pairs: list[Pair] = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "positive" not in record or "negative" not in record:
                raise ValueError(
                    f"{path}:{lineno}: missing 'positive'/'negative' field"
                )
            pairs.append(
                {"positive": record["positive"], "negative": record["negative"]}
            )
    if not pairs:
        raise ValueError(f"{path}: no pairs found")
    return pairs


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def split_pairs(
    pairs: list[Pair], test_frac: float = 0.25, seed: int = 0
) -> tuple[list[Pair], list[Pair]]:
    """Shuffled train/test split that keeps near-duplicates on one side.

    Pairs whose normalized text matches an already-assigned pair follow it to the same side,
    so trivially rephrased prompts cannot leak across the boundary.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))

    side_by_key: dict[str, str] = {}
    train: list[Pair] = []
    test: list[Pair] = []
    for pair in shuffled:
        key = _normalize(pair["positive"]) + " || " + _normalize(pair["negative"])
        side = side_by_key.get(key)
        if side is None:
            side = "test" if len(test) < n_test else "train"
            side_by_key[key] = side
        (test if side == "test" else train).append(pair)
    return train, test


def difference_in_means(model, pairs: list[Pair], hook_point: str, position: int = -1):
    """Mean(positive) - mean(negative) activation at hook_point, final position by default.

    The model is a TransformerLens HookedTransformer; runs are no-grad, one prompt at a time
    (pair sets are small and prompts are variable-length; batching buys nothing here).
    """
    import torch

    def _activation(prompt: str):
        _, cache = model.run_with_cache(prompt)
        return cache[hook_point][0, position, :].detach().float()

    with torch.no_grad():
        pos = torch.stack([_activation(p["positive"]) for p in pairs]).mean(dim=0)
        neg = torch.stack([_activation(p["negative"]) for p in pairs]).mean(dim=0)
    return pos - neg
