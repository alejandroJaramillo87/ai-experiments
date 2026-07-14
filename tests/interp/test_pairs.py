"""Unit tests for interp.miniprojects.common.pairs (CPU only)."""

from __future__ import annotations

import json

import pytest

from interp.miniprojects.common.pairs import load_pairs, split_pairs

PAIRS = [
    {"positive": f"I love thing number {i}", "negative": f"I hate thing number {i}"}
    for i in range(20)
]


def _write_jsonl(path, records):
    with open(path, "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return str(path)


def test_load_pairs_roundtrip(tmp_path):
    path = _write_jsonl(tmp_path / "pairs.jsonl", PAIRS)
    assert load_pairs(path) == PAIRS


def test_load_pairs_rejects_missing_field(tmp_path):
    path = _write_jsonl(tmp_path / "bad.jsonl", [{"positive": "only half a pair"}])
    with pytest.raises(ValueError, match="missing 'positive'/'negative'"):
        load_pairs(path)


def test_load_pairs_rejects_empty(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="no pairs"):
        load_pairs(str(path))


def test_split_sizes_and_partition():
    train, test = split_pairs(PAIRS, test_frac=0.25, seed=1)
    assert len(train) + len(test) == len(PAIRS)
    assert len(test) == 5
    assert not any(p in train for p in test)


def test_split_deterministic():
    assert split_pairs(PAIRS, seed=7) == split_pairs(PAIRS, seed=7)


def test_split_keeps_near_duplicates_on_one_side():
    # Same text modulo case/punctuation must never straddle the boundary
    duplicated = PAIRS + [
        {
            "positive": p["positive"].upper() + "!",
            "negative": p["negative"].upper() + "!",
        }
        for p in PAIRS
    ]
    for seed in range(5):
        train, test = split_pairs(duplicated, test_frac=0.5, seed=seed)

        def keys(side):
            import re

            def norm(t):
                return re.sub(r"\W+", " ", t.lower()).strip()

            return {norm(p["positive"]) + " || " + norm(p["negative"]) for p in side}

        assert keys(train).isdisjoint(keys(test))


def test_split_rejects_bad_frac():
    with pytest.raises(ValueError, match="test_frac"):
        split_pairs(PAIRS, test_frac=1.5)
