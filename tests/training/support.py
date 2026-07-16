"""Shared module-level constants and builders for tests/training.

Module constants cannot come from fixtures, so REPO_ROOT and the reference
recipe paths live here; per-test scaffolding stays in conftest.py fixtures.
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

REFERENCE_DIR = os.path.join(REPO_ROOT, "training", "recipes", "reference")
TINY_RECIPE = os.path.join(REFERENCE_DIR, "tiny-e2e.yaml")


def reference_recipe(name: str, overrides: list[str] | None = None):
    """Load a recipe from training/recipes/reference with optional overrides."""
    from training.recipes import load_recipe

    return load_recipe(os.path.join(REFERENCE_DIR, name), overrides or [])


def make_shard_dir(base, tokens, *, name="shards", dataset="synthetic",
                   shard_tokens=2000, dtype=np.uint16, tokenizer_hash="abc123"):
    """Write a synthetic token shard dir under base/name with a full manifest."""
    from training.data.shards import ShardWriter, write_manifest

    out = str(base / name)
    writer = ShardWriter(out, dtype, shard_tokens)
    writer.write_tokens(tokens)
    records = writer.close()
    write_manifest(out, {
        "dataset": dataset, "split": None, "revision": None, "license": "unknown",
        "tokenizer": "synthetic", "tokenizer_hash": tokenizer_hash,
        "dtype": str(np.dtype(dtype)), "eos_id": 1, "shard_tokens": shard_tokens,
        "shards": records, "total_tokens": len(tokens), "total_documents": 1,
    })
    return out


def make_source(tmp_path, name: str, tokens, tokenizer_hash: str = "abc123",
                shard_tokens: int = 64) -> str:
    """Mixture/replay source: shard dir whose manifest dataset is its name."""
    return make_shard_dir(tmp_path, tokens, name=name, dataset=name,
                          shard_tokens=shard_tokens, tokenizer_hash=tokenizer_hash)


def write_benchmarks_yaml(tmp_path, rows, *, name="toy", text_fields=("question",),
                          config=None, revision=None) -> str:
    """Write a one-benchmark yaml plus the JSONL dataset it points at."""
    import json

    import yaml

    bench_jsonl = tmp_path / f"{name}.jsonl"
    with open(bench_jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    body = {"version": 1, "ngram_size": 13, "benchmarks": [{
        "name": name, "dataset": str(bench_jsonl),
        "config": config, "revision": revision,
        "splits": ["test"], "text_fields": list(text_fields)}]}
    path = tmp_path / "benchmarks.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(body, f)
    return str(path)


def build_bpe_tokenizer(out_dir: str, texts, specials, vocab_size: int = 512) -> str:
    """Train and save a tiny byte-level BPE tokenizer for offline tests."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=list(specials),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    os.makedirs(out_dir, exist_ok=True)
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    return str(out_dir)
