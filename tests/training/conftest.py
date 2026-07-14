"""Fixtures for the D2 data pipeline tests: a tiny byte-level BPE tokenizer and a
synthetic JSONL corpus, both built in tmp dirs so tests run without network or GPU."""

from __future__ import annotations

import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


@pytest.fixture(autouse=True)
def train_lock_path(tmp_path, monkeypatch):
    """Point the D9 run lock at a per-test path: the sandbox has no
    /mnt/ai-data, and tests must not contend on a shared lock."""
    monkeypatch.setenv("TRAIN_LOCK_PATH", str(tmp_path / ".train.lock"))

DOCUMENTS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "jackdaws love my big sphinx of quartz",
    "waltz bad nymph for quick jigs vex",
    "quick zephyrs blow vexing daft jim",
] * 4   # 32 documents so multi-shard runs have material


@pytest.fixture(scope="session")
def documents() -> list[str]:
    return DOCUMENTS


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory) -> str:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512, special_tokens=["<|bos|>", "<|eos|>", "<|pad|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(DOCUMENTS, trainer)

    out = tmp_path_factory.mktemp("tokenizer")
    tok.save(str(out / "tokenizer.json"))
    return str(out)


@pytest.fixture()
def jsonl_path(tmp_path) -> str:
    path = tmp_path / "corpus.jsonl"
    with open(path, "w") as f:
        for text in DOCUMENTS:
            f.write(json.dumps({"text": text}) + "\n")
    return str(path)
