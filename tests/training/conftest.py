"""Fixtures for the D2 data pipeline tests: a tiny byte-level BPE tokenizer and a
synthetic JSONL corpus, both built in tmp dirs so tests run without network or GPU."""

from __future__ import annotations

import json

import pytest
from support import build_bpe_tokenizer  # noqa: F401  (imports insert REPO_ROOT)


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
    out = tmp_path_factory.mktemp("tokenizer")
    return build_bpe_tokenizer(
        str(out), DOCUMENTS, specials=["<|bos|>", "<|eos|>", "<|pad|>"])


@pytest.fixture()
def jsonl_path(tmp_path) -> str:
    path = tmp_path / "corpus.jsonl"
    with open(path, "w") as f:
        for text in DOCUMENTS:
            f.write(json.dumps({"text": text}) + "\n")
    return str(path)
