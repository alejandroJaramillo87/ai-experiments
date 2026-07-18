"""Fixtures for the D2 data pipeline tests: a tiny byte-level BPE tokenizer and a
synthetic JSONL corpus, both built in tmp dirs so tests run without network or GPU."""

from __future__ import annotations

import json
import logging

import pytest
from support import build_bpe_tokenizer  # noqa: F401  (imports insert REPO_ROOT)

from common.obs import core as obs_core


@pytest.fixture(autouse=True)
def train_lock_path(tmp_path, monkeypatch):
    """Point the D9 run lock at a per-test path: the sandbox has no
    /mnt/ai-data, and tests must not contend on a shared lock."""
    monkeypatch.setenv("TRAIN_LOCK_PATH", str(tmp_path / ".train.lock"))


@pytest.fixture()
def obs_isolation(tmp_path, monkeypatch):
    """For tests that call cli.main() in-process: the traced dispatch
    (roadmap OBS-4) configures the obs core, so point the event file at tmp
    and restore context and handler state after the test."""
    monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path / "obs-logs"))
    token = obs_core._context.set(obs_core._RunContext())
    yield
    obs_core._context.reset(token)
    logger = logging.getLogger("obs")
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)


DOCUMENTS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz judge my vow",
    "how vexingly quick daft zebras jump",
    "the five boxing wizards jump quickly",
    "jackdaws love my big sphinx of quartz",
    "waltz bad nymph for quick jigs vex",
    "quick zephyrs blow vexing daft jim",
] * 4  # 32 documents so multi-shard runs have material


@pytest.fixture(scope="session")
def documents() -> list[str]:
    return DOCUMENTS


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("tokenizer")
    return build_bpe_tokenizer(
        str(out), DOCUMENTS, specials=["<|bos|>", "<|eos|>", "<|pad|>"]
    )


@pytest.fixture()
def jsonl_path(tmp_path) -> str:
    path = tmp_path / "corpus.jsonl"
    with open(path, "w") as f:
        for text in DOCUMENTS:
            f.write(json.dumps({"text": text}) + "\n")
    return str(path)
