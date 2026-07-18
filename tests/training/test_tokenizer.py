"""Phase 0 D3 tokenizer training tests: special-token ids, reserved block,
reproducibility, round-trip losslessness, overwrite refusal, and hash-chain
integration with the D2 loader. Synthetic corpus only; no network or GPU."""

from __future__ import annotations

import json
import os

import pytest

from tests.training.conftest import DOCUMENTS
from training.data import tokenizer as tokenizer_mod
from training.data.prepare import load_tokenizer

VOCAB = 320  # small but above the 275-id floor (256-byte alphabet + 19 specials)

# Mixed-script and digit-heavy material so round-trip covers multi-byte UTF-8
EXTRA_DOCUMENTS = [
    "pi is 3.14159 and e is 2.71828",
    "snowman ☃ and accents: café naïve über",
    "code: for i in range(10): print(i * 2)",
    "year 2026, port 8002, id 49152",
]
CORPUS = DOCUMENTS + EXTRA_DOCUMENTS


@pytest.fixture(scope="module")
def trained():
    return tokenizer_mod.train(CORPUS, VOCAB)


@pytest.fixture()
def corpus_jsonl(tmp_path) -> str:
    path = tmp_path / "corpus.jsonl"
    with open(path, "w") as f:
        for text in CORPUS:
            f.write(json.dumps({"text": text}) + "\n")
    return str(path)


def write_artifact(tok, out_dir, force=False):
    return tokenizer_mod.write_artifact(
        tok,
        str(out_dir),
        corpus=["corpus.jsonl"],
        text_field="text",
        sampled_bytes=1,
        sampled_documents=len(CORPUS),
        vocab_size=VOCAB,
        metrics={},
        references={},
        force=force,
        command="test",
    )


def test_special_token_ids_fixed(trained):
    assert trained.token_to_id("<|bos|>") == 0
    assert trained.token_to_id("<|eos|>") == 1
    assert trained.token_to_id("<|pad|>") == 2


def test_reserved_block_present(trained):
    for i in range(16):
        assert trained.token_to_id(f"<|reserved_{i}|>") == 3 + i


def test_vocab_size_respected(trained):
    assert trained.get_vocab_size() <= VOCAB


def test_digits_split_individually(trained):
    ids = trained.encode("49152").ids
    assert len(ids) == 5  # one token per digit, never merged


def test_roundtrip_lossless(trained):
    for text in CORPUS:
        assert (
            trained.decode(trained.encode(text).ids, skip_special_tokens=False) == text
        )


def test_vocab_floor_rejected():
    with pytest.raises(ValueError, match="below minimum"):
        tokenizer_mod.train(CORPUS, 100)


def test_reproducible_artifact_hash(tmp_path):
    a = write_artifact(tokenizer_mod.train(CORPUS, VOCAB), tmp_path / "a")
    b = write_artifact(tokenizer_mod.train(CORPUS, VOCAB), tmp_path / "b")
    assert a["tokenizer_hash"] == b["tokenizer_hash"]


def test_overwrite_refused_without_force(trained, tmp_path):
    write_artifact(trained, tmp_path / "art")
    with pytest.raises(FileExistsError, match="--force"):
        write_artifact(trained, tmp_path / "art")
    write_artifact(trained, tmp_path / "art", force=True)


def test_artifact_loads_through_d2_loader(trained, tmp_path):
    manifest = write_artifact(trained, tmp_path / "art")
    loaded = load_tokenizer(str(tmp_path / "art"))
    assert loaded.eos_id == 1  # <|eos|> resolves first
    assert loaded.content_hash == manifest["tokenizer_hash"]  # D2/D5 hash chain holds
    assert loaded.vocab_size == trained.get_vocab_size()


def test_manifest_contents(trained, tmp_path):
    write_artifact(trained, tmp_path / "art")
    with open(tmp_path / "art" / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["special_tokens"]["<|eos|>"] == 1
    assert (
        len([k for k in manifest["special_tokens"] if k.startswith("<|reserved_")])
        == 16
    )
    assert manifest["tokenizers_version"]
    assert manifest["vocab_size"] == trained.get_vocab_size()
    assert manifest["requested_vocab_size"] == VOCAB


def test_manifest_records_actual_vocab_size(tmp_path):
    # KB-02: a corpus too small to reach the requested vocab must not produce
    # a manifest attesting the requested size
    requested = 5000
    trained = tokenizer_mod.train(CORPUS, requested)
    assert trained.get_vocab_size() < requested  # corpus exhausts merges early
    manifest = tokenizer_mod.write_artifact(
        trained,
        str(tmp_path / "art"),
        corpus=["corpus.jsonl"],
        text_field="text",
        sampled_bytes=1,
        sampled_documents=len(CORPUS),
        vocab_size=requested,
        metrics={},
        references={},
        command="test",
    )
    assert manifest["vocab_size"] == trained.get_vocab_size()
    assert manifest["requested_vocab_size"] == requested


def test_evaluate_metrics(trained):
    metrics = tokenizer_mod.evaluate(trained, {"web": CORPUS[:8], "code": CORPUS[8:]})
    for stats in metrics.values():
        assert stats["fertility"] > 0
        assert stats["compression"] > 0
        assert stats["roundtrip_failures"] == 0


def test_evaluate_reference_matches_shape(trained, tmp_path):
    write_artifact(trained, tmp_path / "art")
    reference = load_tokenizer(str(tmp_path / "art"))
    result = tokenizer_mod.evaluate_reference(reference, {"web": CORPUS[:4]})
    assert set(result["web"]) == {"documents", "tokens", "fertility", "compression"}


def test_corpus_sampler_budget_and_heldout(corpus_jsonl):
    budget = sum(len(t.encode("utf-8")) for t in CORPUS[:5])
    sampler = tokenizer_mod.CorpusSampler([corpus_jsonl], "text", budget)
    trained_docs = list(sampler.training_documents())
    assert trained_docs == CORPUS[:5]
    assert sampler.sampled_bytes >= budget
    heldout = sampler.heldout_documents(3)
    assert heldout == CORPUS[5:8]


def test_corpus_sampler_exhaustion(corpus_jsonl):
    sampler = tokenizer_mod.CorpusSampler([corpus_jsonl], "text", None)
    assert list(sampler.training_documents()) == CORPUS
    assert sampler.heldout_documents(3) == []


def test_expand_corpus_sorted_and_missing(tmp_path):
    for name in ("b.jsonl", "a.jsonl"):
        (tmp_path / name).write_text("{}")
    paths = tokenizer_mod.expand_corpus([str(tmp_path / "*.jsonl")])
    assert [os.path.basename(p) for p in paths] == ["a.jsonl", "b.jsonl"]
    with pytest.raises(FileNotFoundError):
        tokenizer_mod.expand_corpus([str(tmp_path / "missing-*.jsonl")])


def test_trained_artifact_feeds_d2_shards(trained, tmp_path, corpus_jsonl):
    """Integration: shards prepared with the artifact carry its manifest hash."""
    from training.data import prepare as prepare_mod

    artifact = write_artifact(trained, tmp_path / "art")
    shard_manifest = prepare_mod.prepare(
        corpus_jsonl, str(tmp_path / "shards"), str(tmp_path / "art")
    )
    assert shard_manifest["tokenizer_hash"] == artifact["tokenizer_hash"]
    assert shard_manifest["eos_id"] == 1
