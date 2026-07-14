"""End-to-end tests for the JSONL tokenize-and-shard path (Phase 0 D2).

HF dataset paths need network and the datasets package; they are exercised by the
host-side acceptance run, not here.
"""


import pytest

from training.data.prepare import iter_jsonl, load_tokenizer, prepare
from training.data.shards import ShardedTokenDataset, load_manifest


def test_load_tokenizer_from_directory(tokenizer_dir):
    tok = load_tokenizer(tokenizer_dir)
    assert tok.eos_id >= 0
    assert tok.vocab_size <= 512
    assert len(tok.content_hash) == 16   # xxhash64 hex
    ids = tok.encode_batch(["the quick brown fox"])
    assert len(ids) == 1 and len(ids[0]) > 0


def test_load_tokenizer_requires_eos(tokenizer_dir, tmp_path):
    with pytest.raises(ValueError, match="EOS"):
        load_tokenizer(tokenizer_dir, eos_token="<|missing|>")


def test_iter_jsonl_missing_field(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"body": "no text field"}\n')
    with pytest.raises(KeyError, match="text"):
        list(iter_jsonl(str(path), "text"))


def test_prepare_jsonl_manifest_and_round_trip(tokenizer_dir, jsonl_path, tmp_path, documents):
    out = str(tmp_path / "shards")
    manifest = prepare(jsonl_path, out, tokenizer_dir,
                       shard_tokens=64, command="test", license_str="cc0")

    assert manifest["total_documents"] == len(documents)
    assert manifest["total_tokens"] == sum(r["tokens"] for r in manifest["shards"])
    assert manifest["dtype"] == "uint16"
    assert manifest["license"] == "cc0"
    assert sum(manifest["doc_length_histogram"].values()) == len(documents)
    assert load_manifest(out)["manifest_version"] == 1

    # Round-trip: decoding the token stream reproduces the source documents
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{tokenizer_dir}/tokenizer.json")
    import numpy as np
    stream = []
    for rec in manifest["shards"]:
        stream.extend(np.fromfile(f"{out}/{rec['file']}", dtype=np.uint16).tolist())
    docs, current = [], []
    for token_id in stream:
        if token_id == manifest["eos_id"]:
            docs.append(tok.decode(current))
            current = []
        else:
            current.append(token_id)
    assert docs == documents


def test_prepare_deterministic(tokenizer_dir, jsonl_path, tmp_path):
    a = prepare(jsonl_path, str(tmp_path / "a"), tokenizer_dir, shard_tokens=64, command="test")
    b = prepare(jsonl_path, str(tmp_path / "b"), tokenizer_dir, shard_tokens=64, command="test")
    assert [r["xxhash64"] for r in a["shards"]] == [r["xxhash64"] for r in b["shards"]]
    assert a["tokenizer_hash"] == b["tokenizer_hash"]


def test_prepare_max_tokens_cap(tokenizer_dir, jsonl_path, tmp_path):
    full = prepare(jsonl_path, str(tmp_path / "full"), tokenizer_dir, command="test")
    capped = prepare(jsonl_path, str(tmp_path / "capped"), tokenizer_dir,
                     max_tokens=20, command="test")
    assert capped["total_tokens"] < full["total_tokens"]
    assert capped["total_tokens"] >= 20   # stops at the document that crosses the cap


def test_prepare_refuses_overwrite(tokenizer_dir, jsonl_path, tmp_path):
    out = str(tmp_path / "shards")
    prepare(jsonl_path, out, tokenizer_dir, command="test")
    with pytest.raises(FileExistsError):
        prepare(jsonl_path, out, tokenizer_dir, command="test")


def test_prepared_shards_load_with_matching_tokenizer_hash(tokenizer_dir, jsonl_path, tmp_path):
    out = str(tmp_path / "shards")
    manifest = prepare(jsonl_path, out, tokenizer_dir, shard_tokens=64, command="test")
    tok = load_tokenizer(tokenizer_dir)
    ds = ShardedTokenDataset(out, window=16, tokenizer_hash=tok.content_hash)
    first = ds.next_window()
    assert first is not None and len(first) == 16
    assert manifest["total_tokens"] - ds.dropped_tokens == len(ds) * 16
