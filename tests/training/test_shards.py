"""Unit tests for the shard writer, manifest verification, and mmap reader (Phase 0 D2)."""

import numpy as np
import pytest
from support import make_shard_dir

from training.data.shards import (
    ShardedTokenDataset,
    ShardIntegrityError,
    ShardWriter,
    TokenizerMismatchError,
    dtype_for_vocab,
    hash_file,
    length_bucket,
    load_manifest,
    verify_shards,
)


def make_shards(tmp_path, tokens, shard_tokens=10, dtype=np.uint16, tokenizer_hash="abc123"):
    return make_shard_dir(tmp_path, tokens, shard_tokens=shard_tokens,
                          dtype=dtype, tokenizer_hash=tokenizer_hash)


def test_dtype_selection():
    assert dtype_for_vocab(65_535) == np.dtype(np.uint16)
    assert dtype_for_vocab(65_536) == np.dtype(np.uint32)


def test_length_buckets():
    assert length_bucket(1) == "1-1"
    assert length_bucket(5) == "4-7"
    assert length_bucket(1024) == "1024-2047"


def test_writer_splits_shards_and_accounts_tokens(tmp_path):
    tokens = list(range(25))
    out = make_shards(tmp_path, tokens, shard_tokens=10)
    manifest = load_manifest(out)
    assert [rec["tokens"] for rec in manifest["shards"]] == [10, 10, 5]
    assert sum(rec["tokens"] for rec in manifest["shards"]) == manifest["total_tokens"]
    verify_shards(out, manifest)


def test_shard_bytes_round_trip(tmp_path):
    tokens = list(range(25))
    out = make_shards(tmp_path, tokens, shard_tokens=10)
    manifest = load_manifest(out)
    read_back = []
    for rec in manifest["shards"]:
        read_back.extend(np.fromfile(f"{out}/{rec['file']}", dtype=np.uint16))
    assert read_back == tokens


def test_uint32_shard_round_trip(tmp_path):
    # Gemma 4's 262,144-token vocab makes uint32 the production dtype for
    # every real shard; pin the write/read/verify path above the uint16 range.
    tokens = [0, 65_536, 200_000, 262_143]
    out = make_shards(tmp_path, tokens, shard_tokens=10, dtype=np.uint32)
    manifest = load_manifest(out)
    assert manifest["dtype"] == "uint32"
    verify_shards(out, manifest)
    read_back = []
    for rec in manifest["shards"]:
        read_back.extend(np.fromfile(f"{out}/{rec['file']}", dtype=np.uint32))
    assert read_back == tokens
    ds = ShardedTokenDataset(out, window=4)
    assert ds[0].tolist() == tokens


@pytest.mark.parametrize("bad", [
    [70_000],                                # list above the uint16 range
    np.array([70_000], dtype=np.int64),      # int64 ndarray would wrap silently
    np.array([-1], dtype=np.int64),          # negative ids are never valid
], ids=["list", "int64-array", "negative"])
def test_writer_rejects_out_of_range_ids(tmp_path, bad):
    writer = ShardWriter(str(tmp_path / "shards"), np.uint16, shard_tokens=10)
    with pytest.raises(ValueError, match="out of range"):
        writer.write_tokens(bad)


def test_writer_accepts_dtype_boundary_ids(tmp_path):
    out = make_shards(tmp_path, [0, 65_535], shard_tokens=10)
    manifest = load_manifest(out)
    verify_shards(out, manifest)


def test_writer_deterministic(tmp_path):
    tokens = list(range(1_000))
    a = load_manifest(make_shards(tmp_path / "a", tokens))
    b = load_manifest(make_shards(tmp_path / "b", tokens))
    assert [r["xxhash64"] for r in a["shards"]] == [r["xxhash64"] for r in b["shards"]]


def test_verify_detects_corruption(tmp_path):
    out = make_shards(tmp_path, list(range(25)))
    manifest = load_manifest(out)
    shard = f"{out}/{manifest['shards'][0]['file']}"
    data = bytearray(open(shard, "rb").read())
    data[0] ^= 0xFF
    open(shard, "wb").write(data)
    with pytest.raises(ShardIntegrityError, match="content hash"):
        verify_shards(out, manifest)


def test_verify_detects_truncation(tmp_path):
    out = make_shards(tmp_path, list(range(25)))
    manifest = load_manifest(out)
    shard = f"{out}/{manifest['shards'][0]['file']}"
    open(shard, "ab").write(b"\x00\x00")
    with pytest.raises(ShardIntegrityError, match="size"):
        verify_shards(out, manifest)


def test_reader_windows_do_not_cross_shards(tmp_path):
    # 3 shards of 10, 10, 5 tokens; window 4 -> 2 + 2 + 1 windows, 2 + 2 + 1 dropped tokens
    out = make_shards(tmp_path, list(range(25)), shard_tokens=10)
    ds = ShardedTokenDataset(out, window=4)
    assert len(ds) == 5
    assert ds.dropped_tokens == 5
    windows = [w.tolist() for w in ds]
    assert windows[0] == [0, 1, 2, 3]
    assert windows[1] == [4, 5, 6, 7]        # tokens 8, 9 dropped at shard 0 tail
    assert windows[2] == [10, 11, 12, 13]
    assert windows[4] == [20, 21, 22, 23]    # token 24 dropped at shard 2 tail
    assert ds[3].tolist() == [14, 15, 16, 17]


def test_reader_cursor_save_restore(tmp_path):
    out = make_shards(tmp_path, list(range(25)), shard_tokens=10)
    ds = ShardedTokenDataset(out, window=4)
    ds.next_window()
    ds.next_window()
    ds.next_window()
    state = ds.state_dict()
    expected = ds.next_window()

    restored = ShardedTokenDataset(out, window=4)
    restored.load_state_dict(state)
    assert restored.next_window().tolist() == expected.tolist()


def test_reader_cursor_rejects_window_mismatch(tmp_path):
    out = make_shards(tmp_path, list(range(25)), shard_tokens=10)
    ds = ShardedTokenDataset(out, window=4)
    with pytest.raises(ValueError, match="window"):
        ds.load_state_dict({"shard": 0, "offset": 0, "window": 8})


def test_reader_exhausts_then_returns_none(tmp_path):
    out = make_shards(tmp_path, list(range(25)), shard_tokens=10)
    ds = ShardedTokenDataset(out, window=4)
    assert len(list(ds)) == 5
    assert ds.next_window() is None


def test_reader_rejects_tokenizer_mismatch(tmp_path):
    out = make_shards(tmp_path, list(range(25)), tokenizer_hash="expected")
    with pytest.raises(TokenizerMismatchError):
        ShardedTokenDataset(out, window=4, tokenizer_hash="other")
    ShardedTokenDataset(out, window=4, tokenizer_hash="expected")


def test_hash_file_streams(tmp_path):
    path = tmp_path / "blob"
    path.write_bytes(b"x" * 100)
    import xxhash
    assert hash_file(str(path)) == xxhash.xxh64(b"x" * 100).hexdigest()
