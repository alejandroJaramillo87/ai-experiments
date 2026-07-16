"""D9 RAM-disk staging: cap respected with clean fallback, hash-verified
copies, reuse of a valid staged set, corruption detection."""

import os
import shutil

import numpy as np
import pytest
from omegaconf import OmegaConf
from support import make_shard_dir

from training.data.shards import load_manifest
from training.data.staging import shard_set_bytes, stage_shards


@pytest.fixture()
def shard_dir(tmp_path) -> str:
    rng = np.random.default_rng(3)
    tokens = rng.integers(0, 500, size=2500).tolist()
    return make_shard_dir(tmp_path, tokens, shard_tokens=1000)


def staging_cfg(dir: str, max_gb: float = 1.0):
    return OmegaConf.create({"enabled": True, "dir": dir, "max_gb": max_gb})


def test_stages_and_verifies_under_cap(shard_dir, tmp_path):
    stage = str(tmp_path / "staging")
    os.makedirs(stage)
    result = stage_shards(shard_dir, staging_cfg(stage))
    assert result != shard_dir
    assert result.startswith(stage)
    staged = load_manifest(result)
    for rec in staged["shards"]:
        assert os.path.isfile(os.path.join(result, rec["file"]))


def test_falls_back_when_over_cap(shard_dir, tmp_path, capsys):
    stage = str(tmp_path / "staging")
    os.makedirs(stage)
    tiny_cap = shard_set_bytes(shard_dir, load_manifest(shard_dir)) / 2**30 / 2
    result = stage_shards(shard_dir, staging_cfg(stage, max_gb=tiny_cap))
    assert result == shard_dir
    assert "SKIP" in capsys.readouterr().err


def test_falls_back_when_staging_dir_absent(shard_dir, tmp_path, capsys):
    result = stage_shards(shard_dir, staging_cfg(str(tmp_path / "missing")))
    assert result == shard_dir
    assert "SKIP" in capsys.readouterr().err


def test_reuses_valid_staged_copy(shard_dir, tmp_path, capsys):
    stage = str(tmp_path / "staging")
    os.makedirs(stage)
    first = stage_shards(shard_dir, staging_cfg(stage))
    second = stage_shards(shard_dir, staging_cfg(stage))
    assert first == second
    assert "reusing" in capsys.readouterr().err


def test_post_copy_hash_mismatch_is_an_error_not_a_fallback(
    shard_dir, tmp_path, monkeypatch
):
    # A staging medium that corrupts a shard during the copy must fail loudly,
    # not silently fall back to the source: a truncated copy can never train.
    stage = str(tmp_path / "staging")
    os.makedirs(stage)
    real_copyfile = shutil.copyfile

    def corrupting_copyfile(src, dst):
        real_copyfile(src, dst)
        # Flip the first byte so the freshly staged file cannot hash-match.
        with open(dst, "r+b") as f:
            first = f.read(1)
            f.seek(0)
            f.write(bytes([first[0] ^ 0xFF]))

    monkeypatch.setattr(shutil, "copyfile", corrupting_copyfile)
    with pytest.raises(OSError, match="staging medium corrupted the data"):
        stage_shards(shard_dir, staging_cfg(stage))


def test_restages_after_staged_corruption(shard_dir, tmp_path):
    stage = str(tmp_path / "staging")
    os.makedirs(stage)
    staged = stage_shards(shard_dir, staging_cfg(stage))
    rec = load_manifest(staged)["shards"][0]
    with open(os.path.join(staged, rec["file"]), "r+b") as f:
        f.write(b"\xff\xff")
    # The corrupted copy fails reuse validation and is re-copied cleanly
    result = stage_shards(shard_dir, staging_cfg(stage))
    assert result == staged
    from training.data.shards import hash_file

    assert hash_file(os.path.join(result, rec["file"])) == rec["xxhash64"]
