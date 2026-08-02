"""In-process handler tests for python -m training preflight and download.

Exercises cmd through build_parser() + args.func(args), the same argv path
production takes. Preflight is pure filesystem inspection, so it runs for real
against synthetic shard directories built by support.make_shard_dir; nothing is
faked except the tokenizer load, which would otherwise require a real tokenizer
on disk.

The behavior under test is the contract Prelude C rests on: every asset a recipe
references must be present, every failure is reported rather than only the first,
and a missing asset exits non-zero.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from support import TINY_RECIPE, make_shard_dir

from training import cli


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


@pytest.fixture
def staged(tmp_path, monkeypatch):
    """A tiny-e2e recipe whose shard and tokenizer assets both exist.

    Returns (shard_dir, tokenizer_dir). The tokenizer is a bare directory: the
    content-hash comparison is exercised separately by stubbing load_tokenizer,
    because building a real tokenizer here would test the tokenizers library
    rather than preflight.
    """
    shard_dir = make_shard_dir(tmp_path, np.arange(256, dtype=np.uint16))
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    import training.data.prepare as prepare_mod

    class _Loaded:
        content_hash = "abc123"  # matches make_shard_dir's manifest default

    monkeypatch.setattr(prepare_mod, "load_tokenizer", lambda *a, **k: _Loaded())
    return shard_dir, str(tokenizer_dir)


class TestPreflight:
    def test_all_assets_present_exits_zero(self, staged, capsys):
        shard_dir, tokenizer_dir = staged
        rc = run_cli(
            [
                "preflight",
                "--config",
                TINY_RECIPE,
                f"data.source={shard_dir}",
                f"tokenizer.path={tokenizer_dir}",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 0
        assert "[OK] All referenced assets are present and verified" in err
        # tiny-e2e is a scratch recipe: there is no base model to stage.
        assert "trains from scratch" in err

    def test_missing_shard_dir_exits_nonzero_and_names_it(self, staged, capsys):
        _, tokenizer_dir = staged
        missing = "/nonexistent/shards/tiny-e2e"
        rc = run_cli(
            [
                "preflight",
                "--config",
                TINY_RECIPE,
                f"data.source={missing}",
                f"tokenizer.path={tokenizer_dir}",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "shard directory not found" in err
        assert missing in err

    def test_missing_tokenizer_exits_nonzero(self, staged, capsys):
        shard_dir, _ = staged
        rc = run_cli(
            [
                "preflight",
                "--config",
                TINY_RECIPE,
                f"data.source={shard_dir}",
                "tokenizer.path=/nonexistent/tokenizers/tiny-e2e",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "tokenizer.path: not found" in err

    def test_reports_every_failure_not_just_the_first(self, staged, capsys):
        """The whole point of preflight: one pass lists everything to stage."""
        rc = run_cli(
            [
                "preflight",
                "--config",
                TINY_RECIPE,
                "data.source=/nonexistent/shards",
                "tokenizer.path=/nonexistent/tokenizer",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "Missing or unusable assets: 2" in err
        assert "shard directory not found" in err
        assert "tokenizer.path: not found" in err

    def test_remediation_advice_matches_what_is_missing(self, staged, capsys):
        """A scratch recipe has no model to download; do not suggest it.

        The first container run of this command printed `make train-download`
        for tiny-e2e, which trains from scratch -- naming a step that does not
        exist for that recipe. Advice is derived from the failures now.
        """
        rc = run_cli(
            [
                "preflight",
                "--config",
                TINY_RECIPE,
                "data.source=/nonexistent/shards",
                "tokenizer.path=/nonexistent/tokenizer",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "train-download" not in err
        assert "make train-tokenizer" in err
        assert "make prepare-corpus" in err

    def test_corrupt_shard_is_caught(self, staged, capsys):
        """verify_shards runs here rather than inside the training loop."""
        shard_dir, tokenizer_dir = staged
        shard_files = [f for f in os.listdir(shard_dir) if f.endswith(".bin")]
        assert shard_files, "fixture produced no shard files"
        target = os.path.join(shard_dir, shard_files[0])
        with open(target, "r+b") as f:
            f.write(b"\xff\xff")  # content hash no longer matches the manifest

        rc = run_cli(
            [
                "preflight",
                "--config",
                TINY_RECIPE,
                f"data.source={shard_dir}",
                f"tokenizer.path={tokenizer_dir}",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "content hash" in err

    def test_tokenizer_hash_mismatch_is_caught(self, tmp_path, monkeypatch, capsys):
        """A tokenizer that did not build these shards must not pass.

        The check itself lives in validate() (_check_tokenizer_matches_shards,
        validate.py:725-756), which preflight runs first and returns on. This
        asserts the contract -- non-zero exit naming both hashes -- rather than
        preflight's own wording, because preflight deliberately does not repeat
        a comparison the validation pass already owns.
        """
        shard_dir = make_shard_dir(tmp_path, np.arange(256, dtype=np.uint16))
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()

        import training.data.prepare as prepare_mod

        class _Loaded:
            content_hash = "different-hash"

        monkeypatch.setattr(prepare_mod, "load_tokenizer", lambda *a, **k: _Loaded())

        rc = run_cli(
            [
                "preflight",
                "--config",
                TINY_RECIPE,
                f"data.source={shard_dir}",
                f"tokenizer.path={tokenizer_dir}",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "does not match shard manifest hash" in err
        assert "different-hash" in err  # both hashes named, so the fix is obvious
        assert "abc123" in err


class TestDownload:
    def test_refuses_to_run_when_offline(self, monkeypatch, capsys):
        """Offline is the image default; reaching download means the wrong service."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        rc = run_cli(
            ["download", "--model", "google/gemma-4-E2B", "--revision", "deadbeef"]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "cannot reach the hub" in err
        assert "make train-download" in err

    def test_revision_is_required(self):
        """An unpinned download is the problem this command exists to prevent."""
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["download", "--model", "google/gemma-4-E2B"])
