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


SFT_RECIPE = os.path.join(os.path.dirname(TINY_RECIPE), "sft-qlora-e2b.yaml")


class TestPreflightSft:
    """An SFT recipe reads the prepared F1 dataset, not token shards.

    The reference SFT recipes set data.source to the same directory as
    sft.dataset. Until 2026-09-19 preflight verified that directory as a shard
    directory and died in verify_shards on a KeyError (an SFT manifest has no
    dtype or shards keys), so the one gate meant to run before a fine-tune
    could not pass on a prepared dataset. FT.recipe-gates in the fine-tuning runbook is
    the workstation run of the same contract.
    """

    @pytest.fixture
    def sft_staged(self, tmp_path, monkeypatch):
        """(dataset_dir, tokenizer_dir) for the SFT recipe.

        The dataset is prepared for real through prepare_sft with the chat
        tokenizer test_sft builds, so the manifest kind, template version and
        file hashes are the trainer's own.
        """
        import json

        from support import build_bpe_tokenizer
        from test_sft import CASES, CHAT_TEXTS, CONTROL_TOKENS

        from training.data.sft import prepare_sft

        tokenizer_dir = build_bpe_tokenizer(
            str(tmp_path / "chat_tokenizer"), CHAT_TEXTS, specials=CONTROL_TOKENS
        )
        source = tmp_path / "chat.jsonl"
        with open(source, "w") as f:
            for case in CASES.values():
                f.write(json.dumps({"messages": case}) + "\n")
        dataset_dir = str(tmp_path / "sft")
        prepare_sft(str(source), dataset_dir, tokenizer_dir, "gemma4", holdout=0.2)
        return dataset_dir, tokenizer_dir

    @pytest.fixture
    def base_stubbed(self, monkeypatch):
        """Base identity stubbed, for the cases that override model.hf.id to
        the chat tokenizer's directory: it has no config.json to resolve."""
        import training.export.preflight as export_preflight

        monkeypatch.setattr(
            export_preflight, "resolve_base_identity", lambda *a, **k: None
        )

    def argv(self, dataset_dir: str, tokenizer_dir: str) -> list[str]:
        # No tokenizer override: the recipe carries no tokenizer block, and
        # the base directory (here the chat tokenizer's) is what the hash
        # check reads (KB-80). The base identity itself is stubbed.
        return [
            "preflight",
            "--config",
            SFT_RECIPE,
            f"sft.dataset={dataset_dir}",
            f"data.source={dataset_dir}",
            f"model.hf.id={tokenizer_dir}",
        ]

    def test_prepared_dataset_passes(self, sft_staged, base_stubbed, capsys):
        dataset_dir, tokenizer_dir = sft_staged
        rc = run_cli(self.argv(dataset_dir, tokenizer_dir))
        err = capsys.readouterr().err
        assert rc == 0, err
        assert "[OK] All referenced assets are present and verified" in err
        assert "shard" not in err
        assert "tokenizer.path" not in err

    def test_missing_base_is_one_problem_not_two(self, sft_staged, monkeypatch, capsys):
        """KB-80: an absent base is reported as the model, not also as its tokenizer."""
        import training.export.preflight as export_preflight
        from training.export.errors import ExportError

        dataset_dir, tokenizer_dir = sft_staged

        def missing(*a, **k):
            raise ExportError("config.json not found locally")

        monkeypatch.setattr(export_preflight, "resolve_base_identity", missing)
        rc = run_cli(self.argv(dataset_dir, tokenizer_dir))
        err = capsys.readouterr().err
        assert rc == 1
        assert "Missing or unusable assets: 1" in err
        assert "- model.hf: config.json not found locally" in err
        assert "tokenizer.path" not in err
        assert err.count("Command:") == 1
        assert "make train-download" in err

    def test_explicit_tokenizer_path_is_still_checked(
        self, sft_staged, base_stubbed, capsys
    ):
        # validate's hash check reads data.source and skips only when that
        # directory has no manifest, so the test names one that exists on no
        # host: the recipe's own is prepared on the workstation, where leaving
        # it in place made validate load the tokenizer first and report
        # `cannot load` before preflight's presence check ran. Preflight
        # itself skips the shard checks for an SFT recipe, so the override
        # adds no line of its own.
        dataset_dir, tokenizer_dir = sft_staged
        rc = run_cli(
            [
                "preflight",
                "--config",
                SFT_RECIPE,
                f"sft.dataset={dataset_dir}",
                "data.source=/nonexistent/datasets/unprepared",
                f"model.hf.id={tokenizer_dir}",
                "tokenizer.path=/nonexistent/tokenizers/pinned",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "tokenizer.path: not found: /nonexistent/tokenizers/pinned" in err
        assert "make train-tokenizer" in err

    def test_committed_id_resolves_to_the_staged_directory(
        self, sft_staged, tmp_path, monkeypatch, capsys
    ):
        """KB-79: the recipe keeps its hub id and preflight reads the directory
        train-download staged, with no override and no stub: the real
        resolve_base_identity reads the staged config.json and the tokenizer
        hash check reads the tokenizer.json beside it."""
        import shutil

        from training.models import staged

        dataset_dir, tokenizer_dir = sft_staged
        root = tmp_path / "models" / "hf"
        base = root / "gemma-4-E2B"
        shutil.copytree(tokenizer_dir, base)
        (base / "config.json").write_text("{}")
        monkeypatch.setattr(staged, "STAGED_ROOTS", (str(root),))
        rc = run_cli(
            [
                "preflight",
                "--config",
                SFT_RECIPE,
                f"sft.dataset={dataset_dir}",
                f"data.source={dataset_dir}",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 0, err
        assert "[OK] All referenced assets are present and verified" in err
        assert "tokenizer.path" not in err

    def test_unstaged_id_is_the_model_line_not_the_tokenizer(
        self, sft_staged, monkeypatch, capsys
    ):
        """FT.recipe-gates 2026-09-19: with the dataset on disk, validate's hash
        check fetched the hub id and failed first, hiding preflight's own
        report. An unstaged id is now preflight's line alone."""
        import training.export.preflight as export_preflight
        from training.export.errors import ExportError
        from training.models import staged

        monkeypatch.setattr(staged, "STAGED_ROOTS", ())

        # The offline container's cache miss; the sandbox has no transformers
        # to produce it, and the hub branch is not what this case tests.
        def cache_miss(model, **kwargs):
            assert model == "google/gemma-4-E2B"
            raise ExportError(f"Could not resolve base model config: {model}")

        monkeypatch.setattr(export_preflight, "resolve_base_identity", cache_miss)
        dataset_dir, _ = sft_staged
        rc = run_cli(
            [
                "preflight",
                "--config",
                SFT_RECIPE,
                f"sft.dataset={dataset_dir}",
                f"data.source={dataset_dir}",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1, err
        assert "Recipe validation failed" not in err
        assert "Missing or unusable assets: 1" in err
        assert (
            "- model.hf: Could not resolve base model config: google/gemma-4-E2B" in err
        )
        assert "tokenizer.path" not in err
        assert "make train-download" in err

    def test_missing_dataset_names_it_and_prepare_sft(
        self, sft_staged, base_stubbed, capsys
    ):
        _, tokenizer_dir = sft_staged
        missing = "/nonexistent/datasets/sft/ultrachat-e2b"
        rc = run_cli(self.argv(missing, tokenizer_dir))
        err = capsys.readouterr().err
        assert rc == 1
        assert "Missing or unusable assets: 1" in err
        assert f"sft.dataset: dataset directory not found: {missing}" in err
        assert "make prepare-sft" in err
        assert "prepare-corpus" not in err

    def test_corrupted_dataset_is_reported_not_raised(self, sft_staged, capsys):
        dataset_dir, tokenizer_dir = sft_staged
        from training.data.sft import TRAIN_FILE

        with open(os.path.join(dataset_dir, TRAIN_FILE), "a") as f:
            f.write("{}\n")
        rc = run_cli(self.argv(dataset_dir, tokenizer_dir))
        err = capsys.readouterr().err
        assert rc == 1
        assert "sft.dataset:" in err
        assert "content hash" in err

    def test_sft_manifest_named_as_shards_is_reported_not_raised(
        self, sft_staged, capsys
    ):
        """A scratch recipe pointed at an SFT dataset: the pre-fix crash."""
        dataset_dir, tokenizer_dir = sft_staged
        rc = run_cli(
            [
                "preflight",
                "--config",
                TINY_RECIPE,
                f"data.source={dataset_dir}",
                f"tokenizer.path={tokenizer_dir}",
            ]
        )
        err = capsys.readouterr().err
        assert rc == 1
        assert "data.source: manifest is not a shard manifest" in err


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
