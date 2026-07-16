"""P2-D4 replay buffer tests: budget subsampling and refusal, RAM-floor
refusal against a mocked psutil, permutation uniformity and determinism,
mixture integration (fraction accounting from source tags, resume draw
equality), the calculator replay term, and the synthetic generator CLI.
Sandbox-only: synthetic shards, no GPU or network."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest
from support import REPO_ROOT, make_source

from training.continual.replay import (
    GIB,
    ReplayBudgetError,
    ReplayBuffer,
    ReplayRamFloorError,
)
from training.data.mixture import MixtureDataset

WINDOW = 8


@pytest.fixture()
def replay_dir(tmp_path):
    """320 tokens -> 40 windows of 8, spread over 5 shards of 64 tokens.
    Window i holds the constant value i so provenance is checkable."""
    tokens = np.repeat(np.arange(40, dtype=np.uint16), WINDOW)
    return make_source(tmp_path, "replay", tokens)


class TestLoading:
    def test_loads_all_windows_when_budget_fits(self, replay_dir):
        buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0)
        assert len(buf) == 40
        values = sorted(int(buf[i][0]) for i in range(len(buf)))
        assert values == list(range(40))

    def test_windows_are_intact(self, replay_dir):
        buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0)
        for i in range(len(buf)):
            window = buf[i]
            assert window.shape == (WINDOW,)
            assert (window == window[0]).all()

    def test_rejects_source_without_a_full_window(self, tmp_path):
        path = make_source(tmp_path, "tiny", [5] * 4)
        with pytest.raises(ValueError, match="no complete window"):
            ReplayBuffer(path, window=WINDOW, seed=0)

    def test_window_mismatch_refused_by_mixture(self, replay_dir, tmp_path):
        buf = ReplayBuffer(replay_dir, window=4, seed=0)
        domain = make_source(tmp_path, "domain", [100] * 320)
        with pytest.raises(ValueError, match="window"):
            MixtureDataset(
                [("domain", domain, 0.8), ("replay", buf, 0.2)], window=WINDOW, seed=0
            )


class TestBudget:
    def test_subsamples_to_budget(self, replay_dir, tmp_path):
        # 10 windows of 8 uint16 tokens = 160 bytes
        buf = ReplayBuffer(
            replay_dir, window=WINDOW, max_bytes=160, seed=0, run_dir=str(tmp_path)
        )
        assert len(buf) == 10
        values = {int(buf[i][0]) for i in range(len(buf))}
        assert values < set(range(40))  # a strict subset of the source

    def test_subsample_is_deterministic(self, replay_dir):
        first = ReplayBuffer(replay_dir, window=WINDOW, max_bytes=160, seed=0)
        second = ReplayBuffer(replay_dir, window=WINDOW, max_bytes=160, seed=0)
        for i in range(len(first)):
            assert (first[i] == second[i]).all()

    def test_refuses_budget_below_one_window(self, replay_dir):
        with pytest.raises(ReplayBudgetError, match="no complete window"):
            ReplayBuffer(replay_dir, window=WINDOW, max_bytes=8, seed=0)

    def test_subsample_manifest_written(self, replay_dir, tmp_path):
        ReplayBuffer(
            replay_dir,
            window=WINDOW,
            max_bytes=160,
            seed=7,
            run_dir=str(tmp_path),
            name="web",
        )
        with open(tmp_path / "replay_manifest_web.json") as f:
            record = json.load(f)
        assert record["loaded_windows"] == 10
        assert record["source_windows"] == 40
        assert record["subsample_seed"] == 7
        windows = np.load(tmp_path / record["windows_file"])
        assert len(windows) == 10
        assert (np.sort(windows) == windows).all()

    def test_full_load_manifest_written(self, replay_dir, tmp_path):
        ReplayBuffer(replay_dir, window=WINDOW, seed=0, run_dir=str(tmp_path))
        with open(tmp_path / "replay_manifest_replay.json") as f:
            record = json.load(f)
        assert record["loaded_windows"] == 40
        assert record["subsample_seed"] is None
        assert record["windows_file"] is None


class TestRamFloor:
    def _mock_available_ram(self, monkeypatch, available: int) -> None:
        monkeypatch.setattr(
            "training.continual.replay._available_ram_bytes", lambda: available
        )

    def test_refuses_load_below_floor(self, replay_dir, monkeypatch):
        # Load needs 640 bytes; 1GB available minus that is under a 2GB floor.
        self._mock_available_ram(monkeypatch, available=1 * GIB)
        with pytest.raises(ReplayRamFloorError, match="watchdog RAM floor"):
            ReplayBuffer(replay_dir, window=WINDOW, seed=0, ram_floor_gb=2.0)

    def test_loads_above_floor(self, replay_dir, monkeypatch):
        self._mock_available_ram(monkeypatch, available=4 * GIB)
        buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0, ram_floor_gb=2.0)
        assert len(buf) == 40

    def test_missing_psutil_skips_check(self, replay_dir, monkeypatch):
        monkeypatch.setattr(
            "training.continual.replay._available_ram_bytes", lambda: None
        )
        buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0, ram_floor_gb=1e6)
        assert len(buf) == 40


class TestPermutation:
    def test_order_is_shuffled(self, replay_dir):
        buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0)
        order = [int(buf[i][0]) for i in range(len(buf))]
        assert order != sorted(order)

    def test_order_is_deterministic_across_instances(self, replay_dir):
        first = ReplayBuffer(replay_dir, window=WINDOW, seed=3)
        second = ReplayBuffer(replay_dir, window=WINDOW, seed=3)
        assert all((first[i] == second[i]).all() for i in range(len(first)))

    def test_out_of_range_raises(self, replay_dir):
        buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0)
        with pytest.raises(IndexError):
            buf[40]
        with pytest.raises(IndexError):
            buf[-1]


class TestMixtureIntegration:
    @pytest.fixture()
    def mixed(self, tmp_path, replay_dir):
        domain = make_source(tmp_path, "domain", [100] * 640)
        buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0)
        return MixtureDataset(
            [("domain", domain, 0.8), ("replay", buf, 0.2)], window=WINDOW, seed=0
        )

    def test_replay_fraction_within_one_percent(self, mixed):
        draws = 10_000
        replay_draws = sum(mixed.source_at(i) == 1 for i in range(draws))
        assert abs(replay_draws / draws - 0.2) < 0.01

    def test_replay_windows_come_from_the_buffer(self, mixed):
        for i in range(500):
            window, source = mixed[i]
            if source == 1:
                assert int(window[0]) < 40  # buffer values are 0..39
            else:
                assert int(window[0]) == 100

    def test_resume_reproduces_identical_draws(self, tmp_path, replay_dir):
        domain = make_source(tmp_path, "domain2", [100] * 640)

        def build():
            buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0)
            return MixtureDataset(
                [("domain", domain, 0.8), ("replay", buf, 0.2)], window=WINDOW, seed=0
            )

        straight = build()
        expected = [straight[i] for i in range(200)]
        resumed = build()  # fresh instance reading from the resume point only
        for i in range(100, 200):
            window, source = resumed[i]
            assert source == expected[i][1]
            assert (window == expected[i][0]).all()

    def test_mixture_rejects_hash_mismatch_with_buffer(self, tmp_path, replay_dir):
        from training.data.shards import TokenizerMismatchError

        other = make_source(tmp_path, "other", [100] * 320, tokenizer_hash="different")
        buf = ReplayBuffer(replay_dir, window=WINDOW, seed=0)
        with pytest.raises(TokenizerMismatchError):
            MixtureDataset(
                [("other", other, 0.8), ("replay", buf, 0.2)], window=WINDOW, seed=0
            )


class TestValidate:
    def recipe(
        self,
        tmp_path,
        tokenizer_dir,
        mixture_sources,
        replay_sources,
        extra_replay="",
        overrides=(),
    ):
        from training.recipes import load_recipe

        def lines(sources):
            return "\n".join(
                f"    - {{name: {name}, path: {path}, weight: {weight}}}"
                for name, path, weight in sources
            )

        text = f"""
run: {{name: replay-test}}
model:
  scratch: {{layers: 1, hidden: 32, heads: 2, kv_heads: 1, vocab: 512}}
data: {{seq_len: {WINDOW}, micro_batch: 1}}
mixture:
  enabled: true
  sources:
{lines(mixture_sources)}
replay:
  enabled: true
  fraction: 0.2
  sources:
{lines(replay_sources)}
{extra_replay}
tokenizer: {{path: {tokenizer_dir}}}
optimizer: {{lr: 1.0e-3}}
schedule: {{total_steps: 4}}
"""
        path = tmp_path / "recipe.yaml"
        path.write_text(text)
        return load_recipe(str(path), overrides=overrides)

    def matched_sources(self, tmp_path, tokenizer_dir):
        from training.data.prepare import load_tokenizer

        tok_hash = load_tokenizer(tokenizer_dir).content_hash
        mixture = [
            (
                "domain",
                make_source(tmp_path, "domain", [1] * 64, tokenizer_hash=tok_hash),
                1.0,
            )
        ]
        replay = [
            (
                "replay_web",
                make_source(tmp_path, "replay_web", [2] * 64, tokenizer_hash=tok_hash),
                1.0,
            )
        ]
        return mixture, replay

    def test_valid_replay_recipe_passes(self, tmp_path, tokenizer_dir):
        from training.recipes import validate

        mixture, replay = self.matched_sources(tmp_path, tokenizer_dir)
        validate(self.recipe(tmp_path, tokenizer_dir, mixture, replay))

    def test_requires_mixture_enabled(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        mixture, replay = self.matched_sources(tmp_path, tokenizer_dir)
        recipe = self.recipe(
            tmp_path,
            tokenizer_dir,
            mixture,
            replay,
            overrides=["mixture.enabled=false", f"data.source={mixture[0][1]}"],
        )
        with pytest.raises(RecipeValidationError, match="requires mixture.enabled"):
            validate(recipe)

    def test_fraction_out_of_range_is_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        mixture, replay = self.matched_sources(tmp_path, tokenizer_dir)
        for bad in ("0.0", "1.0", "1.5"):
            recipe = self.recipe(
                tmp_path,
                tokenizer_dir,
                mixture,
                replay,
                overrides=[f"replay.fraction={bad}"],
            )
            with pytest.raises(RecipeValidationError, match="replay.fraction"):
                validate(recipe)

    def test_max_gb_must_be_positive(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        mixture, replay = self.matched_sources(tmp_path, tokenizer_dir)
        recipe = self.recipe(
            tmp_path, tokenizer_dir, mixture, replay, overrides=["replay.max_gb=-1"]
        )
        with pytest.raises(RecipeValidationError, match="replay.max_gb"):
            validate(recipe)

    def test_max_gb_beyond_host_budget_is_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        mixture, replay = self.matched_sources(tmp_path, tokenizer_dir)
        # 128GB total minus the default 8GB floor leaves 120GB.
        recipe = self.recipe(
            tmp_path, tokenizer_dir, mixture, replay, overrides=["replay.max_gb=121"]
        )
        with pytest.raises(RecipeValidationError, match="host budget"):
            validate(recipe)

    def test_empty_sources_are_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate
        from training.recipes.loader import load_recipe

        mixture, _ = self.matched_sources(tmp_path, tokenizer_dir)
        text = f"""
run: {{name: replay-test}}
model:
  scratch: {{layers: 1, hidden: 32, heads: 2, kv_heads: 1, vocab: 512}}
data: {{seq_len: {WINDOW}, micro_batch: 1}}
mixture:
  enabled: true
  sources:
    - {{name: domain, path: {mixture[0][1]}, weight: 1.0}}
replay: {{enabled: true, fraction: 0.2}}
tokenizer: {{path: {tokenizer_dir}}}
optimizer: {{lr: 1.0e-3}}
schedule: {{total_steps: 4}}
"""
        path = tmp_path / "recipe.yaml"
        path.write_text(text)
        with pytest.raises(RecipeValidationError, match="replay.sources"):
            validate(load_recipe(str(path)))

    def test_name_collision_with_mixture_is_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        mixture, replay = self.matched_sources(tmp_path, tokenizer_dir)
        replay = [("domain", replay[0][1], 1.0)]  # collides with the mixture name
        recipe = self.recipe(tmp_path, tokenizer_dir, mixture, replay)
        with pytest.raises(RecipeValidationError, match="unique across"):
            validate(recipe)

    def test_non_positive_weight_is_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        mixture, replay = self.matched_sources(tmp_path, tokenizer_dir)
        replay = [(replay[0][0], replay[0][1], 0.0)]
        recipe = self.recipe(tmp_path, tokenizer_dir, mixture, replay)
        with pytest.raises(RecipeValidationError, match="weight"):
            validate(recipe)

    def test_hash_mismatch_with_mixture_is_refused(self, tmp_path, tokenizer_dir):
        from training.recipes import RecipeValidationError, validate

        mixture, _ = self.matched_sources(tmp_path, tokenizer_dir)
        replay = [
            (
                "replay_web",
                make_source(
                    tmp_path, "replay_bad", [2] * 64, tokenizer_hash="different"
                ),
                1.0,
            )
        ]
        recipe = self.recipe(tmp_path, tokenizer_dir, mixture, replay)
        with pytest.raises(RecipeValidationError, match="tokenized identically"):
            validate(recipe)


class TestBudgetResolution:
    def tiny_recipe(self, overrides=()):
        from support import reference_recipe

        return reference_recipe("tiny-e2e.yaml", list(overrides))

    def test_explicit_max_gb_is_authoritative(self):
        from training.continual.replay import resolve_max_bytes

        recipe = self.tiny_recipe(["replay.enabled=true", "replay.max_gb=4"])
        assert resolve_max_bytes(recipe) == 4 * GIB

    def test_unset_max_gb_derives_calculator_headroom(self):
        from training.continual.replay import resolve_max_bytes

        # tiny-e2e: no offload, so headroom = 128GB - watchdog floor (8GB).
        recipe = self.tiny_recipe(["replay.enabled=true"])
        assert resolve_max_bytes(recipe) == (128 - 8) * GIB

    def test_host_ram_bytes_includes_explicit_replay_budget(self):
        from training.calculator import host_ram_bytes, spec_from_recipe

        recipe = self.tiny_recipe(["replay.enabled=true", "replay.max_gb=4"])
        assert host_ram_bytes(spec_from_recipe(recipe), recipe) == 4 * GIB

    def test_host_ram_bytes_ignores_derived_budget(self):
        # Unset max_gb derives FROM the headroom; counting it here would be
        # circular (resolved P2-D4 decision 2).
        from training.calculator import host_ram_bytes, spec_from_recipe

        recipe = self.tiny_recipe(["replay.enabled=true"])
        assert host_ram_bytes(spec_from_recipe(recipe), recipe) == 0


class TestLoopExpansion:
    def test_replay_block_expands_into_mixture_sources(self, tmp_path, replay_dir):
        from omegaconf import OmegaConf

        from training.loop import _sources_with_replay
        from training.recipes.schema import Recipe

        domain = make_source(tmp_path, "domain", [100] * 640)
        recipe = OmegaConf.merge(
            OmegaConf.structured(Recipe),
            OmegaConf.create(
                {
                    "run": {"name": "expand-test"},
                    "data": {"seq_len": WINDOW},
                    "mixture": {
                        "enabled": True,
                        "sources": [{"name": "domain", "path": domain, "weight": 1.0}],
                    },
                    "replay": {
                        "enabled": True,
                        "fraction": 0.25,
                        "max_gb": 1.0,
                        "sources": [
                            {"name": "replay_web", "path": replay_dir, "weight": 3.0},
                            {"name": "replay_code", "path": replay_dir, "weight": 1.0},
                        ],
                    },
                    "watchdog": {"ram_floor_gb": 0.001},
                }
            ),
        )
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        sources = _sources_with_replay(recipe, [("domain", domain, 1.0)], str(run_dir))

        by_name = {name: (src, weight) for name, src, weight in sources}
        assert by_name["domain"][1] == pytest.approx(0.75)
        assert by_name["replay_web"][1] == pytest.approx(0.25 * 0.75)
        assert by_name["replay_code"][1] == pytest.approx(0.25 * 0.25)
        assert isinstance(by_name["replay_web"][0], ReplayBuffer)
        assert (run_dir / "replay_manifest_replay_web.json").is_file()
        assert (run_dir / "replay_manifest_replay_code.json").is_file()

        mixed = MixtureDataset(sources, window=WINDOW, seed=0)
        draws = 10_000
        replay_draws = sum(mixed.source_at(i) in (1, 2) for i in range(draws))
        assert abs(replay_draws / draws - 0.25) < 0.01


class TestGeneratorScript:
    ARGV = [sys.executable, "-m", "training", "make-replay-synthetic"]

    def test_generates_loadable_shard_set(self, tmp_path):
        out = str(tmp_path / "synth")
        result = subprocess.run(
            self.ARGV
            + [
                "--out",
                out,
                "--tokens",
                "4096",
                "--vocab",
                "512",
                "--seed",
                "1",
                "--shard-tokens",
                "1024",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        buf = ReplayBuffer(out, window=WINDOW, seed=0)
        assert len(buf) == 4096 // WINDOW
        assert buf.manifest["tokenizer_hash"] == "synthetic"

    def test_rejects_zero_tokens(self, tmp_path):
        result = subprocess.run(
            self.ARGV + ["--out", str(tmp_path / "x"), "--tokens", "0"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert result.returncode == 2
