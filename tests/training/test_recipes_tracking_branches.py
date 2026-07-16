"""Branch coverage for recipe validation and tracking-sink construction.

Covers the cross-field validate() arms not exercised by test_recipes.py
(schedule bounds, sft/distill artifact checks, organism/data/staging/watchdog
domains, mixture and replay path checks, tokenizer-load failures) and the
Tracker sink constructors in training/tracking.py. The tensorboard, wandb,
and mlflow libraries are absent in the sandbox, so sink construction runs
against stub modules injected into sys.modules; the import-failure arms use a
None sys.modules entry, which makes the lazy import raise ImportError.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
from support import REFERENCE_DIR, make_shard_dir, make_source

from training.recipes import RecipeValidationError, load_recipe, validate
from training.recipes.schema import TrackingConfig
from training.tracking import Tracker


def tiny(overrides):
    return load_recipe(os.path.join(REFERENCE_DIR, "tiny-e2e.yaml"), overrides)


def sft(overrides):
    return load_recipe(os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml"), overrides)


def assert_problem(recipe, fragment: str) -> None:
    with pytest.raises(RecipeValidationError) as excinfo:
        validate(recipe)
    assert any(fragment in p for p in excinfo.value.problems), (
        f"no problem containing {fragment!r} in {excinfo.value.problems}"
    )


class TestScheduleBounds:
    def test_negative_warmup_steps(self):
        assert_problem(
            tiny(["schedule.warmup_steps=-1"]),
            "schedule.warmup_steps: must be >= 0, got -1",
        )

    def test_nonpositive_total_steps(self):
        assert_problem(
            tiny(["schedule.total_steps=0"]),
            "schedule.total_steps: must be positive, got 0",
        )

    def test_wsd_requires_total_steps(self, tmp_path):
        # OmegaConf.merge never overrides a set value with MISSING, so the
        # recipe must omit total_steps in the file rather than via a dotlist.
        path = tmp_path / "wsd.yaml"
        path.write_text("""
run: {name: wsd-branch-test}
model:
  scratch: {layers: 1, hidden: 32, heads: 2, kv_heads: 1, vocab: 512}
data: {source: /nonexistent/shards, seq_len: 64, micro_batch: 1}
tokenizer: {path: /nonexistent/tok}
optimizer: {lr: 1.0e-3}
schedule: {type: wsd}
""")
        assert_problem(
            load_recipe(str(path)), "schedule.total_steps: required when type is wsd"
        )


class TestPeftSftDistillBranches:
    def test_nonpositive_peft_alpha(self):
        assert_problem(tiny(["peft.alpha=0"]), "peft.alpha: must be positive, got 0")

    def test_sft_dataset_dir_without_manifest(self, tmp_path):
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        assert_problem(
            sft([f"sft.dataset={dataset}"]),
            f"sft.dataset: {dataset} has no manifest.json",
        )

    def test_distill_teacher_quantize_domain(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "distill-logit-gemma-e2b.yaml"),
            ["distill.teacher_quantize=int8"],
        )
        assert_problem(recipe, "distill.teacher_quantize: 'int8' not in")

    def test_distill_top_k_minimum(self):
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "distill-logit-gemma-e2b.yaml"),
            ["distill.top_k=0"],
        )
        assert_problem(recipe, "distill.top_k: must be >= 1, got 0")

    def test_response_manifest_with_invalid_json_rejected(self, tmp_path):
        # An unparseable manifest must be treated like a manifest without a
        # teacher block, not crash validation with a JSONDecodeError.
        manifest = tmp_path / "manifest.json"
        manifest.write_text("{not json")
        recipe = load_recipe(
            os.path.join(REFERENCE_DIR, "distill-response-gemma-e2b.yaml"),
            [f"distill.teacher_manifest={manifest}"],
        )
        assert_problem(
            recipe, f"distill.teacher_manifest: {manifest} is not an F4 manifest"
        )


class TestOrganismBounds:
    # prompts_dir/out_dir are provided so the only reported problem is the
    # bound under test.
    BASE = [
        "organism.enabled=true",
        "organism.prompts_dir=prompts",
        "organism.out_dir=out",
    ]

    @pytest.mark.parametrize(
        "override,fragment",
        [
            ("organism.elicitations=0", "organism.elicitations: must be >= 1, got 0"),
            (
                "organism.samples_per_prompt=0",
                "organism.samples_per_prompt: must be >= 1, got 0",
            ),
            (
                "organism.hint_rate_min=0.0",
                "organism.hint_rate_min: must be in (0, 1], got 0.0",
            ),
            (
                "organism.recall_min=1.5",
                "organism.recall_min: must be in (0, 1], got 1.5",
            ),
            (
                "organism.judge_score_min=11",
                "organism.judge_score_min: must be in [0, 10], got 11",
            ),
            (
                "organism.locality_epsilon=0",
                "organism.locality_epsilon: must be positive, got 0.0",
            ),
        ],
    )
    def test_bound_reported(self, override, fragment):
        assert_problem(sft(self.BASE + [override]), fragment)


class TestDataLoaderBounds:
    def test_negative_workers(self):
        assert_problem(tiny(["data.workers=-1"]), "data.workers: must be >= 0, got -1")

    def test_prefetch_factor_minimum(self):
        assert_problem(
            tiny(["data.prefetch_factor=0"]),
            "data.prefetch_factor: must be >= 1, got 0",
        )


def mixture_recipe(tmp_path, source_lines: str, replay_lines: str = ""):
    """Minimal mixture recipe; replay_lines adds an enabled replay block."""
    replay = ""
    if replay_lines:
        replay = (
            f"replay:\n  enabled: true\n  fraction: 0.2\n  sources:\n{replay_lines}\n"
        )
    text = f"""
run: {{name: mix-branch-test}}
model:
  scratch: {{layers: 1, hidden: 32, heads: 2, kv_heads: 1, vocab: 512}}
data: {{seq_len: 64, micro_batch: 1}}
mixture:
  enabled: true
  sources:
{source_lines}
{replay}
tokenizer: {{path: /nonexistent/tok}}
optimizer: {{lr: 1.0e-3}}
schedule: {{total_steps: 4}}
"""
    path = tmp_path / "recipe.yaml"
    path.write_text(text)
    return load_recipe(str(path))


class TestMixtureBranches:
    def test_incompatible_with_sft(self):
        assert_problem(
            sft(["mixture.enabled=true"]), "mixture: incompatible with sft.enabled=true"
        )

    def test_empty_sources(self):
        assert_problem(
            tiny(["mixture.enabled=true"]),
            "mixture.sources: must list at least one source",
        )

    def test_duplicate_names(self, tmp_path):
        recipe = mixture_recipe(
            tmp_path,
            "    - {name: a, path: /nonexistent/one, weight: 1.0}\n"
            "    - {name: a, path: /nonexistent/two, weight: 1.0}",
        )
        assert_problem(recipe, "mixture.sources: names must be unique, got ['a', 'a']")

    def test_empty_source_name(self, tmp_path):
        recipe = mixture_recipe(
            tmp_path, "    - {name: '', path: /nonexistent/one, weight: 1.0}"
        )
        assert_problem(recipe, "mixture.sources[0].name: must not be empty")

    def test_source_dir_without_manifest(self, tmp_path):
        bare = tmp_path / "bare"
        bare.mkdir()
        recipe = mixture_recipe(
            tmp_path, f"    - {{name: a, path: {bare}, weight: 1.0}}"
        )
        assert_problem(recipe, f"mixture.sources: {bare} has no manifest.json")

    def test_unloadable_tokenizer_reported(self, tmp_path):
        # One present source with a manifest hash forces the tokenizer
        # comparison; the nonexistent tokenizer.path must become a problem,
        # not an exception out of validate().
        src = make_source(tmp_path, "a", [1] * 64)
        recipe = mixture_recipe(
            tmp_path, f"    - {{name: a, path: {src}, weight: 1.0}}"
        )
        assert_problem(recipe, "tokenizer.path: cannot load /nonexistent/tok")


class TestReplayBranches:
    def test_empty_replay_source_name(self, tmp_path):
        recipe = mixture_recipe(
            tmp_path,
            "    - {name: dom, path: /nonexistent/dom, weight: 1.0}",
            "    - {name: '', path: /nonexistent/rep, weight: 1.0}",
        )
        assert_problem(recipe, "replay.sources[0].name: must not be empty")

    def test_replay_dir_without_manifest(self, tmp_path):
        bare = tmp_path / "bare"
        bare.mkdir()
        recipe = mixture_recipe(
            tmp_path,
            "    - {name: dom, path: /nonexistent/dom, weight: 1.0}",
            f"    - {{name: rep, path: {bare}, weight: 1.0}}",
        )
        assert_problem(recipe, f"replay.sources: {bare} has no manifest.json")


class TestStagingWatchdogAffinity:
    def test_staging_bounds(self):
        # Both staging checks run even with staging disabled: a recipe carrying
        # a broken block must fail at load, not when someone enables it.
        recipe = tiny(["staging.max_gb=0", "staging.dir=''"])
        assert_problem(recipe, "staging.max_gb: must be positive, got 0.0")
        assert_problem(recipe, "staging.dir: must not be empty")

    def test_watchdog_grad_norm_and_patience(self):
        recipe = tiny(["watchdog.grad_norm_max=-1", "watchdog.patience=0"])
        assert_problem(recipe, "watchdog.grad_norm_max: must be >= 0")
        assert_problem(recipe, "watchdog.patience: must be >= 1, got 0")

    def test_affinity_overlap_reported(self):
        recipe = tiny(["affinity.main=0-7", "affinity.workers=4-12"])
        assert_problem(recipe, "masks overlap on cpus [4, 5, 6, 7]")


class TestTokenizerLoadFailure:
    def test_shard_source_with_unloadable_tokenizer(self, tmp_path):
        # A prepared shard directory forces the hash comparison; the failing
        # tokenizer load must be reported as a problem, not raised.
        shards = make_shard_dir(tmp_path, [1] * 100)
        recipe = tiny([f"data.source={shards}", "tokenizer.path=/nonexistent/tok"])
        assert_problem(recipe, "tokenizer.path: cannot load /nonexistent/tok")


# Tracker sink construction. Stub sink classes record every forwarded call so
# tests assert the init/log/close contract, not merely that code ran.


class FakeSummaryWriter:
    instances: list[FakeSummaryWriter] = []

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scalars = []
        self.closed = False
        FakeSummaryWriter.instances.append(self)

    def add_scalar(self, key, value, step):
        self.scalars.append((key, value, step))

    def close(self):
        self.closed = True


class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, metrics, step):
        self.logged.append((metrics, step))

    def finish(self):
        self.finished = True


def cfg(**kwargs):
    base = {"tensorboard": False, "wandb": False, "mlflow": False}
    base.update(kwargs)
    return TrackingConfig(**base)


def install_torch_tensorboard(monkeypatch):
    """Install stub torch.utils.tensorboard modules exposing FakeSummaryWriter."""
    torch_mod = types.ModuleType("torch")
    utils_mod = types.ModuleType("torch.utils")
    tb_mod = types.ModuleType("torch.utils.tensorboard")
    tb_mod.SummaryWriter = FakeSummaryWriter
    utils_mod.tensorboard = tb_mod
    torch_mod.utils = utils_mod
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.utils", utils_mod)
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tb_mod)


def test_tensorboard_sink_construction_and_writethrough(tmp_path, monkeypatch):
    install_torch_tensorboard(monkeypatch)
    monkeypatch.setattr(FakeSummaryWriter, "instances", [])

    tracker = Tracker(cfg(tensorboard=True), "run-tb", str(tmp_path))
    assert len(FakeSummaryWriter.instances) == 1
    writer = FakeSummaryWriter.instances[0]
    assert writer.log_dir == os.path.join(str(tmp_path), "tensorboard")

    tracker.log({"loss": 1.5}, step=3)
    assert writer.scalars == [("loss", 1.5, 3)]
    tracker.close()
    assert writer.closed
    assert tracker._sinks == []


def test_tensorboard_import_failure_disables_sink(tmp_path, monkeypatch, capsys):
    # A None sys.modules entry makes the lazy torch import raise ImportError;
    # construction must survive with the sink disabled and a stderr warning.
    monkeypatch.setitem(sys.modules, "torch", None)
    tracker = Tracker(cfg(tensorboard=True), "run-tb", str(tmp_path))
    assert tracker._sinks == []
    assert "tensorboard sink disabled" in capsys.readouterr().err


def test_wandb_sink_offline_default_and_forwarding(tmp_path, monkeypatch):
    run = FakeWandbRun()
    init_calls = []
    wandb_mod = types.ModuleType("wandb")

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        return run

    wandb_mod.init = fake_init
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)
    # setenv-then-delenv leaves WANDB_MODE unset here while recording the
    # original state, so the value Tracker writes is undone at teardown.
    monkeypatch.setenv("WANDB_MODE", "sentinel")
    monkeypatch.delenv("WANDB_MODE")

    tracker = Tracker(
        cfg(wandb=True, wandb_project="proj-x"),
        "run-wb",
        str(tmp_path),
        config={"lr": 1e-3},
    )
    # Offline-first contract: an unset WANDB_MODE defaults to offline.
    assert os.environ["WANDB_MODE"] == "offline"
    assert init_calls == [
        {
            "project": "proj-x",
            "name": "run-wb",
            "dir": str(tmp_path),
            "config": {"lr": 1e-3},
        }
    ]

    tracker.log({"loss": 2.0}, step=7)
    assert run.logged == [({"loss": 2.0}, 7)]
    tracker.close()
    assert run.finished
    assert tracker._sinks == []


def test_wandb_import_failure_keeps_preset_mode(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setenv("WANDB_MODE", "online")
    tracker = Tracker(cfg(wandb=True), "run-wb", str(tmp_path))
    assert tracker._sinks == []
    assert "wandb sink disabled" in capsys.readouterr().err
    # setdefault must not clobber an explicitly configured mode
    assert os.environ["WANDB_MODE"] == "online"


def test_mlflow_sink_local_store_param_filter_and_forwarding(tmp_path, monkeypatch):
    calls = {
        "uri": [],
        "experiment": [],
        "start": [],
        "params": [],
        "metrics": [],
        "ended": 0,
    }
    mlflow_mod = types.ModuleType("mlflow")
    mlflow_mod.set_tracking_uri = lambda uri: calls["uri"].append(uri)
    mlflow_mod.set_experiment = lambda name: calls["experiment"].append(name)
    mlflow_mod.start_run = lambda run_name: calls["start"].append(run_name)
    mlflow_mod.log_params = lambda params: calls["params"].append(params)

    def log_metrics(metrics, step):
        calls["metrics"].append((metrics, step))

    def end_run():
        calls["ended"] += 1

    mlflow_mod.log_metrics = log_metrics
    mlflow_mod.end_run = end_run
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_mod)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    config = {"lr": 0.001, "name": "x", "steps": 10, "flag": True, "nested": {"a": 1}}
    tracker = Tracker(cfg(mlflow=True), "run-ml", str(tmp_path), config=config)
    # No MLFLOW_TRACKING_URI: logs to a file store under the run directory.
    assert calls["uri"] == [f"file://{os.path.join(str(tmp_path), 'mlruns')}"]
    assert calls["experiment"] == ["run-ml"]
    assert calls["start"] == ["run-ml"]
    # Non-scalar params are dropped (mlflow rejects params over 500 chars).
    assert calls["params"] == [{"lr": 0.001, "name": "x", "steps": 10, "flag": True}]

    tracker.log({"loss": 0.5}, step=2)
    assert calls["metrics"] == [({"loss": 0.5}, 2)]
    tracker.close()
    assert calls["ended"] == 1
    assert tracker._sinks == []


def test_mlflow_import_failure_disables_sink(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    tracker = Tracker(cfg(mlflow=True), "run-ml", str(tmp_path))
    assert tracker._sinks == []
    assert "mlflow sink disabled" in capsys.readouterr().err


def test_close_survives_a_failing_sink(tmp_path, capsys):
    class ExplodingRun:
        def finish(self):
            raise RuntimeError("sink down")

    tracker = Tracker(cfg(), "run-x", str(tmp_path))
    tracker._sinks = [("wandb", ExplodingRun())]
    tracker.close()
    assert "wandb sink disabled" in capsys.readouterr().err
    assert tracker._sinks == []
