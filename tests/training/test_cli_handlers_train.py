"""CLI handler contracts for the run subcommands: cmd_train and cmd_train_sft.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 failure), the styled
[OK]/[WARN]/[ERROR] lines, and --json stdout purity. The GPU pipelines are
replaced at their home modules (training.runtime.loop / training.tune.sft)
with fakes returning the real result dataclasses; recipe loading and
validation execute for real (they are torch-free). check_memory_fit is always faked: the real one
resolves HF model specs through transformers, absent in the sandbox.
"""

from __future__ import annotations

import json
import os

import pytest
from support import REFERENCE_DIR

from training import cli
from training.runtime.loop import TrainResult
from training.runtime.watchdog import WatchdogAbort
from training.tune.sft import SftPreflightError, SftResult

TINY = os.path.join(REFERENCE_DIR, "tiny-e2e.yaml")
SFT = os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml")

FIT_OK = "Memory fit: 20.61 GB reserved estimate of 32 GB VRAM"
FIT_FORCED = (
    "Memory fit: predicted 40.00 GB exceeds the 32 GB budget "
    "(--force: proceeding anyway)"
)

WATCHDOG_LINE = "Watchdog abort: loss_spike value 12.5 crossed threshold 6.25"


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def train_result(**kw) -> TrainResult:
    base = {
        "run_dir": "/runs/tiny-e2e/1",
        "checkpoint_root": "/ckpt/tiny-e2e",
        "final_step": 20,
        "tokens_seen": 2560,
        "losses": [(10, 2.5), (20, 1.25)],
    }
    base.update(kw)
    return TrainResult(**base)


def sft_result(**kw) -> SftResult:
    base = {
        "run_dir": "/runs/sft/1",
        "checkpoint_root": "/ckpt/sft",
        "final_step": 2000,
        "backend": "trl",
        "train_loss": 1.5,
        "holdout_loss": 1.8,
        "trainable_report": "trainable params: 4,096 of 1,000,000",
        "losses": [(1000, 2.0), (2000, 1.5)],
    }
    base.update(kw)
    return SftResult(**base)


def patch_fit(monkeypatch, message: str = FIT_OK, fits: bool = True) -> list:
    """Replace training.tune.sft.check_memory_fit; returns the recorded calls."""
    calls: list = []

    def fake_fit(recipe, force):
        calls.append((recipe, force))
        return fits, message

    monkeypatch.setattr("training.tune.sft.check_memory_fit", fake_fit)
    return calls


def assert_watchdog_output(out: str) -> None:
    assert f"[ERROR] {WATCHDOG_LINE}" in out
    assert "Emergency checkpoint saved" in out
    assert "checkpoint.resume=latest" in out


# -- cmd_train ----------------------------------------------------------------


class TestCmdTrain:
    def test_success_prints_run_summary(self, monkeypatch, capsys):
        seen = {}

        def fake_train(recipe):
            seen["recipe"] = recipe
            return train_result()

        monkeypatch.setattr("training.runtime.loop.train", fake_train)
        assert run_cli(["train", "--config", TINY]) == 0
        out = capsys.readouterr().err
        assert "[RUN] Training tiny-e2e: 20 total steps" in out
        assert "[OK] Run complete: step 20, 2,560 tokens seen" in out
        assert "Final loss: 1.250000" in out
        assert "Run directory: /runs/tiny-e2e/1" in out
        assert "Checkpoints:   /ckpt/tiny-e2e" in out
        assert seen["recipe"].run.name == "tiny-e2e"

    def test_resume_verb_and_empty_losses(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "training.runtime.loop.train", lambda recipe: train_result(losses=[])
        )
        assert run_cli(["train", "--config", TINY, "checkpoint.resume=latest"]) == 0
        out = capsys.readouterr().err
        assert "[RUN] Resuming (latest) tiny-e2e: 20 total steps" in out
        assert "Final loss" not in out

    def test_json_stdout_is_pure_json(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "training.runtime.loop.train", lambda recipe: train_result()
        )
        assert run_cli(["train", "--config", TINY, "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["final_step"] == 20
        assert data["tokens_seen"] == 2560
        assert data["losses"] == [[10, 2.5], [20, 1.25]]
        assert data["run_dir"] == "/runs/tiny-e2e/1"
        assert data["checkpoint_root"] == "/ckpt/tiny-e2e"

    def test_watchdog_abort_exits_1(self, monkeypatch, capsys):
        def aborting(recipe):
            raise WatchdogAbort("loss_spike", 12.5, 6.25)

        monkeypatch.setattr("training.runtime.loop.train", aborting)
        assert run_cli(["train", "--config", TINY]) == 1
        assert_watchdog_output(capsys.readouterr().err)

    def test_runtime_failure_exits_1_with_solution(self, monkeypatch, capsys):
        def failing(recipe):
            raise RuntimeError("shard dir missing")

        monkeypatch.setattr("training.runtime.loop.train", failing)
        assert run_cli(["train", "--config", TINY]) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Training failed: shard dir missing" in out
        assert "Solution: check the recipe's data.source shards" in out

    @pytest.mark.parametrize("subcommand", ["train", "train-sft"])
    def test_unloadable_recipe_exits_1(self, subcommand, tmp_path, capsys):
        # No pipeline patching needed: the handler must bail before any run.
        assert run_cli([subcommand, "--config", str(tmp_path / "absent.yaml")]) == 1
        assert "[ERROR] Recipe load failed" in capsys.readouterr().err


# -- cmd_train-sft --------------------------------------------------------------


class TestCmdTrainSft:
    def test_success_prints_full_report(self, monkeypatch, capsys):
        patch_fit(monkeypatch)
        monkeypatch.setattr(
            "training.tune.sft.train_sft", lambda recipe, force=False: sft_result()
        )
        assert run_cli(["train-sft", "--config", SFT]) == 0
        out = capsys.readouterr().err
        assert "[RUN] Fine-tuning sft-qlora-e2b [trl]: 2000 total steps" in out
        assert f"[OK] {FIT_OK}" in out
        assert "[OK] Run complete: step 2000 [trl]" in out
        assert "  [INFO] trainable params: 4,096 of 1,000,000" in out
        assert "Final train loss: 1.500000" in out
        assert "Held-out loss:    1.800000" in out
        assert "Run directory: /runs/sft/1" in out

    def test_forced_fit_warns_resume_verb_and_force_passthrough(
        self, monkeypatch, capsys
    ):
        fit_calls = patch_fit(monkeypatch, message=FIT_FORCED)
        train_calls = []

        def fake_train(recipe, force=False):
            train_calls.append(force)
            return sft_result(trainable_report="", train_loss=None, holdout_loss=None)

        monkeypatch.setattr("training.tune.sft.train_sft", fake_train)
        assert (
            run_cli(
                ["train-sft", "--config", SFT, "--force", "checkpoint.resume=latest"]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[RUN] Resuming (latest) sft-qlora-e2b [trl]" in out
        assert f"[WARN] {FIT_FORCED}" in out
        # Optional report fields absent: their lines must not print.
        assert "[INFO]" not in out
        assert "Final train loss" not in out
        assert "Held-out loss" not in out
        assert fit_calls[0][1] is True
        assert train_calls == [True]

    def test_preflight_error_exits_1(self, monkeypatch, capsys):
        patch_fit(monkeypatch)

        def failing(recipe, force=False):
            raise SftPreflightError("dataset manifest hash mismatch")

        monkeypatch.setattr("training.tune.sft.train_sft", failing)
        assert run_cli(["train-sft", "--config", SFT]) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Preflight failed: dataset manifest hash mismatch" in out

    def test_watchdog_abort_exits_1(self, monkeypatch, capsys):
        patch_fit(monkeypatch)

        def aborting(recipe, force=False):
            raise WatchdogAbort("loss_spike", 12.5, 6.25)

        monkeypatch.setattr("training.tune.sft.train_sft", aborting)
        assert run_cli(["train-sft", "--config", SFT]) == 1
        assert_watchdog_output(capsys.readouterr().err)

    def test_runtime_failure_exits_1_with_solution(self, monkeypatch, capsys):
        patch_fit(monkeypatch)

        def failing(recipe, force=False):
            raise FileNotFoundError("no such dataset")

        monkeypatch.setattr("training.tune.sft.train_sft", failing)
        assert run_cli(["train-sft", "--config", SFT]) == 1
        out = capsys.readouterr().err
        assert "[ERROR] SFT run failed: no such dataset" in out
        assert "Solution: check sft.dataset" in out

    def test_json_stdout_is_pure_json(self, monkeypatch, capsys):
        patch_fit(monkeypatch)
        monkeypatch.setattr(
            "training.tune.sft.train_sft",
            lambda recipe, force=False: sft_result(holdout_loss=None),
        )
        assert run_cli(["train-sft", "--config", SFT, "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["final_step"] == 2000
        assert data["backend"] == "trl"
        assert data["train_loss"] == 1.5
        assert data["holdout_loss"] is None
        assert data["losses"] == [[1000, 2.0], [2000, 1.5]]
