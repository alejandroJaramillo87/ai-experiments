"""CLI handler contracts for the run subcommands: cmd_train, cmd_train_sft,
cmd_train_distill, and cmd_distill_cache.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 failure), the styled
[OK]/[WARN]/[ERROR] lines, and --json stdout purity. The GPU pipelines are
replaced at their home modules (training.loop / training.sft /
training.distill) with fakes returning the real result dataclasses; recipe
loading, validation, and the distill-cache preflight refusals execute for
real (they are torch-free). check_memory_fit is always faked: the real one
resolves HF model specs through transformers, absent in the sandbox.
"""

from __future__ import annotations

import json
import os

import pytest
from support import REFERENCE_DIR

from training import cli
from training.distill import DistillPreflightError
from training.loop import TrainResult
from training.sft import SftPreflightError, SftResult
from training.watchdog import WatchdogAbort

TINY = os.path.join(REFERENCE_DIR, "tiny-e2e.yaml")
SFT = os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml")
LOGIT = os.path.join(REFERENCE_DIR, "distill-logit-gemma-e2b.yaml")
RESPONSE = os.path.join(REFERENCE_DIR, "distill-response-gemma-e2b.yaml")

FIT_OK = "Memory fit: 20.61 GB reserved estimate of 32 GB VRAM"
FIT_FORCED = ("Memory fit: predicted 40.00 GB exceeds the 32 GB budget "
              "(--force: proceeding anyway)")

WATCHDOG_LINE = "Watchdog abort: loss_spike value 12.5 crossed threshold 6.25"


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def train_result(**kw) -> TrainResult:
    base = {"run_dir": "/runs/tiny-e2e/1", "checkpoint_root": "/ckpt/tiny-e2e",
                "final_step": 20, "tokens_seen": 2560, "losses": [(10, 2.5), (20, 1.25)]}
    base.update(kw)
    return TrainResult(**base)


def sft_result(**kw) -> SftResult:
    base = {"run_dir": "/runs/sft/1", "checkpoint_root": "/ckpt/sft",
                "final_step": 2000, "backend": "trl", "train_loss": 1.5,
                "holdout_loss": 1.8,
                "trainable_report": "trainable params: 4,096 of 1,000,000",
                "losses": [(1000, 2.0), (2000, 1.5)]}
    base.update(kw)
    return SftResult(**base)


def patch_fit(monkeypatch, message: str = FIT_OK, fits: bool = True) -> list:
    """Replace training.sft.check_memory_fit; returns the recorded calls."""
    calls: list = []

    def fake_fit(recipe, force):
        calls.append((recipe, force))
        return fits, message

    monkeypatch.setattr("training.sft.check_memory_fit", fake_fit)
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

        monkeypatch.setattr("training.loop.train", fake_train)
        assert run_cli(["train", "--config", TINY]) == 0
        out = capsys.readouterr().out
        assert "[RUN] Training tiny-e2e: 20 total steps" in out
        assert "[OK] Run complete: step 20, 2,560 tokens seen" in out
        assert "Final loss: 1.250000" in out
        assert "Run directory: /runs/tiny-e2e/1" in out
        assert "Checkpoints:   /ckpt/tiny-e2e" in out
        assert seen["recipe"].run.name == "tiny-e2e"

    def test_resume_verb_and_empty_losses(self, monkeypatch, capsys):
        monkeypatch.setattr("training.loop.train",
                            lambda recipe: train_result(losses=[]))
        assert run_cli(["train", "--config", TINY,
                        "checkpoint.resume=latest"]) == 0
        out = capsys.readouterr().out
        assert "[RUN] Resuming (latest) tiny-e2e: 20 total steps" in out
        assert "Final loss" not in out

    def test_json_stdout_is_pure_json(self, monkeypatch, capsys):
        monkeypatch.setattr("training.loop.train",
                            lambda recipe: train_result())
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

        monkeypatch.setattr("training.loop.train", aborting)
        assert run_cli(["train", "--config", TINY]) == 1
        assert_watchdog_output(capsys.readouterr().out)

    def test_runtime_failure_exits_1_with_solution(self, monkeypatch, capsys):
        def failing(recipe):
            raise RuntimeError("shard dir missing")

        monkeypatch.setattr("training.loop.train", failing)
        assert run_cli(["train", "--config", TINY]) == 1
        out = capsys.readouterr().out
        assert "[ERROR] Training failed: shard dir missing" in out
        assert "Solution: check the recipe's data.source shards" in out

    @pytest.mark.parametrize(
        "subcommand", ["train", "train-sft", "train-distill", "distill-cache"])
    def test_unloadable_recipe_exits_1(self, subcommand, tmp_path, capsys):
        # No pipeline patching needed: the handler must bail before any run.
        assert run_cli([subcommand, "--config",
                        str(tmp_path / "absent.yaml")]) == 1
        assert "[ERROR] Recipe load failed" in capsys.readouterr().out


# -- cmd_train-sft --------------------------------------------------------------

class TestCmdTrainSft:
    def test_success_prints_full_report(self, monkeypatch, capsys):
        patch_fit(monkeypatch)
        monkeypatch.setattr("training.sft.train_sft",
                            lambda recipe, force=False: sft_result())
        assert run_cli(["train-sft", "--config", SFT]) == 0
        out = capsys.readouterr().out
        assert "[RUN] Fine-tuning sft-qlora-e2b [trl]: 2000 total steps" in out
        assert f"[OK] {FIT_OK}" in out
        assert "[OK] Run complete: step 2000 [trl]" in out
        assert "  [INFO] trainable params: 4,096 of 1,000,000" in out
        assert "Final train loss: 1.500000" in out
        assert "Held-out loss:    1.800000" in out
        assert "Run directory: /runs/sft/1" in out

    def test_forced_fit_warns_resume_verb_and_force_passthrough(
            self, monkeypatch, capsys):
        fit_calls = patch_fit(monkeypatch, message=FIT_FORCED)
        train_calls = []

        def fake_train(recipe, force=False):
            train_calls.append(force)
            return sft_result(trainable_report="", train_loss=None,
                              holdout_loss=None)

        monkeypatch.setattr("training.sft.train_sft", fake_train)
        assert run_cli(["train-sft", "--config", SFT, "--force",
                        "checkpoint.resume=latest"]) == 0
        out = capsys.readouterr().out
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

        monkeypatch.setattr("training.sft.train_sft", failing)
        assert run_cli(["train-sft", "--config", SFT]) == 1
        out = capsys.readouterr().out
        assert "[ERROR] Preflight failed: dataset manifest hash mismatch" in out

    def test_watchdog_abort_exits_1(self, monkeypatch, capsys):
        patch_fit(monkeypatch)

        def aborting(recipe, force=False):
            raise WatchdogAbort("loss_spike", 12.5, 6.25)

        monkeypatch.setattr("training.sft.train_sft", aborting)
        assert run_cli(["train-sft", "--config", SFT]) == 1
        assert_watchdog_output(capsys.readouterr().out)

    def test_runtime_failure_exits_1_with_solution(self, monkeypatch, capsys):
        patch_fit(monkeypatch)

        def failing(recipe, force=False):
            raise FileNotFoundError("no such dataset")

        monkeypatch.setattr("training.sft.train_sft", failing)
        assert run_cli(["train-sft", "--config", SFT]) == 1
        out = capsys.readouterr().out
        assert "[ERROR] SFT run failed: no such dataset" in out
        assert "Solution: check sft.dataset" in out

    def test_json_stdout_is_pure_json(self, monkeypatch, capsys):
        patch_fit(monkeypatch)
        monkeypatch.setattr(
            "training.sft.train_sft",
            lambda recipe, force=False: sft_result(holdout_loss=None))
        assert run_cli(["train-sft", "--config", SFT, "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["final_step"] == 2000
        assert data["backend"] == "trl"
        assert data["train_loss"] == 1.5
        assert data["holdout_loss"] is None
        assert data["losses"] == [[1000, 2.0], [2000, 1.5]]


# -- cmd_train-distill ----------------------------------------------------------

class TestCmdTrainDistill:
    def test_response_tier_success(self, monkeypatch, capsys):
        patch_fit(monkeypatch)
        monkeypatch.setattr(
            "training.distill.train_distill",
            lambda recipe, force=False: sft_result(
                run_dir="/runs/distill/1", final_step=1000,
                trainable_report="", train_loss=None, holdout_loss=None))
        assert run_cli(["train-distill", "--config", RESPONSE]) == 0
        out = capsys.readouterr().out
        assert ("[RUN] Distilling distill-response-gemma-e2b [tier=response]: "
                "1000 total steps") in out
        assert f"[OK] {FIT_OK}" in out
        assert "[OK] Run complete: step 1000 [trl]" in out
        assert "Run directory: /runs/distill/1" in out

    def test_logit_tier_detail_and_report(self, monkeypatch, capsys):
        patch_fit(monkeypatch, message=FIT_FORCED)
        monkeypatch.setattr(
            "training.distill.train_distill",
            lambda recipe, force=False: sft_result(final_step=1000))
        assert run_cli(["train-distill", "--config", LOGIT, "--force"]) == 0
        out = capsys.readouterr().out
        assert ("[RUN] Distilling distill-logit-gemma-e2b [tier=logit "
                "mode=online teacher=google/gemma-4-E4B-it]: "
                "1000 total steps") in out
        assert f"[WARN] {FIT_FORCED}" in out
        assert "  [INFO] trainable params: 4,096 of 1,000,000" in out
        assert "Final train loss: 1.500000" in out
        assert "Held-out loss:    1.800000" in out

    def test_preflight_error_exits_1(self, monkeypatch, capsys):
        patch_fit(monkeypatch)

        def failing(recipe, force=False):
            raise DistillPreflightError("teacher tokenizer parity failed")

        monkeypatch.setattr("training.distill.train_distill", failing)
        assert run_cli(["train-distill", "--config", LOGIT]) == 1
        out = capsys.readouterr().out
        assert "[ERROR] Preflight failed: teacher tokenizer parity failed" in out

    def test_watchdog_abort_exits_1(self, monkeypatch, capsys):
        patch_fit(monkeypatch)

        def aborting(recipe, force=False):
            raise WatchdogAbort("loss_spike", 12.5, 6.25)

        monkeypatch.setattr("training.distill.train_distill", aborting)
        assert run_cli(["train-distill", "--config", LOGIT]) == 1
        assert_watchdog_output(capsys.readouterr().out)

    def test_runtime_failure_exits_1_with_solution(self, monkeypatch, capsys):
        patch_fit(monkeypatch)

        def failing(recipe, force=False):
            raise ValueError("bad alpha")

        monkeypatch.setattr("training.distill.train_distill", failing)
        assert run_cli(["train-distill", "--config", LOGIT]) == 1
        out = capsys.readouterr().out
        assert "[ERROR] Distillation run failed: bad alpha" in out
        assert "Solution: check sft.dataset, distill.teacher" in out

    def test_json_stdout_is_pure_json(self, monkeypatch, capsys):
        patch_fit(monkeypatch)
        monkeypatch.setattr(
            "training.distill.train_distill",
            lambda recipe, force=False: sft_result(final_step=1000))
        assert run_cli(["train-distill", "--config", LOGIT, "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["final_step"] == 1000
        assert data["tier"] == "logit"
        assert data["backend"] == "trl"
        assert data["holdout_loss"] == 1.8


# -- cmd_distill-cache ----------------------------------------------------------

CACHE_MANIFEST = {
    "examples": 1024, "total_tokens": 200000, "top_k": 64,
    "teacher": "google/gemma-4-E4B-it", "teacher_revision": "abc1234",
}


class TestCmdDistillCache:
    # Refusals raised before build_logit_cache imports torch run for real.
    @pytest.mark.parametrize("config,message", [
        (TINY, "distill.enabled=true and tier=logit"),
        (LOGIT, "no cache path: set distill.cache_path or pass --out"),
    ], ids=["distill-disabled", "no-cache-path"])
    def test_real_preflight_refusals_exit_1(self, config, message, capsys):
        assert run_cli(["distill-cache", "--config", config]) == 1
        out = capsys.readouterr().out
        assert "[ERROR] Preflight failed: " in out
        assert message in out

    def test_existing_cache_exits_1(self, monkeypatch, capsys):
        def failing(recipe, out_dir=None, progress=None):
            raise FileExistsError("cache exists: /c (pass a new --out)")

        monkeypatch.setattr("training.distill.build_logit_cache", failing)
        assert run_cli(["distill-cache", "--config", LOGIT,
                        "--out", "/c"]) == 1
        out = capsys.readouterr().out
        assert "[ERROR] cache exists: /c (pass a new --out)" in out
        assert "Cache pre-pass failed" not in out

    def test_runtime_failure_exits_1_with_solution(self, monkeypatch, capsys):
        def failing(recipe, out_dir=None, progress=None):
            raise RuntimeError("teacher OOM")

        monkeypatch.setattr("training.distill.build_logit_cache", failing)
        assert run_cli(["distill-cache", "--config", LOGIT,
                        "--out", "/c"]) == 1
        out = capsys.readouterr().out
        assert "[ERROR] Cache pre-pass failed: teacher OOM" in out
        assert "Solution: check sft.dataset" in out

    def test_success_reports_progress_and_manifest(self, monkeypatch, tmp_path,
                                                   capsys):
        target = str(tmp_path / "cache")
        seen = {}

        def fake_build(recipe, out_dir=None, progress=None):
            seen["out_dir"] = out_dir
            # 8 % 512 == 8 misses the <8 window; 512 and the final total hit it.
            progress(8, 1024)
            progress(512, 1024)
            progress(1024, 1024)
            return dict(CACHE_MANIFEST)

        monkeypatch.setattr("training.distill.build_logit_cache", fake_build)
        assert run_cli(["distill-cache", "--config", LOGIT,
                        "--out", target]) == 0
        out = capsys.readouterr().out
        assert ("[RUN] Caching top-64 teacher logprobs: "
                f"google/gemma-4-E4B-it -> {target}") in out
        assert "  [INFO] Examples cached: 512/1024" in out
        assert "  [INFO] Examples cached: 1024/1024" in out
        assert "Examples cached: 8/1024" not in out
        assert ("[OK] Cache written: 1024 examples, 200,000 token positions, "
                "top_k 64") in out
        assert "Teacher: google/gemma-4-E4B-it (revision abc1234)" in out
        assert seen["out_dir"] == target

    def test_json_stdout_is_pure_manifest(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(
            "training.distill.build_logit_cache",
            lambda recipe, out_dir=None, progress=None: dict(CACHE_MANIFEST))
        # No --out: the target falls back to the recipe's distill.cache_path.
        assert run_cli([
            "distill-cache", "--config", LOGIT, "--json",
            "distill.mode=cached", "distill.teacher_quantize=none",
            f"distill.cache_path={tmp_path / 'cache'}"]) == 0
        assert json.loads(capsys.readouterr().out) == CACHE_MANIFEST
