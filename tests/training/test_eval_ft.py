"""eval-ft command contracts that run without containers: the prompt-set
preflight (KB-21) and the restore-failure exit path (KB-20). The container
and serving phases are faked at their module seams; everything else is the
real command."""

from __future__ import annotations

import argparse
import json
import os

from common.output import Printer
from training import serving
from training.commands import eval_ft


def judge_args(tmp_path, **overrides):
    prompts = tmp_path / "prompts.jsonl"
    if not prompts.exists():
        with open(prompts, "w") as f:
            f.write(json.dumps({"id": "p1", "prompt": "q"}) + "\n")
    defaults = {
        "evaluators": "judge",
        "name": "eval",
        "out_dir": str(tmp_path / "run"),
        "base": "google/gemma-4-E2B",
        "tuned": "/ckpt/latest",
        "base_gguf": "/app/models/b.gguf",
        "tuned_gguf": "/app/models/t.gguf",
        "judge_gguf": "/app/models/j.gguf",
        "judge_service": "llama-cpu",
        "prompts": str(prompts),
        "dataset": None,
        "max_tokens": 16,
        "timeout": 5,
        "load_4bit": False,
        "max_examples": None,
        "tasks": None,
        "lmeval_limit": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestValidatePromptPreflight:
    def test_missing_prompts_file_fails_validation(self, tmp_path, capsys):
        # KB-21: a missing --prompts file must fail preflight, not crash
        # hours later at sha256_file after the container phases complete
        args = judge_args(tmp_path, prompts=str(tmp_path / "absent.jsonl"))
        assert eval_ft.validate(args, ["judge"], Printer()) is False
        assert "[ERROR] Prompt set:" in capsys.readouterr().err

    def test_malformed_prompts_fail_validation(self, tmp_path, capsys):
        # KB-21: malformed prompt content is a preflight failure too
        bad = tmp_path / "bad.jsonl"
        with open(bad, "w") as f:
            f.write('{"id": "p1"}\n')  # record without a prompt key
        args = judge_args(tmp_path, prompts=str(bad))
        assert eval_ft.validate(args, ["judge"], Printer()) is False
        assert "[ERROR] Prompt set:" in capsys.readouterr().err

    def test_valid_prompts_pass(self, tmp_path):
        args = judge_args(tmp_path)
        assert eval_ft.validate(args, ["judge"], Printer()) is True


class TestRestoreFailureExit:
    def patch_seams(self, monkeypatch):
        def fake_identity(model):
            return {
                "model": model,
                "host_path": model,
                "quant_type": "Q8_0",
                "sha256": "s",
                "manifest": None,
                "thinking_mode": "off",
            }

        monkeypatch.setattr("training.eval.runner.gguf_identity", fake_identity)
        monkeypatch.setattr("training.serving.ensure_training_idle", lambda: None)
        monkeypatch.setattr("training.serving.sha256_file", lambda p: "hash")
        monkeypatch.setattr(
            "training.eval.scorecard.assemble_scorecard",
            lambda metadata, sections: {"schema_version": 1},
        )
        monkeypatch.setattr(
            "training.eval.scorecard.render_markdown", lambda card: "table\n"
        )
        monkeypatch.setattr(eval_ft, "phase_container", lambda *a, **k: {})

    def test_restore_failure_exits_1_with_scorecard_written(
        self, tmp_path, monkeypatch, capsys
    ):
        # KB-20: a failed serving restore must not exit 0 publishing a clean
        # success; the scorecard is still written for the record
        self.patch_seams(monkeypatch)

        def fake_phase_serving(args, out_dir, identities, out, restore_failed):
            restore_failed.append("llama-cpu")
            return {"judge": {"win_rate": 0.5}}

        monkeypatch.setattr(eval_ft, "phase_serving", fake_phase_serving)
        args = judge_args(tmp_path)
        assert eval_ft.cmd(args) == 1
        err = capsys.readouterr().err
        assert "Serving restore failed for: llama-cpu" in err
        assert os.path.isfile(os.path.join(args.out_dir, "scorecard.json"))

    def test_clean_restore_exits_0(self, tmp_path, monkeypatch):
        self.patch_seams(monkeypatch)
        monkeypatch.setattr(
            eval_ft,
            "phase_serving",
            lambda args, out_dir, identities, out, restore_failed: {"judge": {}},
        )
        assert eval_ft.cmd(judge_args(tmp_path)) == 0


class TestPhaseServingRestoreTracking:
    def test_failed_restore_recorded_not_raised(self, tmp_path, monkeypatch, capsys):
        # KB-20: phase_serving records the failed restore and still returns
        # the judge section (the eval itself succeeded)
        prompts = tmp_path / "prompts.jsonl"
        with open(prompts, "w") as f:
            f.write(json.dumps({"id": "p1", "prompt": "q"}) + "\n")
        for side in ("baseline", "tuned"):
            with open(tmp_path / f"responses_{side}.jsonl", "w") as f:
                f.write(json.dumps({"id": "p1", "response": "a"}) + "\n")

        def fake_swap(service, container_path, env=None):
            if container_path is None:
                raise serving.ServingError("compose exited 1")

        monkeypatch.setattr("training.serving.swap_model", fake_swap)
        monkeypatch.setattr("training.serving.wait_healthy", lambda s, *a: 0.1)
        monkeypatch.setattr("training.eval.runner.run_judgments", lambda *a, **k: [])
        monkeypatch.setattr(
            "training.eval.judge.score_judgments",
            lambda judgments: {
                "win_rate": None,
                "tuned_wins": 0,
                "baseline_wins": 0,
                "ties": 0,
                "malformed": 0,
                "errors": 0,
            },
        )
        args = judge_args(tmp_path)
        identities = {
            "baseline": {"model": "/app/models/b.gguf"},
            "tuned": {"model": "/app/models/t.gguf"},
            "judge": {"model": "/app/models/j.gguf"},
        }
        restore_failed: list = []
        section = eval_ft.phase_serving(
            args, str(tmp_path), identities, Printer(), restore_failed
        )
        assert "judge" in section
        assert restore_failed == ["llama-cpu"]
        assert "Restore failed for llama-cpu" in capsys.readouterr().err
