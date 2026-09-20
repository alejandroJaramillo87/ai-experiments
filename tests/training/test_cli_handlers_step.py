"""In-process handler tests for python -m training step.

Exercises the step command's handler and the shared load_validated helper
through build_parser() + args.func(args), the same argv path production
takes. The GPU loop is the only faked seam (training.commands.train_step.step),
with records built from the real StepRecord dataclass so as_dict() and
write_record stay real.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest
from support import REFERENCE_DIR, TINY_RECIPE

from training import cli
from training.commands.train_step import StepRecord

GIB = 1 << 30

# tiny-e2e scratch model: 2 layers, hidden 64, vocab 512 -> the analytic
# param count pinned by tests/training/test_architecture.py.
TINY_PARAMS = 155_968


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def make_record(**overrides) -> StepRecord:
    """A real StepRecord with passing defaults: analytic matches the model."""
    fields = {
        "recipe_run_name": "tiny-e2e",
        "steps": 5,
        "model_param_count": TINY_PARAMS,
        "analytic_param_count": TINY_PARAMS,
        "model_trainable_params": TINY_PARAMS,
        "analytic_trainable_params": TINY_PARAMS,
        "static_measured": int(1.5 * GIB),
        "measured_allocated": int(2.1 * GIB),
        "measured_reserved": int(2.5 * GIB),
        "measured_nvml_peak": int(2.4 * GIB),
        "tokens_per_micro_batch": 256,
    }
    fields.update(overrides)
    return StepRecord(**fields)


def patch_step(monkeypatch, record=None, error=None):
    """Replace the GPU loop, the one seam a sandbox cannot run."""

    def fake_step(recipe, steps):
        if error is not None:
            raise error
        return record

    monkeypatch.setattr("training.commands.train_step.step", fake_step)


class TestStep:
    def test_reports_the_figures_and_exits_zero(self, capsys, monkeypatch):
        patch_step(monkeypatch, record=make_record())
        rc = run_cli(["step", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert "[RUN] Stepping tiny-e2e: 5 steps on cuda:0" in out
        assert (
            f"[OK] Parameter count: {TINY_PARAMS:,} (analytic matches model)"
        ) in out
        assert "Resident after build:   1.50 GB" in out
        assert "Peak allocated:         2.10 GB" in out
        assert "Peak reserved:          2.50 GB" in out
        assert "NVML peak:              2.40 GB" in out
        assert "Tokens per micro-batch: 256" in out
        assert "[OK] Stepped 5 steps" in out
        assert "Activation rate" not in out and "Record:" not in out

    def test_out_flag_writes_record_and_steps_flag_flows_through(
        self, capsys, monkeypatch, tmp_path
    ):
        record = make_record(steps=3)
        patch_step(monkeypatch, record=record)
        out_path = str(tmp_path / "records" / "step.json")
        rc = run_cli(
            ["step", "--config", TINY_RECIPE, "--steps", "3", "--out", out_path]
        )
        out = capsys.readouterr().err
        assert rc == 0
        assert "[RUN] Stepping tiny-e2e: 3 steps on cuda:0" in out
        assert f"[OK] Step record written: {out_path}" in out
        # The real write_record ran: the file round-trips the record.
        with open(out_path) as f:
            assert json.load(f) == record.as_dict()

    def test_param_count_drift_warns_but_still_passes(self, capsys, monkeypatch):
        patch_step(monkeypatch, record=make_record(model_param_count=TINY_PARAMS + 2))
        rc = run_cli(["step", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert (
            f"[WARN] Parameter count drift: model {TINY_PARAMS + 2:,} "
            f"vs analytic {TINY_PARAMS:,}"
        ) in out
        assert (
            "Solution: align the param formulas in training/models/architecture.py"
            in out
        )
        assert "[OK] Parameter count:" not in out

    def test_trainable_count_drift_warns_but_still_passes(self, capsys, monkeypatch):
        record = make_record(
            model_trainable_params=1_234, analytic_trainable_params=1_200
        )
        patch_step(monkeypatch, record=record)
        rc = run_cli(["step", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert "[WARN] Trainable count drift: model 1,234 vs analytic 1,200" in out
        assert "Solution: align adapter_param_count" in out

    def test_runtime_error_from_loop_is_styled_failure(self, capsys, monkeypatch):
        patch_step(monkeypatch, error=RuntimeError("no CUDA device visible"))
        rc = run_cli(["step", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Step failed: no CUDA device visible" in out
        assert (
            "Solution: run on the workstation via 'make train-step CONFIG=...'" in out
        )

    def test_out_of_memory_is_a_geometry_finding(self, capsys, monkeypatch):
        """KB-86: an OOM in the loop carries what was measured and names the
        geometry as the remedy, never the workstation."""
        from training.commands.train_step import StepOutOfMemory

        error = StepOutOfMemory(
            step=1,
            tokens=4096,
            static_measured=int(6.37 * GIB),
            peak_allocated=int(30.57 * GIB),
            total=int(31.36 * GIB),
        )
        patch_step(monkeypatch, error=error)
        rc = run_cli(["step", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert (
            "[ERROR] Step ran out of GPU memory at step 1: resident after "
            "build 6.37 GB, peak allocated 30.57 GB of 31.36 GB, at 4,096 tokens "
            "per micro-batch"
        ) in out
        assert "Solution: lower data.micro_batch or data.seq_len" in out
        assert "run on the workstation" not in out

    def test_without_gpu_fails_styled_for_real(self, capsys, monkeypatch):
        # step() itself is unpatched: it either fails importing torch (this
        # sandbox) or raises its no-CUDA RuntimeError, and both must land in
        # the same styled exit-1 arm. Only torch's CUDA probe is forced, so
        # the GPU workstation takes the same arm as the sandbox.
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is not None:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        rc = run_cli(["step", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Step failed:" in out
        assert "Solution: run on the workstation" in out

    def test_missing_recipe_file_exits_one_before_stepping(self, capsys):
        rc = run_cli(["step", "--config", "/nonexistent/recipe.yaml"])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Recipe load failed:" in out
        assert "[RUN] Stepping" not in out

    def test_invalid_override_is_styled_validation_error(self, capsys):
        rc = run_cli(["step", "--config", TINY_RECIPE, "data.micro_batch=0"])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Recipe validation failed:" in out
        assert "- data.micro_batch: must be >= 1, got 0" in out

    def test_json_mode_emits_only_the_record(self, capsys, monkeypatch):
        record = make_record()
        patch_step(monkeypatch, record=record)
        rc = run_cli(["step", "--config", TINY_RECIPE, "--json"])
        captured = capsys.readouterr()
        assert rc == 0
        assert json.loads(captured.out) == record.as_dict()

    def test_unsupported_recipe_is_refused_without_workstation_hint(
        self, capsys, monkeypatch
    ):
        from training.commands.train_step import StepUnsupported

        patch_step(
            monkeypatch, error=StepUnsupported("memory.offload=zero_offload cannot")
        )
        rc = run_cli(["step", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Step unsupported:" in out
        assert "Solution: run on the workstation" not in out

    def test_memfit_and_calibrate_verbs_are_gone(self):
        for verb in ("memfit", "calibrate"):
            with pytest.raises(SystemExit) as exc:
                run_cli([verb, "--config", TINY_RECIPE])
            assert exc.value.code == 2


class TestStepSupport:
    """_check_supported and _build_step_model (KB-03), stubs only."""

    def test_zero_offload_recipe_is_refused(self):
        from training.commands.train_step import StepUnsupported, _check_supported
        from training.recipes.loader import load_recipe

        recipe = load_recipe(os.path.join(REFERENCE_DIR, "continual-gemma4-e2b.yaml"))
        with pytest.raises(StepUnsupported, match="zero_offload"):
            _check_supported(recipe)

    def test_scratch_and_plain_sft_recipes_are_supported(self):
        from training.commands.train_step import _check_supported
        from training.recipes.loader import load_recipe

        _check_supported(load_recipe(TINY_RECIPE))
        _check_supported(load_recipe(os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml")))

    def test_builder_wraps_peft_only_when_enabled(self, monkeypatch):
        from training.commands import train_step as step_mod
        from training.recipes.loader import load_recipe

        calls = {}

        class Model:
            """Records train(): the built model must step in training mode,
            or HF's checkpointing layers stay inert (KB-89)."""

            config = types.SimpleNamespace(model_type="gemma4")

            def train(self):
                calls.setdefault("trained", []).append(self)
                return self

        base = Model()
        wrapped = Model()
        monkeypatch.setattr(step_mod, "build_model", lambda recipe: base)

        def fake_resolve(recipe, model_type):
            calls["resolved"] = model_type

        def fake_wrap(model, peft_cfg, gradient_checkpointing):
            calls["wrapped"] = model
            calls["checkpointing"] = gradient_checkpointing
            return wrapped

        fake_peft = types.SimpleNamespace(
            resolve_target_modules=fake_resolve, build_peft_model=fake_wrap
        )
        monkeypatch.setitem(sys.modules, "training.models.peft_utils", fake_peft)

        qlora = load_recipe(os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml"))
        assert step_mod._build_step_model(qlora) is wrapped
        # The wrap receives the recipe's checkpointing flag (KB-85), and the
        # wrapped model, not the base, is put in training mode (KB-89).
        assert calls == {
            "resolved": "gemma4",
            "wrapped": base,
            "checkpointing": True,
            "trained": [wrapped],
        }

        calls.clear()
        scratch = load_recipe(TINY_RECIPE)
        assert step_mod._build_step_model(scratch) is base
        assert calls == {"trained": [base]}
