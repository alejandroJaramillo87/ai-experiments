"""In-process handler tests for python -m training memfit and calibrate.

Exercises cmd_memfit, cmd_calibrate, and the shared _load_validated /
_print_estimate_table helpers through build_parser() + args.func(args), the
same argv path production takes. memfit is pure calculator math and runs for
real against the reference recipes; calibrate's GPU loop is the only faked
seam (training.calculator.calibrate.calibrate), with records built from the
real CalibrationRecord dataclass so as_dict() and write_record stay real.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest
from support import REFERENCE_DIR, TINY_RECIPE

from training import cli
from training.calculator.calibrate import CalibrationRecord

GIB = 1 << 30
PRETRAIN_1B = os.path.join(REFERENCE_DIR, "pretrain-1b.yaml")
DISTILL_LOGIT = os.path.join(REFERENCE_DIR, "distill-logit-gemma-e2b.yaml")

# tiny-e2e scratch model: 2 layers, hidden 64, vocab 512 -> the analytic
# param count pinned by tests/training/test_calculator.py.
TINY_PARAMS = 155_968


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


class TestMemfit:
    def test_fitting_recipe_prints_table_and_exits_zero(self, capsys):
        rc = run_cli(["memfit", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert "=== Memory-fit estimate ===" in out
        assert f"Model parameters: {TINY_PARAMS:,}" in out
        assert "Micro-batch 2, sequence length 128" in out
        for component in (
            "weights",
            "gradients",
            "optimizer state",
            "activations",
            "framework overhead",
        ):
            assert component in out
        assert "[OK] Fits: reserved estimate" in out
        assert "within budget 31.00 GB" in out
        # No distillation and no offload: the conditional rows must not print.
        assert "teacher (distill)" not in out
        assert "Host RAM" not in out

    def test_oversized_batch_does_not_fit_and_exits_one(self, capsys):
        rc = run_cli(["memfit", "--config", PRETRAIN_1B, "data.micro_batch=512"])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Does not fit: reserved estimate" in out
        assert "exceeds budget 31.00 GB" in out
        assert "Solution: reduce data.micro_batch or data.seq_len" in out

    def test_solve_batch_reports_search_cap_for_tiny_model(self, capsys):
        # The tiny model fits at any micro-batch, so solve() lands exactly on
        # the MAX_MICRO_BATCH search bound.
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "--solve", "batch"])
        out = capsys.readouterr().err
        assert rc == 0
        assert "[OK] Solve batch: max micro-batch 4096" in out
        assert "(reserved estimate" in out

    def test_solve_seq_reports_search_cap_for_tiny_model(self, capsys):
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "--solve", "seq"])
        out = capsys.readouterr().err
        assert rc == 0
        assert "[OK] Solve seq: max sequence length 1048576" in out

    def test_solve_with_nothing_fitting_exits_one(self, capsys):
        # 100k micro-batch overflows even one SEQ_STEP of sequence length:
        # a failed solve is exit 1 per the exit-code contract, not exit 0
        # with an [ERROR] line.
        rc = run_cli(
            [
                "memfit",
                "--config",
                PRETRAIN_1B,
                "--solve",
                "seq",
                "data.micro_batch=100000",
            ]
        )
        out = capsys.readouterr().err
        assert rc == 1
        assert (
            "[ERROR] Solve seq: no sequence length fits at the recipe's other settings"
        ) in out

    def test_json_mode_emits_only_json(self, capsys):
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "--solve", "batch", "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        result = json.loads(out)  # stdout must parse as one JSON document
        assert result["estimate"]["fits"] is True
        assert result["estimate"]["param_count"] == TINY_PARAMS
        assert result["estimate"]["teacher"] == 0
        assert result["max_batch"] == 4096
        assert result["max_batch_estimate"]["micro_batch"] == 4096

    def test_checkpointing_prints_peak_phase_note(self, capsys):
        # pretrain-1b checkpoints; the total is a temporal max, and the
        # table must say so (KB-56). tiny-e2e does not; no note.
        rc = run_cli(["memfit", "--config", PRETRAIN_1B])
        out = capsys.readouterr().err
        assert rc == 0
        assert "Peak phase:" in out
        rc = run_cli(["memfit", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert "Peak phase:" not in out

    def test_json_carries_peak_phase(self, capsys):
        rc = run_cli(["memfit", "--config", PRETRAIN_1B, "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        result = json.loads(out)
        assert result["estimate"]["peak_phase"] in ("late_backward", "head_backward")

    def test_offload_recipe_reports_host_ram(self, capsys):
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "memory.offload=zero_offload"])
        out = capsys.readouterr().err
        assert rc == 0
        assert "[INFO] Host RAM (offloaded optimizer state + replay budget):" in out
        assert "of 128 GB" in out

    def test_missing_recipe_file_is_styled_load_error(self, capsys):
        rc = run_cli(["memfit", "--config", "/nonexistent/recipe.yaml"])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Recipe load failed:" in out
        assert "Recipe file not found: /nonexistent/recipe.yaml" in out
        assert "Solution: Fix the reported key or type" in out

    def test_invalid_override_is_styled_validation_error(self, capsys):
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "data.micro_batch=0"])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Recipe validation failed:" in out
        assert "- data.micro_batch: must be >= 1, got 0" in out

    def test_distill_recipe_prints_teacher_row(self, capsys, monkeypatch):
        # Online logit distillation resolves both student and teacher through
        # transformers.AutoConfig; a stub keeps the sandbox torch-free while
        # the component math runs for real on the returned tiny config.
        config = types.SimpleNamespace(
            model_type="llama",
            num_hidden_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=256,
            vocab_size=512,
            tie_word_embeddings=True,
        )
        stub = types.ModuleType("transformers")
        stub.AutoConfig = types.SimpleNamespace(
            from_pretrained=lambda model_id, **kw: config
        )
        monkeypatch.setitem(sys.modules, "transformers", stub)

        rc = run_cli(["memfit", "--config", DISTILL_LOGIT])
        out = capsys.readouterr().err
        assert rc == 0
        teacher_rows = [
            line for line in out.splitlines() if "teacher (distill)" in line
        ]
        assert len(teacher_rows) == 1
        # The nf4 teacher weights plus a forward pass are never zero bytes.
        assert "teacher (distill)                 0  " not in teacher_rows[0]
        assert "[OK] Fits:" in out


def make_record(**overrides) -> CalibrationRecord:
    """A real CalibrationRecord with passing defaults: analytic matches the
    model, +4.2% allocated error, reserved covering the NVML peak."""
    fields = {
        "recipe_run_name": "tiny-e2e",
        "steps": 5,
        "model_param_count": TINY_PARAMS,
        "analytic_param_count": TINY_PARAMS,
        "estimate": {"allocated_total": 2 * GIB},
        "measured_allocated": int(2.1 * GIB),
        "measured_reserved": int(2.5 * GIB),
        "measured_nvml_peak": int(2.4 * GIB),
        "allocated_error_pct": 4.2,
        "reserved_underpredicts_nvml": False,
    }
    fields.update(overrides)
    return CalibrationRecord(**fields)


def patch_calibrate(monkeypatch, record=None, error=None):
    """Replace the GPU calibration loop, the one seam a sandbox cannot run."""

    def fake_calibrate(recipe, steps):
        if error is not None:
            raise error
        return record

    monkeypatch.setattr("training.calculator.calibrate.calibrate", fake_calibrate)


class TestCalibrate:
    def test_within_bar_exits_zero(self, capsys, monkeypatch):
        patch_calibrate(monkeypatch, record=make_record())
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert "[RUN] Calibrating tiny-e2e: 5 steps on cuda:0" in out
        assert (
            f"[OK] Parameter count: {TINY_PARAMS:,} (analytic matches model)"
        ) in out
        assert "Estimated allocated: 2.00 GB" in out
        assert "Measured allocated:  2.10 GB" in out
        assert "Measured reserved:   2.50 GB" in out
        assert "NVML peak:           2.40 GB" in out
        assert ("[OK] Allocated error: +4.2% (within the 10% acceptance bar)") in out
        assert "[OK] Reserved estimate covers the NVML peak" in out

    def test_out_flag_writes_record_and_steps_flag_flows_through(
        self, capsys, monkeypatch, tmp_path
    ):
        record = make_record(steps=3)
        patch_calibrate(monkeypatch, record=record)
        out_path = str(tmp_path / "records" / "calibration.json")
        rc = run_cli(
            ["calibrate", "--config", TINY_RECIPE, "--steps", "3", "--out", out_path]
        )
        out = capsys.readouterr().err
        assert rc == 0
        assert "[RUN] Calibrating tiny-e2e: 3 steps on cuda:0" in out
        assert f"[OK] Calibration record written: {out_path}" in out
        # The real write_record ran: the file round-trips the record.
        with open(out_path) as f:
            assert json.load(f) == record.as_dict()

    def test_param_count_drift_warns_but_still_passes(self, capsys, monkeypatch):
        patch_calibrate(
            monkeypatch, record=make_record(model_param_count=TINY_PARAMS + 2)
        )
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert (
            f"[WARN] Parameter count drift: model {TINY_PARAMS + 2:,} "
            f"vs analytic {TINY_PARAMS:,}"
        ) in out
        assert "Solution: align the ffn/param formulas" in out
        assert "[OK] Parameter count:" not in out

    @pytest.mark.parametrize(
        "error_pct,measured,shown",
        # measured_allocated kept consistent with error_pct so the absolute
        # gap also exceeds the 128 MiB floor and the error branch is exercised
        [
            (-12.5, int(2 / 0.875 * GIB), "-12.5%"),
            (10.1, int(2 / 1.101 * GIB), "+10.1%"),
        ],
    )
    def test_allocated_error_beyond_bar_exits_one(
        self, capsys, monkeypatch, error_pct, measured, shown
    ):
        patch_calibrate(
            monkeypatch,
            record=make_record(
                allocated_error_pct=error_pct, measured_allocated=measured
            ),
        )
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert (
            f"[ERROR] Allocated error: {shown} (exceeds the 10% acceptance bar)"
        ) in out

    def test_tiny_model_passes_via_absolute_floor(self, capsys, monkeypatch):
        # Fixed CUDA workspaces dominate a tiny model's measured allocated;
        # the relative error is huge but the absolute gap sits under the floor.
        record = make_record(
            estimate={"allocated_total": 5 << 20},
            measured_allocated=70 << 20,
            allocated_error_pct=-92.9,
        )
        patch_calibrate(monkeypatch, record=record)
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert (
            "[OK] Allocated error: -92.9% (absolute gap 0.06 GB within the "
            "128 MB floor)"
        ) in out

    def test_reserved_underprediction_exits_one(self, capsys, monkeypatch):
        patch_calibrate(
            monkeypatch, record=make_record(reserved_underpredicts_nvml=True)
        )
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Reserved estimate under-predicts the NVML peak" in out
        assert "Solution: raise OVERHEAD_BYTES" in out

    def test_runtime_error_from_loop_is_styled_failure(self, capsys, monkeypatch):
        patch_calibrate(monkeypatch, error=RuntimeError("no CUDA device visible"))
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Calibration failed: no CUDA device visible" in out
        assert "Solution: run on the workstation" in out

    def test_without_gpu_fails_styled_for_real(self, capsys, monkeypatch):
        # calibrate() itself is unpatched: it either fails importing torch
        # (this sandbox) or raises its no-CUDA RuntimeError, and both must
        # land in the same styled exit-1 arm. Only torch's CUDA probe is
        # forced, so the GPU workstation takes the same arm as the sandbox.
        try:
            import torch
        except ImportError:
            torch = None  # type: ignore[assignment]
        if torch is not None:
            monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Calibration failed:" in out
        assert "Solution: run on the workstation" in out

    def test_missing_recipe_file_exits_one_before_calibrating(self, capsys):
        rc = run_cli(["calibrate", "--config", "/nonexistent/recipe.yaml"])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Recipe load failed:" in out
        assert "[RUN] Calibrating" not in out

    def test_json_mode_emits_only_the_record(self, capsys, monkeypatch):
        record = make_record()
        patch_calibrate(monkeypatch, record=record)
        rc = run_cli(["calibrate", "--config", TINY_RECIPE, "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        assert json.loads(out) == record.as_dict()

    def test_unsupported_recipe_is_refused_without_workstation_hint(
        self, capsys, monkeypatch
    ):
        from training.calculator.calibrate import CalibrationUnsupported

        patch_calibrate(
            monkeypatch,
            error=CalibrationUnsupported("memory.offload=zero_offload is not"),
        )
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 1
        assert "[ERROR] Calibration unsupported:" in out
        assert "Solution: run on the workstation" not in out

    def test_trainable_count_drift_warns_but_still_passes(self, capsys, monkeypatch):
        record = make_record(
            model_trainable_params=1_234, analytic_trainable_params=1_200
        )
        patch_calibrate(monkeypatch, record=record)
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().err
        assert rc == 0
        assert "[WARN] Trainable count drift: model 1,234 vs analytic 1,200" in out
        assert "Solution: align adapter_param_count" in out


class TestCalibrateSupport:
    """_check_supported and _build_calibration_model (KB-03), stubs only."""

    def test_zero_offload_recipe_is_refused(self):
        from training.calculator.calibrate import (
            CalibrationUnsupported,
            _check_supported,
        )
        from training.recipes.loader import load_recipe

        recipe = load_recipe(os.path.join(REFERENCE_DIR, "continual-gemma4-e2b.yaml"))
        with pytest.raises(CalibrationUnsupported, match="zero_offload"):
            _check_supported(recipe)

    def test_online_logit_distill_recipe_is_refused(self):
        from training.calculator.calibrate import (
            CalibrationUnsupported,
            _check_supported,
        )
        from training.recipes.loader import load_recipe

        recipe = load_recipe(DISTILL_LOGIT)
        with pytest.raises(CalibrationUnsupported, match="in-process teacher"):
            _check_supported(recipe)

    def test_scratch_and_plain_sft_recipes_are_supported(self):
        from training.calculator.calibrate import _check_supported
        from training.recipes.loader import load_recipe

        _check_supported(load_recipe(TINY_RECIPE))
        _check_supported(load_recipe(os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml")))

    def test_builder_wraps_peft_only_when_enabled(self, monkeypatch):
        from training.calculator import calibrate as calibrate_mod
        from training.recipes.loader import load_recipe

        base = types.SimpleNamespace(config=types.SimpleNamespace(model_type="gemma4"))
        wrapped = object()
        calls = {}
        monkeypatch.setattr(calibrate_mod, "build_model", lambda recipe: base)

        def fake_resolve(recipe, model_type):
            calls["resolved"] = model_type

        def fake_wrap(model, peft_cfg):
            calls["wrapped"] = model
            return wrapped

        fake_peft = types.SimpleNamespace(
            resolve_target_modules=fake_resolve, build_peft_model=fake_wrap
        )
        monkeypatch.setitem(sys.modules, "training.models.peft_utils", fake_peft)

        qlora = load_recipe(os.path.join(REFERENCE_DIR, "sft-qlora-e2b.yaml"))
        assert calibrate_mod._build_calibration_model(qlora) is wrapped
        assert calls == {"resolved": "gemma4", "wrapped": base}

        calls.clear()
        scratch = load_recipe(TINY_RECIPE)
        assert calibrate_mod._build_calibration_model(scratch) is base
        assert calls == {}
