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
        out = capsys.readouterr().out
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
        out = capsys.readouterr().out
        assert rc == 1
        assert "[ERROR] Does not fit: reserved estimate" in out
        assert "exceeds budget 31.00 GB" in out
        assert "Solution: reduce data.micro_batch or data.seq_len" in out

    def test_solve_batch_reports_search_cap_for_tiny_model(self, capsys):
        # The tiny model fits at any micro-batch, so solve() lands exactly on
        # the MAX_MICRO_BATCH search bound.
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "--solve", "batch"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "[OK] Solve batch: max micro-batch 4096" in out
        assert "(reserved estimate" in out

    def test_solve_seq_reports_search_cap_for_tiny_model(self, capsys):
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "--solve", "seq"])
        out = capsys.readouterr().out
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
        out = capsys.readouterr().out
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

    def test_offload_recipe_reports_host_ram(self, capsys):
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "memory.offload=zero_offload"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "[INFO] Host RAM (offloaded optimizer state + replay budget):" in out
        assert "of 128 GB" in out

    def test_missing_recipe_file_is_styled_load_error(self, capsys):
        rc = run_cli(["memfit", "--config", "/nonexistent/recipe.yaml"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "[ERROR] Recipe load failed:" in out
        assert "Recipe file not found: /nonexistent/recipe.yaml" in out
        assert "Solution: Fix the reported key or type" in out

    def test_invalid_override_is_styled_validation_error(self, capsys):
        rc = run_cli(["memfit", "--config", TINY_RECIPE, "data.micro_batch=0"])
        out = capsys.readouterr().out
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
        out = capsys.readouterr().out
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
        out = capsys.readouterr().out
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
        out = capsys.readouterr().out
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
        out = capsys.readouterr().out
        assert rc == 0
        assert (
            f"[WARN] Parameter count drift: model {TINY_PARAMS + 2:,} "
            f"vs analytic {TINY_PARAMS:,}"
        ) in out
        assert "Solution: align the ffn/param formulas" in out
        assert "[OK] Parameter count:" not in out

    @pytest.mark.parametrize("error_pct,shown", [(-12.5, "-12.5%"), (10.1, "+10.1%")])
    def test_allocated_error_beyond_bar_exits_one(
        self, capsys, monkeypatch, error_pct, shown
    ):
        patch_calibrate(monkeypatch, record=make_record(allocated_error_pct=error_pct))
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().out
        assert rc == 1
        assert (
            f"[ERROR] Allocated error: {shown} (exceeds the 10% acceptance bar)"
        ) in out

    def test_reserved_underprediction_exits_one(self, capsys, monkeypatch):
        patch_calibrate(
            monkeypatch, record=make_record(reserved_underpredicts_nvml=True)
        )
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().out
        assert rc == 1
        assert "[ERROR] Reserved estimate under-predicts the NVML peak" in out
        assert "Solution: raise OVERHEAD_BYTES" in out

    def test_runtime_error_from_loop_is_styled_failure(self, capsys, monkeypatch):
        patch_calibrate(monkeypatch, error=RuntimeError("no CUDA device visible"))
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().out
        assert rc == 1
        assert "[ERROR] Calibration failed: no CUDA device visible" in out
        assert "Solution: run on the workstation" in out

    def test_sandbox_without_gpu_fails_styled_for_real(self, capsys):
        # No patching: the real calibrate() either fails importing torch
        # (this sandbox) or raises its no-CUDA RuntimeError (CPU host);
        # both must land in the same styled exit-1 arm.
        rc = run_cli(["calibrate", "--config", TINY_RECIPE])
        out = capsys.readouterr().out
        assert rc == 1
        assert "[ERROR] Calibration failed:" in out
        assert "Solution: run on the workstation" in out

    def test_missing_recipe_file_exits_one_before_calibrating(self, capsys):
        rc = run_cli(["calibrate", "--config", "/nonexistent/recipe.yaml"])
        out = capsys.readouterr().out
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
