"""PEFT glue tests (Phase 1 F2): family-registry lookups, target_modules
"auto" resolution into the config snapshot, and the trainable-parameter
report. CPU-only; torch/peft-dependent paths use stand-in objects, and
build_peft_model itself is exercised on the workstation (F2 acceptance)."""

from __future__ import annotations

import os

import pytest

from training.families import (
    UnknownModelFamilyError,
    family_spec,
    supported_families,
)
from training.peft_utils import (
    TrainableParameterReport,
    build_peft_model,
    resolve_target_modules,
    trainable_parameter_report,
)
from training.recipes import load_recipe, load_snapshot, write_snapshot

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFERENCE_DIR = os.path.join(REPO_ROOT, "training", "recipes", "reference")

LLAMA_LINEAGE = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]


def qlora_recipe(overrides=()):
    return load_recipe(
        os.path.join(REFERENCE_DIR, "sft-qlora-1b.yaml"), list(overrides))


class TestFamilyRegistry:
    def test_supported_families_cover_calculator_and_gemma3n(self):
        families = supported_families()
        for family in ("llama", "mistral", "qwen2", "qwen3",
                       "gemma", "gemma2", "gemma3", "gemma3n"):
            assert family in families

    @pytest.mark.parametrize("family", ["llama", "gemma3n"])
    def test_target_modules_are_llama_lineage(self, family):
        assert list(family_spec(family).lora_target_modules) == LLAMA_LINEAGE

    def test_unknown_family_fails_with_registration_message(self):
        with pytest.raises(UnknownModelFamilyError, match="training/families.py"):
            family_spec("mamba")


class TestResolveTargetModules:
    def test_auto_resolves_and_rewrites_recipe(self):
        recipe = qlora_recipe()
        resolved = resolve_target_modules(recipe, "llama")
        assert resolved == LLAMA_LINEAGE
        assert list(recipe.peft.target_modules) == LLAMA_LINEAGE

    def test_snapshot_never_contains_auto(self, tmp_path):
        recipe = qlora_recipe()
        resolve_target_modules(recipe, "llama")
        path = write_snapshot(recipe, str(tmp_path))
        assert "auto" not in open(path).read()
        reloaded = load_snapshot(path)
        assert list(reloaded.peft.target_modules) == LLAMA_LINEAGE

    def test_explicit_list_passes_through_unchanged(self):
        recipe = qlora_recipe(["peft.target_modules=[q_proj,v_proj]"])
        assert resolve_target_modules(recipe, "llama") == ["q_proj", "v_proj"]
        assert list(recipe.peft.target_modules) == ["q_proj", "v_proj"]

    def test_auto_with_unknown_family_fails(self):
        with pytest.raises(UnknownModelFamilyError):
            resolve_target_modules(qlora_recipe(), "mamba")

    def test_build_peft_model_rejects_unresolved_auto(self):
        recipe = qlora_recipe()
        with pytest.raises(ValueError, match="resolve_target_modules"):
            build_peft_model(object(), recipe.peft)


class _FakeParam:
    def __init__(self, n: int, requires_grad: bool) -> None:
        self._n = n
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._n


class _FakeModel:
    def __init__(self, params) -> None:
        self._params = params

    def parameters(self):
        return iter(self._params)


class TestTrainableParameterReport:
    def test_counts_and_percentage(self):
        model = _FakeModel([_FakeParam(950, False), _FakeParam(50, True)])
        report = trainable_parameter_report(model)
        assert report == TrainableParameterReport(trainable=50, total=1000)
        assert report.percent == 5.0
        assert "50" in str(report) and "percent" in str(report)

    def test_all_trainable_is_visible_misconfiguration(self):
        # The F2 tripwire: a QLoRA run reporting 100 percent trainable.
        report = trainable_parameter_report(_FakeModel([_FakeParam(10, True)]))
        assert report.percent == 100.0

    def test_empty_model(self):
        assert trainable_parameter_report(_FakeModel([])).percent == 0.0
