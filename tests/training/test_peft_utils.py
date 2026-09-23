"""PEFT glue tests (Phase 1 F2): family-registry lookups, target_modules
"auto" resolution into the config snapshot, and the trainable-parameter
report, and the composite-base exclusion (KB-83). CPU-only; torch/peft-dependent
paths use stand-in objects: build_peft_model runs against a fake peft module
here and against the real one on the workstation (F2 acceptance)."""

from __future__ import annotations

import re
import sys
import types
from types import SimpleNamespace

import pytest
from support import reference_recipe

from training.models.families import (
    UnknownModelFamilyError,
    family_spec,
    supported_families,
)
from training.models.peft_utils import (
    COMPOSITE_EXCLUDE_MODULES,
    TrainableParameterReport,
    build_peft_model,
    lora_exclude_modules,
    prepare_quantized_base,
    resolve_target_modules,
    trainable_parameter_report,
)
from training.recipes import load_snapshot, write_snapshot

LLAMA_LINEAGE = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def qlora_recipe(overrides=()):
    return reference_recipe("sft-qlora-e2b.yaml", list(overrides))


class TestFamilyRegistry:
    def test_registry_is_gemma4_only(self):
        assert supported_families() == ("gemma4", "gemma4_text")

    @pytest.mark.parametrize("family", ["gemma4", "gemma4_text"])
    def test_target_modules_are_llama_lineage(self, family):
        assert list(family_spec(family).lora_target_modules) == LLAMA_LINEAGE

    @pytest.mark.parametrize("family", ["mamba", "llama", "mistral", "gemma3n"])
    def test_unknown_family_fails_with_registration_message(self, family):
        with pytest.raises(
            UnknownModelFamilyError, match="training/models/families.py"
        ):
            family_spec(family)


class TestResolveTargetModules:
    def test_auto_resolves_and_rewrites_recipe(self):
        recipe = qlora_recipe()
        resolved = resolve_target_modules(recipe, "gemma4")
        assert resolved == LLAMA_LINEAGE
        assert list(recipe.peft.target_modules) == LLAMA_LINEAGE

    def test_snapshot_never_contains_auto(self, tmp_path):
        recipe = qlora_recipe()
        resolve_target_modules(recipe, "gemma4")
        path = write_snapshot(recipe, str(tmp_path))
        assert "auto" not in open(path).read()
        reloaded = load_snapshot(path)
        assert list(reloaded.peft.target_modules) == LLAMA_LINEAGE

    def test_explicit_list_passes_through_unchanged(self):
        recipe = qlora_recipe(["peft.target_modules=[q_proj,v_proj]"])
        assert resolve_target_modules(recipe, "gemma4") == ["q_proj", "v_proj"]
        assert list(recipe.peft.target_modules) == ["q_proj", "v_proj"]

    def test_auto_with_unknown_family_fails(self):
        with pytest.raises(UnknownModelFamilyError):
            resolve_target_modules(qlora_recipe(), "mamba")

    def test_build_peft_model_rejects_unresolved_auto(self):
        recipe = qlora_recipe()
        with pytest.raises(ValueError, match="resolve_target_modules"):
            build_peft_model(object(), recipe.peft)


# Module keys from docs/models/gemma4-safetensors.txt: the towers reuse the
# decoder's projection names behind Gemma4ClippableLinear wrappers.
TOWER_KEYS = [
    "model.vision_tower.encoder.layers.0.self_attn.q_proj",
    "model.vision_tower.encoder.layers.15.mlp.gate_proj",
    "model.audio_tower.layers.0.self_attn.q_proj",
    "model.audio_tower.layers.11.feed_forward1.ffw_layer_1",
    "model.embed_vision.embedding_projection",
]
LANGUAGE_KEYS = [
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.34.mlp.down_proj",
]
COMPOSITE_CONFIG = SimpleNamespace(model_type="gemma4", text_config=object())
PLAIN_CONFIG = SimpleNamespace(model_type="llama")


class TestExcludeModules:
    @pytest.mark.parametrize("key", TOWER_KEYS)
    def test_pattern_excludes_the_towers(self, key):
        # re.fullmatch is the call PEFT makes on a string exclude_modules.
        assert re.fullmatch(COMPOSITE_EXCLUDE_MODULES, key)

    @pytest.mark.parametrize("key", LANGUAGE_KEYS)
    def test_pattern_keeps_the_language_model(self, key):
        assert re.fullmatch(COMPOSITE_EXCLUDE_MODULES, key) is None

    def test_composite_config_gets_the_pattern(self):
        assert lora_exclude_modules(COMPOSITE_CONFIG) == COMPOSITE_EXCLUDE_MODULES

    def test_single_config_gets_none(self):
        assert lora_exclude_modules(PLAIN_CONFIG) is None


@pytest.fixture
def fake_peft(monkeypatch):
    """A peft stand-in that returns the LoraConfig kwargs it was handed.
    It carries no prepare_model_for_kbit_training: the wrap owns that step
    (KB-85), and a regression to PEFT's helper fails here on AttributeError."""
    module = types.ModuleType("peft")
    module.LoraConfig = lambda **kwargs: kwargs
    module.get_peft_model = lambda model, config: config
    monkeypatch.setitem(sys.modules, "peft", module)
    return module


# Dtype sentinels for the fake torch: identity is all prepare_quantized_base
# compares, so plain strings stand in for torch.float16 and the rest.
FAKE_TORCH = types.SimpleNamespace(float16="fp16", bfloat16="bf16", float32="fp32")


class _FakeData:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def to(self, dtype: str) -> _FakeData:
        return _FakeData(dtype)


class _FakeTensor:
    """A parameter as prepare_quantized_base reads it: dim, dtype, data."""

    def __init__(self, dims: int, dtype: str) -> None:
        self._dims = dims
        self.data = _FakeData(dtype)
        self.requires_grad = True

    def dim(self) -> int:
        return self._dims

    @property
    def dtype(self) -> str:
        return self.data.dtype


class _FakeBase:
    def __init__(self, params) -> None:
        self._params = params
        self.input_grads_enabled = 0

    def parameters(self):
        return iter(self._params)

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled += 1


@pytest.fixture
def fake_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    return FAKE_TORCH


class TestPrepareQuantizedBase:
    """KB-85: the k-bit preparation upcasts the norms and nothing else, and
    checkpointing is the recipe's call."""

    @staticmethod
    def _base():
        return _FakeBase(
            [
                _FakeTensor(1, "bf16"),  # a norm weight
                _FakeTensor(2, "bf16"),  # the embedding table
                _FakeTensor(2, "uint8"),  # a Params4bit storage tensor
                _FakeTensor(1, "fp32"),  # a norm already in fp32
            ]
        )

    def test_every_parameter_is_frozen(self, fake_torch):
        base = prepare_quantized_base(self._base(), gradient_checkpointing=True)
        assert [p.requires_grad for p in base.parameters()] == [False] * 4

    def test_only_one_dimensional_half_parameters_are_upcast(self, fake_torch):
        base = prepare_quantized_base(self._base(), gradient_checkpointing=False)
        assert [p.dtype for p in base.parameters()] == ["fp32", "bf16", "uint8", "fp32"]

    def test_fp16_norms_are_upcast_too(self, fake_torch):
        base = prepare_quantized_base(_FakeBase([_FakeTensor(1, "fp16")]), False)
        assert next(base.parameters()).dtype == "fp32"

    def test_input_grads_follow_the_recipe(self, fake_torch):
        on = prepare_quantized_base(self._base(), gradient_checkpointing=True)
        off = prepare_quantized_base(self._base(), gradient_checkpointing=False)
        assert on.input_grads_enabled == 1
        assert off.input_grads_enabled == 0


def quantized_base(config):
    base = _FakeBase([_FakeTensor(2, "bf16")])
    base.config = config
    return base


class TestBuildPeftModel:
    def test_composite_base_is_confined_to_the_language_model(
        self, fake_peft, fake_torch
    ):
        recipe = qlora_recipe()
        resolve_target_modules(recipe, "gemma4")
        config = build_peft_model(quantized_base(COMPOSITE_CONFIG), recipe.peft)
        assert config["exclude_modules"] == COMPOSITE_EXCLUDE_MODULES
        assert config["target_modules"] == LLAMA_LINEAGE
        assert config["task_type"] == "CAUSAL_LM"

    def test_single_config_base_has_no_exclusion(self, fake_peft, fake_torch):
        recipe = qlora_recipe(["peft.target_modules=[q_proj,v_proj]"])
        config = build_peft_model(quantized_base(PLAIN_CONFIG), recipe.peft)
        assert config["exclude_modules"] is None
        assert config["target_modules"] == ["q_proj", "v_proj"]

    def test_nf4_base_is_prepared_with_the_recipe_checkpointing(
        self, fake_peft, fake_torch
    ):
        # The wrap owns the k-bit preparation (KB-85): the table keeps its
        # dtype and input grads follow the flag the caller passes.
        recipe = qlora_recipe()
        resolve_target_modules(recipe, "gemma4")
        base = quantized_base(COMPOSITE_CONFIG)
        build_peft_model(base, recipe.peft, gradient_checkpointing=True)
        assert next(base.parameters()).dtype == "bf16"
        assert base.input_grads_enabled == 1
        untouched = quantized_base(COMPOSITE_CONFIG)
        build_peft_model(untouched, recipe.peft)
        assert untouched.input_grads_enabled == 0

    def test_unquantized_base_is_not_prepared(self, fake_peft):
        # No torch in sys.modules here: a LoRA base never enters the branch.
        recipe = qlora_recipe(["peft.quantize_base=none"])
        resolve_target_modules(recipe, "gemma4")
        config = build_peft_model(SimpleNamespace(config=PLAIN_CONFIG), recipe.peft)
        assert config["target_modules"] == LLAMA_LINEAGE


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
