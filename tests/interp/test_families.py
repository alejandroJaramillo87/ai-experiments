"""interp/families.py: the per-family hook-point registry (WI-6).

CPU-only: the traversal test uses a hand-rolled stand-in model (nested
attributes plus an indexable layer list), the same seam style as the harvest
fakes; no torch or transformers involved.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from interp import families
from interp.families import (
    GEMMA_SCOPE_MODEL_ID,
    UnknownHookFamilyError,
    family_from_config,
    hook_family,
    hook_name,
    residual_width,
    resolve_submodule,
    supported_families,
    unwrap_text_config,
)


class TestRegistry:
    def test_registered_families(self):
        assert supported_families() == ("gemma2", "gemma4", "gemma4_text", "qwen3")

    def test_hook_family_returns_spec(self):
        spec = hook_family("gemma4")
        assert spec.model_type == "gemma4"
        assert spec.hook_template == "model.layers.{layer}"

    def test_unknown_family_fails_loudly(self):
        with pytest.raises(UnknownHookFamilyError, match="mystery"):
            hook_family("mystery")
        with pytest.raises(UnknownHookFamilyError, match="interp/families.py"):
            hook_family("mystery")

    def test_gemma_scope_anchor_is_gemma2(self):
        # The interp ecosystem anchor (Gemma Scope, Neuronpedia, tracing);
        # its family must stay registered even though the training tier
        # does not register gemma2.
        assert GEMMA_SCOPE_MODEL_ID == "google/gemma-2-2b"
        assert hook_family("gemma2").hook_template == "model.layers.{layer}"


class TestConfigResolution:
    def test_unwrap_descends_into_text_config(self):
        inner = SimpleNamespace(model_type="gemma4_text", hidden_size=2304)
        composite = SimpleNamespace(model_type="gemma4", text_config=inner)
        assert unwrap_text_config(composite) is inner
        assert unwrap_text_config(inner) is inner

    def test_family_from_config_uses_unwrapped_model_type(self):
        composite = SimpleNamespace(
            model_type="gemma4",
            text_config=SimpleNamespace(model_type="gemma4_text", hidden_size=2304),
        )
        assert family_from_config(composite).model_type == "gemma4_text"
        flat = SimpleNamespace(model_type="qwen3", hidden_size=2048)
        assert family_from_config(flat).model_type == "qwen3"

    def test_family_from_config_unknown_type_raises(self):
        with pytest.raises(UnknownHookFamilyError):
            family_from_config(SimpleNamespace(model_type="mystery"))

    def test_residual_width_unwraps_and_defaults_to_zero(self):
        composite = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=2304, hidden_size_per_layer_input=256
            )
        )
        assert residual_width(composite) == 2304
        assert residual_width(SimpleNamespace()) == 0


class TinyBlock:
    def __init__(self, index: int) -> None:
        self.index = index


class TinyStandInModel:
    """Flat text-model layout stand-in: root.model.layers[i] like an HF
    CausalLM (and like an nnsight proxy, which traverses identically)."""

    def __init__(self, n_layers: int = 4) -> None:
        self.model = SimpleNamespace(layers=[TinyBlock(i) for i in range(n_layers)])


class TestResolveSubmodule:
    def test_walks_to_the_indexed_block(self):
        root = TinyStandInModel()
        spec = hook_family("gemma4")
        block = resolve_submodule(root, spec, 2)
        assert isinstance(block, TinyBlock)
        assert block.index == 2

    def test_matches_hook_name_path(self):
        spec = hook_family("qwen3")
        assert hook_name(spec, 15) == "model.layers.15"
        root = TinyStandInModel(n_layers=16)
        assert resolve_submodule(root, spec, 15).index == 15

    def test_missing_layout_raises_attribute_error(self):
        # The harvest dimension gate converts this into HarvestError; the
        # registry itself stays a thin traversal.
        spec = hook_family("gemma4")
        with pytest.raises(AttributeError):
            resolve_submodule(SimpleNamespace(), spec, 0)


class TestFamiliesModuleIsRegistryData:
    def test_all_specs_share_flat_decoder_layout_today(self):
        # When a composite layout lands, this test forces a deliberate update
        # here and in the harvest dimension gate.
        for model_type in supported_families():
            assert hook_family(model_type).hook_template == "model.layers.{layer}"

    def test_hook_name_formats_layer_int(self):
        spec = hook_family("gemma2")
        assert hook_name(spec, 12) == "model.layers.12"
        assert families.hook_name(spec, "7") == "model.layers.7"
