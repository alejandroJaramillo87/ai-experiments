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
        assert supported_families() == ("gemma2", "gemma4", "gemma4_text")

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
        # The interp ecosystem anchor (Gemma Scope, tracing);
        # its family must stay registered even though the training tier
        # does not register gemma2.
        assert GEMMA_SCOPE_MODEL_ID == "google/gemma-2-2b"
        assert hook_family("gemma2").hook_template == "model.layers.{layer}"


class TestConfigResolution:
    def test_unwrap_descends_into_text_config(self):
        inner = SimpleNamespace(model_type="gemma4_text", hidden_size=1536)
        composite = SimpleNamespace(model_type="gemma4", text_config=inner)
        assert unwrap_text_config(composite) is inner
        assert unwrap_text_config(inner) is inner

    def test_family_from_config_uses_unwrapped_model_type(self):
        composite = SimpleNamespace(
            model_type="gemma4",
            text_config=SimpleNamespace(model_type="gemma4_text", hidden_size=1536),
        )
        assert family_from_config(composite).model_type == "gemma4_text"
        flat = SimpleNamespace(model_type="gemma2", hidden_size=2048)
        assert family_from_config(flat).model_type == "gemma2"

    def test_composite_config_resolves_to_the_composite_hook_path(self):
        # Regression: a Gemma 4 checkpoint loaded from its composite config
        # nests the decoder under model.language_model.layers.{n}, sibling to
        # model.vision_tower. Resolving it to the flat model.layers.{n} made
        # every E-series hook fail to attach. Verified against
        # google/gemma-4-E2B at revision d29ff6b4.
        composite = SimpleNamespace(
            model_type="gemma4",
            text_config=SimpleNamespace(model_type="gemma4_text", hidden_size=1536),
        )
        spec = family_from_config(composite)
        assert spec.model_type == "gemma4_text"
        assert hook_name(spec, 0) == "model.language_model.layers.0"

    def test_bare_text_config_keeps_the_flat_hook_path(self):
        text_only = SimpleNamespace(model_type="gemma4_text", hidden_size=1536)
        spec = family_from_config(text_only)
        assert hook_name(spec, 0) == "model.layers.0"

    def test_composite_config_without_a_composite_path_raises(self):
        # Silently falling back to the flat path is the defect above; a
        # family with no composite form must say so.
        composite = SimpleNamespace(
            model_type="wrapper",
            text_config=SimpleNamespace(model_type="gemma2", hidden_size=2048),
        )
        with pytest.raises(UnknownHookFamilyError, match="composite"):
            family_from_config(composite)

    def test_family_from_config_unknown_type_raises(self):
        with pytest.raises(UnknownHookFamilyError):
            family_from_config(SimpleNamespace(model_type="mystery"))

    def test_residual_width_unwraps_and_defaults_to_zero(self):
        composite = SimpleNamespace(
            text_config=SimpleNamespace(
                hidden_size=1536, hidden_size_per_layer_input=256
            )
        )
        assert residual_width(composite) == 1536
        assert residual_width(SimpleNamespace()) == 0


class TinyBlock:
    def __init__(self, index: int) -> None:
        self.index = index


class TinyStandInModel:
    """Flat text-model layout stand-in: root.model.layers[i] like an HF
    CausalLM, walked by attribute and index as resolve_submodule does."""

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
        spec = hook_family("gemma2")
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
    def test_every_spec_registers_the_flat_layout_as_its_base(self):
        # The composite path is an override selected by family_from_config,
        # never a replacement: hook_family() alone still yields the flat
        # layout for every family.
        for model_type in supported_families():
            assert hook_family(model_type).hook_template == "model.layers.{layer}"

    def test_only_the_gemma4_spellings_register_a_composite_layout(self):
        composite = {
            m for m in supported_families() if hook_family(m).composite_hook_template
        }
        assert composite == {"gemma4", "gemma4_text"}
        for model_type in composite:
            assert (
                hook_family(model_type).composite_hook_template
                == "model.language_model.layers.{layer}"
            )

    def test_hook_name_formats_layer_int(self):
        spec = hook_family("gemma2")
        assert hook_name(spec, 12) == "model.layers.12"
        assert families.hook_name(spec, "7") == "model.layers.7"
