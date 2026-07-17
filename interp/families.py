"""Per-family hook-point registry for the interp tier.

Mirrors training/models/families.py: the registry is data, not behavior —
consumers resolve hook points through hook_family()/family_from_config() and
fail loudly on an unknown family rather than guess a module layout. Keyed by
HF config model_type after the text-config unwrap.

gemma2 is registered here although the training tier does not register it:
the interp tier anchors on Gemma-2-2B for pretrained SAEs (Gemma Scope),
Neuronpedia presets, and circuit tracing, because that ecosystem does not
exist for Gemma 4.

Hook paths are HF named-module paths (model.layers.{n}); the output of a
decoder layer is the post-block residual stream. TransformerLens aliases are
not used because HookedTransformer has no Gemma 4 support (P4-D1 resolved
decision).
"""

from __future__ import annotations

from dataclasses import dataclass

# The Gemma-2-2B anchor for everything touching the pretrained-SAE ecosystem
# (Gemma Scope releases, Neuronpedia presets, circuit tracing).
GEMMA_SCOPE_MODEL_ID = "google/gemma-2-2b"
GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"
GEMMA_SCOPE_SAE_ID = "layer_12/width_16k/canonical"


class UnknownHookFamilyError(ValueError):
    """The model's family has no hook-point registry entry."""


@dataclass(frozen=True)
class HookFamilySpec:
    """Hook-point data for one HF model family (keyed by config model_type)."""

    model_type: str
    # HF named-module path template whose output is the post-block residual
    # stream; formatted with layer=<int>.
    hook_template: str


# Every registered family exposes the flat text-model decoder layout; a
# future composite layout gets its own template entry, and the harvest
# hook-dimension gate is the runtime backstop for layout drift.
_FLAT_DECODER_TEMPLATE = "model.layers.{layer}"

_REGISTRY: dict[str, HookFamilySpec] = {
    spec.model_type: spec
    for spec in (
        HookFamilySpec("gemma4", _FLAT_DECODER_TEMPLATE),
        HookFamilySpec("gemma4_text", _FLAT_DECODER_TEMPLATE),
        HookFamilySpec("gemma2", _FLAT_DECODER_TEMPLATE),
        HookFamilySpec("qwen3", _FLAT_DECODER_TEMPLATE),
    )
}


def supported_families() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def hook_family(model_type: str) -> HookFamilySpec:
    try:
        return _REGISTRY[model_type]
    except KeyError:
        raise UnknownHookFamilyError(
            f"model_type {model_type!r} has no hook-point registry entry; "
            f"supported: {', '.join(supported_families())} "
            f"(add it in interp/families.py)"
        ) from None


def unwrap_text_config(hf_config):
    """The text config of a composite multimodal config, or the config itself.

    Gemma 4 nests the language model under text_config; the residual stream
    is hidden_size there, not hidden_size_per_layer_input (PLE).
    """
    return getattr(hf_config, "text_config", hf_config)


def family_from_config(hf_config) -> HookFamilySpec:
    """Resolve the registry entry for a loaded HF config."""
    cfg = unwrap_text_config(hf_config)
    return hook_family(getattr(cfg, "model_type", "unknown"))


def residual_width(hf_config) -> int:
    """hidden_size of the residual stream, 0 when the config lacks it."""
    return int(getattr(unwrap_text_config(hf_config), "hidden_size", 0) or 0)


def hook_name(spec: HookFamilySpec, layer: int) -> str:
    """The family's HF named-module path for a decoder layer's output."""
    return spec.hook_template.format(layer=int(layer))


def resolve_submodule(root, spec: HookFamilySpec, layer: int):
    """Walk root to the layer's decoder block along the family's hook path.

    Generic attribute/index traversal so plain torch modules and nnsight
    proxies resolve identically (both support getattr and integer indexing).
    """
    node = root
    for part in hook_name(spec, layer).split("."):
        node = node[int(part)] if part.isdigit() else getattr(node, part)
    return node
