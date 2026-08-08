"""Per-family hook-point registry for the interp tier.

Mirrors training/models/families.py: the registry is data, not behavior —
consumers resolve hook points through hook_family()/family_from_config() and
fail loudly on an unknown family rather than guess a module layout. Keyed by
HF config model_type after the text-config unwrap.

gemma2 is registered here although the training tier does not register it:
the interp tier anchors on Gemma-2-2B for pretrained SAEs (Gemma Scope),
Neuronpedia presets, and circuit tracing, because that ecosystem does not
exist for Gemma 4.

Hook paths are HF named-module paths; the output of a decoder layer is the
post-block residual stream. TransformerLens aliases are not used because
HookedTransformer has no Gemma 4 support (P4-D1 resolved decision).

A family registers two paths because the module layout depends on which
config the caller loaded, not on the family alone. Gemma 4 loaded from its
composite config nests the decoder under model.language_model.layers.{n},
alongside model.vision_tower; loaded from the bare text config it is the
flat model.layers.{n}. family_from_config() selects on the presence of
text_config.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    # The same, for a model built from a composite (multimodal) config that
    # nests the language model under a submodule. None when the family has
    # no composite form, which makes a composite config an error rather than
    # a silent fall back to the flat path.
    composite_hook_template: str | None = None


_FLAT_DECODER_TEMPLATE = "model.layers.{layer}"

# Verified against google/gemma-4-E2B at revision d29ff6b4 by enumerating
# named_modules() of a meta-device model built from its config: the decoder
# is model.language_model.layers.{n}, sibling to model.vision_tower.
_COMPOSITE_DECODER_TEMPLATE = "model.language_model.layers.{layer}"

_REGISTRY: dict[str, HookFamilySpec] = {
    spec.model_type: spec
    for spec in (
        HookFamilySpec("gemma4", _FLAT_DECODER_TEMPLATE, _COMPOSITE_DECODER_TEMPLATE),
        HookFamilySpec(
            "gemma4_text", _FLAT_DECODER_TEMPLATE, _COMPOSITE_DECODER_TEMPLATE
        ),
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
    """Resolve the registry entry for a loaded HF config.

    A composite config resolves through its text model_type but keeps the
    composite module layout: unwrapping determines the family, not the path.
    Conflating the two is what pointed every Gemma 4 hook at model.layers.
    """
    cfg = unwrap_text_config(hf_config)
    spec = hook_family(getattr(cfg, "model_type", "unknown"))
    if cfg is hf_config:
        return spec
    if spec.composite_hook_template is None:
        raise UnknownHookFamilyError(
            f"model_type {spec.model_type!r} was loaded from a composite "
            f"config but registers no composite hook path; the flat "
            f"{spec.hook_template!r} would not resolve "
            f"(add one in interp/families.py)"
        )
    return replace(spec, hook_template=spec.composite_hook_template)


def residual_width(hf_config) -> int:
    """hidden_size of the residual stream, 0 when the config lacks it."""
    return int(getattr(unwrap_text_config(hf_config), "hidden_size", 0) or 0)


def decoder_depth(hf_config) -> int:
    """num_hidden_layers of the decoder stack, 0 when the config lacks it.

    Read after the same text-config unwrap as residual_width: on a composite
    Gemma 4 config the depth lives on text_config, and the outer config has
    none.
    """
    return int(getattr(unwrap_text_config(hf_config), "num_hidden_layers", 0) or 0)


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
