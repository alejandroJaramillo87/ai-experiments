"""Model-family registry (Phase 1 F2).

One place that knows each supported model family's architecture quirks, keyed
by the HF config model_type. F2 populates the LoRA target-module column;
F1 (chat/instruction data prep) extends each entry with its chat template so
a per-family fix — a PLE-model quirk, a renamed projection — lands once.

The registry is data, not behavior: consumers look entries up through
family_spec() and fail loudly on an unknown family rather than guess
(mirroring the D4 calculator's UnknownArchitectureError policy).
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownModelFamilyError(ValueError):
    """The HF model_type has no verified entry in the family registry."""


# The Llama-lineage projection names shared by every currently supported
# family, including Gemma 3n (E2B/E4B): attention q/k/v/o plus SwiGLU
# gate/up/down. Families that diverge get their own literal list.
_LLAMA_LINEAGE_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass(frozen=True)
class FamilySpec:
    """Per-family architecture facts. F1 adds the chat-template column."""

    model_type: str
    # Module names for peft.target_modules: "auto" (F2 decision 2).
    lora_target_modules: tuple[str, ...]


_REGISTRY: dict[str, FamilySpec] = {
    spec.model_type: spec
    for spec in (
        FamilySpec("llama", _LLAMA_LINEAGE_TARGET_MODULES),
        FamilySpec("mistral", _LLAMA_LINEAGE_TARGET_MODULES),
        FamilySpec("qwen2", _LLAMA_LINEAGE_TARGET_MODULES),
        FamilySpec("qwen3", _LLAMA_LINEAGE_TARGET_MODULES),
        FamilySpec("gemma", _LLAMA_LINEAGE_TARGET_MODULES),
        FamilySpec("gemma2", _LLAMA_LINEAGE_TARGET_MODULES),
        FamilySpec("gemma3", _LLAMA_LINEAGE_TARGET_MODULES),
        FamilySpec("gemma3_text", _LLAMA_LINEAGE_TARGET_MODULES),
        # Gemma 3n (MatFormer/PLE; gemma-4-E2B/E4B checkpoints) keeps the
        # Llama-lineage projection names in its text decoder layers.
        FamilySpec("gemma3n", _LLAMA_LINEAGE_TARGET_MODULES),
        FamilySpec("gemma3n_text", _LLAMA_LINEAGE_TARGET_MODULES),
    )
}


def supported_families() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def family_spec(model_type: str) -> FamilySpec:
    """Look up a family by HF config model_type; fail loudly when unknown."""
    try:
        return _REGISTRY[model_type]
    except KeyError:
        raise UnknownModelFamilyError(
            f"HF model_type '{model_type}' has no entry in the family registry "
            f"{supported_families()}; guessing target modules risks silently "
            "training the wrong layers. Add a verified FamilySpec in "
            "training/families.py.") from None
