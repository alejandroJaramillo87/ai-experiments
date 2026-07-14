"""Model-family registry (Phase 1 F2).

One place that knows each supported model family's architecture quirks, keyed
by the HF config model_type. F2 populates the LoRA target-module column;
F1 (chat/instruction data prep) extends each entry with its chat template so
a per-family fix — a PLE-model quirk, a renamed projection — lands once.

Model policy (Phase 1): Gemma 4 E-models (HF model_type gemma3n) are the
primary family for every fine-tuning deliverable. Qwen3 is the single
fallback family — ungated weights, dense Llama-lineage decoder — kept
registered so a Gemma 4 library-support failure never blocks the phase.
No other family is registered; re-adding one is a one-line FamilySpec with
a verification note.

The registry is data, not behavior: consumers look entries up through
family_spec() and fail loudly on an unknown family rather than guess
(mirroring the D4 calculator's UnknownArchitectureError policy).
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownModelFamilyError(ValueError):
    """The HF model_type has no verified entry in the family registry."""


# The Llama-lineage projection names shared by both registered families:
# attention q/k/v/o plus SwiGLU gate/up/down. A future family that diverges
# gets its own literal list.
_LLAMA_LINEAGE_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Bumped whenever any ChatTemplateSpec changes; recorded in every prepared
# SFT dataset manifest so a template fix invalidates stale artifacts.
CHAT_TEMPLATE_VERSION = 1


@dataclass(frozen=True)
class ChatTemplateSpec:
    """Structural chat template (Phase 1 F1).

    Describes a family's turn framing precisely enough to render a
    conversation segment by segment, so assistant-token spans are known by
    construction rather than recovered by string matching on rendered text
    (F1 decision 3). Loss covers assistant content plus the turn's
    end-of-turn token; everything else — role headers, system and user
    turns, separators — is masked.

    These specs encode the pinned transformers rendering of each family's
    template; the F1 host-side acceptance compares them against the real
    tokenizer's apply_chat_template output before golden fixtures freeze.
    """

    bos: str            # emitted once at sequence start; "" when none
    turn_prefix: str    # role header, formatted with {role}
    eot: str            # end-of-turn token proper (supervised on assistant turns)
    turn_suffix: str    # after eot, always masked
    assistant_role: str  # the template's name for the assistant role
    # "turn": system renders as its own turn. "prefix_first_user": the
    # template has no system turn; system content is prepended to the first
    # user turn, joined by system_sep (the Gemma convention).
    system_style: str
    system_sep: str = "\n\n"


_GEMMA_CHAT_TEMPLATE = ChatTemplateSpec(
    bos="<bos>",
    turn_prefix="<start_of_turn>{role}\n",
    eot="<end_of_turn>",
    turn_suffix="\n",
    assistant_role="model",
    system_style="prefix_first_user",
)

_CHATML_CHAT_TEMPLATE = ChatTemplateSpec(
    bos="",
    turn_prefix="<|im_start|>{role}\n",
    eot="<|im_end|>",
    turn_suffix="\n",
    assistant_role="assistant",
    system_style="turn",
)


@dataclass(frozen=True)
class FamilySpec:
    """Per-family architecture facts and chat-template structure."""

    model_type: str
    # Module names for peft.target_modules: "auto" (F2 decision 2).
    lora_target_modules: tuple[str, ...]
    chat_template: ChatTemplateSpec


_REGISTRY: dict[str, FamilySpec] = {
    spec.model_type: spec
    for spec in (
        # Gemma 4 E-models (MatFormer/PLE; gemma-4-E2B/E4B checkpoints) keep
        # the Llama-lineage projection names in their text decoder layers.
        FamilySpec("gemma3n", _LLAMA_LINEAGE_TARGET_MODULES, _GEMMA_CHAT_TEMPLATE),
        FamilySpec("gemma3n_text", _LLAMA_LINEAGE_TARGET_MODULES, _GEMMA_CHAT_TEMPLATE),
        # Qwen3: the fallback family (ungated, dense decoder; ChatML framing).
        FamilySpec("qwen3", _LLAMA_LINEAGE_TARGET_MODULES, _CHATML_CHAT_TEMPLATE),
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
