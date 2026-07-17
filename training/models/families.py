"""Model-family registry (Phase 1 F2).

One place that knows each supported model family's architecture quirks, keyed
by the HF config model_type. F2 populates the LoRA target-module column;
F1 (chat/instruction data prep) extends each entry with its chat template so
a per-family fix — a PLE-model quirk, a renamed projection — lands once.

Model policy (Phase 1): Gemma 4 E-models (HF model_type gemma4, text config
gemma4_text; google/gemma-4-E2B and -E4B checkpoints, shipped April 2026)
are the primary family for every fine-tuning deliverable. Qwen3 is the
single fallback family — ungated weights, dense Llama-lineage decoder —
kept registered so a Gemma 4 library-support failure never blocks the
phase. No other family is registered; re-adding one is a one-line
FamilySpec with a verification note.

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
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Bumped whenever any ChatTemplateSpec changes; recorded in every prepared
# SFT dataset manifest so a template fix invalidates stale artifacts.
# v2: Gemma 4 corrected from the Gemma 3 <start_of_turn> framing to the
# official gemma4 template (<|turn>role ... <turn|>, native system turn),
# verified against transformers' convert_gemma4_weights.py and the
# chat_template.jinja mirrored in llama.cpp models/templates.
CHAT_TEMPLATE_VERSION = 2


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

    bos: str  # emitted once at sequence start; "" when none
    turn_prefix: str  # role header, formatted with {role}
    eot: str  # end-of-turn token proper (supervised on assistant turns)
    turn_suffix: str  # after eot, always masked
    assistant_role: str  # the template's name for the assistant role
    trim_content: bool  # whether the template applies | trim to content


# Gemma 4 (google/gemma-4-*): <|turn>role\n{content}<turn|>\n per turn, BOS
# once, native system turn, assistant role named "model", content trimmed.
# Thinking-mode tokens (<|think|> in the system turn, <|channel>thought ...
# <channel|> reasoning sections) are an inference-time concern: the official
# template strips thought channels from conversation history, so rendered
# training conversations contain none of them.
_GEMMA4_CHAT_TEMPLATE = ChatTemplateSpec(
    bos="<bos>",
    turn_prefix="<|turn>{role}\n",
    eot="<turn|>",
    turn_suffix="\n",
    assistant_role="model",
    trim_content=True,
)

_CHATML_CHAT_TEMPLATE = ChatTemplateSpec(
    bos="",
    turn_prefix="<|im_start|>{role}\n",
    eot="<|im_end|>",
    turn_suffix="\n",
    assistant_role="assistant",
    trim_content=False,
)


@dataclass(frozen=True)
class FamilySpec:
    """Per-family architecture facts and chat-template structure."""

    model_type: str
    # Module names for peft.target_modules: "auto" (F2 decision 2).
    lora_target_modules: tuple[str, ...]
    chat_template: ChatTemplateSpec
    # The mode a served checkpoint runs in when no thinking-control tokens
    # are injected (the eval pipeline injects none). Gemma 4 thinking is
    # opt-in (<|think|> in the system turn), so the default is off; Qwen3
    # hybrid checkpoints think by default (enable_thinking=True), so a
    # plain chat completion produces <think> traces.
    thinking_default: str = "off"
    # Reasoning-trace markers this family emits; SFT data prep refuses
    # assistant turns containing them (train on answers, not raw traces).
    thinking_markers: tuple[str, ...] = ()


_REGISTRY: dict[str, FamilySpec] = {
    spec.model_type: spec
    for spec in (
        # Gemma 4 E-models (per-layer embeddings; gemma-4-E2B/E4B checkpoints)
        # keep the Llama-lineage projection names in their text decoder
        # layers. The new relative_k_proj is deliberately not a LoRA target.
        FamilySpec(
            "gemma4",
            _LLAMA_LINEAGE_TARGET_MODULES,
            _GEMMA4_CHAT_TEMPLATE,
            thinking_default="off",
            thinking_markers=("<|think|>", "<|channel>thought"),
        ),
        FamilySpec(
            "gemma4_text",
            _LLAMA_LINEAGE_TARGET_MODULES,
            _GEMMA4_CHAT_TEMPLATE,
            thinking_default="off",
            thinking_markers=("<|think|>", "<|channel>thought"),
        ),
        # Qwen3: the fallback family (ungated, dense decoder; ChatML framing).
        # Hybrid checkpoints (the pinned Qwen/Qwen3-1.7B) think by default;
        # SFT datasets carry no think blocks (the trained model is steered
        # toward direct answers rather than supervised on empty traces).
        FamilySpec(
            "qwen3",
            _LLAMA_LINEAGE_TARGET_MODULES,
            _CHATML_CHAT_TEMPLATE,
            thinking_default="on",
            thinking_markers=("<think>", "</think>"),
        ),
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
            "training/models/families.py."
        ) from None
