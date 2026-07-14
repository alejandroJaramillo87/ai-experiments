"""Subject-model loading and chat formatting (interp M3).

load_subject returns a hooked model at bf16 on cuda: a TransformerLens HookedTransformer
where the architecture is supported, an nnsight LanguageModel otherwise. The registry maps
known subject ids to their loading quirks; unknown models fail loudly rather than guess,
because a silently wrong chat template or processing mode corrupts every downstream
measurement.

Interp libraries are imported lazily so host-side `make test` runs without the interp
dependency group installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubjectSpec:
    """Loading quirks for a known subject model."""

    loader: str  # "tlens" | "nnsight"
    # TransformerLens from_pretrained_no_processing preserves the exact computation
    # (no folding/centering); required when activations feed SAEs trained on the raw model.
    no_processing: bool = False
    chat: bool = False  # has a chat template (instruct/it variants)


# Phase 0/1 subjects. Extend deliberately, one entry per model actually used by a scaffold.
REGISTRY: dict[str, SubjectSpec] = {
    "gpt2": SubjectSpec(loader="tlens"),
    "google/gemma-2-2b": SubjectSpec(loader="tlens", no_processing=True),
    "google/gemma-2-2b-it": SubjectSpec(loader="tlens", no_processing=True, chat=True),
    "Qwen/Qwen3-4B": SubjectSpec(loader="nnsight", chat=True),
    "meta-llama/Llama-3.1-8B-Instruct": SubjectSpec(loader="nnsight", chat=True),
}


def load_subject(model_id: str, revision: str | None = None):
    """Load a known subject at bf16 on cuda; unknown ids raise KeyError."""
    if model_id not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown subject model {model_id!r}; registry has: {known}")
    spec = REGISTRY[model_id]
    import torch

    if spec.loader == "tlens":
        from transformer_lens import HookedTransformer

        loader = (
            HookedTransformer.from_pretrained_no_processing
            if spec.no_processing
            else HookedTransformer.from_pretrained
        )
        kwargs = {"dtype": torch.bfloat16}
        if revision is not None:
            kwargs["revision"] = revision
        return loader(model_id, **kwargs).to("cuda")

    from nnsight import LanguageModel

    kwargs = {"device_map": "cuda", "torch_dtype": torch.bfloat16, "dispatch": True}
    if revision is not None:
        kwargs["revision"] = revision
    return LanguageModel(model_id, **kwargs)


def format_chat(model_id: str, messages: list[dict[str, str]]) -> str:
    """Render messages through the model's own chat template, never by hand.

    Hand-rolled templates are the classic silent-corruption path: a missing BOS or wrong
    turn delimiter shifts every activation downstream.
    """
    spec = REGISTRY.get(model_id)
    if spec is None:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown subject model {model_id!r}; registry has: {known}")
    if not spec.chat:
        raise ValueError(f"{model_id} has no chat template; use raw prompts")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
