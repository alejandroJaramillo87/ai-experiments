"""Shared model, optimizer, and scheduler builders (Phase 0 D6).

Extracted from training/calculator/calibrate.py (resolved D6 design decision 3)
so the calibrator and the D6 reference loop instantiate exactly the same model
for a recipe. Scratch specs build through HF LlamaConfig +
AutoModelForCausalLM.from_config; HF hub ids or local paths build through
from_pretrained. Both paths yield HF models, so the D6 checkpoint layout
(save_pretrained sharded safetensors) holds for both.

torch and transformers are imported lazily so the CPU sandbox can import this
module without the training environment installed.
"""

from __future__ import annotations

import sys

from omegaconf import DictConfig

from training.calculator.components import ArchitectureSpec, spec_from_recipe

_flash_fallback_warned = False


def attention_implementation() -> str:
    """flash_attention_2 when CUDA and flash-attn are usable, else sdpa.

    The CUDA training image ships flash-attn (D1); the CPU sandbox running the
    D6 resume test has neither, and sdpa is the deterministic-capable fallback.
    A CUDA host missing flash-attn is a degradation (wheel-tag drift is a
    known build risk), so that fallback is warned once per process; consumers
    stamp the returned value into run and eval metadata for provenance.
    """
    global _flash_fallback_warned
    import torch

    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception as exc:  # broken extension raises beyond ImportError
            if not _flash_fallback_warned:
                print(
                    f"[model] WARN: flash-attn unavailable on a CUDA host "
                    f"({exc}); falling back to sdpa attention",
                    file=sys.stderr,
                )
                _flash_fallback_warned = True
    return "sdpa"


def model_dtype(recipe: DictConfig):
    import torch

    return torch.bfloat16 if recipe.precision.dtype == "bf16" else torch.float32


def build_scratch_model(spec: ArchitectureSpec, recipe: DictConfig):
    """Instantiate the recipe's scratch architecture as an HF Llama model."""
    from transformers import AutoModelForCausalLM, LlamaConfig

    scratch = recipe.model.scratch
    config = LlamaConfig(
        num_hidden_layers=spec.layers,
        hidden_size=spec.hidden,
        num_attention_heads=spec.heads,
        num_key_value_heads=spec.kv_heads,
        intermediate_size=spec.ffn,
        vocab_size=spec.vocab,
        tie_word_embeddings=spec.tie_embeddings,
        rope_theta=scratch.rope_theta if scratch is not None else 10000.0,
        max_position_embeddings=recipe.data.seq_len,
        attn_implementation=attention_implementation(),
    )
    model = AutoModelForCausalLM.from_config(config, dtype=model_dtype(recipe))
    if recipe.memory.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def build_model(recipe: DictConfig):
    """Build the recipe's model: HF id via from_pretrained, scratch via spec.

    QLoRA base quantization (peft.quantize_base: nf4, F2) happens here at
    load; the PEFT adapter wrap is a separate step
    (training/models/peft_utils.py: build_peft_model), owned by the entry point.
    """
    if recipe.model.hf is not None:
        from transformers import AutoModelForCausalLM

        from training.models.peft_utils import bnb_quantization_config

        kwargs = {}
        quantization = (
            bnb_quantization_config(recipe.peft) if recipe.peft.enabled else None
        )
        if quantization is not None:
            kwargs["quantization_config"] = quantization
        model = AutoModelForCausalLM.from_pretrained(
            recipe.model.hf.id,
            dtype=model_dtype(recipe),
            attn_implementation=attention_implementation(),
            **kwargs,
        )
        if recipe.memory.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model
    return build_scratch_model(spec_from_recipe(recipe), recipe)


def build_optimizer(model, recipe: DictConfig, name: str | None = None, params=None):
    """The plain-loop optimizer for a recipe.

    name overrides recipe.optimizer.name (the offload fallback chain resolves
    an effective optimizer, training/runtime/offload.py); params overrides
    model.parameters() so a degraded offload run keeps its trainable filter.
    """
    name = name or recipe.optimizer.name
    params = params if params is not None else model.parameters()
    kwargs = {
        "lr": recipe.optimizer.lr,
        "betas": tuple(recipe.optimizer.betas),
        "eps": recipe.optimizer.eps,
        "weight_decay": recipe.optimizer.weight_decay,
    }
    if name == "adamw":
        import torch

        return torch.optim.AdamW(params, **kwargs)
    if name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, **kwargs)
    if name == "paged_adamw8bit":
        # 8-bit states in CUDA unified memory, paged to host under pressure:
        # the engine-free middle rung of the offload fallback chain.
        import bitsandbytes as bnb

        return bnb.optim.PagedAdamW8bit(params, **kwargs)
    # cpu_adam lives inside the DeepSpeed engine (P2-D5): the loop's offload
    # path builds it through training/runtime/offload.initialize_engine, never here.
    raise ValueError(
        f"The plain loop has no optimizer for '{name}'; "
        "cpu_adam requires memory.offload=zero_offload (P2-D5 engine path)"
    )
