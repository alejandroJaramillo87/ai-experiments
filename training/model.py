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

from omegaconf import DictConfig

from training.calculator.components import ArchitectureSpec, spec_from_recipe


def attention_implementation() -> str:
    """flash_attention_2 when CUDA and flash-attn are usable, else sdpa.

    The CUDA training image ships flash-attn (D1); the CPU sandbox running the
    D6 resume test has neither, and sdpa is the deterministic-capable fallback.
    """
    import torch

    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except ImportError:
            pass
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
        attn_implementation=attention_implementation())
    model = AutoModelForCausalLM.from_config(config, dtype=model_dtype(recipe))
    if recipe.memory.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def build_model(recipe: DictConfig):
    """Build the recipe's model: HF id via from_pretrained, scratch via spec."""
    if recipe.model.hf is not None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            recipe.model.hf.id, dtype=model_dtype(recipe),
            attn_implementation=attention_implementation())
        if recipe.memory.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model
    return build_scratch_model(spec_from_recipe(recipe), recipe)


def build_optimizer(model, recipe: DictConfig):
    import torch

    params = model.parameters()
    kwargs = {
        "lr": recipe.optimizer.lr, "betas": tuple(recipe.optimizer.betas),
        "eps": recipe.optimizer.eps, "weight_decay": recipe.optimizer.weight_decay}
    if recipe.optimizer.name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    if recipe.optimizer.name == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, **kwargs)
    # cpu_adam requires a DeepSpeed engine; the Phase 0 reference loop is a
    # plain single-GPU loop, so the branch lands with the Phase 1 trainer.
    raise ValueError(
        f"The reference loop has no optimizer for '{recipe.optimizer.name}'")


# WSD decay fraction: the final 10 percent of total_steps decays linearly to
# zero; the stable plateau covers everything between warmup and decay.
WSD_DECAY_FRACTION = 0.1


def build_scheduler(optimizer, recipe: DictConfig):
    """LR schedule from the recipe: cosine or wsd, both with linear warmup."""
    import math

    import torch

    warmup = recipe.schedule.warmup_steps
    total = recipe.schedule.total_steps

    def cosine(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    def wsd(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        decay_start = total - int(total * WSD_DECAY_FRACTION)
        if step < decay_start:
            return 1.0
        return max(0.0, (total - step) / max(1, total - decay_start))

    fn = cosine if recipe.schedule.type == "cosine" else wsd
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
