"""Held-out loss evaluator (Phase 1 F7).

Mean cross-entropy over the supervised tokens of an F1 prepared dataset's
held-out split, for an arbitrary model (HF id, checkpoint directory, or
adapter-only checkpoint). Runs in-process in the training container with
inference down; the loss is plain CE over labels != -100, so the number is
directly comparable with the holdout curve the F3/F5 trainers emit.

Matched precision is the caller's contract: evaluate both sides of a
comparison with the same dtype and quantization flags; the returned record
names them so the scorecard can enforce it.

torch and transformers import lazily so the module stays importable on the
host for unit tests of the pure helpers.
"""

from __future__ import annotations

import os
from typing import Any

IGNORE_INDEX = -100


def resolve_model(model: str) -> tuple[str, str | None]:
    """(model_dir_or_id, adapter_dir_or_None) for any supported model form."""
    if not os.path.exists(model):
        return model, None  # hub id
    from training.export.merge import find_adapter, resolve_model_dir

    model_dir = resolve_model_dir(model)
    adapter = find_adapter(model_dir)
    if adapter is not None:
        from training.export.merge import adapter_base_model

        return adapter_base_model(adapter), adapter
    return model_dir, None


def load_model(model: str, load_4bit: bool = False):
    """Model ready for evaluation, adapter applied when present."""
    import torch
    from transformers import AutoModelForCausalLM

    from training.models.model import attention_implementation

    base, adapter = resolve_model(model)
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        # Explicit request with the flash->sdpa fallback; the record stamps
        # the same resolution so provenance matches the actual load.
        "attn_implementation": attention_implementation(),
    }
    if load_4bit:
        from transformers import BitsAndBytesConfig

        # Match the QLoRA training configuration (F3): nf4 storage, bf16 compute
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # Any: the result is either a transformers model or its PeftModel wrap;
    # both only need the duck-typed generate/eval surface
    loaded: Any = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    if adapter is not None:
        from peft import PeftModel

        loaded = PeftModel.from_pretrained(loaded, adapter)
    loaded.eval()
    return loaded


def evaluate_holdout(
    model: str,
    dataset_dir: str,
    load_4bit: bool = False,
    max_examples: int | None = None,
) -> dict:
    """Held-out loss record for one model over one prepared dataset.

    Loss is the token-weighted mean CE over supervised positions, identical
    in definition to the trainers' eval_loss, computed one example at a time
    (the holdout split is stored unpacked; see training/data/sft.py).
    """
    import math

    from training.data.sft import HOLDOUT_FILE, load_sft_dataset

    # Only the holdout split is materialized; the (much larger) train split
    # is hash-verified but never parsed into memory.
    _, holdout = load_sft_dataset(dataset_dir, parse=(HOLDOUT_FILE,))
    if not holdout:
        raise ValueError(
            f"Dataset has no held-out split: {dataset_dir}\n"
            "  Solution: re-run prepare_sft with --holdout > 0"
        )
    if max_examples is not None:
        holdout = holdout[:max_examples]

    import torch

    loaded = load_model(model, load_4bit=load_4bit)
    device = next(loaded.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for sample in holdout:
            input_ids = torch.tensor([sample["input_ids"]], device=device)
            labels = torch.tensor([sample["labels"]], device=device)
            supervised = int((labels != IGNORE_INDEX).sum())
            if supervised == 0:
                continue
            output = loaded(input_ids=input_ids, labels=labels)
            # loss is already the mean over this sample's supervised tokens;
            # re-weight by count so the aggregate is token-weighted
            total_loss += float(output.loss) * supervised
            total_tokens += supervised

    if total_tokens == 0:
        raise ValueError("Held-out split contains no supervised tokens")
    loss = total_loss / total_tokens
    from training.models.model import attention_implementation

    return {
        "model": model,
        "dataset": dataset_dir,
        "examples": len(holdout),
        "supervised_tokens": total_tokens,
        "loss": round(loss, 6),
        "perplexity": round(math.exp(loss), 6),
        "precision": "nf4-bf16" if load_4bit else "bf16",
        # Resolved attention path (flash_attention_2 or the sdpa fallback):
        # provenance alongside precision, same confound rationale.
        "attention_implementation": attention_implementation(),
    }
