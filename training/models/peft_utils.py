"""PEFT construction glue (Phase 1 F2): LoRA, QLoRA, DoRA.

Adoption of Hugging Face PEFT and bitsandbytes; the only logic here is
resolving the recipe's peft group into library objects:

- resolve_target_modules() turns target_modules: "auto" into the concrete
  per-family list via training/models/families.py and writes it back into the
  recipe, so the run's config snapshot never contains "auto".
- bnb_quantization_config() maps quantize_base: nf4 onto a
  BitsAndBytesConfig for from_pretrained (the QLoRA base load).
- build_peft_model() wraps a loaded model with the recipe's adapter.
- trainable_parameter_report() is logged at run start; a QLoRA run
  reporting 100 percent trainable is a misconfiguration caught immediately.

torch/transformers/peft are imported lazily so the CPU sandbox can import
and unit-test the module without the training environment installed.
"""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig

from training.models.families import family_spec
from training.recipes.schema import TARGET_MODULES_AUTO

# torch dtype names for peft.compute_dtype; resolved lazily in _torch_dtype.
_COMPUTE_DTYPE_ATTR = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


def _torch_dtype(compute_dtype: str):
    import torch

    return getattr(torch, _COMPUTE_DTYPE_ATTR[compute_dtype])


def resolve_target_modules(recipe: DictConfig, model_type: str) -> list[str]:
    """Resolve peft.target_modules against the family registry.

    When the recipe says "auto", the family's verified module list replaces
    it in the recipe in place — callers resolve before write_snapshot so
    "auto" never appears in a run snapshot. An explicit list passes through
    untouched. model_type is the HF config's model_type field.
    """
    modules = recipe.peft.target_modules
    if modules == TARGET_MODULES_AUTO:
        resolved = list(family_spec(model_type).lora_target_modules)
        recipe.peft.target_modules = resolved
        return resolved
    return list(modules)


def bnb_quantization_config(peft_cfg: DictConfig):
    """BitsAndBytesConfig for the recipe, or None when the base is unquantized.

    Passed to AutoModelForCausalLM.from_pretrained as quantization_config;
    quantization happens at load, before build_peft_model wraps the model.
    """
    if peft_cfg.quantize_base != "nf4":
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=peft_cfg.double_quant,
        bnb_4bit_compute_dtype=_torch_dtype(peft_cfg.compute_dtype),
    )


def build_peft_model(model, peft_cfg: DictConfig):
    """Wrap a loaded (optionally 4-bit) model with the recipe's adapter.

    target_modules must already be a concrete list (resolve_target_modules
    ran); "auto" here is a programming error, not a user error.
    """
    modules = peft_cfg.target_modules
    if modules == TARGET_MODULES_AUTO:
        raise ValueError(
            "peft.target_modules is still 'auto'; call resolve_target_modules "
            "before build_peft_model so the snapshot carries the concrete list"
        )
    from peft import LoraConfig, get_peft_model

    if peft_cfg.quantize_base == "nf4":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=peft_cfg.rank,
        lora_alpha=peft_cfg.alpha,
        lora_dropout=peft_cfg.dropout,
        target_modules=[str(m) for m in modules],
        use_dora=peft_cfg.method == "dora",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


@dataclass(frozen=True)
class TrainableParameterReport:
    trainable: int
    total: int

    @property
    def percent(self) -> float:
        return 100.0 * self.trainable / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (
            f"trainable parameters: {self.trainable:,} of {self.total:,} "
            f"({self.percent:.2f} percent)"
        )


def trainable_parameter_report(model) -> TrainableParameterReport:
    """Count trainable vs total parameters; logged at run start.

    bitsandbytes 4-bit layers pack two 4-bit weights per stored element, so
    quantized frozen weights are counted at their logical element count when
    the layer exposes it (mirrors PEFT's print_trainable_parameters).
    """
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        if param.__class__.__name__ == "Params4bit" and hasattr(param, "quant_storage"):
            count *= 2
        total += count
        if param.requires_grad:
            trainable += count
    return TrainableParameterReport(trainable=trainable, total=total)
