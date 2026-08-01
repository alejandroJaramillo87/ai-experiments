"""Calibration harness for the memory-fit calculator (Phase 0 D4).

Runs a short real training loop for a recipe on the workstation GPU, records
torch.cuda.max_memory_allocated/reserved and the NVML peak, and writes a
calibration record JSON comparing measured peaks against estimate(). The
acceptance bar: the allocated estimate (a temporal max over backward phases
under gradient checkpointing, else the component sum) within 10 percent of
max_memory_allocated (or within ALLOCATED_FLOOR_BYTES absolute, for tiny
models where fixed allocator-resident workspaces dominate) on the two
reference configs, and the reserved estimate never under-predicting the
NVML peak.

The loop here is deliberately private to calibration: a few optimizer steps
over synthetic batches, no checkpointing, tracking, or resume. It is not the
D6 reference trainer (training/runtime/loop.py), which owns those facilities.

The model is built through the shared builders in training/models/model.py
(resolved design decision 2) so the measured model is exactly what training
runs instantiate: scratch recipes via LlamaConfig +
AutoModelForCausalLM.from_config, model.hf recipes via from_pretrained (with
the recipe's nf4 BitsAndBytesConfig), PEFT-wrapped through
training/models/peft_utils.py when peft.enabled (KB-03). The built model's
true parameter count is recorded so any drift from the analytic count is
caught immediately; under PEFT the trainable count is checked the same way.

Out of scope, refused with CalibrationUnsupported: memory.offload=zero_offload
recipes (the loop here is plain torch, not the DeepSpeed ZeRO engine, so its
measurement would not reflect that runtime) and online logit-distillation
recipes (the estimate prices an in-process teacher this loop does not run,
so the comparison would be skewed by exactly the teacher terms). The teacher
component stays analytic; response-tier and cached-mode distillation
recipes price no in-process teacher and calibrate normally.

GPU-only; torch is imported lazily so the CPU sandbox can import this module.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from omegaconf import DictConfig

from training.calculator.components import spec_from_recipe
from training.calculator.solve import Estimate, estimate
from training.models.model import (
    build_model,
    build_optimizer,  # noqa: F401  (re-export; extracted in D6)
    build_scratch_model,  # noqa: F401  (re-export kept for callers/tests)
)

DEFAULT_STEPS = 5


class CalibrationUnsupported(RuntimeError):
    """The recipe's runtime cannot be measured by the calibration loop."""


# Absolute tolerance for the allocated-error acceptance check. CUDA library
# workspaces (cuBLAS and friends) live inside max_memory_allocated but outside
# the component model; measured at ~65 MB on the RTX 5090. On a tiny reference
# model that fixed cost dominates and no relative bar can pass, so the check
# accepts an estimate within this absolute gap regardless of relative error.
# 128 MiB is above the workspace cost and below 2 percent of any real recipe.
ALLOCATED_FLOOR_BYTES = 128 << 20


@dataclass(frozen=True)
class CalibrationRecord:
    """Estimate versus measurement for one recipe on this machine."""

    recipe_run_name: str
    steps: int
    model_param_count: int  # torch-reported, must match the analytic count
    analytic_param_count: int
    estimate: dict  # Estimate.as_dict()
    measured_allocated: int  # torch.cuda.max_memory_allocated
    measured_reserved: int  # torch.cuda.max_memory_reserved
    measured_nvml_peak: int  # max NVML used-memory sample across steps
    allocated_error_pct: float  # (estimate - measured) / measured * 100
    reserved_underpredicts_nvml: bool
    # Trainable-parameter drift check (KB-03): equals the totals above on a
    # full-parameter run; under PEFT the model count is the adapter and the
    # analytic count is trainable_param_count from the estimate.
    model_trainable_params: int = 0
    analytic_trainable_params: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def _nvml_used_bytes() -> int:
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    finally:
        pynvml.nvmlShutdown()


def _check_supported(recipe: DictConfig) -> None:
    """Refuse recipes whose runtime the calibration loop cannot measure."""
    if recipe.memory.offload == "zero_offload":
        raise CalibrationUnsupported(
            "memory.offload=zero_offload is not calibratable: the calibration "
            "loop is plain torch, not the DeepSpeed ZeRO engine, so its "
            "measurement would not reflect the recipe's runtime"
        )
    d = getattr(recipe, "distill", None)
    if d is not None and d.enabled and d.tier == "logit" and d.mode == "online":
        raise CalibrationUnsupported(
            "online logit distillation is not calibratable: the estimate "
            "prices an in-process teacher the calibration loop does not run; "
            "calibrate the student recipe without distill, or use cached mode"
        )


def _build_calibration_model(recipe: DictConfig):
    """The recipe's actual model (KB-03): shared builders, PEFT-wrapped."""
    model = build_model(recipe)
    if recipe.peft.enabled:
        from training.models.peft_utils import (
            build_peft_model,
            resolve_target_modules,
        )

        resolve_target_modules(recipe, model.config.model_type)
        model = build_peft_model(model, recipe.peft)
    return model


def calibrate(recipe: DictConfig, steps: int = DEFAULT_STEPS) -> CalibrationRecord:
    """Run the calibration loop and return the estimate-vs-measured record."""
    import torch

    _check_supported(recipe)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Calibration requires the workstation GPU; run via "
            "'make train-calibrate CONFIG=...' on the host, not in a sandbox."
        )

    spec = spec_from_recipe(recipe)
    est: Estimate = estimate(recipe)
    device = torch.device("cuda:0")
    torch.manual_seed(recipe.run.seed)
    torch.cuda.init()  # memory-stats API does not lazy-init the CUDA context
    torch.cuda.reset_peak_memory_stats(device)

    model = _build_calibration_model(recipe)
    # A 4-bit base loads on the GPU at from_pretrained; .to() on it raises.
    if not getattr(model, "is_loaded_in_4bit", False):
        model = model.to(device)
    if recipe.memory.compile:
        model = torch.compile(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(model, recipe, params=trainable)

    from training.models.peft_utils import trainable_parameter_report

    # Params4bit-aware logical counts (packed 4-bit elements count double)
    report = trainable_parameter_report(model)
    # Under PEFT the analytic total is base plus adapter; the estimate's
    # trainable count is the adapter alone.
    analytic_total = spec.param_count()
    if recipe.peft.enabled:
        analytic_total += est.trainable_params

    nvml_peak = 0
    for _ in range(steps):
        batch = torch.randint(
            0, spec.vocab, (recipe.data.micro_batch, recipe.data.seq_len), device=device
        )
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        nvml_peak = max(nvml_peak, _nvml_used_bytes())

    measured_allocated = torch.cuda.max_memory_allocated(device)
    measured_reserved = torch.cuda.max_memory_reserved(device)
    error_pct = (est.allocated_total - measured_allocated) / measured_allocated * 100

    return CalibrationRecord(
        recipe_run_name=recipe.run.name,
        steps=steps,
        model_param_count=report.total,
        analytic_param_count=analytic_total,
        estimate=est.as_dict(),
        measured_allocated=measured_allocated,
        measured_reserved=measured_reserved,
        measured_nvml_peak=nvml_peak,
        allocated_error_pct=error_pct,
        reserved_underpredicts_nvml=est.reserved_estimate < nvml_peak,
        model_trainable_params=report.trainable,
        analytic_trainable_params=est.trainable_params,
    )


def write_record(record: CalibrationRecord, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record.as_dict(), f, indent=2)
        f.write("\n")
