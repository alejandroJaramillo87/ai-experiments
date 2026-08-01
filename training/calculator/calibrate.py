"""Calibration harness for the memory-fit calculator (Phase 0 D4).

Runs a short real training loop for a recipe on the workstation GPU, records
torch.cuda.max_memory_allocated/reserved and the NVML peak, and writes a
calibration record JSON comparing measured peaks against estimate(). The
acceptance bar: component sum within 10 percent of max_memory_allocated on
the two reference configs, and the reserved estimate never under-predicting
the NVML peak.

The loop here is deliberately private to calibration: a few optimizer steps
over synthetic batches, no checkpointing, tracking, or resume. It is not the
D6 reference trainer (training/runtime/loop.py), which owns those facilities.

The model is built through the shared builders in training/models/model.py (HF
LlamaConfig + AutoModelForCausalLM.from_config, resolved design decision 2)
so the measured model is exactly what training runs instantiate; the built model's true parameter count is recorded so any
drift from the analytic param_count is caught immediately.

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
    build_optimizer,  # noqa: F401  (re-export; extracted in D6)
    build_scratch_model,
)

# Backward-compatible alias: calibrate.build_model predates the shared module
build_model = build_scratch_model

DEFAULT_STEPS = 5


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


def calibrate(recipe: DictConfig, steps: int = DEFAULT_STEPS) -> CalibrationRecord:
    """Run the calibration loop and return the estimate-vs-measured record."""
    import torch

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

    model = build_scratch_model(spec, recipe).to(device)
    if recipe.memory.compile:
        model = torch.compile(model)
    optimizer = build_optimizer(model, recipe)
    model_params = sum(p.numel() for p in model.parameters())

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
        model_param_count=model_params,
        analytic_param_count=spec.param_count(),
        estimate=est.as_dict(),
        measured_allocated=measured_allocated,
        measured_reserved=measured_reserved,
        measured_nvml_peak=nvml_peak,
        allocated_error_pct=error_pct,
        reserved_underpredicts_nvml=est.reserved_estimate < nvml_peak,
    )


def write_record(record: CalibrationRecord, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(record.as_dict(), f, indent=2)
        f.write("\n")
