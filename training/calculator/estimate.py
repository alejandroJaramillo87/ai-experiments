"""estimate() for the memory-fit calculator (Phase 0 D4, reduced 2026-09-19).

estimate(recipe) prices the static terms from the recipe (weights, gradients,
optimizer state, host RAM), adds the measured activation rate for the recipe's
profile times its tokens per micro-batch when one is on file (measured.py),
and adds the calibrated framework overhead. The fits flag tests the reserved
estimate against 32GB minus a safety margin, and is None for an unmeasured
profile: the estimate is then a static lower bound and says so.

The analytic activation formula, the temporal-max peak model (KB-56) and the
binary-search solve() left with the 2026-09-19 reduction recorded in
docs/training/memory_fit.md; a measured rate per profile replaced them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from omegaconf import DictConfig

from training.calculator import measured
from training.calculator.components import (
    ArchitectureSpec,
    grad_bytes,
    host_ram_bytes,
    optimizer_bytes,
    profile,
    spec_from_recipe,
    trainable_param_count,
    weights_bytes,
)

GIB = 1 << 30

# Calibrated constants (roadmap D4 model, component 5). The overhead started
# at 1.5 GB and was calibrated to 2.5 GB from the 2026-08-01 pretrain-1b
# record: allocator slack 1.38 GB (reserved 8.38 - allocated 7.00) plus CUDA
# context 1.08 GB (NVML 9.46 - reserved 8.38) = 2.46 GB above the measured
# allocated. The safety margin defaults to 1GB below the 32GB device total.
VRAM_TOTAL_BYTES = 32 * GIB
OVERHEAD_BYTES = int(2.5 * GIB)
SAFETY_MARGIN_BYTES = 1 * GIB
HOST_RAM_TOTAL_BYTES = 128 * GIB
VRAM_BUDGET_BYTES = VRAM_TOTAL_BYTES - SAFETY_MARGIN_BYTES


@dataclass(frozen=True)
class Estimate:
    """Per-component bytes and derived totals for one (micro_batch, seq_len)."""

    micro_batch: int
    seq_len: int
    param_count: int
    trainable_params: int  # equals param_count unless peft is enabled
    weights: int
    grads: int
    optimizer: int
    profile: str  # the measured.py key this recipe prices activations under
    rate: int | None  # measured activation bytes per token; None when unmeasured
    activations: int | None  # rate * micro_batch * seq_len; None when unmeasured
    overhead: int
    # Compared to max_memory_allocated: static terms plus activations, or the
    # static terms alone (a lower bound) when the profile is unmeasured.
    allocated_total: int
    reserved_estimate: int  # allocated + overhead, must not under-predict NVML
    fits: bool | None  # None when unmeasured: nothing to test against the budget
    host_ram: int  # offloaded optimizer state + explicit replay budget; 0 when neither

    def as_dict(self) -> dict:
        return asdict(self)


def estimate_at(
    spec: ArchitectureSpec, recipe: DictConfig, micro_batch: int, seq_len: int
) -> Estimate:
    """The estimate for a spec at a geometry other than the recipe's own."""
    weights = weights_bytes(spec, recipe)
    grads = grad_bytes(spec, recipe)
    optimizer = optimizer_bytes(spec, recipe)
    key = profile(spec, recipe)
    rate = measured.RATES.get(key)
    activations = None if rate is None else rate * micro_batch * seq_len
    allocated = weights + grads + optimizer + (activations or 0)
    reserved = allocated + OVERHEAD_BYTES
    return Estimate(
        micro_batch=micro_batch,
        seq_len=seq_len,
        param_count=spec.param_count(),
        trainable_params=trainable_param_count(spec, recipe),
        weights=weights,
        grads=grads,
        optimizer=optimizer,
        profile=key,
        rate=rate,
        activations=activations,
        overhead=OVERHEAD_BYTES,
        allocated_total=allocated,
        reserved_estimate=reserved,
        fits=None if rate is None else reserved <= VRAM_BUDGET_BYTES,
        host_ram=host_ram_bytes(spec, recipe),
    )


def estimate(recipe: DictConfig) -> Estimate:
    """Peak-VRAM estimate for the recipe's own micro-batch and seq_len."""
    return estimate_at(
        spec_from_recipe(recipe), recipe, recipe.data.micro_batch, recipe.data.seq_len
    )
