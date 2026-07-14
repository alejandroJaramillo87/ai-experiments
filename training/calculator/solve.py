"""estimate() and solve() for the memory-fit calculator (Phase 0 D4).

estimate(recipe) sums the component formulas for the recipe's micro-batch and
sequence length; solve(recipe, axis) binary-searches the largest micro-batch
or sequence length whose estimate still fits. The fits flag tests the
reserved-estimate against 32GB minus a safety margin, per the roadmap model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from omegaconf import DictConfig

from training.calculator.components import (
    ArchitectureSpec,
    activation_bytes,
    grad_bytes,
    host_ram_bytes,
    optimizer_bytes,
    spec_from_recipe,
    trainable_param_count,
    weights_bytes,
)

GIB = 1 << 30

# Calibrated constants (roadmap D4 model, component 5). overhead_gb starts at
# 1.5 (CUDA context + allocator slack) and is refined by calibrate.py records;
# the safety margin defaults to 1GB below the 32GB device total.
VRAM_TOTAL_BYTES = 32 * GIB
OVERHEAD_BYTES = int(1.5 * GIB)
SAFETY_MARGIN_BYTES = 1 * GIB
HOST_RAM_TOTAL_BYTES = 128 * GIB

SOLVE_AXES = ("batch", "seq")
# solve() search bounds: micro-batch is a plain integer; sequence length moves
# in 128-token steps (flash-attn/tensor-core friendly, and recipes use powers
# of two).
MAX_MICRO_BATCH = 4096
SEQ_STEP = 128
MAX_SEQ_LEN = 1 << 20


@dataclass(frozen=True)
class Estimate:
    """Per-component bytes and derived totals for one (micro_batch, seq_len)."""

    micro_batch: int
    seq_len: int
    param_count: int
    trainable_params: int         # equals param_count unless peft is enabled
    weights: int
    grads: int
    optimizer: int
    activations: int
    overhead: int
    allocated_total: int      # component sum, compared to max_memory_allocated
    reserved_estimate: int    # allocated + overhead, must not under-predict NVML
    fits: bool
    host_ram: int             # offloaded optimizer state; 0 when offload is off

    def as_dict(self) -> dict:
        return asdict(self)


def _estimate_at(
    spec: ArchitectureSpec, recipe: DictConfig, micro_batch: int, seq_len: int
) -> Estimate:
    weights = weights_bytes(spec, recipe)
    grads = grad_bytes(spec, recipe)
    optimizer = optimizer_bytes(spec, recipe)
    activations = activation_bytes(spec, recipe, micro_batch=micro_batch, seq_len=seq_len)
    allocated = weights + grads + optimizer + activations
    reserved = allocated + OVERHEAD_BYTES
    return Estimate(
        micro_batch=micro_batch,
        seq_len=seq_len,
        param_count=spec.param_count(),
        trainable_params=trainable_param_count(spec, recipe),
        weights=weights,
        grads=grads,
        optimizer=optimizer,
        activations=activations,
        overhead=OVERHEAD_BYTES,
        allocated_total=allocated,
        reserved_estimate=reserved,
        fits=reserved <= VRAM_TOTAL_BYTES - SAFETY_MARGIN_BYTES,
        host_ram=host_ram_bytes(spec, recipe))


def estimate(recipe: DictConfig) -> Estimate:
    """Peak-VRAM estimate for the recipe's own micro-batch and seq_len."""
    spec = spec_from_recipe(recipe)
    return _estimate_at(spec, recipe, recipe.data.micro_batch, recipe.data.seq_len)


def solve(recipe: DictConfig, axis: str) -> Estimate | None:
    """Largest micro-batch ("batch") or sequence length ("seq") that fits.

    Binary search over a monotonic axis (activations grow linearly in both).
    Returns the Estimate at the maximum fitting value, or None when even the
    minimum (batch 1, or one SEQ_STEP) does not fit.
    """
    if axis not in SOLVE_AXES:
        raise ValueError(f"solve axis must be one of {SOLVE_AXES}, got '{axis}'")
    spec = spec_from_recipe(recipe)

    def at(value: int) -> Estimate:
        if axis == "batch":
            return _estimate_at(spec, recipe, value, recipe.data.seq_len)
        return _estimate_at(spec, recipe, recipe.data.micro_batch, value * SEQ_STEP)

    lo, hi = 1, (MAX_MICRO_BATCH if axis == "batch" else MAX_SEQ_LEN // SEQ_STEP)
    if not at(lo).fits:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if at(mid).fits:
            lo = mid
        else:
            hi = mid - 1
    return at(lo)
