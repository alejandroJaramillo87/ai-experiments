"""Memory-fit calculator (Phase 0 D4): pre-run VRAM estimation for a recipe.

estimate(recipe) sums the five components of peak VRAM (weights, gradients,
optimizer state, activations, framework overhead) for the configured
micro-batch and sequence length; solve(recipe, axis) binary-searches the
largest micro-batch or sequence length that fits in 32GB. Input is always a
D5 recipe, never a parallel argument set, so the calculator and the trainer
read identical configuration by construction.

The closed-form component math lives in components.py and is pinned by
tests/training/test_calculator.py; the calibrated constants (framework
overhead, safety margin) are refined against real measurements by
calibrate.py on the workstation.
"""

from training.calculator.components import (
    ArchitectureSpec,
    UnknownArchitectureError,
    activation_bytes,
    adapter_param_count,
    grad_bytes,
    host_ram_bytes,
    optimizer_bytes,
    spec_from_recipe,
    trainable_param_count,
    weights_bytes,
)
from training.calculator.solve import Estimate, estimate, solve

__all__ = [
    "ArchitectureSpec",
    "Estimate",
    "UnknownArchitectureError",
    "activation_bytes",
    "adapter_param_count",
    "estimate",
    "grad_bytes",
    "host_ram_bytes",
    "optimizer_bytes",
    "solve",
    "spec_from_recipe",
    "trainable_param_count",
    "weights_bytes",
]
