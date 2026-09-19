"""Memory-fit calculator (Phase 0 D4, reduced 2026-09-19): pre-run VRAM
estimation for a recipe.

estimate(recipe) prices the static terms of peak VRAM (weights, gradients,
optimizer state) from the recipe's parameter count, adds the activation rate
calibrate.py measured for the recipe's profile (measured.py) times its tokens
per micro-batch, and the calibrated framework overhead. A profile with no
rate on file is reported as unmeasured rather than estimated. Input is always
a D5 recipe, never a parallel argument set, so the calculator and the trainer
read identical configuration by construction.

The static component math lives in components.py and is pinned by
tests/training/test_calculator.py; the rates and the overhead constant are
measurements from the workstation, recorded by calibrate.py runs.
"""

from training.calculator.components import (
    ArchitectureSpec,
    UnknownArchitectureError,
    adapter_param_count,
    grad_bytes,
    host_ram_bytes,
    optimizer_bytes,
    profile,
    spec_from_recipe,
    trainable_param_count,
    weights_bytes,
)
from training.calculator.estimate import Estimate, estimate

__all__ = [
    "ArchitectureSpec",
    "Estimate",
    "UnknownArchitectureError",
    "adapter_param_count",
    "estimate",
    "grad_bytes",
    "host_ram_bytes",
    "optimizer_bytes",
    "profile",
    "spec_from_recipe",
    "trainable_param_count",
    "weights_bytes",
]
