"""RH.1a check 1 on the real Gemma 4 checkpoints: were the shared-KV tail's
key and value projections ever trained?

The modeling code builds k_proj and v_proj only for layers that own a KV
cache; the released weights carry them on every layer. If the surplus was
trained, the CPU backend's owns_cache branch reads the wrong keys and the
output is fluent and wrong. The discriminator is distribution shape, not
scale -- see interp/probes/weight_stats.py -- compared across the four
projections of one shared layer, which isolates the gradient path.

Full tier: reading every attention projection of E4B through safetensors
takes minutes, which hw_smoke's five-minute budget does not allow. Skips
when a variant is not on disk, per the conftest rule for real-model loads.
The thresholds are the ones docs/verification/rust-harness/rust-harness.md
argues from; the 2026-08-09 E2B measurement cleared each by a wide margin.
"""

from __future__ import annotations

import os

import pytest

from interp.probes.weight_stats import (
    GAUSS_RATIO,
    PROJECTIONS,
    group,
    layer_boundary,
    projection_stats,
    text_config,
)
from tests.hardware.conftest import AI_DATA_ROOT

pytestmark = [
    pytest.mark.hw_full,
    pytest.mark.hw_e4b,
    pytest.mark.ssd,
    pytest.mark.interp,
]

VARIANTS = ("gemma-4-E2B", "gemma-4-E4B")

# The shared tail's k/v ratios were pinned to four significant figures on
# E2B (spread 0.0010 and 0.0011 over 20 layers); a trained group scatters by
# an order of magnitude more.
UNTRAINED_SPREAD_MAX = 0.01


def checkpoint_root() -> str:
    return os.environ.get("GEMMA4_CHECKPOINT_ROOT", f"{AI_DATA_ROOT}/models/hf")


@pytest.fixture(scope="module", params=VARIANTS)
def checkpoint(request):
    """One variant's stats, computed once per module: the load is the cost."""
    path = os.path.join(checkpoint_root(), request.param)
    if not os.path.isfile(os.path.join(path, "model.safetensors")):
        pytest.skip(f"{request.param} not on disk at {path}")
    n, boundary = layer_boundary(path)
    return request.param, path, n, boundary, projection_stats(path)


def describe(layer, stats):
    return "  ".join(
        f"{p} ratio {stats[(layer, p)].ratio:.4f} kurt {stats[(layer, p)].kurtosis:+.2f}"
        for p in PROJECTIONS
        if (layer, p) in stats
    )


def test_the_shared_tail_carries_untrained_keys_and_values_beside_trained_queries(
    checkpoint,
):
    name, _, n, boundary, stats = checkpoint
    assert boundary < n, f"{name}: no shared layers; nothing to decide"
    for layer in range(boundary, n):
        row = describe(layer, stats)
        for proj in ("k_proj", "v_proj"):
            s = stats[(layer, proj)]
            assert s.ratio > GAUSS_RATIO and s.kurtosis < 0, (
                f"{name} layer {layer}: {proj} does not look untrained\n  {row}"
            )
        for proj in ("q_proj", "o_proj"):
            s = stats[(layer, proj)]
            assert s.kurtosis > 0, (
                f"{name} layer {layer}: {proj} does not look trained\n  {row}"
            )


def test_the_shared_tail_keys_and_values_are_pinned_to_one_value(checkpoint):
    name, _, n, boundary, stats = checkpoint
    for proj in ("k_proj", "v_proj"):
        ratios = [s.ratio for s in group(stats, proj, range(boundary, n))]
        spread = max(ratios) - min(ratios)
        assert spread < UNTRAINED_SPREAD_MAX, (
            f"{name}: shared {proj} ratios spread {spread:.4f}, "
            f"min {min(ratios):.4f} max {max(ratios):.4f}"
        )


def test_every_owning_layer_is_trained_on_all_four_projections(checkpoint):
    name, _, _, boundary, stats = checkpoint
    for layer in range(boundary):
        for proj in PROJECTIONS:
            assert stats[(layer, proj)].kurtosis > 0, (
                f"{name} layer {layer}: {proj} kurtosis not positive\n"
                f"  {describe(layer, stats)}"
            )


def test_key_width_agrees_with_the_attention_schedule(checkpoint):
    """The runbook read the schedule off k_proj shapes (512 wide on global
    layers); config.json states it in layer_types. The two must agree."""
    name, path, n, _, stats = checkpoint
    layer_types = text_config(path).get("layer_types")
    if not layer_types:
        pytest.skip(f"{name}: config.json carries no layer_types")
    widths = {stats[(layer, "k_proj")].rows for layer in range(n)}
    assert len(widths) == 2, f"{name}: k_proj widths {sorted(widths)}, expected two"
    wide = max(widths)
    from_weights = {i for i in range(n) if stats[(i, "k_proj")].rows == wide}
    from_config = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    assert from_weights == from_config, (
        f"{name}: global layers by k_proj width {sorted(from_weights)} "
        f"but by layer_types {sorted(from_config)}"
    )
