"""The shape statistics behind RH.1a check 1, proved on synthetic draws.

The workstation test (tests/hardware/test_shared_kv_weights.py) asserts a
verdict on the real Gemma 4 checkpoints; these assert that the discriminator
it relies on separates an initialization distribution from a trained one at
all, and that the checkpoint reader finds tensors under both key spellings.
Neither needs torch: safetensors reads float32 through its numpy backend.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from interp.probes.weight_stats import (
    GAUSS_RATIO,
    PROJECTIONS,
    group,
    layer_boundary,
    projection_stats,
    shape_stats,
)

N = 1 << 18


def truncated_normal(rng, size, cut=2.0):
    """What torch.nn.init.trunc_normal_ draws: a Gaussian with |x| > cut
    redrawn, which is how transformers initializes every linear weight."""
    x = rng.standard_normal(size)
    while True:
        bad = np.abs(x) > cut
        if not bad.any():
            return x
        x[bad] = rng.standard_normal(int(bad.sum()))


def test_a_gaussian_sits_at_the_reference_ratio_with_zero_kurtosis():
    s = shape_stats(np.random.default_rng(0).standard_normal(N))
    assert abs(s.ratio - GAUSS_RATIO) < 0.01, s
    assert abs(s.kurtosis) < 0.1, s


def test_a_truncated_normal_lands_above_the_ratio_with_negative_kurtosis():
    s = shape_stats(truncated_normal(np.random.default_rng(1), N))
    assert s.ratio > GAUSS_RATIO, s
    assert s.kurtosis < 0, s


def test_a_heavy_tailed_sample_lands_below_the_ratio_with_positive_kurtosis():
    s = shape_stats(np.random.default_rng(2).laplace(size=N))
    assert s.ratio < GAUSS_RATIO, s
    assert s.kurtosis > 0, s


def test_scale_does_not_move_either_statistic():
    """The reason shape and not std decides: initializer_range makes the scale
    of an untrained tensor indistinguishable from a trained one's."""
    x = truncated_normal(np.random.default_rng(3), N)
    a, b = shape_stats(x), shape_stats(x * 0.02)
    assert a.ratio == pytest.approx(b.ratio)
    assert a.kurtosis == pytest.approx(b.kurtosis)


def test_a_constant_tensor_is_refused_rather_than_divided_by_zero():
    with pytest.raises(ValueError, match="no distribution shape"):
        shape_stats(np.zeros(16))


def test_rows_is_the_first_dimension():
    assert shape_stats(np.random.default_rng(4).standard_normal((512, 64))).rows == 512


def write_config(path, n, shared, nested=True):
    text = {"num_hidden_layers": n, "num_kv_shared_layers": shared}
    body = {"model_type": "gemma4", "text_config": text} if nested else text
    (path / "config.json").write_text(json.dumps(body))


def test_layer_boundary_reads_the_nested_text_config(tmp_path):
    write_config(tmp_path, 35, 20)
    assert layer_boundary(tmp_path) == (35, 15)


def test_layer_boundary_reads_a_flat_config_and_defaults_shared_to_zero(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    assert layer_boundary(tmp_path) == (4, 4)


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    """Three layers, boundary 2. The shared layer's k/v are untouched
    initialization; everything else is heavy-tailed. Layer 0 uses the older
    key spelling and layer 1 omits o_proj, so both reader branches run."""
    from safetensors.numpy import save_file

    rng = np.random.default_rng(5)
    write_config(tmp_path, 3, 1)
    tensors = {}

    def put(layer, proj, w, template=0):
        key = (
            "model.language_model.layers.{l}.self_attn.{p}.weight",
            "language_model.model.layers.{l}.self_attn.{p}.weight",
        )[template].format(l=layer, p=proj)
        tensors[key] = w.astype(np.float32)

    size = (64, 256)
    for proj in PROJECTIONS:
        put(0, proj, rng.laplace(size=size), template=1)
        if proj != "o_proj":
            put(1, proj, rng.laplace(size=size))
    put(2, "q_proj", rng.laplace(size=size))
    put(2, "o_proj", rng.laplace(size=size))
    put(2, "k_proj", truncated_normal(rng, (32, 256)))
    put(2, "v_proj", truncated_normal(rng, (32, 256)))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    return tmp_path


def test_projection_stats_finds_both_spellings_and_skips_a_missing_tensor(
    synthetic_checkpoint,
):
    stats = projection_stats(synthetic_checkpoint)
    assert {p for (i, p) in stats if i == 0} == set(PROJECTIONS)
    assert {p for (i, p) in stats if i == 1} == set(PROJECTIONS) - {"o_proj"}
    assert stats[(2, "k_proj")].rows == 32


def test_projection_stats_separates_the_shared_layer_within_itself(
    synthetic_checkpoint,
):
    stats = projection_stats(synthetic_checkpoint)
    n, boundary = layer_boundary(synthetic_checkpoint)
    shared = range(boundary, n)
    for proj in ("k_proj", "v_proj"):
        for s in group(stats, proj, shared):
            assert s.ratio > GAUSS_RATIO and s.kurtosis < 0, (proj, s)
    for proj in ("q_proj", "o_proj"):
        for s in group(stats, proj, shared):
            assert s.ratio < GAUSS_RATIO and s.kurtosis > 0, (proj, s)
    for proj in PROJECTIONS:
        for s in group(stats, proj, range(boundary)):
            assert s.kurtosis > 0, (proj, s)
