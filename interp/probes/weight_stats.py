"""Distribution-shape statistics for attention projection weights.

Decides whether a tensor was ever updated by training. A tensor allocated at
initialization and never touched keeps the *shape* of its initialization
distribution; a trained one does not. Scale is the wrong statistic:
``initializer_range`` places an untrained tensor at close to the standard
deviation of a trained one, so ``std`` cannot separate them. Two shape
statistics can:

- ``|mean| / std``: a Gaussian gives ``sqrt(2 / pi)``, 0.7979. The truncated
  normal ``transformers`` initializes with gives a higher value because the
  tails are cut. Trained weights become heavy-tailed and give a lower one.
- Excess kurtosis: negative for a truncated normal, positive for trained
  weights.

The question this answers is the shared-KV weight conflict in Gemma 4
(docs/models/gemma-4.md): the modeling code builds ``k_proj`` and ``v_proj``
only for layers that own a KV cache, while the released checkpoints carry
them on every layer. Comparing all four projections within one shared layer
isolates the effect to the gradient path, which no whole-layer comparison
can do: ``q_proj`` and ``o_proj`` are used there and must be trained.

Verified in two places. ``tests/interp/test_weight_stats.py`` proves the
discriminator on synthetic draws in the sandbox;
``tests/hardware/test_shared_kv_weights.py`` asserts the verdict on the real
checkpoints on the workstation (runbook entry RH.1a).

``numpy`` is the only import at module level so the sandbox can collect the
unit tests; ``safetensors`` (and ``torch``, when present) are imported inside
``projection_stats``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# |mean| / std of a zero-mean Gaussian: E|X| / sigma = sqrt(2 / pi).
GAUSS_RATIO = math.sqrt(2 / math.pi)

# The four attention projections, in the order a reader expects them.
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

# Both spellings released Gemma 4 checkpoints have used for the language
# model's decoder stack; the first is the current one.
_KEY_TEMPLATES = (
    "model.language_model.layers.{layer}.self_attn.{proj}.weight",
    "language_model.model.layers.{layer}.self_attn.{proj}.weight",
)

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"


@dataclass(frozen=True)
class ShapeStats:
    """Shape statistics of one weight tensor, plus its shape for the schedule.

    ``ratio`` is ``|mean| / std``; ``kurtosis`` is excess kurtosis, so a
    Gaussian sits at 0. ``rows`` is the tensor's first dimension: on a
    ``k_proj`` it is the key width, which is what tells a sliding layer from
    a global one on a checkpoint whose config does not say.
    """

    ratio: float
    kurtosis: float
    rows: int


def shape_stats(w: np.ndarray) -> ShapeStats:
    """The two shape statistics of ``w``, computed in float64 over all entries."""
    rows = int(w.shape[0]) if w.ndim else 1
    x = np.asarray(w, dtype=np.float64).ravel()
    s = x.std()
    if s == 0.0:
        raise ValueError("a constant tensor has no distribution shape")
    z = x / s
    return ShapeStats(
        ratio=float(np.abs(z).mean()),
        kurtosis=float((z**4).mean() - 3.0),
        rows=rows,
    )


def text_config(checkpoint_dir: str | Path) -> dict:
    """``config.json``'s text config: the nested ``text_config`` of a composite
    checkpoint, or the top level of a text-only one."""
    with open(Path(checkpoint_dir) / CONFIG_FILE) as f:
        config = json.load(f)
    return config.get("text_config", config)


def layer_boundary(checkpoint_dir: str | Path) -> tuple[int, int]:
    """``(num_hidden_layers, boundary)``: layers at or past ``boundary`` read a
    donor's KV cache and, per the modeling code, own no key or value
    projection. ``boundary`` equals the layer count when nothing is shared."""
    cfg = text_config(checkpoint_dir)
    n = int(cfg["num_hidden_layers"])
    shared = int(cfg.get("num_kv_shared_layers", 0) or 0)
    return n, n - shared


def projection_stats(
    checkpoint_dir: str | Path,
) -> dict[tuple[int, str], ShapeStats]:
    """Shape statistics for every attention projection in the checkpoint.

    Keyed by ``(layer, projection)``; a projection the file does not carry
    is absent rather than zero. Tensors are read one at a time, so the
    checkpoint is never resident in full.

    Uses the torch backend when torch is importable, because released Gemma 4
    weights are bf16 and the numpy backend of ``safetensors`` cannot read that
    dtype. Without torch, a float32 or float16 file still reads through numpy,
    which is what the sandbox unit test relies on.
    """
    from safetensors import safe_open

    try:
        import torch  # noqa: F401

        framework = "pt"
    except ImportError:
        framework = "np"

    n, _ = layer_boundary(checkpoint_dir)
    out: dict[tuple[int, str], ShapeStats] = {}
    with safe_open(str(Path(checkpoint_dir) / WEIGHTS_FILE), framework=framework) as f:
        names = set(f.keys())
        for layer in range(n):
            for proj in PROJECTIONS:
                key = next(
                    (
                        t.format(layer=layer, proj=proj)
                        for t in _KEY_TEMPLATES
                        if t.format(layer=layer, proj=proj) in names
                    ),
                    None,
                )
                if key is None:
                    continue
                w = f.get_tensor(key)
                if framework == "pt":
                    w = w.float().numpy()
                out[(layer, proj)] = shape_stats(w)
    return out


def group(
    stats: dict[tuple[int, str], ShapeStats], proj: str, layers: range
) -> list[ShapeStats]:
    """The statistics of ``proj`` over ``layers``, skipping absent tensors."""
    return [stats[(layer, proj)] for layer in layers if (layer, proj) in stats]
