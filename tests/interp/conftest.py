"""Fixtures for the write-back path tests (interp Phase 3 W1-W4).

Everything runs CPU-only, without network, GPU, or services: captures come
from a deterministic fake capture function, services from injected callables.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

D_MODEL = 8
FAKE_FINGERPRINT = "fp-test"


def fake_capture(texts: list[str], layers: list[int]):
    """Deterministic stand-in for the bf16 nnsight capture.

    Class-0 texts (containing the token 'neg') and class-1 texts get linearly
    separable activations plus text-seeded noise, so probes train to high
    AUROC and repeated captures are bit-identical.
    """
    activations = {}
    for layer in layers:
        rows = []
        for text in texts:
            rng = np.random.default_rng(abs(hash((text, layer))) % (2**32))
            base = -1.0 if "neg" in text else 1.0
            row = rng.normal(0, 0.1, D_MODEL).astype(np.float16)
            row[0] += np.float16(base * (1 + layer / 10))
            rows.append(row)
        activations[layer] = np.stack(rows)
    return activations, FAKE_FINGERPRINT


@pytest.fixture
def labeled_dataset(tmp_path):
    """A small separable dataset JSONL: 'neg' texts are label 0."""
    path = tmp_path / "dataset.jsonl"
    with open(path, "w") as f:
        for i in range(120):
            label = i % 2
            text = f"example {i} {'pos' if label else 'neg'} sentence"
            f.write(json.dumps({"text": text, "label": label}) + "\n")
    return str(path)


@pytest.fixture
def capture_dir(tmp_path, labeled_dataset):
    from interp.probes.capture import run_capture

    out = str(tmp_path / "capture")
    run_capture(
        "test-model",
        labeled_dataset,
        [3, 5],
        out,
        model_hash="modelhash-1",
        capture_fn=fake_capture,
    )
    return out


@pytest.fixture
def probe_dir(tmp_path, capture_dir):
    from interp.probes.train import train_probes

    manifest = train_probes(capture_dir, "test-probe", str(tmp_path / "probes"))
    return os.path.join(str(tmp_path / "probes"), manifest["name"])
