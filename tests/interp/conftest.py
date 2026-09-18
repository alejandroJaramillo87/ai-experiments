"""Fixtures for the write-back path tests (interp Phase 3 W1-W4).

Everything runs CPU-only, without network, GPU, or services: captures come
from a deterministic fake capture function, services from injected callables.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

D_MODEL = 8
FAKE_FINGERPRINT = "fp-test"


@pytest.fixture(autouse=True)
def train_lock_path(tmp_path, monkeypatch):
    """Point the D9 run lock at a per-test path: the sandbox has no
    /mnt/ai-data, and tests must not contend on a shared lock."""
    monkeypatch.setenv("TRAIN_LOCK_PATH", str(tmp_path / "test.lock"))


def fake_capture(texts: list[str], layers: list[int]):
    """Deterministic stand-in for the bf16 hooked capture.

    Class-0 texts (containing the token 'neg') and class-1 texts get linearly
    separable activations plus text-seeded noise, so probes train to high
    AUROC and repeated captures are bit-identical. Seeds come from hashlib,
    not the salted builtin hash(), so identical across processes.
    """
    activations = {}
    for layer in layers:
        rows = []
        for text in texts:
            digest = hashlib.sha256(f"{text}|{layer}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:4], "big"))
            base = -1.0 if "neg" in text else 1.0
            row = rng.normal(0, 0.1, D_MODEL).astype(np.float32)
            row[0] += np.float32(base * (1 + layer / 10))
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
