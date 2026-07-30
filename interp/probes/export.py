"""Probe GGUF export for the rust harness in-band monitor.

Converts a registered probe artifact (probe.npz + manifest.json) into a
single GGUF file the rust runtime loads directly, mirroring the
control-vector exporter (interp/steering/export.py): architecture "probe",
scalar metadata as KV entries, and one float32 tensor each for the weights
and the standardization statistics. The rust side (rust-harness
runtime/src/model/probe.rs) reads exactly this layout; scoring parity with
interp/probes/score.py is golden-tested there.
"""

from __future__ import annotations

from interp.probes.registry import Probe, load_probe

ARCH = "probe"


class ProbeExportError(Exception):
    """The probe artifact cannot be written as a probe GGUF."""


def write_probe_gguf_from_probe(probe: Probe, path: str) -> None:
    """Write a loaded probe in the rust-harness probe GGUF layout."""
    import gguf
    import numpy as np

    weights = np.asarray(probe.weights, dtype=np.float32)
    mu = np.asarray(probe.mu, dtype=np.float32)
    sigma = np.asarray(probe.sigma, dtype=np.float32)
    if not (weights.ndim == 1 and weights.shape == mu.shape == sigma.shape):
        raise ProbeExportError(
            f"weights/mu/sigma must share one 1-D width, got "
            f"{weights.shape}, {mu.shape}, {sigma.shape}"
        )
    if probe.layer < 0:
        raise ProbeExportError(f"layer must be non-negative, got {probe.layer}")

    writer = gguf.GGUFWriter(path, ARCH)
    writer.add_string(f"{ARCH}.name", probe.name)
    writer.add_string(
        f"{ARCH}.model_hash", str(probe.manifest.get("model_hash", "unknown"))
    )
    writer.add_uint32(f"{ARCH}.layer", probe.layer)
    writer.add_float32(f"{ARCH}.bias", float(probe.bias))
    writer.add_float32(f"{ARCH}.threshold", float(probe.threshold))
    writer.add_tensor(f"{ARCH}.weights", weights)
    writer.add_tensor(f"{ARCH}.mu", mu)
    writer.add_tensor(f"{ARCH}.sigma", sigma)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def write_probe_gguf(probe_dir: str, path: str) -> None:
    """Load a registered probe (hash-validated) and export it as GGUF."""
    write_probe_gguf_from_probe(load_probe(probe_dir), path)
