"""Probe GGUF export round-trip (rust in-band monitor artifact)."""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

requires_gguf = pytest.mark.skipif(
    importlib.util.find_spec("gguf") is None,
    reason="gguf package required for export tests",
)


@requires_gguf
class TestProbeExport:
    def test_round_trip(self, tmp_path, probe_dir):
        import gguf

        from interp.probes.export import write_probe_gguf
        from interp.probes.registry import load_probe

        out = str(tmp_path / "probe.gguf")
        write_probe_gguf(probe_dir, out)
        probe = load_probe(probe_dir)

        reader = gguf.GGUFReader(out)
        fields = {
            name: reader.fields[name].contents()
            for name in (
                "probe.name",
                "probe.model_hash",
                "probe.layer",
                "probe.bias",
                "probe.threshold",
            )
        }
        assert fields["probe.name"] == "test-probe"
        assert fields["probe.model_hash"] == "modelhash-1"
        assert fields["probe.layer"] == probe.layer
        assert fields["probe.bias"] == pytest.approx(probe.bias)
        assert fields["probe.threshold"] == pytest.approx(probe.threshold)

        tensors = {t.name: t for t in reader.tensors}
        assert set(tensors) == {"probe.weights", "probe.mu", "probe.sigma"}
        np.testing.assert_allclose(
            np.asarray(tensors["probe.weights"].data, dtype=np.float32),
            probe.weights.astype(np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(tensors["probe.mu"].data, dtype=np.float32),
            probe.mu.astype(np.float32),
        )
        np.testing.assert_allclose(
            np.asarray(tensors["probe.sigma"].data, dtype=np.float32),
            probe.sigma.astype(np.float32),
        )

    def test_refuses_tampered_probe(self, tmp_path, probe_dir):
        from interp.probes.export import write_probe_gguf
        from interp.probes.registry import RegistryError

        with open(os.path.join(probe_dir, "probe.npz"), "ab") as f:
            f.write(b"tamper")
        with pytest.raises(RegistryError, match="hash"):
            write_probe_gguf(probe_dir, str(tmp_path / "probe.gguf"))
