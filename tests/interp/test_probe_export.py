"""Probe export round-trip (rust in-band monitor artifact).

The exporter writes two files: a safetensors image of the parameter vectors
and a JSON sidecar of the scalars. Both halves are asserted here, and the
sidecar is additionally checked against the shared report envelope, so a
drifting header fails on the Python side rather than only in the Rust reader.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

requires_safetensors = pytest.mark.skipif(
    importlib.util.find_spec("safetensors") is None,
    reason="safetensors package required for export tests",
)


@requires_safetensors
class TestProbeExport:
    def test_round_trip(self, tmp_path, probe_dir):
        from safetensors.numpy import load_file

        from common.reports import validate_report
        from interp.probes.export import write_probe
        from interp.probes.registry import load_probe

        out = tmp_path / "probe.safetensors"
        write_probe(probe_dir, str(out))
        probe = load_probe(probe_dir)

        sidecar = json.loads(out.with_suffix(".json").read_text())
        validate_report(sidecar)
        assert sidecar["artifact"] == "probe"
        assert sidecar["schema_version"] == 1
        assert sidecar["name"] == "test-probe"
        assert sidecar["model_hash"] == "modelhash-1"
        assert sidecar["layer"] == probe.layer
        assert sidecar["bias"] == pytest.approx(probe.bias)
        assert sidecar["threshold"] == pytest.approx(probe.threshold)
        assert sidecar["width"] == probe.weights.shape[0]
        # The manifest records no site, so the exporter writes none and the
        # reader's default stands. Inventing one would assert a fact nobody
        # recorded.
        assert "site" not in sidecar

        tensors = load_file(str(out))
        assert set(tensors) == {"weights", "mu", "sigma"}
        for name in ("weights", "mu", "sigma"):
            assert tensors[name].dtype == np.float32
            np.testing.assert_allclose(
                tensors[name], getattr(probe, name).astype(np.float32)
            )

    def test_writes_the_site_the_manifest_records(self, tmp_path, probe_dir):
        from interp.probes.export import write_probe_from_probe
        from interp.probes.registry import load_probe

        probe = load_probe(probe_dir)
        probe.manifest["site"] = "layer_output"
        out = tmp_path / "probe.safetensors"
        write_probe_from_probe(probe, str(out))

        sidecar = json.loads(out.with_suffix(".json").read_text())
        assert sidecar["site"] == "layer_output"

    def test_refuses_a_non_finite_scalar(self, tmp_path, probe_dir):
        # JSON has no spelling for NaN: json.dumps would write a bare `NaN`,
        # which the rust reader's parser rejects. Refused here so the failure
        # names the field instead of arriving as a parse error later.
        from interp.probes.export import ProbeExportError, write_probe_from_probe
        from interp.probes.registry import load_probe

        probe = load_probe(probe_dir)
        probe.threshold = float("nan")
        with pytest.raises(ProbeExportError, match="threshold must be finite"):
            write_probe_from_probe(probe, str(tmp_path / "probe.safetensors"))

    def test_refuses_tampered_probe(self, tmp_path, probe_dir):
        from interp.probes.export import write_probe
        from interp.probes.registry import RegistryError

        with open(os.path.join(probe_dir, "probe.npz"), "ab") as f:
            f.write(b"tamper")
        with pytest.raises(RegistryError, match="hash"):
            write_probe(probe_dir, str(tmp_path / "probe.safetensors"))

    def test_the_sidecar_lands_beside_the_image(self, tmp_path, probe_dir):
        # The reader looks for the sidecar at the image's path with a .json
        # extension, so the two must agree about that spelling.
        from interp.probes.export import write_probe

        out = tmp_path / "nested" / "my-probe.safetensors"
        out.parent.mkdir()
        write_probe(probe_dir, str(out))
        assert Path(str(out).removesuffix(".safetensors") + ".json").is_file()
