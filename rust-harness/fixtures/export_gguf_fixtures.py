"""Write the probe and control-vector GGUF fixtures with the real exporters.

The Rust runtime parses two GGUF layouts that only Python writes: the probe
image (interp/probes/export.py) and the control-vector image
(interp/steering/export.py). Every other golden synthesizes those bytes in
Rust with runtime::synth::GgufWriter, which checks the parser against a second
Rust implementation rather than against the writer it exists to read.

This script calls the actual exporters, so the fixtures it produces are
exporter output and not a Rust approximation of it. Contract:
schemas/gguf_artifacts.json. Regenerate with:

    python3 rust-harness/fixtures/export_gguf_fixtures.py rust-harness/fixtures

Requires numpy and gguf; needs no GPU, no checkpoint, and no network. The
values below are dyadic rationals, exact in float32, so the Rust test compares
them for equality rather than within a tolerance.
"""

import sys
from pathlib import Path

import numpy as np

# Run from anywhere: the exporters live in the parent repository, two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

PROBE_NAME = "contract-fixture-probe"
PROBE_MODEL_HASH = "0123456789abcdef"
PROBE_LAYER = 2  # 0-based decoder block, per the probe layout
PROBE_BIAS = 0.5
PROBE_THRESHOLD = 0.25
PROBE_WEIGHTS = [0.5, -1.5, 2.25, -0.125]
PROBE_MU = [0.0, 1.0, -2.0, 4.0]
PROBE_SIGMA = [1.0, 0.5, 2.0, 0.25]

CV_MODEL_HINT = "contract-fixture-model"
# 1-based layer numbering, per the llama.cpp control-vector convention. The
# gap at 2 is deliberate: the Rust loader indexes directions by block and must
# leave an unnamed block unsteered rather than shifting the next one into it.
CV_DIRECTIONS = {
    1: [0.25, -0.5, 1.0, -2.0],
    3: [-0.125, 0.75, -1.25, 3.5],
}


def build_probe():
    """A Probe with the fields the exporter reads, built without a registry.

    load_probe would require a probe.npz directory and its hash validation;
    the exporter only needs the dataclass, so the fixture stays independent of
    the training path.
    """
    from interp.probes.registry import Probe

    return Probe(
        name=PROBE_NAME,
        layer=PROBE_LAYER,
        weights=np.array(PROBE_WEIGHTS, dtype=np.float32),
        bias=PROBE_BIAS,
        mu=np.array(PROBE_MU, dtype=np.float32),
        sigma=np.array(PROBE_SIGMA, dtype=np.float32),
        threshold=PROBE_THRESHOLD,
        manifest={"model_hash": PROBE_MODEL_HASH},
    )


def main():
    if len(sys.argv) != 2:
        print(
            "usage: export_gguf_fixtures.py <out-dir>",
            file=sys.stderr,
        )
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    from interp.probes.export import write_probe_gguf_from_probe
    from interp.steering.export import write_control_vector_gguf

    probe_path = out_dir / "python-probe.gguf"
    write_probe_gguf_from_probe(build_probe(), str(probe_path))

    cv_path = out_dir / "python-control-vector.gguf"
    write_control_vector_gguf(
        {layer: np.array(v, dtype=np.float32) for layer, v in CV_DIRECTIONS.items()},
        str(cv_path),
        model_hint=CV_MODEL_HINT,
    )

    for path in (probe_path, cv_path):
        print(f"[OK] {path.name}: {path.stat().st_size} bytes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
