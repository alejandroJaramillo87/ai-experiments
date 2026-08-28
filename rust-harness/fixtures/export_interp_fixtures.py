"""Write the probe and control-vector fixtures with the real exporters.

The Rust runtime reads two interp artifacts that only Python writes: the
probe (interp/probes/export.py, safetensors plus a JSON sidecar) and the
control vector (interp/steering/export.py). Every other golden
synthesizes those bytes in Rust, which checks the parser against a second
Rust implementation rather than against the writer it exists to read.

This script calls the actual exporters, so the fixtures it produces are
exporter output and not a Rust approximation of it. Contracts:
schemas/probe_artifact.json and schemas/control_vector_artifact.json.
Regenerate with:

    python3 rust-harness/fixtures/export_interp_fixtures.py rust-harness/fixtures

Requires numpy and safetensors; needs no GPU, no checkpoint, and no
network. The values below are dyadic rationals, exact in float32, so the Rust
test compares them for equality rather than within a tolerance.
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
# Named explicitly, so the fixture exercises the sidecar key rather than
# only the reader's default for an artifact that omits it.
PROBE_SITE = "layer_output"

# 0-based block indices. Block 0 is named deliberately: it is the index the
# retired llama.cpp layout refused, so a fixture omitting it would let the
# old floor come back unnoticed. The gap at 2 is also deliberate -- the Rust
# loader indexes directions by block and must leave an unnamed block
# unsteered rather than shifting the next one into it.
CV_DIR_NAME = "python-control-vector"
CV_MODEL_HASH = "fedcba9876543210"
CV_MODEL_ID = "contract-fixture/model"
CV_MODEL_REVISION = "main"
CV_DIRECTIONS = {
    0: [0.25, -0.5, 1.0, -2.0],
    1: [1.5, 0.25, -0.75, 2.0],
    3: [-0.125, 0.75, -1.25, 3.5],
}


def build_direction_set():
    """A DirectionSet with the fields the exporter reads.

    Built directly rather than through extract_repeng or extract_sae_row:
    both need a model, and the exporter needs only the dataclass.
    """
    from interp.steering.extract import DirectionSet

    return DirectionSet(
        directions={
            block: np.array(v, dtype=np.float32) for block, v in CV_DIRECTIONS.items()
        },
        model_id=CV_MODEL_ID,
        model_revision=CV_MODEL_REVISION,
        method="sae_row",
        source={"sae_release": "contract-fixture", "sae_id": "fixture"},
    )


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
        manifest={"model_hash": PROBE_MODEL_HASH, "site": PROBE_SITE},
    )


def main():
    if len(sys.argv) != 2:
        print(
            "usage: export_interp_fixtures.py <out-dir>",
            file=sys.stderr,
        )
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    from interp.probes.export import write_probe_from_probe
    from interp.steering.export import VECTOR_NAME, export_control_vector

    probe_path = out_dir / "python-probe.safetensors"
    write_probe_from_probe(build_probe(), str(probe_path))

    # Through export_control_vector rather than the tensor writer beneath it,
    # so the fixture is the directory the harness and wiring.py actually
    # consume: the image plus the sibling manifest carrying the model hash
    # that harness/src/main.rs reads.
    cv_dir = out_dir / CV_DIR_NAME
    export_control_vector(
        build_direction_set(),
        str(cv_dir),
        model_hash=CV_MODEL_HASH,
    )

    cv_path = cv_dir / VECTOR_NAME
    for path in (
        probe_path,
        probe_path.with_suffix(".json"),
        cv_path,
        cv_dir / "manifest.json",
    ):
        print(f"[OK] {path.name}: {path.stat().st_size} bytes", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
