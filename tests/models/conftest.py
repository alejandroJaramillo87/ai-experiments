"""Shared fixtures for the model header-transcriber tests."""

from pathlib import Path

import pytest

# sys.path carries the repo root already; tests/conftest.py inserts it.
from tests.models import specbuild

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Was written by the upstream `gguf` package through interp/steering/export.py,
# back when the control vector was a GGUF; R10 moved that artifact to
# safetensors and this file stopped being regenerable. It is kept frozen and
# moved here, out of rust-harness/fixtures, because rust-harness no longer
# reads or writes GGUF at all.
#
# Freezing costs nothing that matters. The property this fixture is here for
# is that neither the bash reader under test nor the Python reader it is
# checked against had a hand in producing it, and a file no in-tree code can
# rewrite has that property more strongly than one regenerated on demand. Any
# small exporter-written GGUF serves: this suite is about the container, and
# the container is unchanged -- training/export still produces GGUF for
# llama.cpp, which is why scripts/models/gguf_header_transcript.sh survives.
TINY_GGUF = REPO_ROOT / "tests" / "models" / "fixtures" / "tiny-exporter-written.gguf"


@pytest.fixture
def spec() -> specbuild.ModelSpec:
    return specbuild.ModelSpec()


@pytest.fixture
def model_dir(tmp_path, spec) -> Path:
    return specbuild.build_model_dir(tmp_path, spec)


@pytest.fixture
def transcribe(run_script, tmp_path):
    """Runs a transcriber and returns (CompletedProcess, parsed sections).

    The transcript path is per-call so a test can transcribe more than once
    and compare, which is how the mutation tests detect an output that does
    not depend on its input.
    """
    counter = {"n": 0}

    def _run(script: str, *args: str, **kwargs):
        counter["n"] += 1
        out = tmp_path / f"transcript-{counter['n']}.txt"
        result = run_script(script, "--out", str(out), *args, **kwargs)
        sections = specbuild.read_sections(out) if out.exists() else {}
        return result, sections, out

    return _run


@pytest.fixture
def reference_parse_gguf():
    """The independent GGUF reader the bash script is checked against.

    Skips rather than fails when numpy is absent, since the reader returns
    tensor payloads as arrays.
    """
    pytest.importorskip("numpy", reason="the reference reader returns numpy arrays")
    from tests.models.gguf_reference_reader import parse_gguf

    return parse_gguf
