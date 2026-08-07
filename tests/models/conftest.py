"""Shared fixtures for the model header-transcriber tests."""

import sys
from pathlib import Path

import pytest

# sys.path carries the repo root already; tests/conftest.py inserts it.
from tests.models import specbuild

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The committed fixture written by runtime/src/synth.rs::GgufWriter, and the
# independent pure-Python reader of it. Together they are the one cross-check
# in this suite that shares no author with the code under test.
RUST_FIXTURE_DIR = REPO_ROOT / "rust-harness" / "fixtures"
TINY_GGUF = RUST_FIXTURE_DIR / "tiny-gemma4.gguf"


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
    """The in-tree independent GGUF reader from the rust-harness fixtures.

    Imported by path because rust-harness is not a Python package. Skips
    rather than fails when numpy is absent, since reference_logits.py imports
    it at module scope for the forward pass this suite does not use.
    """
    numpy = pytest.importorskip("numpy", reason="reference_logits.py imports numpy")
    del numpy
    sys.path.insert(0, str(RUST_FIXTURE_DIR))
    try:
        import reference_logits

        return reference_logits.parse_gguf
    finally:
        sys.path.remove(str(RUST_FIXTURE_DIR))
