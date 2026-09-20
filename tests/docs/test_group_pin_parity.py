"""Tests for scripts/docs/check_group_pin_parity.py. The gate reads pyproject.toml
rather than a document, and its one finding is a package pinned to two different
exact versions across the core dependencies, the extras and the dependency groups,
which `uv lock --check` cannot see because each literal `==` string resolves on its
own. The parts worth pinning: PEP 503 name normalization so `Foo_Bar` and `foo-bar`
are one package, a non-exact constraint contributing no pin, an environment marker
truncated before it can mangle the version, the empty-set guard, and the exit code
on one agreeing and one disagreeing pin; then the real gate over the repository's
own pyproject.toml.

The module is loaded by path; scripts/ is not an importable package.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "docs" / "check_group_pin_parity.py"

AGREEING = """\
[project]
name = "x"
dependencies = ["numpy==2.0.0", "Foo_Bar==1.0", "loose>=3"]

[project.optional-dependencies]
extra = ["numpy==2.0.0 ; python_version >= '3.12'"]

[dependency-groups]
sandbox = ["foo-bar==1.0", "numpy==2.0.0"]
"""

DISAGREEING = AGREEING.replace('sandbox = ["foo-bar==1.0"', 'sandbox = ["foo-bar==1.1"')


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_group_pin_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(pyproject: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--pyproject", str(pyproject)],
        capture_output=True,
        text=True,
        env={"NO_COLOR": "1", "PATH": "/usr/bin:/bin"},
    )


class TestPins:
    def test_names_normalize_per_pep_503(self, mod):
        assert mod.normalize("Foo_Bar") == mod.normalize("foo-bar") == "foo-bar"

    def test_only_exact_pins_count(self, mod):
        assert mod.pins(["a==1.0", "b>=2", "c"]) == {"a": "1.0"}

    def test_an_environment_marker_does_not_reach_the_version(self, mod):
        assert mod.pins(["numpy==2.0.0 ; python_version >= '3.12'"]) == {
            "numpy": "2.0.0"
        }


class TestGate:
    def test_agreeing_pins_pass(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(AGREEING, encoding="utf-8")
        result = run(pyproject)
        assert result.returncode == 0, result.stderr
        assert "every package pinned twice carries the same version" in result.stderr

    def test_one_disagreeing_pin_fails_and_is_named(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(DISAGREEING, encoding="utf-8")
        result = run(pyproject)
        assert result.returncode == 1
        assert "1 package carries conflicting pins" in result.stderr
        assert "foo-bar:" in result.stderr
        assert "1.0" in result.stderr and "1.1" in result.stderr

    def test_a_pyproject_with_no_pins_fails_the_empty_set_guard(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text('[project]\nname = "x"\n', encoding="utf-8")
        result = run(pyproject)
        assert result.returncode == 1
        assert "found no exact pins" in result.stderr

    def test_the_gate_passes_on_the_repository(self):
        result = run(REPO_ROOT / "pyproject.toml")
        assert result.returncode == 0, result.stderr
