"""Tests for scripts/utils/check_py_deps_install.py.

Unit tests load the module directly (importing it is light; torch is only
imported inside functions). End-to-end tests run the tool as a subprocess —
host-dependent and slow (torch import), by design part of `make test`.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parent.parent.parent
    / "scripts" / "utils" / "check_py_deps_install.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_py_deps_install", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_printer_plain_when_not_tty(mod, capsys):
    out = mod.Printer(quiet=False)
    out.ok("CUDA availability: True")
    out.warn("indented", indent=2)
    captured = capsys.readouterr().out
    assert "[OK] CUDA availability: True\n" in captured
    assert "  [WARN] indented\n" in captured
    assert "\x1b[" not in captured


def test_printer_quiet_suppresses_everything(mod, capsys):
    out = mod.Printer(quiet=True)
    out.ok("hidden")
    out.error("hidden")
    out.line("hidden")
    assert capsys.readouterr().out == ""


def test_pkg_version_known_and_unknown(mod):
    assert mod._pkg_version("pytest") is not None
    assert mod._pkg_version("no-such-dist-xyz") is None


def test_pkg_version_dist_name_mapping(mod):
    # llama_cpp imports as llama_cpp but is distributed as llama-cpp-python.
    assert mod.DIST_NAMES["llama_cpp"] == "llama-cpp-python"
    assert mod._pkg_version("llama_cpp") is not None


def test_quiet_import_success_and_failure(mod):
    module, noise = mod.quiet_import("json")
    assert module is not None
    assert noise == ""
    module, _ = mod.quiet_import("no_such_module_xyz")
    assert module is None


def run_tool(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "NO_COLOR": "1"},
        timeout=120,
    )


def test_json_mode_is_pure_json():
    result = run_tool("--json")
    assert result.returncode in (0, 1)
    payload = json.loads(result.stdout)
    assert set(payload) == {"gpu", "packages"}
    assert payload["gpu"]["pytorch_version"] is not None


def test_human_mode_output_contract():
    result = run_tool()
    assert result.returncode in (0, 1)
    assert "\x1b[" not in result.stdout
    assert "Validation summary:" in result.stdout
