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
    / "scripts"
    / "utils"
    / "check_py_deps_install.py"
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


def test_pkg_version_dist_name_mapping(mod, monkeypatch):
    # llama_cpp imports as llama_cpp but is distributed as llama-cpp-python.
    # The mapping is pure logic; the distribution need not be installed here.
    assert mod.DIST_NAMES["llama_cpp"] == "llama-cpp-python"
    asked = []

    def fake_version(name):
        asked.append(name)
        return "0.3.9"

    monkeypatch.setattr(mod, "version", fake_version)
    assert mod._pkg_version("llama_cpp") == "0.3.9"
    assert asked == ["llama-cpp-python"]


def test_installed_but_unimportable_is_not_misreported(mod, monkeypatch, capsys):
    # A distribution whose import raises (bitsandbytes without torch) is
    # installed; reporting "not installed" directs the operator to install
    # a package that is present.
    monkeypatch.setattr(mod, "quiet_import",
                        lambda name: (None, "boom traceback"))
    monkeypatch.setattr(mod, "_pkg_version", lambda name: "0.46.1")
    monkeypatch.setattr(mod, "KEY_PACKAGES",
                        {"bitsandbytes": ("bitsandbytes", True)})
    packages = mod.check_key_packages(mod.Printer(quiet=False))
    captured = capsys.readouterr().out
    assert packages["bitsandbytes"]["installed"] is True
    assert packages["bitsandbytes"]["version"] == "0.46.1"
    assert "failed to import" in captured
    assert "not installed" not in captured


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
    # The contract under test is the output format, not the host's health
    # (tests/utils/README.md): schema assertions only, no host-value claims.
    result = run_tool("--json")
    assert result.returncode in (0, 1)
    payload = json.loads(result.stdout)
    assert set(payload) == {"gpu", "packages"}
    assert "pytorch_version" in payload["gpu"]


def test_human_mode_output_contract():
    result = run_tool()
    assert result.returncode in (0, 1)
    assert "\x1b[" not in result.stdout
    assert "Validation summary:" in result.stdout
