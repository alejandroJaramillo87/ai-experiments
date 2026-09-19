"""training/commands/smoke_test.py: the D1 smoke command, in-process.

The sandbox has no torch, no CUDA, and none of the CUDA-built packages, so a
fixture installs stub modules into sys.modules for every import the command
makes. Each case drives one branch: the build-mode SKIP degradation, the
workstation-mode failures, and the all-pass path with its manifest."""

from __future__ import annotations

import argparse
import json
import sys
import types

import pytest
from support import REPO_ROOT  # noqa: F401  (inserts the repo root on sys.path)

from training.commands import smoke_test

PURE_IMPORTS = [
    "transformers",
    "trl",
    "peft",
    "datasets",
    "accelerate",
    "hydra",
    "safetensors",
    "pynvml",
]


class _Tensor:
    shape = (1, 8, 4, 32)


class _Param:
    def __init__(self, *_a, **_k):
        self.grad = None


class _Linear4bit:
    def __init__(self, *_a, **_k):
        pass

    def cuda(self):
        return self

    def __call__(self, _x):
        out = _Tensor()
        out.shape = (2, 64)
        return out


class _CPUAdam:
    def __init__(self, _params):
        pass

    def step(self):
        pass


def _torch_stub(*, available: bool, capability=(12, 0)) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0-stub"
    torch.version = types.SimpleNamespace(cuda="13.0")
    torch.float16 = "float16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_capability=lambda: capability,
        get_device_name=lambda _i: "Stub GPU",
    )
    torch.nn = types.SimpleNamespace(Parameter=_Param)
    torch.randn = lambda *a, **k: _Tensor()
    return torch


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    """Install the stub module tree; returns a mutable dict of the stubs."""
    installed: dict[str, types.ModuleType] = {}

    def install(name: str, module: types.ModuleType | None = None):
        module = module or types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        installed[name] = module
        return module

    install("torch", _torch_stub(available=False))
    for mod in PURE_IMPORTS:
        install(mod)
    fa = install("flash_attn")
    fa.flash_attn_func = lambda q, k, v: q
    bnb = install("bitsandbytes")
    bnb.nn = types.SimpleNamespace(Linear4bit=_Linear4bit)
    install("deepspeed")
    install("deepspeed.ops")
    adam = install("deepspeed.ops.adam")
    adam.DeepSpeedCPUAdam = _CPUAdam

    manifest = tmp_path / "runs" / "_smoke" / "versions.json"
    monkeypatch.setattr(smoke_test, "WORKSTATION_MANIFEST", str(manifest))
    installed["manifest_path"] = manifest  # type: ignore[assignment]
    return installed


def run(build: bool) -> int:
    return smoke_test.cmd(argparse.Namespace(build=build))


def test_build_mode_without_cuda_skips_and_passes(stubs, capsys):
    rc = run(build=True)
    out = capsys.readouterr().err
    assert rc == 0
    assert "[SKIP] torch: CUDA unavailable on build host" in out
    assert "[SKIP] flash-attn: skipped (CUDA unproven)" in out
    assert "[SKIP] bitsandbytes: skipped (CUDA unproven)" in out
    assert "[OK] deepspeed: DeepSpeedCPUAdam step (pre-built op loaded)" in out
    assert "[OK] ALL SMOKE STEPS PASSED" in out
    assert not stubs["manifest_path"].exists()


def test_workstation_mode_without_cuda_fails_every_cuda_step(stubs, capsys):
    rc = run(build=False)
    out = capsys.readouterr().err
    assert rc == 1
    assert "[ERROR] torch: CUDA not available" in out
    assert "Solution: run on the workstation" in out
    assert "[ERROR] FAILED: torch-cuda, flash-attn, bitsandbytes" in out


def test_wrong_capability_fails_torch_capability(stubs, monkeypatch, capsys):
    monkeypatch.setitem(
        sys.modules, "torch", _torch_stub(available=True, capability=(9, 0))
    )
    rc = run(build=True)
    out = capsys.readouterr().err
    assert rc == 1
    assert "[ERROR] Unexpected device capability (9, 0), require (12, 0)" in out
    assert "torch-capability" in out


def test_failing_pure_import_is_named(stubs, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "trl")

    real_import = __import__

    def fake_import(name, *a, **k):
        if name == "trl":
            raise ImportError("no trl")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", fake_import)
    rc = run(build=True)
    out = capsys.readouterr().err
    assert rc == 1
    assert "[ERROR] import trl failed: no trl" in out
    assert "import-trl" in out


def test_failing_cpuadam_names_the_prebuilt_op(stubs, capsys):
    def boom(_params):
        raise RuntimeError("op not built")

    sys.modules["deepspeed.ops.adam"].DeepSpeedCPUAdam = boom
    rc = run(build=True)
    out = capsys.readouterr().err
    assert rc == 1
    assert "[ERROR] DeepSpeedCPUAdam failed: op not built" in out
    assert "Note: AVX-512 miscompile on Zen 5 is upstream issue #7556" in out
    assert "deepspeed-cpuadam" in out


def test_all_pass_writes_the_manifest(stubs, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", _torch_stub(available=True))
    rc = run(build=False)
    out = capsys.readouterr().err
    assert rc == 0
    assert "[OK] torch 0.0.0-stub: CUDA available, Stub GPU sm_120" in out
    assert "[OK] flash-attn: flash_attn_func forward" in out
    assert "[OK] bitsandbytes: 4-bit Linear forward" in out
    assert "[OK] ALL SMOKE STEPS PASSED" in out
    manifest = json.loads(stubs["manifest_path"].read_text())
    assert manifest["gpu"] == "Stub GPU"
    assert manifest["capability"] == [12, 0]
    assert manifest["cuda"] == "13.0"
    assert set(PURE_IMPORTS) <= set(manifest["packages"])
    assert f"manifest written: {stubs['manifest_path']}" in out
