"""docker/training/entrypoint.sh: the environment echo and the D9 LD_PRELOAD
slot.

Run as a subprocess against a scratch PATH that carries python but no
nvidia-smi, the sandbox case. The exec'd command reports the LD_PRELOAD it
inherited, so the slot is asserted without loading any real interposer."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest
from support import REPO_ROOT

ENTRYPOINT = os.path.join(REPO_ROOT, "docker", "training", "entrypoint.sh")
BASH = shutil.which("bash") or "/bin/bash"
REPORT = "echo LD=${LD_PRELOAD:-unset}; exit 7"


@pytest.fixture
def scratch_path(tmp_path) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # python3 under the name the entrypoint calls, plus the tools it and the
    # exec'd sh need; nvidia-smi is deliberately absent.
    for name, tool in (("python", "python3"), ("sh", "sh"), ("awk", "awk")):
        found = shutil.which(tool)
        assert found, f"{tool} is required on PATH"
        (bin_dir / name).symlink_to(found)
    return str(bin_dir)


def run(path: str, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = {"PATH": path, "NO_COLOR": "1", "TORCH_CUDA_ARCH_LIST": "12.0"}
    env.update(extra_env or {})
    return subprocess.run(
        [BASH, ENTRYPOINT, "sh", "-c", REPORT],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )


def test_echoes_environment_and_execs_command(scratch_path):
    result = run(scratch_path)
    assert result.returncode == 7
    assert "=== Training container ===" in result.stdout
    assert "CUDA arch list: 12.0" in result.stdout
    assert (
        "[WARN] nvidia-smi not found: no GPU visible in this container" in result.stdout
    )
    assert "[RUN] exec: sh -c " in result.stdout
    assert "LD=unset" in result.stdout
    assert "[INFO] LD_PRELOAD:" not in result.stdout


def test_train_ld_preload_sets_the_slot(scratch_path):
    result = run(scratch_path, {"TRAIN_LD_PRELOAD": "/x/shim.so"})
    assert "[INFO] LD_PRELOAD: /x/shim.so" in result.stdout
    assert "LD=/x/shim.so" in result.stdout


def test_train_ld_preload_prepends_to_existing_ld_preload(scratch_path):
    result = run(
        scratch_path, {"TRAIN_LD_PRELOAD": "/x/shim.so", "LD_PRELOAD": "/y.so"}
    )
    assert "[INFO] LD_PRELOAD: /x/shim.so:/y.so" in result.stdout
    assert "LD=/x/shim.so:/y.so" in result.stdout


def test_no_color_output_carries_no_escape_codes(scratch_path):
    result = run(scratch_path)
    assert "\x1b[" not in result.stdout
