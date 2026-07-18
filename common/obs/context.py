"""Run-context capture (roadmap OBS-7).

run_context() reports the environment a command ran in — git SHA, Python
and platform strings, key package versions, and GPU identity — so a
recorded failure is diagnosable and reproducible. Package versions come
from importlib.metadata, never from importing the packages, keeping the
CLI's lazy-import contract; the GPU query is best-effort pynvml and the
key is omitted on any failure. Environment variables are never captured.
"""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from common.provenance import git_sha

# One list covers both CLIs; packages not installed are omitted per record.
PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "datasets",
    "sae_lens",
    "transformer_lens",
    "nnsight",
)


def _gpu_info() -> dict[str, str] | None:
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(name, bytes):
            name = name.decode()
        if isinstance(driver, bytes):
            driver = driver.decode()
        return {"name": name, "driver": driver}
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def run_context() -> dict[str, Any]:
    """Facts needed to reproduce a run; never raises, never reads the env."""
    packages: dict[str, str] = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    context: dict[str, Any] = {
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }
    gpu = _gpu_info()
    if gpu is not None:
        context["gpu"] = gpu
    return context
