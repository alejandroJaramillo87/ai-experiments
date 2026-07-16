"""Version-probe helpers shared by the tier smoke tests.

Both smoke tests (training and interp) record library and driver versions in
their manifests; the probes live here so the two cannot drift.
"""

from __future__ import annotations


def pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def driver_version() -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlSystemGetDriverVersion()
    except Exception:  # noqa: BLE001
        return "unknown"
