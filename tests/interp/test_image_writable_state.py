"""Every writable-state dir in the GPU images must land outside the rootfs.

docker-compose.interp.yaml and docker-compose.training.yaml both run their
service with `read_only: true`, and both Dockerfiles set HOME=/app -- which is
on that read-only rootfs. Any library that follows the XDG defaults and writes
under $HOME therefore fails at runtime, not at build time, and often not until
teardown: SAELens uploads the SAE checkpoint as a wandb artifact at the end of
SAETrainingRunner.run(), so an unredirected staging dir failed `make sae-train`
*after* a completed training run, discarding the result.

The Dockerfiles already redirect Jupyter, IPython, matplotlib, Triton and
XDG_CACHE_HOME. This pins the same treatment for wandb, whose data dir is a
separate variable from its cache dir and is not covered by XDG_CACHE_HOME.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from tests.interp.conftest import REPO_ROOT as _REPO_ROOT_STR

REPO_ROOT = pathlib.Path(_REPO_ROOT_STR)

DOCKERFILES = [
    "docker/interp/Dockerfile.interp",
    "docker/training/Dockerfile.training",
]

# wandb resolves each of these independently; WANDB_CACHE_DIR does not cover
# artifact staging, which is what WANDB_DATA_DIR controls.
REQUIRED_ENV = ["WANDB_DATA_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR"]


def _env_values(dockerfile: str) -> dict[str, str]:
    text = (REPO_ROOT / dockerfile).read_text()
    return {
        m.group(1): m.group(2).strip().strip('"')
        for m in re.finditer(r"^ENV\s+(\w+)=(.*)$", text, re.MULTILINE)
    }


@pytest.mark.parametrize("dockerfile", DOCKERFILES)
@pytest.mark.parametrize("var", REQUIRED_ENV)
def test_wandb_dirs_are_off_the_read_only_rootfs(dockerfile: str, var: str) -> None:
    env = _env_values(dockerfile)
    assert var in env, f"{dockerfile}: {var} unset; wandb would write under HOME"
    value = env[var]
    assert value.startswith("/tmp/") or value.startswith("/mnt/ai-data/"), (
        f"{dockerfile}: {var}={value} is on the read-only rootfs; "
        "point it at the /tmp tmpfs or the ai-data bind mount"
    )
