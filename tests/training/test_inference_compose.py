"""The vllm-gpu service's tmpfs mounts, held to what its JIT compilers need.

The service runs on a read-only root with tmpfs mounts for every directory
the engine writes. Two of those directories receive compiled kernels that
are then loaded with dlopen: Triton's cache at /app/.triton, and FlashInfer's
at $HOME/.cache/flashinfer, which is /app/.cache in this image. A tmpfs is
mounted noexec unless the mount says otherwise, and the loader refuses to map
a shared object from a noexec mount ("failed to map segment from shared
object"). On 2026-09-26 the first model whose head size the shipped
flashinfer-jit-cache wheel did not cover (Gemma 4 E4B-it, 512) sent the
engine down that path and it restart-looped until the health budget expired
(KB-101); the Triton line carried exec and the FlashInfer line did not. This
test holds both.
"""

import os

import yaml

REPO = os.path.join(os.path.dirname(__file__), "..", "..")
COMPOSE = os.path.join(REPO, "docker-compose.inference.yaml")

# Directories a JIT compiler writes a shared object to and then loads.
JIT_CACHES = ["/app/.triton", "/app/.cache"]


def _tmpfs(service: str) -> dict[str, set[str]]:
    """Map each tmpfs mount point to its option set."""
    with open(COMPOSE) as f:
        compose = yaml.safe_load(f)
    mounts = {}
    for line in compose["services"][service]["tmpfs"]:
        path, _, options = line.partition(":")
        mounts[path] = set(options.split(",")) if options else set()
    return mounts


def test_vllm_jit_caches_are_executable():
    mounts = _tmpfs("vllm-gpu")
    for path in JIT_CACHES:
        assert path in mounts, f"{path} is not a tmpfs mount on vllm-gpu"
        assert "exec" in mounts[path], (
            f"{path} is mounted noexec; a kernel compiled there cannot be loaded (KB-101)"
        )
