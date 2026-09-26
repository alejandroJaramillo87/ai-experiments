"""The vllm-gpu image pins every package its serving contract depends on.

docker/vllm-gpu/Dockerfile.vllm-gpu installs the official vLLM wheel with
`uv pip install`, so the container holds exactly the pins its install lines
name plus whatever those resolve on build day. On 2026-09-26 the first Gemma 4
checkpoint the image was asked to serve died at engine config (KB-100): vLLM
0.25.0 read `head_dim` as a global attribute, and the transformers release
the unpinned resolve had picked up makes it per-layer for Gemma 4 and raises
when it is read globally. The two packages move together on that contract,
so transformers is pinned beside vLLM; FlashInfer's two packages already were,
and the vLLM wheel's own `flashinfer-cubin` requirement is exact, so all four
are held here.

Two further checks close the "fact in two places" trap the Dockerfile sets
for itself: every `==${NAME}` in an install line resolves to a declared ARG,
and the version the header comment states for vLLM and FlashInfer is the ARG's
value, so a bump that edits one and not the other fails here rather than in a
reader's notes.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from support import REPO_ROOT as _REPO_ROOT_STR

REPO_ROOT = pathlib.Path(_REPO_ROOT_STR)
DOCKERFILE = REPO_ROOT / "docker" / "vllm-gpu" / "Dockerfile.vllm-gpu"

# The packages whose versions form the serving contract (KB-100).
PINNED = ["vllm", "transformers", "flashinfer-cubin", "flashinfer-jit-cache"]

# Header comment phrase -> ARG whose value it must state.
HEADER_FACTS = {
    r"# - vLLM (\S+) official cu130 wheel": "VLLM_VERSION",
    r"# - FlashInfer (\S+) attention kernels": "FLASHINFER_VERSION",
}


def _text() -> str:
    return DOCKERFILE.read_text()


def _args(text: str) -> dict[str, str]:
    return dict(re.findall(r"^ARG (\w+)=(\S+)$", text, re.MULTILINE))


def _install_specs(text: str) -> dict[str, str]:
    """Map each package named on a `uv pip install` line to its spec text."""
    specs: dict[str, str] = {}
    # Join backslash continuations so one RUN reads as one line.
    joined = re.sub(r"\\\n\s*", " ", text)
    for command in re.split(r"&&|\n", joined):
        if "uv pip install" not in command:
            continue
        for token in command.split("uv pip install", 1)[1].split():
            if token.startswith("-") or token.startswith("https://"):
                continue
            name = re.split(r"[=<>!~]", token, maxsplit=1)[0]
            specs[name] = token
    return specs


@pytest.mark.parametrize("package", PINNED)
def test_contract_packages_are_pinned_exactly(package):
    specs = _install_specs(_text())
    assert package in specs, f"{package} is not installed by {DOCKERFILE.name}"
    assert "==" in specs[package], (
        f"{package} is installed unpinned ({specs[package]}); the build would "
        "resolve whatever satisfies vLLM's range that day (KB-100)"
    )


def test_every_pin_variable_is_a_declared_arg():
    text = _text()
    args = _args(text)
    for name in re.findall(r"==\$\{(\w+)\}", text):
        assert name in args, (
            f"${{{name}}} is used in an install line but never declared as ARG"
        )


def test_header_states_the_arg_versions():
    text = _text()
    args = _args(text)
    for pattern, arg in HEADER_FACTS.items():
        match = re.search(pattern, text, re.MULTILINE)
        assert match, f"header comment no longer carries the line matching {pattern!r}"
        assert match.group(1) == args[arg], (
            f"header says {match.group(1)} where ARG {arg}={args[arg]}: one fact in two places"
        )
