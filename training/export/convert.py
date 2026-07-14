"""GGUF conversion and quantization against the pinned llama.cpp (Phase 0 D7).

Wraps the pinned checkout's convert_hf_to_gguf.py and llama-quantize binary as
subprocesses. Scope is GGUF-convertible architectures only: the staged model's
architecture is checked against the pinned converter's registration list before
any conversion work, failing with a clear message otherwise (Phase 3 scratch
models must emit an HF-compatible config of a supported family — a constraint
on Phase 3, not solved here).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

PIN_FILE = os.path.join(os.path.dirname(__file__), "llama_cpp_pin.txt")

# Locations baked by docker/training/Dockerfile.export; overridable for tests
DEFAULT_LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", "/opt/llama.cpp")

# Matches converter registrations: @ModelBase.register("LlamaForCausalLM", ...)
# (older revisions use @Model.register)
_REGISTER_RE = re.compile(r"@\w+\.register\(([^)]*)\)")
_QUOTED_RE = re.compile(r"\"([A-Za-z0-9_]+)\"")


class ExportError(RuntimeError):
    """Export pipeline failure with a user-facing message."""


def read_pin(path: str = PIN_FILE) -> str:
    """The pinned llama.cpp commit SHA; validates the pin file's format."""
    with open(path) as f:
        sha = f.read().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ExportError(f"Invalid llama.cpp pin (expected 40-hex SHA): {path}")
    return sha


def converter_path(llama_cpp_dir: str = DEFAULT_LLAMA_CPP_DIR) -> str:
    return os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")


def quantize_path(llama_cpp_dir: str = DEFAULT_LLAMA_CPP_DIR) -> str:
    return os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")


def supported_architectures(convert_script: str) -> set[str]:
    """Architecture names the pinned converter registers."""
    with open(convert_script) as f:
        text = f.read()
    names: set[str] = set()
    for match in _REGISTER_RE.finditer(text):
        names.update(_QUOTED_RE.findall(match.group(1)))
    if not names:
        raise ExportError(
            f"No architecture registrations found in {convert_script}; "
            "the pinned converter layout changed — update supported_architectures()")
    return names


def check_architecture(staging_dir: str, convert_script: str) -> str:
    """Assert the staged model's architecture is convertible; returns it."""
    config_path = os.path.join(staging_dir, "config.json")
    with open(config_path) as f:
        architectures = json.load(f).get("architectures") or []
    if not architectures:
        raise ExportError(f"config.json declares no architectures: {config_path}")
    arch = architectures[0]
    supported = supported_architectures(convert_script)
    if arch not in supported:
        raise ExportError(
            f"Architecture not supported by the pinned converter: {arch}\n"
            f"  Pinned llama.cpp: {read_pin()}\n"
            f"  Scope is GGUF-convertible families (Llama/Qwen/SmolLM); bump the\n"
            f"  pin in training/export/llama_cpp_pin.txt (with a note) if a newer\n"
            f"  llama.cpp supports this architecture")
    return arch


def _run(cmd: list[str], log_path: str) -> None:
    """Run a pipeline step, teeing output to a log file; raise on failure."""
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log_path) as log:
            tail = "".join(log.readlines()[-15:])
        raise ExportError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"  Log: {log_path}\n{tail}")


def convert_to_gguf(staging_dir: str, out_path: str, outtype: str = "bf16",
                    llama_cpp_dir: str = DEFAULT_LLAMA_CPP_DIR) -> None:
    """convert_hf_to_gguf.py: staged HF dir -> unquantized GGUF."""
    _run([sys.executable, converter_path(llama_cpp_dir), staging_dir,
          "--outfile", out_path, "--outtype", outtype],
         out_path + ".convert.log")


def quantize(in_path: str, out_path: str, quant_type: str,
             llama_cpp_dir: str = DEFAULT_LLAMA_CPP_DIR) -> None:
    """llama-quantize: unquantized GGUF -> quantized GGUF."""
    _run([quantize_path(llama_cpp_dir), in_path, out_path, quant_type],
         out_path + ".quantize.log")
