#!/usr/bin/env python3
"""
PyTorch GPU environment validation utility.
Verifies PyTorch GPU setup, Blackwell (sm_120) support, and AI package dependencies.

Output follows .styles/PYTHON_OUTPUT_STYLE.md: bracketed status indicators, colorized
for a TTY, plain when piped or under NO_COLOR, and pure JSON under --json.
"""

import argparse
import contextlib
import io
import json
import os
import re
import subprocess
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version
from typing import Any

# The shared Printer is stdlib-only, so importing it keeps this script safe to
# run against a broken Python environment; it does require the repo checkout.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.output import Printer  # noqa: E402

# Exit codes
EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

# Key AI packages: display name -> (import name, required).
# Import name differs from the distribution name for several packages. Optional
# packages (required=False) report WARN rather than ERROR when absent; vllm lives
# in the GPU containers and is not necessarily installed in the host env.
KEY_PACKAGES = {
    "transformers": ("transformers", True),
    "accelerate": ("accelerate", True),
    "datasets": ("datasets", True),
    "huggingface_hub": ("huggingface_hub", True),
    "bitsandbytes": ("bitsandbytes", True),
    "peft": ("peft", True),
    "sentence_transformers": ("sentence_transformers", True),
    "vllm": ("vllm", False),
}

# Distribution names for importlib.metadata.version() when they differ from the
# import name used above.
DIST_NAMES = {
    "huggingface_hub": "huggingface-hub",
    "sentence_transformers": "sentence-transformers",
}

# Import-time stderr signatures that indicate a package imported but its native/CUDA
# backend failed to load. Import "succeeds" so a naive check reports OK; we translate
# these into a clean WARN instead. Maps a signature substring -> short reason.
BACKEND_ERROR_SIGNATURES = {
    "Configured CUDA binary not found": "CUDA backend unavailable",
    "library load error": "native backend load error",
}


def quiet_import(module_name: str) -> tuple[Any | None, str]:
    """Import a module, capturing its import-time warnings and stderr.

    Heavy AI packages print warnings (pynvml FutureWarning via torch) or full
    tracebacks (bitsandbytes with a missing CUDA binary) at import time even when the
    import itself succeeds. Capturing keeps that noise out of the tool's output while
    letting the caller inspect it.

    Returns:
        (module or None if import failed, captured stderr text).
    """
    buf = io.StringIO()
    try:
        with warnings.catch_warnings(), contextlib.redirect_stderr(buf):
            warnings.simplefilter("ignore")
            module = __import__(module_name)
        return module, buf.getvalue()
    except Exception:
        return None, buf.getvalue()


def _pkg_version(import_name: str) -> str | None:
    """Return the installed distribution version, or None if unavailable."""
    dist = DIST_NAMES.get(import_name, import_name)
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def check_pytorch_gpu_setup(out: Printer) -> dict[str, Any]:
    """Check PyTorch GPU configuration, CUDA compatibility, and Blackwell support.

    Returns:
        Dictionary of validation results (see keys in the initializer below).
    """
    out.line("PyTorch GPU environment check:")
    out.line("Verifying PyTorch GPU, CUDA, cuDNN, and Blackwell (sm_120) support")
    out.line()

    result: dict[str, Any] = {
        "success": False,
        "pytorch_version": None,
        "cuda_available": False,
        "pytorch_cuda_version": None,
        "system_cuda_version": None,
        "gpu_detected": False,
        "gpu_name": None,
        "compute_capability": None,
        "arch_list": None,
        "sm_120_supported": None,
        "bf16_supported": None,
        "cudnn_available": False,
    }

    # Import PyTorch (captures the pynvml FutureWarning emitted at import time).
    torch, _ = quiet_import("torch")
    if torch is None:
        out.error("PyTorch installation: not installed")
        return result
    result["pytorch_version"] = torch.__version__
    out.ok(f"PyTorch version: {torch.__version__}")
    out.ok(f"Python version: {sys.version.split(' ')[0]}")
    out.line()

    # Check system CUDA version using nvcc
    try:
        nvcc_output = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True
        )
        if nvcc_output.returncode == 0:
            for line in nvcc_output.stdout.split("\n"):
                if "release" in line:
                    match = re.search(r"release (\d+\.\d+)", line)
                    if match:
                        result["system_cuda_version"] = match.group(1)
                        out.ok(f"System CUDA (nvcc): {result['system_cuda_version']}")
                        break
    except FileNotFoundError:
        out.warn("System CUDA (nvcc): not found in PATH")

    # Check PyTorch CUDA version
    if hasattr(torch.version, "cuda"):
        result["pytorch_cuda_version"] = torch.version.cuda
        out.ok(f"PyTorch CUDA build: {torch.version.cuda}")

    # Check if CUDA is available to PyTorch
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        out.ok(f"CUDA availability: {result['cuda_available']}")
    else:
        out.error(f"CUDA availability: {result['cuda_available']}")
        out.error("GPU configuration: PyTorch cannot access CUDA")
        out.line("  Solution: Verify NVIDIA drivers and PyTorch CUDA build")
        out.line("  Common causes: driver issues, CPU-only PyTorch, version mismatch")
        return result

    # Check for version mismatch (major version)
    if result["system_cuda_version"] and result["pytorch_cuda_version"]:
        sys_major = result["system_cuda_version"].split(".")[0]
        pytorch_cuda_major = result["pytorch_cuda_version"].split(".")[0]

        if sys_major != pytorch_cuda_major:
            out.warn("CUDA version compatibility: version mismatch")
            out.line(
                f"  System: {result['system_cuda_version']}, PyTorch: {result['pytorch_cuda_version']}"
            )
            out.line("  Solution: Reinstall PyTorch for system CUDA version")

    # GPU detection
    device_count = torch.cuda.device_count()
    result["gpu_detected"] = device_count > 0
    if device_count > 0:
        out.ok(f"GPU devices: {device_count} detected")
    else:
        out.error(f"GPU devices: {device_count} detected")

    if device_count > 0:
        result["gpu_name"] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        result["compute_capability"] = f"{capability[0]}.{capability[1]}"
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        out.line(f"  Primary GPU: {result['gpu_name']}")
        out.line(f"  CUDA capability: {result['compute_capability']}")
        out.line(f"  Memory: {total_memory:.1f}GB")

        # Blackwell (sm_120) support: warn if the GPU is compute capability 12.x but
        # the installed torch build lacks sm_120 kernels (falls back to JIT or fails).
        # arch_list looks like ['sm_90', 'sm_100', 'sm_120', ...].
        arch_list = torch.cuda.get_arch_list()
        result["arch_list"] = arch_list
        if capability[0] >= 12:
            has_sm120 = any("120" in a for a in arch_list)
            result["sm_120_supported"] = has_sm120
            if has_sm120:
                out.ok("Blackwell sm_120: present in torch arch list", indent=2)
            else:
                out.warn(
                    f"Blackwell sm_120: not in torch build ({arch_list})", indent=2
                )
                out.line("    Solution: install a torch build with sm_120 (cu130)")

        # bf16 support (relevant for Blackwell inference/training precision)
        result["bf16_supported"] = torch.cuda.is_bf16_supported()
        if result["bf16_supported"]:
            out.ok(f"bfloat16 support: {result['bf16_supported']}", indent=2)
        else:
            out.warn(f"bfloat16 support: {result['bf16_supported']}", indent=2)

    # Check cuDNN
    result["cudnn_available"] = torch.backends.cudnn.is_available()
    if result["cudnn_available"]:
        out.ok(f"cuDNN: version {torch.backends.cudnn.version()}")
    else:
        out.warn("cuDNN: not available")

    # Test GPU operation with a real correctness assertion (not just "it ran").
    out.line()
    try:
        tensor = torch.randn(10, 10, device="cuda")
        gpu_out = tensor * 2 + 1
        expected = tensor.cpu() * 2 + 1
        if torch.allclose(gpu_out.cpu(), expected, atol=1e-5):
            out.ok("GPU tensor test: operations functional and correct")
            result["success"] = True
        else:
            out.error("GPU tensor test: result mismatch")
    except Exception as e:
        out.error(f"GPU tensor test: {str(e)}")

    return result


def check_key_packages(out: Printer) -> dict[str, dict[str, Any]]:
    """Check key AI package availability, versions, and native-backend health.

    Returns:
        Mapping of package name to {installed, version, required, warning}.
    """
    out.line()
    out.line("AI package availability:")

    packages: dict[str, dict[str, Any]] = {}
    for name, (import_name, required) in KEY_PACKAGES.items():
        module, noise = quiet_import(import_name)
        imported = module is not None
        # Distribution metadata is independent of importability: a package
        # whose import raises (bitsandbytes without torch) is still
        # installed, and telling the operator to install it misdiagnoses.
        pkg_version = _pkg_version(import_name)
        installed = imported or pkg_version is not None

        # Detect a package that imported but whose native/CUDA backend failed to load
        # (e.g. bitsandbytes with a missing libbitsandbytes_cuda*.so).
        warning = None
        if imported:
            for signature, reason in BACKEND_ERROR_SIGNATURES.items():
                if signature in noise:
                    warning = reason
                    break
        elif installed:
            warning = "import failed"

        packages[name] = {
            "installed": installed,
            "version": pkg_version,
            "required": required,
            "warning": warning,
        }

        version_suffix = f" {pkg_version}" if pkg_version else ""
        if installed and not imported:
            out.error(
                f"{name}:{version_suffix} installed but failed to import",
                indent=2,
            )
            out.line(
                "    Solution: fix the import error (often a missing "
                "dependency such as torch), not a reinstall"
            )
        elif installed and warning:
            out.warn(f"{name}:{version_suffix} ({warning})", indent=2)
            out.line(
                f"    Solution: reinstall {name} matching the installed CUDA (cu130)"
            )
        elif installed:
            out.ok(
                f"{name}:{version_suffix if pkg_version else ' installed'}", indent=2
            )
        elif required:
            out.error(f"{name}: not installed", indent=2)
        else:
            out.warn(f"{name}: not installed (optional)", indent=2)

    return packages


def check_running_interpreter(out: Printer) -> bool:
    """Whether this is running inside a virtual environment at all.

    This file is executable and carries a `#!/usr/bin/env python3` shebang, so
    `./check_py_deps_install.py` resolves python3 from PATH -- an interpreter
    carrying none of the project's dependencies. Every check below then reports
    `not installed`, which is a false negative about the machine rather than a
    fact about the environment, and is indistinguishable from a genuinely broken
    install. Refuse instead of answering the wrong question.

    Any virtual environment passes, not this project's alone: the interp and
    training images run their own at /opt/venv, and the agent sandbox at
    /opt/pyenv.
    """
    if sys.prefix != sys.base_prefix:
        return True

    out.error(f"Running outside a virtual environment: {sys.executable}")
    out.line("  Found: the project dependencies live in .venv, not on this interpreter")
    out.line("  Solution: run it under uv, which selects the project environment")
    out.line("  Command: uv run python scripts/utils/check_py_deps_install.py")
    return False


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PyTorch GPU environment validation utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python check_py_deps_install.py
  python check_py_deps_install.py --json
        """,
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON only (suppresses human-readable text)",
    )

    return parser


def main() -> int:
    """Run all validation checks.

    Returns:
        Exit code: 0 when the GPU setup validates, 1 otherwise.
    """
    parser = create_parser()
    args = parser.parse_args()

    # Guarded before the quiet Printer is built: a --json caller still needs to
    # see why nothing came back, and this is human output, so stderr carries it
    # and stdout stays free of anything that is not JSON.
    if not check_running_interpreter(Printer(quiet=False)):
        return EXIT_USAGE

    out = Printer(quiet=args.json)

    out.line("PyTorch GPU validation utility")
    out.line()
    try:
        gpu_result = check_pytorch_gpu_setup(out)
        packages = check_key_packages(out)

        # Summary
        out.line()
        out.line("Validation summary:")
        if gpu_result["success"]:
            out.ok("PyTorch GPU setup")
        else:
            out.error("PyTorch GPU setup")
        if gpu_result["system_cuda_version"]:
            out.line(f"System CUDA: {gpu_result['system_cuda_version']}")
        if gpu_result["pytorch_cuda_version"]:
            out.line(f"PyTorch CUDA: {gpu_result['pytorch_cuda_version']}")

        if args.json:
            print(json.dumps({"gpu": gpu_result, "packages": packages}, indent=2))

        return EXIT_SUCCESS if gpu_result["success"] else EXIT_FAILURE

    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    sys.exit(main())
