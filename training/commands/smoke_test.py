"""Training container smoke test (Phase 0 D1).

Verifies the training stack imports and that CUDA-compiled paths actually run on sm_120:
flash-attn, bitsandbytes 4-bit, the pre-built DeepSpeed CPUAdam op, and torchao float8.
The optional unsloth backend (Phase 1 F3) is probed warn-only. Emits a versions
manifest for reproducibility.

Two modes:
  --build   Run at image-build time. CUDA asserts degrade to [SKIP] on a GPU-less build host
            (the import and CPU steps still gate the build); every step that CAN run must pass.
  (default) Run on the workstation via `make train-smoke`. CUDA capability (12, 0) is enforced;
            any failure exits non-zero.

Usage: python -m training smoke [--build]

Exit codes: 0 success, 1 a required step failed, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import json
import os

from common.output import Printer
from common.smoke import driver_version as _driver_version
from common.smoke import pkg_version as _pkg_version

# Manifest destination on the workstation (writable training root, roadmap D1)
WORKSTATION_MANIFEST = "/mnt/ai-data/training/runs/_smoke/versions.json"
REQUIRED_CAPABILITY = (12, 0)  # RTX 5090, Blackwell sm_120


def configure(sub) -> None:
    parser = sub.add_parser(
        "smoke",
        help="training container smoke test: imports and sm_120 CUDA paths (D1)",
        description="Training stack smoke test",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="build-time mode: CUDA asserts degrade to SKIP on a GPU-less host",
    )
    parser.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    out.line("=== Training stack smoke test ===")
    out.line()

    failures: list[str] = []
    packages: dict[str, str] = {}
    manifest: dict[str, object] = {"packages": packages}

    # Step 1: torch + CUDA capability
    cuda_ok = False
    try:
        import torch

        packages["torch"] = torch.__version__
        manifest["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            name = torch.cuda.get_device_name(0)
            manifest["gpu"] = name
            manifest["capability"] = list(cap)
            if cap == REQUIRED_CAPABILITY:
                out.ok(
                    f"torch {torch.__version__}: CUDA available, {name} sm_{cap[0]}{cap[1]}"
                )
                cuda_ok = True
            else:
                out.error(
                    f"Unexpected device capability {cap}, require {REQUIRED_CAPABILITY}"
                )
                failures.append("torch-capability")
        elif args.build:
            out.skip(
                "torch: CUDA unavailable on build host (enforced by make train-smoke)"
            )
        else:
            out.error("torch: CUDA not available")
            out.line(
                "  Solution: run on the workstation with the GPU reservation attached"
            )
            failures.append("torch-cuda")
    except Exception as exc:  # noqa: BLE001
        out.error(f"torch import failed: {exc}")
        failures.append("torch")

    # Step 2: pure imports (no CUDA required)
    import_targets = [
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "torchao",
        "liger_kernel",
        "hydra",
        "safetensors",
        "pynvml",
    ]
    for mod in import_targets:
        try:
            __import__(mod)
            dist = {"hydra": "hydra-core", "liger_kernel": "liger-kernel"}.get(mod, mod)
            packages[mod] = _pkg_version(dist)
            out.ok(f"import {mod}: {packages[mod]}")
        except Exception as exc:  # noqa: BLE001
            out.error(f"import {mod} failed: {exc}")
            failures.append(f"import-{mod}")

    # The remaining steps exercise CUDA kernels; skip cleanly when CUDA is unproven.
    def _needs_cuda(step: str) -> bool:
        if cuda_ok:
            return True
        (out.skip if args.build else out.error)(f"{step}: skipped (CUDA unproven)")
        if not args.build:
            failures.append(step)
        return False

    # Step 3: flash-attn forward
    if _needs_cuda("flash-attn"):
        try:
            import torch
            from flash_attn import flash_attn_func

            q = torch.randn(1, 8, 4, 32, dtype=torch.float16, device="cuda")
            o = flash_attn_func(q, q, q)
            assert o.shape == q.shape
            out.ok("flash-attn: flash_attn_func forward")
        except Exception as exc:  # noqa: BLE001
            out.error(f"flash-attn forward failed: {exc}")
            failures.append("flash-attn")

    # Step 4: bitsandbytes 4-bit quantized linear
    if _needs_cuda("bitsandbytes"):
        try:
            import bitsandbytes as bnb
            import torch

            layer = bnb.nn.Linear4bit(64, 64).cuda()
            y = layer(torch.randn(2, 64, device="cuda", dtype=torch.float16))
            assert y.shape == (2, 64)
            out.ok("bitsandbytes: 4-bit Linear forward")
        except Exception as exc:  # noqa: BLE001
            out.error(f"bitsandbytes 4-bit forward failed: {exc}")
            failures.append("bitsandbytes")

    # Step 5: DeepSpeed CPUAdam (proves the pre-built Zen 5 op loads and steps)
    try:
        import torch
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        p = torch.nn.Parameter(torch.randn(16))
        p.grad = torch.randn(16)
        opt = DeepSpeedCPUAdam([p])
        opt.step()
        out.ok("deepspeed: DeepSpeedCPUAdam step (pre-built op loaded)")
    except Exception as exc:  # noqa: BLE001
        out.error(f"DeepSpeedCPUAdam failed: {exc}")
        out.line("  Context: pre-built CPUAdam op (image build DS_BUILD_CPU_ADAM=1)")
        out.line("  Note: AVX-512 miscompile on Zen 5 is upstream issue #7556")
        failures.append("deepspeed-cpuadam")

    # Step 6: torchao float8 training (proves torch._scaled_mm on sm_120)
    if _needs_cuda("torchao-float8"):
        try:
            import torch
            from torchao.float8 import convert_to_float8_training

            mlp = (
                torch.nn.Sequential(
                    torch.nn.Linear(32, 32, bias=False),
                    torch.nn.Linear(32, 32, bias=False),
                )
                .cuda()
                .to(torch.bfloat16)
            )
            convert_to_float8_training(mlp)
            x = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
            mlp(x).sum().backward()
            out.ok("torchao: convert_to_float8_training forward+backward")
        except Exception as exc:  # noqa: BLE001
            out.error(f"torchao float8 failed: {exc}")
            failures.append("torchao-float8")

    # Step 7: unsloth optional backend (Phase 1 F3 decision 2). Warn-only:
    # unsloth is a throughput optimization documented as may-lag; its failure
    # must not block the image. Gemma 4 PLE support is proven by the F3
    # acceptance run, not here; this step proves the pin imports and exposes
    # the FastLanguageModel entry point train-sft's unsloth backend calls.
    try:
        from unsloth import FastLanguageModel  # noqa: F401

        packages["unsloth"] = _pkg_version("unsloth")
        out.ok(f"import unsloth: {packages['unsloth']} (FastLanguageModel present)")
    except Exception as exc:  # noqa: BLE001
        packages["unsloth"] = "unavailable"
        out.warn(f"unsloth unavailable: {exc}")
        out.line(
            "  Context: optional sft.backend=unsloth (F3); trl backend is unaffected"
        )
        out.line(
            "  Solution: repin unsloth against the torch/transformers "
            "pins, or keep sft.backend=trl"
        )

    # Step 8: versions manifest
    try:
        import torch

        try:
            manifest["driver"] = (
                torch.cuda.is_available() and _driver_version() or "unknown"
            )
        except Exception:  # noqa: BLE001
            manifest["driver"] = "unknown"
    except Exception:  # noqa: BLE001
        pass

    out.line()
    out.line("Versions manifest:")
    out.line(json.dumps(manifest, indent=2))

    # Persist on the workstation only (skip on GPU-less build hosts / missing mount)
    if not args.build:
        try:
            os.makedirs(os.path.dirname(WORKSTATION_MANIFEST), exist_ok=True)
            with open(WORKSTATION_MANIFEST, "w") as fh:
                json.dump(manifest, fh, indent=2)
            out.ok(f"manifest written: {WORKSTATION_MANIFEST}")
        except OSError as exc:
            out.warn(f"manifest not persisted: {exc}")

    out.line()
    if failures:
        out.error(f"FAILED: {', '.join(failures)}")
        return 1
    out.ok("ALL SMOKE STEPS PASSED")
    return 0
