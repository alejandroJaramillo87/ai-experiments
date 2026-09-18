"""Interpretability workbench smoke test (interp Phase 0 I1).

Verifies the interp stack imports and actually runs on sm_120: a Hugging Face forward with a
decoder-block hook, the bf16 substrate policy, and (when HF_TOKEN grants access) the gated
chain: Gemma-2-2B plus a Gemma Scope SAE via SAELens. Emits a versions
manifest for reproducibility.

Two modes:
  --build   Run at image-build time. CUDA and network asserts degrade to [SKIP] on a GPU-less
            or offline build host (the import steps still gate the build); every step that
            CAN run must pass.
  (default) Run on the workstation via `make interp-smoke`. CUDA capability (12, 0) is
            enforced; any failure exits non-zero. The gated Gemma step warns and skips when
            HF_TOKEN is absent but is required for Phase 0 acceptance.

Exit codes: 0 success, 1 a required step failed, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import json
import os

from common.output import Printer
from common.smoke import driver_version as _driver_version
from common.smoke import pkg_version as _pkg_version
from interp.families import (
    GEMMA_SCOPE_LAYER,
    GEMMA_SCOPE_RELEASE,
    GEMMA_SCOPE_SAE_ID,
)
from interp.families import (
    GEMMA_SCOPE_MODEL_ID as GEMMA_ID,
)

# Manifest destination on the workstation (writable interp root, roadmap storage layout)
WORKSTATION_MANIFEST = "/mnt/ai-data/interp/versions.json"
REQUIRED_CAPABILITY = (12, 0)  # RTX 5090, Blackwell sm_120
PROMPT = "The quick brown fox jumps over the lazy dog"


def configure(sub) -> None:
    p = sub.add_parser(
        "smoke",
        help="interpretability workbench smoke test (I1)",
        description="Interpretability workbench smoke test",
    )
    p.add_argument(
        "--build",
        action="store_true",
        help="build-time mode: CUDA and network asserts degrade to SKIP",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    out.line("=== Interpretability workbench smoke test ===")
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
                "torch: CUDA unavailable on build host (enforced by make interp-smoke)"
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

    # Step 2: pure imports plus the anti-downgrade check. The installed torch must match the
    # uv.lock pin exactly: any interp package that dragged torch off the cu130 build would
    # show up here even if it slipped past the build.
    import_targets = [
        "sae_lens",
        "transformers",
        "datasets",
        "einops",
        "pynvml",
    ]
    for mod in import_targets:
        try:
            __import__(mod)
            dist = {
                "sae_lens": "sae-lens",
                "pynvml": "nvidia-ml-py",
            }.get(mod, mod)
            ver = _pkg_version(dist)
            if ver == "unknown" and mod == "pynvml":
                ver = _pkg_version("pynvml")
            packages[mod] = ver
            out.ok(f"import {mod}: {ver}")
        except Exception as exc:  # noqa: BLE001
            out.error(f"import {mod} failed: {exc}")
            failures.append(f"import-{mod}")
    try:
        import torch

        locked = _pkg_version("torch")
        if torch.__version__ == locked:
            out.ok(f"torch pin intact: {locked}")
        else:
            out.error(f"torch runtime {torch.__version__} != installed dist {locked}")
            failures.append("torch-pin")
    except Exception:  # noqa: BLE001
        pass  # already reported in step 1

    # The remaining steps need CUDA and (first run) network; skip cleanly at build time.
    def _needs_cuda(step: str) -> bool:
        if cuda_ok:
            return True
        (out.skip if args.build else out.error)(f"{step}: skipped (CUDA unproven)")
        if not args.build:
            failures.append(step)
        return False

    # Step 3: Hugging Face forward on gpt2 with a forward hook on decoder block 6, the same
    # register_forward_hook pattern harvest and probe capture use on the workstation models.
    hf_model = None
    hf_tokenizer = None
    if _needs_cuda("hf-hook"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            hf_model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda").eval()
            hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            fired = []

            def _record(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                fired.append(tuple(tensor.shape))

            handle = hf_model.get_submodule("transformer.h.6").register_forward_hook(
                _record
            )
            try:
                inputs = hf_tokenizer(PROMPT, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    logits = hf_model(**inputs).logits
            finally:
                handle.remove()
            assert fired, "decoder-block hook did not fire"
            assert fired[0][-1] == hf_model.config.n_embd, f"hook width {fired[0][-1]}"
            assert logits.shape[-1] == hf_model.config.vocab_size
            out.ok("hf-hook: gpt2 forward, block 6 forward hook fired")
        except Exception as exc:  # noqa: BLE001
            out.error(f"hf-hook forward failed: {exc}")
            failures.append("hf-hook")

    # Step 4: bf16 substrate policy (interpretability runs at bf16, never quantized)
    if _needs_cuda("bf16") and hf_model is not None and hf_tokenizer is not None:
        try:
            import torch

            inputs = hf_tokenizer(PROMPT, return_tensors="pt").to("cuda")
            with torch.no_grad():
                logits = hf_model.to(torch.bfloat16)(**inputs).logits
            assert torch.isfinite(logits.float()).all(), "non-finite logits at bf16"
            out.ok("bf16: gpt2 forward, finite logits")
        except Exception as exc:  # noqa: BLE001
            out.error(f"bf16 forward failed: {exc}")
            failures.append("bf16")

    # Step 5 (gated): Gemma-2-2B at bf16 plus one Gemma Scope SAE via SAELens, the residual
    # read through the family registry's hook path. Proves the full hooked-model-plus-
    # pretrained-SAE dependency chain on this GPU.
    if not os.environ.get("HF_TOKEN"):
        out.warn(
            "gemma-scope: skipped, HF_TOKEN not set (required for Phase 0 acceptance)"
        )
        out.line(
            "  Solution: accept the Gemma license on Hugging Face, set HF_TOKEN in .env.interp"
        )
    elif _needs_cuda("gemma-scope"):
        try:
            import torch
            from sae_lens import SAE
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from interp import families

            gemma = (
                AutoModelForCausalLM.from_pretrained(GEMMA_ID, dtype=torch.bfloat16)
                .to("cuda")
                .eval()
            )
            tokenizer = AutoTokenizer.from_pretrained(GEMMA_ID)
            spec = families.family_from_config(gemma.config)
            captured = []

            def _grab(_module, _inputs, output):
                captured.append(output[0] if isinstance(output, tuple) else output)

            handle = gemma.get_submodule(
                families.hook_name(spec, GEMMA_SCOPE_LAYER)
            ).register_forward_hook(_grab)
            try:
                with torch.no_grad():
                    gemma(**tokenizer(PROMPT, return_tensors="pt").to("cuda"))
            finally:
                handle.remove()
            resid = captured[0]
            sae = SAE.from_pretrained(
                GEMMA_SCOPE_RELEASE, GEMMA_SCOPE_SAE_ID, device="cuda"
            )
            feats = sae.encode(resid.to(sae.dtype))
            assert torch.isfinite(feats.float()).all()
            out.ok("gemma-scope: Gemma-2-2B bf16 forward, Gemma Scope SAE encode")
        except Exception as exc:  # noqa: BLE001
            out.error(f"gemma-scope step failed: {exc}")
            failures.append("gemma-scope")

    # Step 6: versions manifest
    manifest["driver"] = _driver_version()
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
