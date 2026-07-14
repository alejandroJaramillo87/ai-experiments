#!/usr/bin/env python3
"""Interpretability workbench smoke test (interp Phase 0 I1).

Verifies the hooked-execution stack imports and actually runs on sm_120: a TransformerLens
hooked forward, an nnsight trace, the bf16 substrate policy, and (when HF_TOKEN grants
access) the gated chain: Gemma-2-2B plus a Gemma Scope SAE via SAELens. Emits a versions
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
import sys

# Manifest destination on the workstation (writable interp root, roadmap storage layout)
WORKSTATION_MANIFEST = "/mnt/ai-data/interp/versions.json"
REQUIRED_CAPABILITY = (12, 0)  # RTX 5090, Blackwell sm_120
PROMPT = "The quick brown fox jumps over the lazy dog"
GEMMA_ID = "google/gemma-2-2b"
GEMMA_SCOPE_RELEASE = "gemma-scope-2b-pt-res-canonical"
GEMMA_SCOPE_SAE_ID = "layer_12/width_16k/canonical"


class Printer:
    """Styled status output (see .styles/PYTHON_OUTPUT_STYLE.md)."""

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        use_color = not quiet and not os.environ.get("NO_COLOR") and sys.stdout.isatty()
        if use_color:
            self._c = {
                "OK": "\033[0;32m",
                "WARN": "\033[1;33m",
                "ERROR": "\033[0;31m",
                "INFO": "\033[0;36m",
                "RUN": "\033[0;34m",
                "SKIP": "\033[0;34m",
            }
            self._nc = "\033[0m"
        else:
            self._c = {}
            self._nc = ""

    def _status(self, tag: str, message: str, indent: int = 0) -> None:
        if self.quiet:
            return
        color = self._c.get(tag, "")
        print(f"{' ' * indent}{color}[{tag}]{self._nc} {message}")

    def ok(self, m, indent=0):
        self._status("OK", m, indent)

    def warn(self, m, indent=0):
        self._status("WARN", m, indent)

    def error(self, m, indent=0):
        self._status("ERROR", m, indent)

    def info(self, m, indent=0):
        self._status("INFO", m, indent)

    def skip(self, m, indent=0):
        self._status("SKIP", m, indent)

    def line(self, message: str = "") -> None:
        if not self.quiet:
            print(message)


def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interpretability workbench smoke test"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="build-time mode: CUDA and network asserts degrade to SKIP",
    )
    args = parser.parse_args()

    out = Printer()
    out.line("=== Interpretability workbench smoke test ===")
    out.line()

    failures: list[str] = []
    manifest: dict[str, object] = {"packages": {}}

    # Step 1: torch + CUDA capability
    cuda_ok = False
    try:
        import torch

        manifest["packages"]["torch"] = torch.__version__
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
        "transformer_lens",
        "nnsight",
        "nnterp",
        "sae_lens",
        "transformers",
        "datasets",
        "einops",
        "circuitsvis",
        "pynvml",
    ]
    for mod in import_targets:
        try:
            __import__(mod)
            dist = {
                "transformer_lens": "transformer-lens",
                "sae_lens": "sae-lens",
                "pynvml": "nvidia-ml-py",
            }.get(mod, mod)
            ver = _pkg_version(dist)
            if ver == "unknown" and mod == "pynvml":
                ver = _pkg_version("pynvml")
            manifest["packages"][mod] = ver
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

    # Step 3: TransformerLens hooked forward on gpt2-small
    tl_model = None
    if _needs_cuda("transformer-lens"):
        try:
            from transformer_lens import HookedTransformer

            tl_model = HookedTransformer.from_pretrained("gpt2").to("cuda")
            fired = []
            logits = tl_model.run_with_hooks(
                PROMPT,
                fwd_hooks=[
                    ("blocks.6.hook_resid_post", lambda t, hook: fired.append(t.shape))
                ],
            )
            assert fired, "residual-stream hook did not fire"
            assert logits.shape[-1] == tl_model.cfg.d_vocab
            out.ok("transformer-lens: gpt2-small forward, resid_post hook fired")
        except Exception as exc:  # noqa: BLE001
            out.error(f"transformer-lens hooked forward failed: {exc}")
            failures.append("transformer-lens")

    # Step 4: nnsight trace on the HF gpt2 model
    if _needs_cuda("nnsight"):
        try:
            from nnsight import LanguageModel

            lm = LanguageModel("gpt2", device_map="cuda", dispatch=True)
            with lm.trace(PROMPT):
                hidden = lm.transformer.h[6].output[0].save()
            assert hidden.shape[-1] == 768
            out.ok("nnsight: gpt2 trace, intermediate activation read")
        except Exception as exc:  # noqa: BLE001
            out.error(f"nnsight trace failed: {exc}")
            failures.append("nnsight")

    # Step 5: bf16 substrate policy (interpretability runs at bf16, never quantized)
    if _needs_cuda("bf16") and tl_model is not None:
        try:
            import torch

            logits = tl_model.to(torch.bfloat16)(PROMPT)
            assert torch.isfinite(logits.float()).all(), "non-finite logits at bf16"
            out.ok("bf16: gpt2-small forward, finite logits")
        except Exception as exc:  # noqa: BLE001
            out.error(f"bf16 forward failed: {exc}")
            failures.append("bf16")

    # Step 6 (gated): Gemma-2-2B at bf16 plus one Gemma Scope SAE via SAELens.
    # Proves the full hooked-model-plus-pretrained-SAE dependency chain on this GPU.
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
            from transformer_lens import HookedTransformer

            gemma = HookedTransformer.from_pretrained_no_processing(
                GEMMA_ID, dtype=torch.bfloat16
            ).to("cuda")
            _, cache = gemma.run_with_cache(PROMPT)
            resid = cache["blocks.12.hook_resid_post"]
            sae = SAE.from_pretrained(
                GEMMA_SCOPE_RELEASE, GEMMA_SCOPE_SAE_ID, device="cuda"
            )
            feats = sae.encode(resid.to(sae.dtype))
            assert torch.isfinite(feats.float()).all()
            out.ok("gemma-scope: Gemma-2-2B bf16 forward, Gemma Scope SAE encode")
        except Exception as exc:  # noqa: BLE001
            out.error(f"gemma-scope step failed: {exc}")
            failures.append("gemma-scope")

    # Step 7: versions manifest
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


def _driver_version() -> str:
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlSystemGetDriverVersion()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
