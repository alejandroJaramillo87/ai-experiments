"""CPU hardware suite (WI-3): Zen 5 feature flags and a CPUAdam step at the
smoke tier; affinity placement and the CPUAdam step-time bound at the full
tier.
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = pytest.mark.cpu

# Zen 5 ships full-width AVX-512; these flags are what the znver5 CPUAdam
# build and the huge-page inference path rely on.
REQUIRED_CPU_FLAGS = ("avx512f", "avx512bw", "avx512vl", "avx512_bf16")

# Step-time bound for the pre-built CPUAdam op. The P2-D5 design treats an
# optimizer step that dominates step time as the raise-grad_accum signal;
# the pathology this bound front-runs is the AVX-512 miscompile (upstream
# issue #7556), which is an order of magnitude off, not percentage points.
# 100M fp32 params move ~1.6GB of state per step; on DDR5-6000 with working
# vectorization that is well under half a second.
CPU_ADAM_PARAMS = 100_000_000
CPU_ADAM_STEP_BUDGET_S = 0.5


def read_cpuinfo_flags() -> set[str]:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    return set()


@pytest.mark.hw_smoke
class TestCpuSmoke:
    def test_zen5_avx512_flags_visible(self):
        flags = read_cpuinfo_flags()
        missing = [f for f in REQUIRED_CPU_FLAGS if f not in flags]
        assert not missing, f"CPU flags missing: {missing} (expected Zen 5)"

    def test_cpu_adam_single_step(self, torch):
        """One optimizer step on the pre-built extension completes; a
        miscompiled op crashes or hangs here rather than mid-run."""
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        param = torch.nn.Parameter(torch.zeros(1024, dtype=torch.float32))
        param.grad = torch.ones_like(param)
        DeepSpeedCPUAdam([param], lr=1e-3).step()
        assert torch.isfinite(param).all()
        assert not torch.equal(param, torch.zeros(1024))


@pytest.mark.hw_full
class TestCpuFull:
    def test_affinity_pinning_matches_actual_placement(self):
        """resolve_masks + apply_mask place the process exactly where the
        recipe says (D9: main on CCD0, workers on CCD1)."""
        from omegaconf import OmegaConf

        from training.recipes.schema import Recipe
        from training.runtime import affinity as aff

        recipe = OmegaConf.structured(Recipe)
        recipe.affinity.enabled = True
        original = os.sched_getaffinity(0)
        try:
            main_mask, workers_mask = aff.resolve_masks(recipe.affinity)
            aff.apply_mask(main_mask)
            assert os.sched_getaffinity(0) == main_mask
            assert main_mask.isdisjoint(workers_mask), (
                "main and worker masks overlap; dataloader workers would "
                "contend with the training process"
            )
        finally:
            os.sched_setaffinity(0, original)

    def test_cpu_adam_step_time_bound(self, torch):
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        param = torch.nn.Parameter(torch.zeros(CPU_ADAM_PARAMS, dtype=torch.float32))
        param.grad = torch.ones_like(param)
        optimizer = DeepSpeedCPUAdam([param], lr=1e-3)
        optimizer.step()  # warm-up: first step initializes moment tensors
        start = time.perf_counter()
        optimizer.step()
        elapsed = time.perf_counter() - start
        assert elapsed <= CPU_ADAM_STEP_BUDGET_S, (
            f"CPUAdam step took {elapsed:.2f}s for {CPU_ADAM_PARAMS} params "
            f"(budget {CPU_ADAM_STEP_BUDGET_S}s); suspect the AVX-512 "
            "miscompile (issue #7556) — rebuild the training image"
        )
