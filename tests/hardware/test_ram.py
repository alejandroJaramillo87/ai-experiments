"""Host-RAM hardware suite (WI-3): huge pages and pinned buffers at the
smoke tier; replay-buffer residency and ZeRO-2 host-RAM headroom at the
full tier.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.ram

# Replay residency probe size: large enough that RSS growth is unambiguous,
# small enough to keep hw_full runtime in seconds (the P2-D4 acceptance at
# real scale is a runbook entry).
REPLAY_BYTES = 1 << 30  # 1GB


def meminfo() -> dict[str, int]:
    values = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0])  # kB units except HugePages_*
    return values


@pytest.mark.hw_smoke
class TestRamSmoke:
    @pytest.mark.skipif(
        os.environ.get("HW_HUGEPAGES") != "1",
        reason="optional huge-page pool check; opt in with HW_HUGEPAGES=1",
    )
    def test_huge_pages_available(self):
        """The validated 26% CPU-inference optimization depends on a
        non-zero huge-page pool (AGENTS.md). Optional: the pool is a
        CPU-inference optimization, not a training prerequisite, so the
        check only runs when HW_HUGEPAGES=1 is set."""
        info = meminfo()
        assert info.get("HugePages_Total", 0) > 0, (
            "no huge pages configured; expected the workstation's 2MB pool "
            "(see docs/optimizations and docker/llama-cpu/hugepages/)"
        )

    def test_small_pinned_buffer_allocates(self, cuda):
        buffer = cuda.empty(64 << 20, dtype=cuda.uint8, pin_memory=True)
        assert buffer.is_pinned()


@pytest.mark.hw_full
class TestRamFull:
    def test_replay_buffer_ram_resident(self, tmp_path):
        """A ReplayBuffer loads its byte budget into process RSS (P2-D4:
        replay windows are served from RAM, not re-read from disk)."""
        import numpy as np
        import psutil
        from support import make_shard_dir

        from training.continual.replay import ReplayBuffer

        window = 1024
        tokens_needed = REPLAY_BYTES // 2  # uint16 tokens
        rng = np.random.default_rng(11)
        shard_dir = make_shard_dir(
            tmp_path,
            rng.integers(0, 512, size=tokens_needed).tolist(),
            shard_tokens=tokens_needed // 8,
        )
        process = psutil.Process()
        rss_before = process.memory_info().rss
        buffer = ReplayBuffer(
            shard_dir,
            window=window,
            max_bytes=REPLAY_BYTES,
            seed=3,
            run_dir=str(tmp_path),
            name="hw-replay",
        )
        assert len(buffer) > 0
        # Touch every window once so lazily-mapped pages become resident.
        for i in range(len(buffer)):
            buffer[i]
        grown = process.memory_info().rss - rss_before
        loaded = len(buffer) * window * 2  # uint16
        assert grown >= loaded // 2, (
            f"RSS grew {grown} bytes for {loaded} loaded bytes; replay "
            "windows do not appear RAM-resident"
        )

    def test_zero2_offload_host_ram_headroom(self):
        """The continual recipe's predicted host RAM (fp32 master weights
        plus Adam moments plus replay budget) fits the machine with
        headroom — the same invariant validate() enforces, checked against
        the real machine."""
        import psutil

        from tests.hardware.conftest import REPO_ROOT
        from training.calculator.components import host_ram_bytes, spec_from_recipe
        from training.recipes import load_recipe

        recipe = load_recipe(
            os.path.join(
                REPO_ROOT,
                "training",
                "recipes",
                "reference",
                "continual-gemma4-e2b.yaml",
            )
        )
        predicted = host_ram_bytes(spec_from_recipe(recipe), recipe)
        total = psutil.virtual_memory().total
        assert predicted < total * 0.75, (
            f"predicted host RAM {predicted / (1 << 30):.1f}GB exceeds 75% "
            f"of the machine's {total / (1 << 30):.0f}GB"
        )
