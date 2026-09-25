"""SSD/storage hardware suite (WI-3): mount layout and staging writability
at the smoke tier; staging throughput, checkpoint round trip, the harvest
disk preflight, and the D9 data_wait_ms bar at the full tier.
"""

from __future__ import annotations

import os
import shutil
import time

import pytest

from tests.hardware.conftest import AI_DATA_ROOT, REPO_ROOT

pytestmark = pytest.mark.ssd

EXPECTED_SUBDIRS = ("models", "training", "interp")

# Sanity bar only (the benchmark suite owns real numbers): tmpfs staging
# moves GB/s and the NVMe pool >1GB/s sequential; anything under this is a
# broken mount or a filled disk, not noise.
STAGING_MIN_MB_S = 200.0
STAGING_PROBE_BYTES = 1 << 30  # 1GB

# D9 acceptance bar: mean data_wait_ms under 5% of mean step time
# (docs/training/base/heterogeneous_compute.md).
DATA_WAIT_FRACTION_BAR = 0.05


def staging_dir() -> str:
    """The compose tmpfs mount inside the training container, /dev/shm on
    the bare host — both are the RAM-backed staging medium D9 targets."""
    return "/staging" if os.path.isdir("/staging") else "/dev/shm"


def tiny_causal_lm():
    """A real PreTrainedModel, built from a config so nothing is downloaded.

    save_checkpoint writes model/ with save_pretrained and load_checkpoint
    calls tie_weights (training/runtime/checkpoint.py:144,331); both are
    PreTrainedModel APIs that a bare nn.Module does not carry. The HF format
    is deliberate rather than incidental — D7 GGUF export consumes model/ —
    so the round trip has to be exercised through a real one. Dimensions are
    the smallest that still produce sharded safetensors and tied embeddings.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            intermediate_size=64,
            vocab_size=128,
        )
    )


@pytest.mark.hw_smoke
@pytest.mark.platform
class TestSsdSmoke:
    def test_ai_data_mounted_with_expected_layout(self):
        assert os.path.isdir(AI_DATA_ROOT), f"{AI_DATA_ROOT} is not mounted"
        missing = [
            name
            for name in EXPECTED_SUBDIRS
            if not os.path.isdir(os.path.join(AI_DATA_ROOT, name))
        ]
        assert not missing, f"{AI_DATA_ROOT} missing subdirectories: {missing}"

    def test_tmpfs_staging_writable(self, tmp_path):
        target = os.path.join(staging_dir(), "hw-smoke-probe")
        with open(target, "wb") as f:
            f.write(b"x" * (1 << 20))
        os.remove(target)


@pytest.mark.hw_full
class TestSsdFull:
    @pytest.mark.platform
    def test_shard_staging_write_throughput(self):
        target = os.path.join(staging_dir(), "hw-throughput-probe")
        payload = os.urandom(1 << 24)  # 16MB block, written repeatedly
        written = 0
        start = time.perf_counter()
        try:
            with open(target, "wb") as f:
                while written < STAGING_PROBE_BYTES:
                    f.write(payload)
                    written += len(payload)
                f.flush()
                os.fsync(f.fileno())
        finally:
            if os.path.exists(target):
                os.remove(target)
        elapsed = time.perf_counter() - start
        mb_s = written / (1 << 20) / elapsed
        assert mb_s >= STAGING_MIN_MB_S, (
            f"staging wrote at {mb_s:.0f}MB/s (< {STAGING_MIN_MB_S}MB/s); "
            "check the tmpfs mount and disk pressure"
        )

    @pytest.mark.training
    def test_checkpoint_write_restore_roundtrip(self, torch, tmp_path):
        """D6 checkpoint save/load round trip on the SSD-backed training
        root: weights, optimizer, scheduler, and data state survive."""
        from training.runtime import checkpoint as ckpt

        root = os.path.join(AI_DATA_ROOT, "training", ".hw-test-checkpoints")
        os.makedirs(root, exist_ok=True)
        try:
            model = tiny_causal_lm()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            data_state = {"format": "index", "next_index": 7, "window": 64}
            meta = {"tokens_seen": 448, "loss": 1.0, "recipe_hash": "hw"}
            ckpt.save_checkpoint(root, 1, model, optimizer, scheduler, data_state, meta)

            restored = tiny_causal_lm()
            restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            restored_sched = torch.optim.lr_scheduler.LambdaLR(
                restored_opt, lambda _: 1.0
            )
            # Random init differs between the two instances, so the equality
            # assertion below has something to prove. Without this the test
            # would pass on two identical models that never round-tripped.
            assert any(
                not torch.equal(a, b)
                for a, b in zip(
                    model.state_dict().values(),
                    restored.state_dict().values(),
                    strict=True,
                )
            )

            class Slot:
                state = None

                def load_state_dict(self, state):
                    Slot.state = state

            path = ckpt.resolve_resume(root, "latest")
            loaded_meta = ckpt.load_checkpoint(
                path, restored, restored_opt, restored_sched, Slot()
            )
            assert loaded_meta["step"] == 1
            assert Slot.state["next_index"] == 7
            for a, b in zip(
                model.state_dict().values(),
                restored.state_dict().values(),
                strict=True,
            ):
                assert torch.equal(a, b)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    @pytest.mark.interp
    def test_harvest_disk_preflight_reads_real_disk(self):
        """The P4-D2 preflight (harvests can reach ~1.4TB per layer) refuses
        an impossible projection against the real mount and passes a small
        one."""
        from interp.harvest import HarvestError, disk_preflight

        out_root = os.path.join(AI_DATA_ROOT, "interp", "activations")
        target = out_root if os.path.isdir(out_root) else AI_DATA_ROOT
        disk_preflight(target, 1 << 20, floor_gb=100.0)  # 1MB projection: passes
        with pytest.raises(HarvestError, match="disk"):
            disk_preflight(target, 10 << 50, floor_gb=100.0)  # 10PB: refused

    @pytest.mark.gpu
    @pytest.mark.training
    def test_data_wait_under_the_d9_bar(self, cuda, tmp_path, train_lock):
        """Mean data_wait_ms stays under 5% of mean step time on a short
        prefetch-loader run (docs/training/base/heterogeneous_compute.md)."""
        import numpy as np
        from support import make_shard_dir

        from training.recipes import load_recipe
        from training.runtime.loop import train

        rng = np.random.default_rng(13)
        shard_dir = make_shard_dir(
            tmp_path, rng.integers(0, 512, size=40_000).tolist(), shard_tokens=8000
        )
        recipe = load_recipe(
            os.path.join(
                REPO_ROOT, "training", "recipes", "reference", "tiny-e2e.yaml"
            ),
            [
                f"data.source={shard_dir}",
                f"run.output_root={tmp_path / 'runs'}",
                "data.seq_len=64",
                "data.workers=2",
                "schedule.total_steps=20",
                "schedule.warmup_steps=2",
                "checkpoint.interval_steps=20",
            ],
        )
        start = time.perf_counter()
        result = train(recipe)
        wall = time.perf_counter() - start
        mean_step_ms = wall / result.final_step * 1000.0
        fraction = result.data_wait_ms_mean / mean_step_ms
        assert fraction < DATA_WAIT_FRACTION_BAR, (
            f"data_wait_ms mean {result.data_wait_ms_mean:.1f}ms is "
            f"{fraction:.1%} of the {mean_step_ms:.1f}ms mean step "
            f"(bar {DATA_WAIT_FRACTION_BAR:.0%})"
        )
