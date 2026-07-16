"""Unit tests for the D10 operational watchdogs.

Per the acceptance criteria, synthetic NaN loss, temperature, RAM, and disk
readings are injected (NVML/psutil are never touched — check_sample takes a
plain dict) and the configured action is asserted, including the emergency
checkpoint path through the loop and the retention exemption. Signal-logic
tests are torch-free; the loop integration test skips without torch.
"""

import json
import math
import os

import pytest

from training.recipes.schema import WatchdogConfig
from training.watchdog import (
    LOSS_WARMUP_SAMPLES,
    Watchdog,
    WatchdogAbort,
)


class FakeTracker:
    def __init__(self):
        self.logged = []

    def log(self, metrics, step):
        self.logged.append((metrics, step))


@pytest.fixture()
def run_dir(tmp_path):
    return str(tmp_path)


def make_watchdog(run_dir, tracker=None, **overrides):
    cfg = WatchdogConfig(**overrides)
    return Watchdog(cfg, run_dir, tracker=tracker)


def read_alerts(run_dir):
    path = os.path.join(run_dir, "alerts.jsonl")
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def test_nan_loss_aborts_and_records_alert(run_dir):
    tracker = FakeTracker()
    wd = make_watchdog(run_dir, tracker)
    with pytest.raises(WatchdogAbort) as exc:
        wd.on_step(1, float("nan"), 1.0)
    assert exc.value.signal == "loss_non_finite"
    alerts = read_alerts(run_dir)
    assert alerts[0]["signal"] == "loss_non_finite"
    assert alerts[0]["action"] == "abort"
    assert math.isnan(alerts[0]["value"])
    assert {"time", "step", "value", "threshold"} <= alerts[0].keys()
    assert any("watchdog/loss_non_finite" in m for m, _ in tracker.logged)


def test_loss_spike_aborts_after_warmup_only(run_dir):
    wd = make_watchdog(run_dir, loss_spike_factor=3.0)
    # A huge first loss must not trip the rule: the median window is unarmed.
    wd.on_step(1, 100.0, 1.0)
    for step in range(2, LOSS_WARMUP_SAMPLES + 2):
        wd.on_step(step, 2.0, 1.0)
    with pytest.raises(WatchdogAbort) as exc:
        wd.on_step(LOSS_WARMUP_SAMPLES + 2, 2.0 * 3.0 + 1.0, 1.0)
    assert exc.value.signal == "loss_spike"


def test_loss_below_spike_threshold_continues(run_dir):
    wd = make_watchdog(run_dir, loss_spike_factor=3.0)
    for step in range(1, 20):
        wd.on_step(step, 2.0 + 0.01 * step, 1.0)
    assert read_alerts(run_dir) == []


def test_grad_norm_patience(run_dir):
    wd = make_watchdog(run_dir, grad_norm_max=10.0, patience=3)
    wd.on_step(1, 1.0, 50.0)
    wd.on_step(2, 1.0, 50.0)
    wd.on_step(3, 1.0, 5.0)  # recovery resets the counter
    wd.on_step(4, 1.0, 50.0)
    wd.on_step(5, 1.0, 50.0)
    with pytest.raises(WatchdogAbort) as exc:
        wd.on_step(6, 1.0, 50.0)
    assert exc.value.signal == "grad_norm"


def test_grad_norm_disabled_by_default(run_dir):
    wd = make_watchdog(run_dir)
    for step in range(1, 10):
        wd.on_step(step, 1.0, 1e9)
    assert read_alerts(run_dir) == []


def test_gpu_temp_alerts_then_aborts_when_sustained(run_dir):
    wd = make_watchdog(run_dir, gpu_temp_limit_c=90, sustain_s=120.0)
    wd.check_sample({"gpu_temp_c": 95.0}, now=0.0)
    assert [a["action"] for a in read_alerts(run_dir)] == ["alert"]
    wd.check_sample({"gpu_temp_c": 96.0}, now=60.0)  # not yet sustained
    wd.check_sample({"gpu_temp_c": 96.0}, now=130.0)
    assert read_alerts(run_dir)[-1]["action"] == "abort"
    with pytest.raises(WatchdogAbort) as exc:
        wd.on_step(1, 1.0, 1.0)  # latched abort surfaces on the next step
    assert exc.value.signal == "gpu_temp"


def test_gpu_temp_recovery_clears_excursion(run_dir):
    wd = make_watchdog(run_dir, gpu_temp_limit_c=90, sustain_s=120.0)
    wd.check_sample({"gpu_temp_c": 95.0}, now=0.0)
    wd.check_sample({"gpu_temp_c": 80.0}, now=60.0)  # recovered
    wd.check_sample({"gpu_temp_c": 95.0}, now=200.0)  # new excursion, clock resets
    wd.on_step(1, 1.0, 1.0)  # must not raise
    actions = [a["action"] for a in read_alerts(run_dir)]
    assert actions == ["alert", "alert"]


def test_power_sustained_alerts_but_never_aborts(run_dir):
    wd = make_watchdog(run_dir, power_w=600.0, sustain_s=120.0)
    wd.check_sample({"gpu_power_w": 650.0}, now=0.0)
    wd.check_sample({"gpu_power_w": 650.0}, now=130.0)
    wd.check_sample({"gpu_power_w": 650.0}, now=260.0)
    wd.on_step(1, 1.0, 1.0)  # must not raise
    actions = [a["action"] for a in read_alerts(run_dir)]
    assert actions == ["alert", "alert"]  # edge + one sustained, rate-limited


def test_ram_floor_alerts_then_half_floor_aborts(run_dir):
    wd = make_watchdog(run_dir, ram_floor_gb=8.0)
    wd.check_sample({"ram_available_gb": 6.0}, now=0.0)
    wd.on_step(1, 1.0, 1.0)  # alert only, no abort
    assert read_alerts(run_dir)[-1]["action"] == "alert"
    wd.check_sample({"ram_available_gb": 3.0}, now=10.0)
    assert read_alerts(run_dir)[-1]["action"] == "abort"
    with pytest.raises(WatchdogAbort) as exc:
        wd.on_step(2, 1.0, 1.0)
    assert exc.value.signal == "ram_floor"


def test_disk_floor_blocks_and_unblocks_checkpoint_writes(run_dir):
    wd = make_watchdog(run_dir, disk_free_floor_gb=100)
    assert not wd.checkpoint_writes_blocked
    wd.check_sample({"disk_free_gb": 50.0}, now=0.0)
    assert wd.checkpoint_writes_blocked
    assert read_alerts(run_dir)[-1]["action"] == "block"
    wd.on_step(1, 1.0, 1.0)  # the run continues; no abort
    wd.check_sample({"disk_free_gb": 150.0}, now=10.0)
    assert not wd.checkpoint_writes_blocked
    assert len(read_alerts(run_dir)) == 1  # recovery logs, but is not an alert


def test_absent_readings_are_ignored(run_dir):
    wd = make_watchdog(run_dir)
    wd.check_sample({}, now=0.0)  # machine without NVML/psutil
    wd.on_step(1, 1.0, 1.0)
    assert read_alerts(run_dir) == []


def test_alert_hook_receives_records_and_failures_are_ignored(run_dir):
    received = []

    def hook(record):
        received.append(record)
        raise RuntimeError("hook boom")

    cfg = WatchdogConfig()
    wd = Watchdog(cfg, run_dir, alert_hook=hook)
    with pytest.raises(WatchdogAbort):
        wd.on_step(1, float("inf"), 1.0)
    assert received[0]["signal"] == "loss_non_finite"


def test_retention_exempts_emergency_checkpoints(tmp_path):
    from training import checkpoint as ckpt

    root = str(tmp_path)
    for step, emergency in [(1, False), (2, False), (3, True), (4, False),
                            (5, False), (6, False)]:
        d = os.path.join(root, ckpt.step_dir_name(step))
        os.makedirs(d)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"step": step, **({"emergency": True} if emergency else {})}, f)
    deleted = ckpt.apply_retention(root, retain_last=2, retain_every=1000)
    # last two (5, 6) plus the emergency save (3) survive
    assert deleted == [1, 2, 4]
    assert ckpt.list_checkpoints(root) == [3, 5, 6]


def test_loop_saves_emergency_checkpoint_on_abort(tmp_path, monkeypatch):
    """Integration: a watchdog abort mid-run leaves a resumable checkpoint
    marked emergency=true and surfaces WatchdogAbort to the caller."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import numpy as np
    from support import TINY_RECIPE, make_shard_dir

    from training.loop import train
    from training.recipes import load_recipe

    rng = np.random.default_rng(7)
    tokens = rng.integers(0, 512, size=6000).tolist()
    shard_dir = make_shard_dir(tmp_path, tokens, shard_tokens=2000)
    recipe = load_recipe(TINY_RECIPE, [
            f"data.source={shard_dir}",
            f"run.output_root={tmp_path / 'runs'}",
            "data.seq_len=64",
            "schedule.total_steps=8",
            "checkpoint.interval_steps=100",  # no periodic save before the abort
            "watchdog.enabled=true",
            "watchdog.interval_s=3600",  # sampler effectively idle in the test
        ])

    def abort_at_step_3(self, step, loss, grad_norm):
        if step >= 3:
            raise WatchdogAbort("loss_non_finite", float("nan"), 0.0)

    monkeypatch.setattr(Watchdog, "on_step", abort_at_step_3)
    with pytest.raises(WatchdogAbort):
        train(recipe)

    from training import checkpoint as ckpt

    root = str(tmp_path / "runs" / "tiny-e2e" / "checkpoints")
    assert ckpt.list_checkpoints(root) == [3]
    with open(os.path.join(root, ckpt.step_dir_name(3), "meta.json")) as f:
        meta = json.load(f)
    assert meta["emergency"] is True
    assert meta["step"] == 3
