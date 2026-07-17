"""Branch tests for the D10 watchdog fail-soft paths.

Complements test_watchdog.py (signal logic) with the seams it leaves out:
the NVML/psutil samplers under every per-reading failure mode, the sampler
thread's never-kill-the-run error handler, the disabled-sampler start(), and
the alert sink degradations (unwritable run dir, tracker that raises). NVML
and psutil are absent in the sandbox, so both are injected via sys.modules —
the sampler bodies themselves execute for real.
"""

import json
import os
import threading
from types import SimpleNamespace

import pytest

from training.recipes.schema import WatchdogConfig
from training.runtime.watchdog import Watchdog, WatchdogAbort, sample_gpu, sample_host


class FakeMemInfo:
    used = 8 * 2**30


def _raise(exc):
    def fn(*args, **kwargs):
        raise exc

    return fn


def make_nvml(**overrides):
    """Scriptable pynvml stand-in with benign defaults; tests override
    individual entry points with raising functions to hit each except arm."""
    calls = {"shutdown": 0}

    def shutdown():
        calls["shutdown"] += 1

    stub = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        NVML_CLOCK_SM=1,
        nvmlClocksThrottleReasonGpuIdle=0x1,
        nvmlInit=lambda: None,
        nvmlShutdown=shutdown,
        nvmlDeviceGetHandleByIndex=lambda idx: "handle0",
        nvmlDeviceGetTemperature=lambda h, kind: 62,
        nvmlDeviceGetPowerUsage=lambda h: 450_000,  # milliwatts
        nvmlDeviceGetMemoryInfo=lambda h: FakeMemInfo,
        nvmlDeviceGetClockInfo=lambda h, kind: 2550,
        nvmlDeviceGetCurrentClocksThrottleReasons=lambda h: 0x4,
        _calls=calls,
    )
    for name, value in overrides.items():
        setattr(stub, name, value)
    return stub


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


# -- sample_gpu ---------------------------------------------------------------


def test_sample_gpu_reports_all_readings(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "pynvml", make_nvml())
    sample = sample_gpu()
    assert sample == {
        "gpu_temp_c": 62.0,
        "gpu_power_w": 450.0,  # 450000 mW scaled to watts
        "gpu_vram_used_gb": 8.0,
        "gpu_sm_clock_mhz": 2550.0,
        "gpu_throttle_active": 1.0,  # 0x4 remains after masking idle 0x1
    }


def test_sample_gpu_masks_idle_throttle_reason(monkeypatch):
    # The GPU-idle reason fires on every idle card; only non-idle reasons
    # count as an active throttle.
    import sys

    stub = make_nvml(nvmlDeviceGetCurrentClocksThrottleReasons=lambda h: 0x1)
    monkeypatch.setitem(sys.modules, "pynvml", stub)
    assert sample_gpu()["gpu_throttle_active"] == 0.0


def test_sample_gpu_init_failure_yields_empty_sample(monkeypatch):
    import sys

    stub = make_nvml(nvmlInit=_raise(RuntimeError("driver not loaded")))
    monkeypatch.setitem(sys.modules, "pynvml", stub)
    assert sample_gpu() == {}


def test_sample_gpu_omits_readings_that_fail(monkeypatch):
    # Every per-reading NVML call fails: the sample is empty rather than an
    # exception, and shutdown still runs so the driver handle is released.
    import sys

    err = _raise(RuntimeError("not supported"))
    stub = make_nvml(
        nvmlDeviceGetTemperature=err,
        nvmlDeviceGetPowerUsage=err,
        nvmlDeviceGetMemoryInfo=err,
        nvmlDeviceGetClockInfo=err,
        nvmlDeviceGetCurrentClocksThrottleReasons=err,
    )
    monkeypatch.setitem(sys.modules, "pynvml", stub)
    assert sample_gpu() == {}
    assert stub._calls["shutdown"] == 1


def test_sample_gpu_survives_handle_and_shutdown_failure(monkeypatch):
    import sys

    stub = make_nvml(
        nvmlDeviceGetHandleByIndex=_raise(RuntimeError("no device")),
        nvmlShutdown=_raise(RuntimeError("shutdown failed")),
    )
    monkeypatch.setitem(sys.modules, "pynvml", stub)
    assert sample_gpu() == {}


# -- sample_host --------------------------------------------------------------


def test_sample_host_reads_stub_psutil_and_real_disk(tmp_path, monkeypatch):
    import sys

    stub = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=64 * 2**30))
    monkeypatch.setitem(sys.modules, "psutil", stub)
    sample = sample_host(str(tmp_path))
    assert sample["ram_available_gb"] == 64.0
    assert sample["disk_free_gb"] > 0  # real shutil.disk_usage on tmp_path


def test_sample_host_without_psutil_or_disk_path_yields_empty(tmp_path, monkeypatch):
    # None in sys.modules forces the import to fail even if psutil were
    # installed; a missing path makes shutil.disk_usage raise. Both failures
    # must yield absent readings, never an exception.
    import sys

    monkeypatch.setitem(sys.modules, "psutil", None)
    assert sample_host(str(tmp_path / "no_such_dir")) == {}


# -- sampler lifecycle --------------------------------------------------------


def test_start_is_noop_when_sampling_disabled(run_dir):
    wd = make_watchdog(run_dir, interval_s=0.0)
    wd.start()
    assert wd._thread is None
    assert not any(t.name == "watchdog-sampler" for t in threading.enumerate())
    wd.stop()  # must be safe without a thread


def test_start_twice_reuses_the_running_thread(run_dir):
    wd = make_watchdog(run_dir, interval_s=60.0)
    wd.start()
    first = wd._thread
    assert first is not None
    wd.start()
    assert wd._thread is first
    wd.stop()  # the stop event wakes the 60s wait immediately
    assert wd._thread is None


class OneShotStop:
    """Stop-event stand-in: the sampler loop runs exactly one iteration in
    the calling thread, so the error path is exercised deterministically."""

    def __init__(self):
        self.calls = 0

    def wait(self, timeout=None):
        self.calls += 1
        return self.calls > 1


def test_sampler_error_is_logged_and_ignored(run_dir, monkeypatch, capsys):
    import training.runtime.watchdog as wd_mod

    monkeypatch.setattr(wd_mod, "sample_gpu", _raise(RuntimeError("nvml boom")))
    wd = make_watchdog(run_dir)
    stop = OneShotStop()
    wd._stop = stop
    wd._sampler_loop()  # must swallow the error and exit on the next wait
    assert stop.calls == 2
    err = capsys.readouterr().err
    assert "[watchdog] WARN: sampler error ignored: nvml boom" in err


# -- alert sink degradations --------------------------------------------------


def test_missing_run_dir_warns_but_abort_still_raises(tmp_path, capsys):
    # The alerts file being unwritable must not mask the abort itself.
    missing = str(tmp_path / "no_such_dir")
    wd = make_watchdog(missing)
    with pytest.raises(WatchdogAbort) as exc:
        wd.on_step(1, float("nan"), 1.0)
    assert exc.value.signal == "loss_non_finite"
    err = capsys.readouterr().err
    assert "ABORT loss_non_finite" in err
    assert "cannot write alerts.jsonl" in err
    assert read_alerts(missing) == []


class RaisingTracker:
    def log(self, metrics, step):
        raise RuntimeError("tracker down")


def test_tracker_failure_does_not_suppress_alert_record(run_dir):
    wd = make_watchdog(run_dir, tracker=RaisingTracker(), ram_floor_gb=8.0)
    wd.check_sample({"ram_available_gb": 6.0}, now=0.0)  # must not raise
    alerts = read_alerts(run_dir)
    assert [a["signal"] for a in alerts] == ["ram_floor"]
    assert alerts[0]["action"] == "alert"
