"""Operational watchdogs (Phase 0 D10).

Automated abort-or-alert for unattended runs, on the failure modes that
destroy runs or hardware value: numerical divergence, thermal and power
anomalies, RAM exhaustion under offload, and disk exhaustion under
checkpointing. NVML-based; DCGM is deliberately not used (datacenter-oriented,
degraded on GeForce — NVML exposes everything needed on the RTX 5090).

One in-process object, hooked into the D6 loop at its per-step metrics point:

- on_step(step, loss, grad_norm) evaluates the step-level signals (NaN/inf or
  exploding loss, sustained gradient-norm excursion) and raises WatchdogAbort
  when one trips. The loop takes an emergency checkpoint, then re-raises.
- A background thread samples NVML (temperature, power, clocks, throttle
  reasons, VRAM) and psutil (available RAM, disk free on the checkpoint
  filesystem) every watchdog.interval_s, evaluates the hardware signals, and
  forwards every sample to the D6 tracker — the watchdog doubles as the run's
  hardware telemetry source (D8 consumes it). A thread cannot raise into the
  training step, so hardware aborts are latched and raised by the next
  on_step call.

Actions: log (stderr + tracking sink), alert (append to <run_dir>/alerts.jsonl
plus a pluggable hook — Phase 0 ships file+log only), abort (WatchdogAbort;
the loop saves an emergency checkpoint marked emergency=true, exempt from
retention). Every trigger records signal, value, threshold, and timestamp.

All sampling is fail-soft: a machine without NVML or psutil yields absent
readings, never an exception — the watchdog must not be able to kill the run
it protects except through an explicit abort. torch is never imported here so
the CPU sandbox (and unit tests with mocked samplers) can use the module.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import threading
import time
from collections import deque
from statistics import median

ALERTS_FILENAME = "alerts.jsonl"
LOG_PREFIX = "[watchdog]"

# The loss-spike rule arms only after this many finite samples: early-training
# loss is noisy and a factor-of-median rule on 1-2 samples is a false positive.
LOSS_WARMUP_SAMPLES = 5


class WatchdogAbort(RuntimeError):
    """A watchdog signal crossed its abort threshold. Carries the trigger so
    the CLI can report signal, value, and threshold without parsing text."""

    def __init__(self, signal: str, value: float, threshold: float) -> None:
        self.signal = signal
        self.value = value
        self.threshold = threshold
        super().__init__(
            f"{signal}: value {value:.4g} crossed abort threshold {threshold:.4g}")


def sample_gpu() -> dict:
    """Best-effort NVML sample of device 0. Readings the driver does not
    expose are omitted; a machine without NVML yields {}."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return {}
    sample: dict = {}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        try:
            sample["gpu_temp_c"] = float(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            sample["gpu_power_w"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except Exception:
            pass
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            sample["gpu_vram_used_gb"] = mem.used / 2**30
        except Exception:
            pass
        try:
            sample["gpu_sm_clock_mhz"] = float(pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_SM))
        except Exception:
            pass
        try:
            reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
            idle = getattr(pynvml, "nvmlClocksThrottleReasonGpuIdle", 0)
            sample["gpu_throttle_active"] = float(bool(reasons & ~idle))
        except Exception:
            pass
    except Exception:
        pass
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return sample


def sample_host(disk_path: str) -> dict:
    """Best-effort host sample: psutil available RAM plus disk free on the
    filesystem holding disk_path (the checkpoint root)."""
    sample: dict = {}
    try:
        import psutil

        sample["ram_available_gb"] = psutil.virtual_memory().available / 2**30
    except Exception:
        pass
    try:
        sample["disk_free_gb"] = shutil.disk_usage(disk_path).free / 2**30
    except Exception:
        pass
    return sample


class Watchdog:
    """Evaluate the D10 signals against a recipe's watchdog block.

    config is the recipe's watchdog node (WatchdogConfig fields); tracker is
    the D6 Tracker (or None); disk_path is the filesystem the disk-floor check
    watches (the checkpoint root); alert_hook, when set, receives each alert
    record dict (future email/ntfy — Phase 0 ships file+log only).
    """

    def __init__(self, config, run_dir: str, tracker=None,
                 disk_path: str | None = None, alert_hook=None) -> None:
        self.cfg = config
        self.run_dir = run_dir
        self.tracker = tracker
        self.disk_path = disk_path or run_dir
        self.alert_hook = alert_hook

        self._losses: deque[float] = deque(maxlen=max(int(config.loss_window), 1))
        self._grad_exceeds = 0
        # Per-signal state for sustain/edge logic: monotonic time the excursion
        # began (None when clear).
        self._since: dict[str, float | None] = {
            "gpu_temp": None, "gpu_power": None, "gpu_throttle": None}
        self._alerted: set[str] = set()
        self._pending_abort: WatchdogAbort | None = None
        self._blocked = False
        self._last_step = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the background sampler thread (no-op when interval_s <= 0)."""
        if self.cfg.interval_s <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._sampler_loop, name="watchdog-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.cfg.interval_s + 5)
            self._thread = None

    def _sampler_loop(self) -> None:
        while not self._stop.wait(self.cfg.interval_s):
            try:
                sample = {**sample_gpu(), **sample_host(self.disk_path)}
                with self._lock:
                    self.check_sample(sample, time.monotonic())
                if self.tracker is not None and sample:
                    self.tracker.log(sample, self._last_step)
            except Exception as exc:
                # The watchdog must never kill the run it protects.
                print(f"{LOG_PREFIX} WARN: sampler error ignored: {exc}",
                      file=sys.stderr)

    # -- step-level signals --------------------------------------------------

    def on_step(self, step: int, loss: float, grad_norm: float) -> None:
        """Evaluate step signals; raise WatchdogAbort when one trips. Also
        raises any abort latched by the sampler thread since the last step."""
        self._last_step = step
        with self._lock:
            pending, self._pending_abort = self._pending_abort, None
        if pending is not None:
            raise pending

        if not math.isfinite(loss):
            self._abort("loss_non_finite", loss, 0.0)
        if len(self._losses) >= LOSS_WARMUP_SAMPLES:
            baseline = median(self._losses)
            limit = self.cfg.loss_spike_factor * baseline
            if loss > limit:
                self._abort("loss_spike", loss, limit)
        self._losses.append(loss)

        if self.cfg.grad_norm_max > 0:
            if not math.isfinite(grad_norm) or grad_norm > self.cfg.grad_norm_max:
                self._grad_exceeds += 1
                if self._grad_exceeds >= self.cfg.patience:
                    self._abort("grad_norm", grad_norm, self.cfg.grad_norm_max)
            else:
                self._grad_exceeds = 0

    # -- sampled hardware signals --------------------------------------------

    def check_sample(self, sample: dict, now: float) -> None:
        """Evaluate one hardware sample (sampler thread; unit tests call it
        directly with synthetic readings). Aborts are latched, not raised."""
        self._check_sustained(
            "gpu_temp", sample.get("gpu_temp_c"),
            float(self.cfg.gpu_temp_limit_c), now, abort_on_sustain=True)
        self._check_sustained(
            "gpu_power", sample.get("gpu_power_w"),
            float(self.cfg.power_w), now, abort_on_sustain=False)
        throttle = sample.get("gpu_throttle_active")
        if throttle is not None:
            # Active throttle reasons are data for D8, not usually fatal.
            self._check_sustained(
                "gpu_throttle", throttle, 0.5, now, abort_on_sustain=False)

        ram = sample.get("ram_available_gb")
        if ram is not None:
            floor = float(self.cfg.ram_floor_gb)
            if ram < floor / 2:
                self._alert("ram_floor", ram, floor / 2, action="abort")
                self._pending_abort = self._pending_abort or WatchdogAbort(
                    "ram_floor", ram, floor / 2)
            elif ram < floor:
                self._edge_alert("ram_floor", ram, floor)
            else:
                self._alerted.discard("ram_floor")

        disk = sample.get("disk_free_gb")
        if disk is not None:
            floor = float(self.cfg.disk_free_floor_gb)
            if disk < floor:
                if not self._blocked:
                    self._blocked = True
                    self._alert("disk_floor", disk, floor, action="block")
            elif self._blocked:
                self._blocked = False
                print(f"{LOG_PREFIX} disk_floor cleared: {disk:.1f} GB free; "
                      f"checkpoint writes resume", file=sys.stderr)

    def _check_sustained(self, signal: str, value: float | None,
                         threshold: float, now: float,
                         abort_on_sustain: bool) -> None:
        """Alert on the rising edge of an excursion; after sustain_s either
        latch an abort (thermal) or alert again (power/throttle: data only)."""
        if value is None:
            return
        if value <= threshold:
            self._since[signal] = None
            self._alerted.discard(signal)
            self._alerted.discard(signal + "_sustained")
            return
        if self._since[signal] is None:
            self._since[signal] = now
            self._alert(signal, value, threshold, action="alert")
        elif now - self._since[signal] >= self.cfg.sustain_s:
            if abort_on_sustain:
                self._alert(signal, value, threshold, action="abort")
                self._pending_abort = self._pending_abort or WatchdogAbort(
                    signal, value, threshold)
            else:
                self._edge_alert(signal + "_sustained", value, threshold)

    # -- actions ---------------------------------------------------------------

    @property
    def checkpoint_writes_blocked(self) -> bool:
        """True while disk free is below the floor: the loop skips periodic
        checkpoint saves (retention ran first; this is the backstop). The
        abort-path emergency save is still permitted."""
        return self._blocked

    def _edge_alert(self, signal: str, value: float, threshold: float) -> None:
        if signal not in self._alerted:
            self._alerted.add(signal)
            self._alert(signal, value, threshold, action="alert")

    def _alert(self, signal: str, value: float, threshold: float,
               action: str) -> None:
        """Record one trigger everywhere post-mortems look: stderr, the run's
        alerts.jsonl, the tracking stream, and the pluggable hook."""
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "step": self._last_step,
            "signal": signal,
            "value": value,
            "threshold": threshold,
            "action": action,
        }
        print(f"{LOG_PREFIX} {action.upper()} {signal}: value {value:.4g}, "
              f"threshold {threshold:.4g}", file=sys.stderr)
        try:
            with open(os.path.join(self.run_dir, ALERTS_FILENAME), "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"{LOG_PREFIX} WARN: cannot write {ALERTS_FILENAME}: {exc}",
                  file=sys.stderr)
        if self.tracker is not None:
            try:
                self.tracker.log({f"watchdog/{signal}": value}, self._last_step)
            except Exception:
                pass
        if self.alert_hook is not None:
            try:
                self.alert_hook(record)
            except Exception as exc:
                print(f"{LOG_PREFIX} WARN: alert hook failed: {exc}",
                      file=sys.stderr)

    def _abort(self, signal: str, value: float, threshold: float) -> None:
        self._alert(signal, value, threshold, action="abort")
        raise WatchdogAbort(signal, value, threshold)
