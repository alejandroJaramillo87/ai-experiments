"""Metric tracking fan-out (Phase 0 D6).

One Tracker.log(metrics, step) call fans out to every sink enabled in the
recipe's tracking block: tensorboard, wandb, mlflow. Offline-first: wandb runs
in offline mode unless WANDB_MODE is already set (this box may be
network-isolated during runs); syncing is a manual `wandb sync` step. mlflow
logs to a local file store under the run directory unless MLFLOW_TRACKING_URI
is set.

Sinks are imported lazily and are individually optional: a sink that fails to
initialize or log is disabled with a warning on stderr rather than killing a
multi-day run. This fan-out is the stable surface D10 telemetry plugs into.

The jsonl sink (roadmap OBS-10) writes <run_dir>/metrics.jsonl, the
observability system's durable per-run record; unlike the external
integrations it is on by default.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
from datetime import datetime
from typing import Any, Literal, TextIO, cast

from common.obs.core import current_cli, current_command, current_run_id
from common.reports import report_header

_WandbMode = Literal["online", "offline", "disabled", "shared"]


def _warn(sink: str, exc: Exception, path: str = "") -> None:
    where = f" ({path})" if path else ""
    print(f"[tracking] WARN: {sink} sink disabled{where}: {exc}", file=sys.stderr)


def _jsonl_default(value: Any) -> float | None:
    """Coerce non-JSON-native values (numpy scalars) for the jsonl sink.

    Non-finite results map to null: json.dumps would otherwise emit bare
    NaN/Infinity literals, which strict parsers (jq, browser JSON.parse)
    reject — and a NaN loss is exactly the record a post-mortem reads.
    """
    coerced = float(value)  # TypeError for non-numeric -> record skipped
    return coerced if math.isfinite(coerced) else None


class _JsonlSink:
    """Append-only metrics.jsonl writer (roadmap OBS-10).

    No I/O until the first log call, which opens the file line-buffered and
    writes the header record — so a run that never logs leaves no file, and
    a killed run keeps every completed step. Thread-safe: the training loop
    and the watchdog sampler thread both log into it, and TextIOWrapper is
    not thread-safe on its own.
    """

    def __init__(self, run_dir: str, header: dict | None) -> None:
        self._path = os.path.join(run_dir, "metrics.jsonl")
        self._extra = dict(header or {})
        self._file: TextIO | None = None
        self._lock = threading.Lock()
        # Identity is captured here, on the constructing (dispatching)
        # thread: contextvars do not cross threads, and the first log call
        # can come from the watchdog sampler, which has no bound context.
        # Empty outside a dispatched CLI, where the events join is
        # unavailable by construction.
        self._run_id = current_run_id() or ""
        self._cli = current_cli() or ""
        self._command = current_command() or ""

    def log(self, metrics: dict, step: int) -> None:
        ts = datetime.now().astimezone().isoformat(timespec="milliseconds")
        record = {"step": step, "ts": ts, **metrics}
        sanitized = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in record.items()
        }
        try:
            # keys pass as logged; numpy scalars coerce via the default
            line = json.dumps(sanitized, default=_jsonl_default, allow_nan=False)
        except (TypeError, ValueError) as exc:
            # a bad value is a data error in one record, not a sink fault:
            # skip the record, keep the sink
            print(
                f"[tracking] WARN: metrics record at step {step} skipped"
                f" ({self._path}): {exc}",
                file=sys.stderr,
            )
            return
        with self._lock:
            if self._file is None:
                self._file = open(self._path, "a", buffering=1, encoding="utf-8")
                self._file.write(json.dumps(self._header_record()) + "\n")
            self._file.write(line + "\n")

    def _header_record(self) -> dict:
        # Shared report envelope so validate_report accepts the file.
        record = report_header("run_metrics")
        record["run_id"] = self._run_id
        record["cli"] = self._cli
        record["command"] = self._command
        record.update(self._extra)
        return record

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class Tracker:
    """Fan one log(dict, step) out to the sinks enabled by TrackingConfig."""

    def __init__(
        self,
        tracking,
        run_name: str,
        run_dir: str,
        config: dict | None = None,
        header: dict | None = None,
    ) -> None:
        self._sinks: list[tuple[str, Any]] = []
        # jsonl first: it is the durable record, so a failure in an optional
        # external sink can never starve it of a log call.
        if tracking.metrics_jsonl:
            self._init_jsonl(run_dir, header)
        if tracking.tensorboard:
            self._init_tensorboard(run_dir)
        if tracking.wandb:
            self._init_wandb(tracking, run_name, run_dir, config)
        if tracking.mlflow:
            self._init_mlflow(run_name, run_dir, config)

    def _init_jsonl(self, run_dir: str, header: dict | None) -> None:
        try:
            self._sinks.append(("jsonl", _JsonlSink(run_dir, header)))
        except Exception as exc:
            _warn("jsonl", exc)

    def _init_tensorboard(self, run_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
            self._sinks.append(("tensorboard", writer))
        except Exception as exc:
            _warn("tensorboard", exc)

    def _init_wandb(
        self, tracking, run_name: str, run_dir: str, config: dict | None
    ) -> None:
        try:
            import wandb

            # mode as an init argument, not os.environ.setdefault: the env
            # mutation outlived a failed init and forced every later
            # in-process wandb use offline
            # An unexpected WANDB_MODE value falls back to offline rather
            # than failing init mid-run
            wandb_mode = os.environ.get("WANDB_MODE", "offline")
            if wandb_mode not in ("online", "offline", "disabled", "shared"):
                wandb_mode = "offline"
            run = wandb.init(
                project=tracking.wandb_project,
                name=run_name,
                dir=run_dir,
                config=config or {},
                mode=cast(_WandbMode, wandb_mode),
            )
            self._sinks.append(("wandb", run))
        except Exception as exc:
            _warn("wandb", exc)

    def _init_mlflow(self, run_name: str, run_dir: str, config: dict | None) -> None:
        started = False
        try:
            import mlflow

            if not os.environ.get("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri(f"file://{os.path.join(run_dir, 'mlruns')}")
            mlflow.set_experiment(run_name)
            mlflow.start_run(run_name=run_name)
            started = True
            if config:
                # mlflow rejects params over 500 chars; flat scalar params only
                mlflow.log_params(
                    {
                        k: v
                        for k, v in config.items()
                        if isinstance(v, int | float | str | bool)
                    }
                )
            self._sinks.append(("mlflow", mlflow))
        except Exception as exc:
            _warn("mlflow", exc)
            if started:
                # a failure after start_run (log_params on an oversize
                # value) left the run active, poisoning every later
                # Tracker in the process with "run already active"
                try:
                    mlflow.end_run()
                except Exception:
                    pass

    def log(self, metrics: dict, step: int) -> None:
        for name, sink in list(self._sinks):
            try:
                if name == "tensorboard":
                    for key, value in metrics.items():
                        sink.add_scalar(key, value, step)
                elif name == "wandb":
                    sink.log(metrics, step=step)
                elif name == "mlflow":
                    sink.log_metrics(metrics, step=step)
                elif name == "jsonl":
                    sink.log(metrics, step)
            except Exception as exc:
                _warn(name, exc, sink._path if name == "jsonl" else "")
                try:
                    self._sinks.remove((name, sink))
                except ValueError:
                    continue  # another thread already evicted it
                # best-effort teardown so an evicted sink does not leak
                # its fd, writer thread, or active mlflow run
                try:
                    self._close_sink(name, sink)
                except Exception:
                    pass

    @staticmethod
    def _close_sink(name: str, sink: Any) -> None:
        if name == "tensorboard":
            sink.close()
        elif name == "wandb":
            sink.finish()
        elif name == "mlflow":
            sink.end_run()
        elif name == "jsonl":
            sink.close()

    def close(self) -> None:
        for name, sink in self._sinks:
            try:
                self._close_sink(name, sink)
            except Exception as exc:
                _warn(name, exc)
        self._sinks.clear()
