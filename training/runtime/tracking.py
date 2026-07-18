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
import os
import sys
from datetime import datetime
from typing import Any, TextIO

from common.obs.core import current_cli, current_command, current_run_id
from common.reports import report_header


def _warn(sink: str, exc: Exception) -> None:
    print(f"[tracking] WARN: {sink} sink disabled: {exc}", file=sys.stderr)


class _JsonlSink:
    """Append-only metrics.jsonl writer (roadmap OBS-10).

    No I/O until the first log call, which opens the file line-buffered and
    writes the header record — so a run that never logs leaves no file, and
    a killed run keeps every completed step.
    """

    def __init__(self, run_dir: str, header: dict | None) -> None:
        self._path = os.path.join(run_dir, "metrics.jsonl")
        self._extra = dict(header or {})
        self._file: TextIO | None = None

    def log(self, metrics: dict, step: int) -> None:
        if self._file is None:
            self._file = open(self._path, "a", buffering=1, encoding="utf-8")
            self._file.write(json.dumps(self._header_record()) + "\n")
        ts = datetime.now().astimezone().isoformat(timespec="milliseconds")
        # default=float lets numpy scalars through; keys pass as logged
        self._file.write(
            json.dumps({"step": step, "ts": ts, **metrics}, default=float) + "\n"
        )

    def _header_record(self) -> dict:
        # Shared report envelope so validate_report accepts the file; the
        # ambient identity is empty outside a dispatched CLI, where the
        # events join is unavailable by construction.
        record = report_header("run_metrics")
        record["run_id"] = current_run_id() or ""
        record["cli"] = current_cli() or ""
        record["command"] = current_command() or ""
        record.update(self._extra)
        return record

    def close(self) -> None:
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
            os.environ.setdefault("WANDB_MODE", "offline")
            import wandb

            run = wandb.init(
                project=tracking.wandb_project,
                name=run_name,
                dir=run_dir,
                config=config or {},
            )
            self._sinks.append(("wandb", run))
        except Exception as exc:
            _warn("wandb", exc)

    def _init_mlflow(self, run_name: str, run_dir: str, config: dict | None) -> None:
        try:
            import mlflow

            if not os.environ.get("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri(f"file://{os.path.join(run_dir, 'mlruns')}")
            mlflow.set_experiment(run_name)
            mlflow.start_run(run_name=run_name)
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
                _warn(name, exc)
                self._sinks.remove((name, sink))

    def close(self) -> None:
        for name, sink in self._sinks:
            try:
                if name == "tensorboard":
                    sink.close()
                elif name == "wandb":
                    sink.finish()
                elif name == "mlflow":
                    sink.end_run()
                elif name == "jsonl":
                    sink.close()
            except Exception as exc:
                _warn(name, exc)
        self._sinks.clear()
