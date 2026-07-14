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
"""

from __future__ import annotations

import os
import sys


def _warn(sink: str, exc: Exception) -> None:
    print(f"[tracking] WARN: {sink} sink disabled: {exc}", file=sys.stderr)


class Tracker:
    """Fan one log(dict, step) out to the sinks enabled by TrackingConfig."""

    def __init__(self, tracking, run_name: str, run_dir: str,
                 config: dict | None = None) -> None:
        self._sinks: list[tuple[str, object]] = []
        if tracking.tensorboard:
            self._init_tensorboard(run_dir)
        if tracking.wandb:
            self._init_wandb(tracking, run_name, run_dir, config)
        if tracking.mlflow:
            self._init_mlflow(run_name, run_dir, config)

    def _init_tensorboard(self, run_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=os.path.join(run_dir, "tensorboard"))
            self._sinks.append(("tensorboard", writer))
        except Exception as exc:
            _warn("tensorboard", exc)

    def _init_wandb(self, tracking, run_name: str, run_dir: str,
                    config: dict | None) -> None:
        try:
            os.environ.setdefault("WANDB_MODE", "offline")
            import wandb

            run = wandb.init(
                project=tracking.wandb_project, name=run_name,
                dir=run_dir, config=config or {})
            self._sinks.append(("wandb", run))
        except Exception as exc:
            _warn("wandb", exc)

    def _init_mlflow(self, run_name: str, run_dir: str,
                     config: dict | None) -> None:
        try:
            import mlflow

            if not os.environ.get("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri(f"file://{os.path.join(run_dir, 'mlruns')}")
            mlflow.set_experiment(run_name)
            mlflow.start_run(run_name=run_name)
            if config:
                # mlflow rejects params over 500 chars; flat scalar params only
                mlflow.log_params({k: v for k, v in config.items()
                                   if isinstance(v, int | float | str | bool)})
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
            except Exception as exc:
                _warn(name, exc)
        self._sinks.clear()
