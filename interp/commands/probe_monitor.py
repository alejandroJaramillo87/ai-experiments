"""Replay a window of serving logs and score registered probes (W4).

Workbench batch job: refuses to start while inference services answer their
healthchecks. Exit codes: 0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
import os

from common.output import Printer


def configure(sub) -> None:
    p = sub.add_parser(
        "probe-monitor",
        help="replay a window of serving logs and score registered probes (W4)",
        description=__doc__,
    )
    p.add_argument("--window", default="7d", help="e.g. 7d or 12h")
    p.add_argument(
        "--probes",
        required=True,
        help="comma-separated probe names under the probes root",
    )
    p.add_argument("--probes-root", default="/mnt/ai-data/interp/probes")
    p.add_argument("--model", required=True, help="replay model (bf16 HF)")
    p.add_argument("--model-hash", required=True)
    p.add_argument(
        "--log-dir",
        default="/mnt/ai-data/interp/logs/llama-gpu",
        help="W3 log directory for one target service",
    )
    p.add_argument("--monitor-root", default="/mnt/ai-data/interp/monitor")
    p.add_argument("--max-records", type=int, default=None)
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from interp.manifest import ManifestError
    from interp.probes.monitor import MonitorError, run_monitor
    from interp.probes.registry import RegistryError

    probe_dirs = [
        os.path.join(args.probes_root, name) for name in args.probes.split(",")
    ]
    try:
        out.run(f"Monitor replay: window {args.window}, {len(probe_dirs)} probes")
        report = run_monitor(
            probe_dirs,
            args.model,
            args.model_hash,
            args.log_dir,
            monitor_root=args.monitor_root,
            window=args.window,
            max_records=args.max_records,
        )
    except (MonitorError, RegistryError, ManifestError, OSError) as exc:
        out.error(f"Monitor failed: {exc}")
        return 1

    counts = report["counts"]
    out.ok(
        f"Replayed {counts['replayed']}/{counts['total']} records "
        f"({counts['flagged']} flagged, skipped {counts['skipped']})"
    )
    for name, dist in report["probes"].items():
        out.ok(
            f"{name}: mean {dist['mean']}, p95 {dist['p95']}, "
            f"over-threshold {dist['fraction_over_threshold']}"
        )
    out.ok(f"Report written: {report['out_dir']}/report.json")
    return 0
