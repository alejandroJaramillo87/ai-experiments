"""Aggregate a window of served-request logs into a probe report (W4).

Reads the scores the serving process already took; runs no model, needs no
GPU, and does not care whether inference is up. Exit codes: 0 success,
1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse

from common.output import Printer


def configure(sub) -> None:
    p = sub.add_parser(
        "probe-monitor",
        help="aggregate a window of served-request logs into a probe report (W4)",
        description=__doc__,
    )
    p.add_argument("--window", default="7d", help="e.g. 7d or 12h")
    p.add_argument(
        "--probes",
        default=None,
        help="comma-separated probe names to report; default every probe in the window",
    )
    p.add_argument(
        "--log-dir",
        default="/mnt/ai-data/interp/logs/harness",
        help="served-request log directory for one service",
    )
    p.add_argument("--monitor-root", default="/mnt/ai-data/interp/monitor")
    p.add_argument("--max-records", type=int, default=None)
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    from interp.manifest import ManifestError
    from interp.probes.monitor import MonitorError, run_monitor

    probes = args.probes.split(",") if args.probes else None
    try:
        out.run(f"Monitor: window {args.window} under {args.log_dir}")
        report = run_monitor(
            args.log_dir,
            monitor_root=args.monitor_root,
            window=args.window,
            probes=probes,
            max_records=args.max_records,
        )
    except (MonitorError, ManifestError, OSError) as exc:
        out.error(f"Monitor failed: {exc}")
        return 1

    counts = report["counts"]
    out.ok(
        f"Reported {counts['reported']}/{counts['total']} records "
        f"({counts['flagged']} flagged, skipped {counts['skipped']})"
    )
    for name, dist in report["probes"].items():
        out.ok(
            f"{name}: mean {dist['mean']}, p95 {dist['p95']}, "
            f"over-threshold {dist['fraction_over_threshold']}"
        )
    out.ok(f"Report written: {report['out_dir']}/report.json")
    return 0
