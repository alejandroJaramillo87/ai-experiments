"""Render a run's metrics and logs into a dashboard (OBS-11, OBS-12).

Reads <run_dir>/metrics.jsonl, locates the event file for the header's
run_id under ${OBS_LOG_DIR}, and writes <run_dir>/run.html — one HTML file
with the charts, data, and vendored uPlot inlined, referencing no external
URL. With --serve, an ephemeral local server renders the same page live
instead, polling the growing files; no run.html is written. Exit codes:
0 success, 1 failure, 2 invalid usage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common.obs import dashboard
from common.obs import serve as obs_serve
from common.output import Printer


def configure(sub) -> None:
    p = sub.add_parser(
        "report",
        help="render a run dashboard from metrics.jsonl",
        description=__doc__,
    )
    p.add_argument(
        "--run",
        required=True,
        help="run directory containing metrics.jsonl",
    )
    p.add_argument(
        "--events",
        help="event file path (default: located from the header run_id "
        "under OBS_LOG_DIR)",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="serve a live dashboard instead of writing run.html",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind address for --serve (0.0.0.0 exposes to the LAN; "
        "there is no authentication)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8010,
        help="port for --serve (default 8010)",
    )
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    out = Printer()
    run_dir = Path(args.run)
    if not run_dir.is_dir():
        out.error(f"Run directory not found: {run_dir}")
        out.line("  Solution: pass a run directory created by a training command")
        return 1
    if args.serve:
        # metrics.jsonl may not exist yet for a run that just started; the
        # server resolves the events path lazily from the header.
        events_path = Path(args.events) if args.events else None
        return obs_serve.run(
            args.host,
            args.port,
            run_dir,
            events_path,
            dashboard.build_live_shell(),
            out,
        )
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        out.error(f"Metrics file not found: {metrics_path}")
        out.line("  Solution: the run must have logged at least one step (OBS-10)")
        return 1

    header, metrics, skipped = dashboard.read_metrics(metrics_path)
    if skipped:
        out.warn(f"Malformed metrics lines skipped: {skipped}")
    if not header:
        out.warn("Metrics file has no run_metrics header")

    events: list[dict] = []
    events_path = (
        Path(args.events) if args.events else dashboard.default_events_path(header)
    )
    if events_path is None:
        out.warn("Event file not located (no run identity in header)")
    elif not events_path.is_file():
        out.warn(f"Event file not found: {events_path}")
    else:
        events, ev_skipped = dashboard.read_events(events_path)
        if ev_skipped:
            out.warn(f"Malformed event lines skipped: {ev_skipped}")

    out_path = run_dir / "run.html"
    out_path.write_text(
        dashboard.build_snapshot(header, metrics, events), encoding="utf-8"
    )
    out.ok(f"Wrote {out_path}")
    return 0
