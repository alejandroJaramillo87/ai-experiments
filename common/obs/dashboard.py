"""Snapshot dashboard builder (roadmap OBS-11).

Pure page assembly: reads the vendored uPlot assets and viewer.js, inlines
them with the run's data, and returns one self-contained HTML string that
references no external URL. training/commands/report.py owns the CLI shell
(argument parsing, Printer output, exit codes).
"""

from __future__ import annotations

import html
import json
import os
from functools import cache
from pathlib import Path

from common.obs.core import DEFAULT_LOG_DIR

_OBS_DIR = Path(__file__).resolve().parent

_PAGE_CSS = """
body { margin: 0; font-family: system-ui, sans-serif; background: #fafafa;
       color: #222; }
#root { max-width: 1140px; margin: 0 auto; padding: 12px; }
.header { display: flex; flex-wrap: wrap; gap: 16px; padding: 8px 0 12px;
          border-bottom: 1px solid #ddd; }
.hdr-item { display: inline-flex; gap: 6px; font-size: 13px; }
.hdr-key { color: #777; }
.hdr-val { font-family: monospace; }
.chart { margin: 16px 0; background: #fff; border: 1px solid #e2e2e2;
         padding: 8px; }
.log-title { font-weight: bold; margin-bottom: 6px; }
.log-controls { display: flex; gap: 8px; margin-bottom: 6px; }
.log-controls select, .log-controls input { font: 13px monospace;
                                            padding: 2px 6px; }
.log-controls input { flex: 1; }
.log-body { height: 320px; overflow-y: auto; background: #fff;
            border: 1px solid #e2e2e2; font: 12px/1.5 monospace;
            padding: 6px 8px; white-space: pre-wrap; }
.log-row { border-bottom: 1px solid #f4f4f4; }
.log-warning { color: #8a6d00; }
.log-error { color: #b30000; }
.muted { color: #999; }
"""


@cache
def _asset(name: str) -> str:
    if name == "viewer.js":
        return (_OBS_DIR / "viewer.js").read_text(encoding="utf-8")
    return (_OBS_DIR / "vendor" / "uplot" / name).read_text(encoding="utf-8")


def read_metrics(path: Path) -> tuple[dict, list[dict], int]:
    """Parse a metrics.jsonl into (header, records, skipped_line_count).

    The first record carrying ``report == "run_metrics"`` becomes the
    header; malformed or non-object lines are skipped and counted.
    """
    header: dict = {}
    records: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(value, dict):
            skipped += 1
            continue
        if value.get("report") == "run_metrics" and not header:
            header = value
        else:
            records.append(value)
    return header, records, skipped


def read_events(path: Path) -> tuple[list[dict], int]:
    """Parse an event file into (records, skipped_line_count)."""
    records: list[dict] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(value, dict):
            records.append(value)
        else:
            skipped += 1
    return records, skipped


def default_events_path(header: dict, log_dir: str | None = None) -> Path | None:
    """Locate the run's event file from its metrics header.

    ``${OBS_LOG_DIR}/<cli>/<run_id>.jsonl``, or None when the header lacks
    the run identity (a tracker constructed outside a dispatched CLI).
    """
    run_id = header.get("run_id")
    cli = header.get("cli")
    if not run_id or not cli:
        return None
    root = log_dir or os.environ.get("OBS_LOG_DIR") or DEFAULT_LOG_DIR
    return Path(root) / cli / f"{run_id}.jsonl"


def inline_json(value: object) -> str:
    """Serialize for embedding inside a <script> block.

    "</" is escaped to "<\\/" so a payload containing "</script>" cannot
    terminate the enclosing block.
    """
    return json.dumps(value).replace("</", "<\\/")


def _page(data_script: str, title: str) -> str:
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        _asset("uPlot.min.css"),
        _PAGE_CSS,
        "</style>",
        "</head>",
        "<body>",
        '<div id="root"></div>',
        "<script>",
        _asset("uPlot.iife.min.js"),
        "</script>",
        "<script>",
        _asset("viewer.js"),
        "</script>",
        data_script,
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def build_snapshot(header: dict, metrics: list[dict], events: list[dict]) -> str:
    """The complete self-contained run.html page."""
    data_script = (
        "<script>\n"
        'obsRender(document.getElementById("root"), {\n'
        f"  header: {inline_json(header or None)},\n"
        f"  metrics: {inline_json(metrics)},\n"
        f"  events: {inline_json(events)}\n"
        "});\n"
        "</script>"
    )
    title = str(header.get("run_id") or "run dashboard")
    return _page(data_script, title)
