"""Run catalog: one summary line per completed run (roadmap OBS-9).

The dispatch wrapper appends a record to ${OBS_LOG_DIR}/<cli>/runs.jsonl
when a command ends, and the runs subcommand renders the file. Append and
read are both best-effort: a machine without the log root records
nothing, and a malformed line is skipped rather than fatal.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from common.obs.core import DEFAULT_LOG_DIR, current_run_dir
from common.output import Printer


def catalog_path(cli: str) -> Path:
    """Resolve the per-CLI catalog file the same way configure resolves it."""
    root = Path(os.environ.get("OBS_LOG_DIR") or DEFAULT_LOG_DIR)
    return root / cli / "runs.jsonl"


def record_run(
    cli: str, *, run_id: str, command: str, exit_code: int, duration_s: float
) -> None:
    """Append one summary line for a completed run; never raises.

    A single write on an append-mode handle (O_APPEND), so concurrent
    commands from both CLIs interleave whole lines. No directory is
    created here: on a machine where configure degraded to console-only,
    the catalog skips the same way.
    """
    record = {
        "run_id": run_id,
        "command": command,
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "duration_s": duration_s,
        "exit_code": exit_code,
        "run_dir": current_run_dir(),
    }
    try:
        with open(catalog_path(cli), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=repr) + "\n")
    except OSError:
        return


def read_runs(path: Path) -> list[dict[str, Any]]:
    """Parse the catalog, silently skipping malformed lines."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def run_command(cli: str, args: argparse.Namespace) -> int:
    """Shared handler behind the training and interp runs subcommands."""
    out = Printer(quiet=args.json)
    records = read_runs(catalog_path(cli))
    if not args.all:
        # the catalog records every invocation, including runs itself;
        # hide the browsing entries unless asked
        records = [r for r in records if r.get("command") != "runs"]
    if args.failed:
        # a row missing exit_code has an unknown outcome, not a failure
        records = [
            r
            for r in records
            if isinstance(r.get("exit_code"), int) and r["exit_code"] != 0
        ]
    if args.limit > 0:
        records = records[-args.limit :]
    if args.json:
        print(json.dumps(records, indent=2))
        return 0
    if not records:
        out.info("No recorded runs")
        return 0
    command_width = max(7, *(len(str(r.get("command", ""))) for r in records))
    out.line(
        f"{'FINISHED':<19}  {'RUN ID':<24}  "
        f"{'COMMAND':<{command_width}}  {'EXIT':>4}  {'DURATION':>10}"
    )
    for r in records:
        # seconds precision is enough for a table; the record keeps the rest
        finished = str(r.get("timestamp", ""))[:19]
        try:
            duration = f"{float(r.get('duration_s', 0) or 0):.3f}s"
        except (TypeError, ValueError):
            # foreign or hand-edited row: keep the row, mark the field
            duration = "?"
        out.line(
            f"{finished:<19}  {str(r.get('run_id', '')):<24}  "
            f"{str(r.get('command', '')):<{command_width}}  "
            f"{r.get('exit_code', '')!s:>4}  {duration:>10}"
        )
    return 0
