"""Live dashboard server (roadmap OBS-12).

A stateless viewer over the run's JSON Lines files: GET /data/* returns
the complete lines appended past a client-owned byte offset, the page
extends its charts and log tail, and killing the server loses nothing —
the files are the record.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

from common.obs.dashboard import default_events_path
from common.output import Printer


def read_new_lines(path: Path, offset: int) -> tuple[int, list[str]]:
    """Complete lines appended past a byte offset, with the new offset.

    A trailing partial line is held back until its newline arrives; a
    missing file or an offset at or past EOF returns (offset, []).
    """
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read()
    except OSError:
        return offset, []
    end = chunk.rfind(b"\n")
    if end < 0:
        return offset, []
    complete = chunk[: end + 1]
    return offset + len(complete), complete.decode("utf-8", "replace").splitlines()


def _parse_offset(query: str) -> int:
    values = parse_qs(query).get("offset", [])
    try:
        offset = int(values[0]) if values else 0
    except ValueError:
        offset = 0
    return max(offset, 0)


class DashboardServer(ThreadingHTTPServer):
    """Serves the live page shell and the two offset endpoints."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        run_dir: Path,
        events_path: Path | None,
        shell: str,
    ) -> None:
        super().__init__(address, _Handler)
        self.run_dir = run_dir
        self.shell = shell
        self._events_path = events_path

    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.jsonl"

    def resolve_events_path(self) -> Path | None:
        """The --events override, else lazily from the metrics header.

        The server may start before the run writes its header, so a miss
        is retried on the next poll and the located path is cached. Only
        the first line is read; the sink writes the header first.
        """
        if self._events_path is not None:
            return self._events_path
        try:
            with open(self.metrics_path(), encoding="utf-8") as f:
                first = f.readline()
        except OSError:
            return None
        if not first.endswith("\n"):
            return None
        try:
            record = json.loads(first)
        except json.JSONDecodeError:
            return None
        if not isinstance(record, dict) or record.get("report") != "run_metrics":
            return None
        located = default_events_path(record)
        if located is not None:
            self._events_path = located
        return located


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        srv = cast(DashboardServer, self.server)
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._respond(200, "text/html; charset=utf-8", srv.shell.encode("utf-8"))
        elif parsed.path == "/data/metrics":
            self._data(srv.metrics_path(), parsed.query)
        elif parsed.path == "/data/events":
            self._data(srv.resolve_events_path(), parsed.query)
        else:
            self._respond(404, "text/plain; charset=utf-8", b"not found\n")

    def _data(self, path: Path | None, query: str) -> None:
        offset = _parse_offset(query)
        lines: list[str] = []
        new_offset = offset
        if path is not None:
            new_offset, lines = read_new_lines(path, offset)
        body = json.dumps({"offset": new_offset, "lines": lines}).encode("utf-8")
        self._respond(200, "application/json", body)

    def _respond(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # a 2-second poll would otherwise print two lines per cycle


def run(
    host: str,
    port: int,
    run_dir: Path,
    events_path: Path | None,
    shell: str,
    out: Printer,
) -> int:
    """Serve until interrupted; Ctrl-C is a clean exit, nothing is lost."""
    server = DashboardServer((host, port), run_dir, events_path, shell)
    try:
        bound_port = server.server_address[1]
        out.info(f"Live dashboard: http://{host}:{bound_port}/")
        out.line("  Stop: Ctrl-C (the files are the record)")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
