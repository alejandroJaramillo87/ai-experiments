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

# Per-request read bound: the client's offset loop fetches the rest, and an
# unbounded read of a huge first poll would hold the whole file in memory.
READ_CHUNK_LIMIT = 8 * 1024 * 1024


def read_new_lines(path: Path, offset: int) -> tuple[int, list[str]]:
    """Complete lines appended past a byte offset, with the new offset.

    A trailing partial line is held back until its newline arrives; a
    missing file or an offset at EOF returns (offset, []). A file now
    shorter than the offset was truncated or replaced: the offset resets
    to 0 so the next polls re-read from the start instead of stalling
    silently forever. Reads are capped per request; the client's next
    poll continues from the returned offset.
    """
    try:
        if path.stat().st_size < offset:
            return 0, []
        with open(path, "rb") as f:
            f.seek(offset)
            chunk = f.read(READ_CHUNK_LIMIT)
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
        is retried on the next poll and the located path is cached. A
        bounded prefix is scanned rather than only line 1: a torn or
        foreign first line must not permanently blank the log pane when
        the real header sits on a later line.
        """
        if self._events_path is not None:
            return self._events_path
        try:
            with open(self.metrics_path(), encoding="utf-8", errors="replace") as f:
                prefix = f.read(65536)
        except OSError:
            return None
        lines = prefix.splitlines(keepends=True)
        for line in lines[:50]:
            if not line.endswith("\n"):
                break  # incomplete tail: retried next poll
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("report") == "run_metrics":
                located = default_events_path(record)
                if located is not None:
                    self._events_path = located
                return located
        return None


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

    def do_HEAD(self) -> None:
        # liveness probes (curl -I) get the headers a GET / would carry
        srv = cast(DashboardServer, self.server)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(srv.shell.encode("utf-8"))))
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _respond(self, status: int, content_type: str, body: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # a reloaded tab aborts its in-flight poll; without this the
            # server prints a full traceback per abort, defeating the
            # deliberate request-log suppression below
            pass

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
    try:
        server = DashboardServer((host, port), run_dir, events_path, shell)
    except OSError as exc:
        out.error(f"Cannot bind {host}:{port}: {exc}")
        out.line("  Solution: pass a free port with --port (0 picks one)")
        return 1
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
