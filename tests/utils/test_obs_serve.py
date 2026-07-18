"""Unit tests for the live dashboard server (roadmap OBS-12)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from common.obs import serve
from common.output import Printer

SHELL = "<html>live-shell</html>"


@pytest.fixture()
def server(tmp_path):
    srv = serve.DashboardServer(("127.0.0.1", 0), tmp_path, None, SHELL)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def get(srv, path):
    port = srv.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as resp:
        return resp.status, resp.headers.get("Content-Type"), resp.read()


def get_json(srv, path):
    status, _, body = get(srv, path)
    assert status == 200
    return json.loads(body)


# -- read_new_lines -----------------------------------------------------------


def test_read_new_lines_returns_only_complete_lines(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text("a\nb\npartial")
    offset, lines = serve.read_new_lines(path, 0)
    assert lines == ["a", "b"]
    assert offset == 4
    path.write_text("a\nb\npartial\n")
    offset, lines = serve.read_new_lines(path, offset)
    assert lines == ["partial"]
    assert offset == 12


def test_read_new_lines_chains_offsets_without_gaps(tmp_path):
    path = tmp_path / "f.jsonl"
    offset = 0
    collected = []
    for chunk in ("one\n", "two\nthree\n", "four\n"):
        with open(path, "a") as f:
            f.write(chunk)
        offset, lines = serve.read_new_lines(path, offset)
        collected.extend(lines)
    assert collected == ["one", "two", "three", "four"]
    assert offset == path.stat().st_size


def test_read_new_lines_missing_file(tmp_path):
    assert serve.read_new_lines(tmp_path / "absent.jsonl", 5) == (5, [])


def test_read_new_lines_offset_at_eof(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text("a\n")
    assert serve.read_new_lines(path, 2) == (2, [])


# -- endpoints ----------------------------------------------------------------


def test_metrics_endpoint_growing_file(tmp_path, server):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"step": 1}\n')
    payload = get_json(server, "/data/metrics?offset=0")
    assert payload["lines"] == ['{"step": 1}']
    with open(metrics, "a") as f:
        f.write('{"step": 2}\n{"step": 3}\n{"par')
    payload = get_json(server, f"/data/metrics?offset={payload['offset']}")
    assert payload["lines"] == ['{"step": 2}', '{"step": 3}']
    # the held-back partial line arrives once completed
    with open(metrics, "a") as f:
        f.write('tial": 4}\n')
    payload = get_json(server, f"/data/metrics?offset={payload['offset']}")
    assert payload["lines"] == ['{"partial": 4}']


def test_events_endpoint_before_header_returns_empty(tmp_path, server):
    assert get_json(server, "/data/events?offset=0") == {"offset": 0, "lines": []}


def test_events_endpoint_resolves_from_header_lazily(tmp_path, server, monkeypatch):
    monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path / "logs"))
    events = tmp_path / "logs" / "training" / "R-9.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text('{"event": "cmd.start"}\n')
    # header written after server start: the first miss retries next poll
    header = {"report": "run_metrics", "run_id": "R-9", "cli": "training"}
    (tmp_path / "metrics.jsonl").write_text(json.dumps(header) + "\n")
    payload = get_json(server, "/data/events?offset=0")
    assert payload["lines"] == ['{"event": "cmd.start"}']
    assert server._events_path == events  # cached once located


def test_index_serves_shell(server, tmp_path):
    status, content_type, body = get(server, "/")
    assert status == 200
    assert content_type == "text/html; charset=utf-8"
    assert body.decode() == SHELL


def test_unknown_path_404(server, tmp_path):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server, "/other")
    assert exc.value.code == 404


def test_bad_offset_treated_as_zero(tmp_path, server):
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\n')
    for query in ("offset=junk", "offset=-3", ""):
        payload = get_json(server, f"/data/metrics?{query}")
        assert payload["lines"] == ['{"step": 1}']


def test_run_returns_zero_on_interrupt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        serve.DashboardServer,
        "serve_forever",
        lambda self: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    rc = serve.run("127.0.0.1", 0, tmp_path, None, SHELL, Printer())
    assert rc == 0
    assert "Live dashboard" in capsys.readouterr().err


def test_truncated_file_resets_offset_instead_of_stalling(tmp_path):
    path = tmp_path / "f.jsonl"
    path.write_text("one\ntwo\n")
    offset, lines = serve.read_new_lines(path, 0)
    assert lines == ["one", "two"]
    # the run directory is reused: a shorter replacement must reset the
    # client to the start, not return empty forever
    path.write_text("new\n")
    offset, lines = serve.read_new_lines(path, offset)
    assert (offset, lines) == (0, [])
    offset, lines = serve.read_new_lines(path, offset)
    assert lines == ["new"]
    assert offset == 4


def test_read_is_capped_per_request(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "READ_CHUNK_LIMIT", 8)
    path = tmp_path / "f.jsonl"
    path.write_text("aaa\nbbb\nccc\n")
    offset, lines = serve.read_new_lines(path, 0)
    assert lines == ["aaa", "bbb"]  # cut at the last newline within the cap
    offset, lines = serve.read_new_lines(path, offset)
    assert lines == ["ccc"]
    assert offset == 12


def test_events_resolve_when_header_is_not_line_one(tmp_path, server, monkeypatch):
    monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path / "logs"))
    events = tmp_path / "logs" / "training" / "R-8.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text('{"event": "cmd.start"}\n')
    # a torn first line (the pre-lock sink could write one) must not
    # permanently blank the log pane; the scan finds the later header
    header = {"report": "run_metrics", "run_id": "R-8", "cli": "training"}
    (tmp_path / "metrics.jsonl").write_text(
        "torn-not-json\n" + json.dumps({"step": 1}) + "\n" + json.dumps(header) + "\n"
    )
    payload = get_json(server, "/data/events?offset=0")
    assert payload["lines"] == ['{"event": "cmd.start"}']


def test_head_request_serves_headers_only(server, tmp_path):
    port = server.server_address[1]
    req = urllib.request.Request(f"http://127.0.0.1:{port}/", method="HEAD")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        assert resp.headers["Content-Length"] == str(len(SHELL.encode()))
        assert resp.read() == b""


def test_bind_failure_returns_styled_error(tmp_path, server, capsys):
    from common.output import Printer

    taken = server.server_address[1]
    rc = serve.run("127.0.0.1", taken, tmp_path, None, SHELL, Printer())
    assert rc == 1
    err = capsys.readouterr().err
    assert "Cannot bind" in err
    assert "--port" in err
