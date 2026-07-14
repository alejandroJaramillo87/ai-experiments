"""Serving request log proxy (interp Phase 3 W3).

A single-file reverse proxy in front of one OpenAI-compatible service.
Forwards every request byte-for-byte (streaming and non-streaming) and appends
one JSONL record per completed completion request: timestamp, target service,
model id, request messages, response text, token counts, latency. llama-server
has no structured request logging and patching it would violate
adopt-over-build; this sidecar is the smallest durable log the W4 monitor can
replay.

A malformed or interrupted request logs a partial record (error field set)
without crashing the proxy or the underlying service. Records land at
LOG_DIR/<target-service>/YYYY-MM-DD.jsonl. Retention is a documented cleanup
policy (30 days default) enforced by the monitor job's preamble, not here.
"""

from __future__ import annotations

import datetime
import json
import os
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

TARGET_URL = os.environ.get("TARGET_URL", "http://llama-gpu:8002").rstrip("/")
TARGET_SERVICE = os.environ.get("TARGET_SERVICE", "llama-gpu")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8012"))
LOG_DIR = os.environ.get("LOG_DIR", "/logs")

LOGGED_PATHS = ("/v1/chat/completions", "/v1/completions")
# Hop-by-hop headers must not be forwarded verbatim
DROP_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "te",
    "upgrade",
    "proxy-authenticate",
    "proxy-authorization",
    "trailer",
}

app = FastAPI()
client = httpx.AsyncClient(base_url=TARGET_URL, timeout=None)


def append_record(record: dict) -> None:
    """Append one JSONL record; logging failure must never break the proxy."""
    try:
        day = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
        directory = os.path.join(LOG_DIR, TARGET_SERVICE)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{day}.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[logproxy] WARN: could not append log record: {exc}", flush=True)


def base_record(request_body: dict | None, latency_s: float | None) -> dict:
    record = {
        "timestamp": datetime.datetime.now(datetime.UTC).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        ),
        "target_service": TARGET_SERVICE,
        "model": None,
        "messages": None,
        "prompt": None,
        "response_text": None,
        "usage": None,
        "latency_s": latency_s,
        "error": None,
    }
    if isinstance(request_body, dict):
        record["model"] = request_body.get("model")
        record["messages"] = request_body.get("messages")
        record["prompt"] = request_body.get("prompt")
    return record


def extract_response(body: dict) -> tuple[str | None, dict | None]:
    """Pull response text and usage from a non-streaming completion body."""
    try:
        choice = body["choices"][0]
        text = choice.get("message", {}).get("content") or choice.get("text")
        return text, body.get("usage")
    except (KeyError, IndexError, TypeError):
        return None, None


def extract_streamed(chunks: list[bytes]) -> tuple[str | None, dict | None]:
    """Reassemble response text and usage from accumulated SSE chunks."""
    text_parts: list[str] = []
    usage = None
    try:
        for line in b"".join(chunks).split(b"\n"):
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            event = json.loads(payload)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                part = delta.get("content") or choice.get("text")
                if part:
                    text_parts.append(part)
        return ("".join(text_parts) or None), usage
    except (json.JSONDecodeError, KeyError, TypeError):
        return ("".join(text_parts) or None), usage


@app.api_route(
    "/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"]
)
async def proxy(request: Request, path: str):
    raw_body = await request.body()
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in DROP_HEADERS
    }
    url = f"/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    logged = request.method == "POST" and f"/{path}" in LOGGED_PATHS

    request_body = None
    wants_stream = False
    if logged:
        try:
            request_body = json.loads(raw_body)
            wants_stream = bool(request_body.get("stream"))
        except json.JSONDecodeError:
            request_body = None

    start = time.monotonic()
    upstream_request = client.build_request(
        request.method, url, headers=headers, content=raw_body
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        if logged:
            record = base_record(request_body, round(time.monotonic() - start, 3))
            record["error"] = f"upstream request failed: {exc}"
            append_record(record)
        return Response(
            content=json.dumps({"error": f"logproxy: upstream unreachable: {exc}"}),
            status_code=502,
            media_type="application/json",
        )

    response_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in DROP_HEADERS
    }

    if logged and wants_stream:

        async def relay():
            chunks: list[bytes] = []
            error = None
            try:
                async for chunk in upstream.aiter_raw():
                    chunks.append(chunk)
                    yield chunk
            except httpx.HTTPError as exc:
                error = f"stream interrupted: {exc}"
            finally:
                await upstream.aclose()
                record = base_record(request_body, round(time.monotonic() - start, 3))
                record["response_text"], record["usage"] = extract_streamed(chunks)
                record["status_code"] = upstream.status_code
                record["error"] = error
                append_record(record)

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
        )

    try:
        body = await upstream.aread()
    except httpx.HTTPError as exc:
        if logged:
            record = base_record(request_body, round(time.monotonic() - start, 3))
            record["error"] = f"read interrupted: {exc}"
            append_record(record)
        return Response(
            content=json.dumps({"error": f"logproxy: upstream read failed: {exc}"}),
            status_code=502,
            media_type="application/json",
        )
    finally:
        await upstream.aclose()

    if logged:
        record = base_record(request_body, round(time.monotonic() - start, 3))
        record["status_code"] = upstream.status_code
        try:
            record["response_text"], record["usage"] = extract_response(
                json.loads(body)
            )
        except json.JSONDecodeError:
            record["error"] = "response body is not JSON"
        append_record(record)

    return Response(
        content=body, status_code=upstream.status_code, headers=response_headers
    )


if __name__ == "__main__":
    print(
        f"[logproxy] forwarding 0.0.0.0:{LISTEN_PORT} -> {TARGET_URL} "
        f"(service {TARGET_SERVICE}), logging to {LOG_DIR}",
        flush=True,
    )
    uvicorn.run(app, host="0.0.0.0", port=LISTEN_PORT, log_level="warning")
