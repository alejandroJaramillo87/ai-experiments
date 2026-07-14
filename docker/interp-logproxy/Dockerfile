# Serving request log proxy (interp Phase 3 W3)
#
# Single-file FastAPI reverse proxy; see logproxy.py for the design. Deliberately tiny:
# slim Python base, three pinned wheels, non-root, read-only rootfs (compose). Pins match
# the repository's core fastapi/uvicorn pins where they exist.

FROM python:3.12-slim

RUN pip install --no-cache-dir \
    fastapi==0.139.0 \
    uvicorn==0.51.0 \
    httpx==0.28.1

RUN useradd --create-home --uid 1000 appuser 2>/dev/null || true
COPY --chmod=644 docker/interp-logproxy/logproxy.py /app/logproxy.py

USER 1000
WORKDIR /app
EXPOSE 8012

CMD ["python", "/app/logproxy.py"]
