"""Measure the benchmark client's own timing resolution.

BQ.5. Every latency this repository publishes is a difference between two stamps
taken in `bench/client.py`, and that stamp is taken after the reading thread wins the
global interpreter lock. At concurrency 16 there are sixteen threads decoding sockets
and parsing JSON, so the client is an instrument with an error of its own and nothing
has measured it.

This is the principle the measurement campaign applies to the workstation, turned on
the tool: a difference smaller than the instrument's resolution is not a small result,
it is not a result. The output is a number, not a verdict.

The scripted server sends frames a known interval apart, so the error is the observed
interval minus the injected one. What the run cannot separate is the client's error
from the server's own scheduling jitter, which is why the single-stream figure is
reported alongside: it is the floor the concurrent figures should be read against.

    python3 tests/bench/resolution.py [--concurrency 1,4,16] [--json]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import pathlib
import statistics
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench import adapters, client  # noqa: E402
from common.output import Printer  # noqa: E402
from common.serving import SERVICES  # noqa: E402
from tests.bench import fakes  # noqa: E402


def measure(level: int, tokens: int, gap_s: float) -> dict:
    """Run `level` streams at once against one scripted server, and reduce the error."""
    frames = [fakes.Frame.of(fakes.text_chunk(f" t{i}"), gap_s) for i in range(tokens)]
    frames.append(
        fakes.Frame.of(
            fakes.usage_frame(prompt_tokens=8, completion_tokens=tokens), gap_s
        )
    )

    with fakes.ScriptedServer(frames) as base_url:
        port = int(base_url.rsplit(":", 1)[1])
        spec = dataclasses.replace(SERVICES["llama-cpu"], port=port)
        adapter = adapters.LlamaCppAdapter(spec)

        def one() -> list[float]:
            result = client.stream(adapter, {"prompt": "x"}, timeout=30.0)
            arrivals = result.arrivals
            return [(b - a) for a, b in zip(arrivals, arrivals[1:], strict=False)]

        if level == 1:
            series = [one()]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                series = [f.result() for f in [pool.submit(one) for _ in range(level)]]

    errors_ms = [
        abs(interval - gap_s) * 1000.0 for stream in series for interval in stream
    ]
    return {
        "concurrency": level,
        "intervals": len(errors_ms),
        "median_error_ms": statistics.median(errors_ms),
        "p95_error_ms": sorted(errors_ms)[max(0, int(0.95 * len(errors_ms)) - 1)],
        "max_error_ms": max(errors_ms),
        "injected_gap_ms": gap_s * 1000.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the bench client's stamp error under concurrency."
    )
    parser.add_argument("--concurrency", default="1,4,16")
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--gap-ms", type=float, default=20.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        levels = [int(part) for part in args.concurrency.split(",") if part]
    except ValueError:
        print(
            "[ERROR] concurrency: expected a comma-separated list of integers",
            file=sys.stderr,
        )
        return 2
    if not levels or any(level < 1 for level in levels):
        print(
            "[ERROR] concurrency: every level must be one or greater", file=sys.stderr
        )
        return 2

    out = Printer(quiet=args.json)
    out.line("=== bench client timing resolution ===")
    out.line()
    out.info(f"Injected gap: {args.gap_ms:.1f} ms over {args.tokens} tokens per stream")

    rows = []
    for level in levels:
        out.run(f"Concurrency {level}: streaming")
        row = measure(level, args.tokens, args.gap_ms / 1000.0)
        rows.append(row)
        out.ok(
            f"Concurrency {level}: median {row['median_error_ms']:.2f} ms, "
            f"p95 {row['p95_error_ms']:.2f} ms, max {row['max_error_ms']:.2f} ms "
            f"over {row['intervals']} intervals",
            indent=2,
        )

    baseline = rows[0]["median_error_ms"]
    worst = max(row["median_error_ms"] for row in rows)
    out.line()
    if worst > args.gap_ms * 0.25:
        out.warn(
            f"Stamp error reaches {worst:.2f} ms against a {args.gap_ms:.1f} ms "
            "interval: the client is a material part of what it measures"
        )
    else:
        out.info(
            f"Stamp error stays under a quarter of the interval "
            f"(single-stream median {baseline:.2f} ms)"
        )

    if args.json:
        print(json.dumps({"levels": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
