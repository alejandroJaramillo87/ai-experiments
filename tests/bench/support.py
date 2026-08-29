"""Builders for synthetic bench_result documents."""

from __future__ import annotations

from bench import report


def stat(median: float) -> dict:
    return {
        "median": median,
        "p95": median,
        "p99": median,
        "min": median,
        "max": median,
        "stdev": 0.0,
        "mean": median,
        "n": 5,
        "p99_meaningful": False,
    }


def result(
    *,
    arm: str = "base",
    ttft_ms: float = 100.0,
    decode_tok_s: float = 35.0,
    service: str = "llama-cpu",
    architecture: str = "moe",
    quantization: str = "IQ4_XS",
    prompt_source: str = "natural",
    profile_sha256: str = "abc123",
    cpuset: str = "8-15",
    git_sha: str = "deadbeef",
) -> dict:
    document = report.build(
        run={
            "run_id": f"run-{arm}",
            "profile": "quick",
            "profile_sha256": profile_sha256,
            "arm": arm,
            "order": "interleaved",
            "repeats": 5,
            "warmup_discarded": 1,
            "seed": 42,
        },
        conditions={
            "service": service,
            "engine": "llama.cpp",
            "port": 8001,
            "model": {
                "name": "m",
                "architecture": architecture,
                "quantization": quantization,
                "context_length": 32768,
            },
            "prompt_source": prompt_source,
            "host": {"client_cpuset": cpuset, "service_cpuset": "0-15"},
        },
        phases={
            "prefill": {
                "timings_source": "server",
                "points": [
                    {
                        "point": "prefill/p128/g1/c1",
                        "prompt_tokens": 128,
                        "max_tokens": 1,
                        "concurrency": 1,
                        "metrics": {"ttft_ms": stat(ttft_ms)},
                    }
                ],
                "slope": {
                    "prefill_tok_s_slope": 2500.0,
                    "intercept_ms": 12.0,
                    "r2": 0.99,
                },
            },
            "decode": {
                "timings_source": "client",
                "points": [
                    {
                        "point": "decode/p128/g256/c1",
                        "prompt_tokens": 128,
                        "max_tokens": 256,
                        "concurrency": 1,
                        "metrics": {"decode_tok_s": stat(decode_tok_s)},
                    }
                ],
            },
        },
    )
    document["git_sha"] = git_sha
    return document
