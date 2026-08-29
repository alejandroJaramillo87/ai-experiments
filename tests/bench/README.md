# tests/bench

Unit coverage for `bench/`. Everything here runs in the sandbox: no GPU, no container,
no running service, no torch. A stub stream function stands in for the engine, so the
whole path — cells, warm-up discard, reduction, envelope, file layout — is exercised
without the workstation.

No test here carries a hardware marker, so all of it is in the default selection
(`pyproject.toml` deselects only `hw_smoke`, `hw_full` and `hw_e4b`).

| File | Covers |
|------|--------|
| `test_metrics.py` | statistics against hand-computed fixtures; TTFT, ITL, TPOT and decode rate from synthetic arrival lists; the `n=1` and `n=2` edges; the prefill slope on a clean line and on one with an outlier |
| `test_adapters.py` | per-engine request bodies and wire-format decoding, including that only text-bearing chunks count as token arrivals and that token counts come from `usage` |
| `test_workloads.py` | profile parsing, per-service overrides, and that repeats are round-robin over points rather than blocked |
| `test_registry.py` | the registry against `schemas/bench_models.json`, including the quantization cross-check and the MoE field requirement |
| `test_runner.py` | the run loop end to end against a fake engine, and that the assembled report validates against the shared envelope |
| `test_delta.py` | the five verdicts, the noise floor, and the refusals that stop an incomparable pair |
| `test_cli.py` | the parser surface, arm parsing, and that `--json` puts nothing but JSON on stdout |
| `support.py` | builders for synthetic `bench_result` documents |

What cannot be proven here is any actual number. Those come from the workstation through
the `verify-host` handshake.
