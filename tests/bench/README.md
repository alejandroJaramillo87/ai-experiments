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
| `test_workloads.py` | profile parsing, per-service overrides, that repeats are round-robin over points rather than blocked, and the prompt slicing |
| `test_registry.py` | the registry's required fields, the quantization cross-check against the model filename, and the MoE field requirement |
| `test_runner.py` | the run loop end to end against a fake engine, and that the assembled report validates against the shared envelope |
| `test_delta.py` | the five verdicts, the noise floor, and the refusals that stop an incomparable pair |
| `test_cli.py` | the parser surface, arm parsing, and that `--json` puts nothing but JSON on stdout |
| `test_client_timing.py` | that a gap between frames becomes a gap of the same size in the latency series, over both a scripted response and a real socket; the HTTP and connection error branches; and the withheld-frame reproductions recorded as BQ-F01 |
| `test_conditions.py` | every path where a host fact is unavailable, which must yield an absent field rather than an exception: `collect` runs after all measurement and outside the caller's handler |
| `test_compare.py` | the compare command end to end: that an empty or one-sided comparison is not a clean bill, that the exit code follows the graded set rather than what the table displays, that a noise floor has to be one, and that an unreadable input is a diagnosed error rather than a traceback |
| `test_report.py` | the write-and-read round trip, and the two failure modes of `read` that no caller handles |
| `test_properties.py` | properties over seeded generated input: the ordering the statistics field names imply, that `classify` is total, that a result compared with itself is never a regression, and that a fit recovers the line it was given |
| `fakes.py` | a scripted response carrying its own delays, and a real server on an ephemeral port for what a substituted response object cannot reach |
| `fixtures/` | wire shapes derived from the pinned engine source, with `provenance.json` naming the serializer behind each |
| `mutate.py` | the mutation harness for `bench/metrics.py` and `bench/delta.py`; not a pytest module, and there is no pass mark |
| `resolution.py` | measures the client's own stamp error under concurrency; not a pytest module, and the number is the deliverable |
| `support.py` | builders for synthetic `bench_result` documents |

The fixtures under `fixtures/` are **source-derived, not captured**. They were written
from the engine versions the Dockerfiles pin, because reading the source shows every
branch a response can take while a capture shows only the one a given request took. A
capture at [bench.md](../../docs/verification/bench/bench.md) entry B1 confirms them
against a live service. Until then `provenance.json` records exactly which inference
each fixture rests on, so a wrong one is traceable rather than mysterious.

What cannot be proven here is any actual number. Those come from the workstation through
the `verify-host` handshake. The campaign that produced the modules above, and the
findings still open against `bench/`, are recorded in
[bench-qa.md](../../docs/verification/bench/bench-qa.md).
