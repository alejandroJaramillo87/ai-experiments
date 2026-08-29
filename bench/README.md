# bench

Phase-separated inference benchmarks for the three serving services that stream. Entry point is
`python3 -m bench`; the protocol it implements is
[.claude/skills/benchmarking/SKILL.md](../.claude/skills/benchmarking/SKILL.md).

## Why this exists

The protocol predates the tooling. The skill has always required a discarded warm-up,
repeated samples, median and spread rather than a best-of, matched quantization, and a
5% regression threshold "outside the run-to-run spread". What implemented it was a
vendored copy of upstream llama.cpp's load generator plus two shell wrappers, which
performed a single un-warmed run and reported arithmetic means to a text file. Every
rule was enforced by the operator remembering it.

Three consequences shaped this package:

- **Prefill and decode were conflated.** The old generator captured each request's
  per-token arrival array and then pooled every request's arrivals into one flat list,
  so the deltas were never differenced. A single tokens/second figure moves when either
  phase moves and identifies neither.
- **Results carried no conditions.** The only persisted field was a total runtime, so
  `docs/optimizations/README.md`'s 26.2% figure cannot be reproduced: it was measured at
  `THREADS=12` against a service that now runs 16, on a mixture-of-experts model, and
  nothing in the artifact says so.
- **Architecture was unrecorded.** Nothing in the tree stated which served model is
  dense and which is MoE, which is how an MoE measurement came to be published as the
  CPU track's headline number.

## Consumes

- `common/serving.py` — the service registry: ports, health paths, concurrency, compose
  recreate. Nothing here re-derives a port, and the health path is the registry's, not a
  guess. (The retired wrappers guessed, and could not reach `vllm-gpu` at all.)
- `common/reports.py` — the shared JSON envelope, which
  `docs/observability/report_schema.md` requires of every emitter.
- `common/output.py`, `common/provenance.py`, `common/obs/` — status output, `git_sha`,
  traced dispatch and the run catalog.
- `schemas/bench_models.json` — architecture facts, stamped into every report.
- `scripts/benchmark/datasets/wikitext-2-raw/` — the corpus `prompts/natural.txt` is
  sliced from.

Stdlib only, like `common/serving.py`: it runs on the host without resolving the uv
environment.

## Consumed by

Nothing imports `bench`. It is a host-side CLI whose output is read by a human, by
`bench compare`, and by whatever renders a result into a blog post.

## Entry points

```
python3 -m bench run     --service <name> --model <slug> [--profile quick|publish] [--arm ...]
python3 -m bench noise   --service <name> --model <slug>
python3 -m bench compare <baseline.json> <candidate.json> --noise <noise.json>
python3 -m bench models  [--json]
```

## Layout

| Module | Role |
|--------|------|
| `cli.py`, `__main__.py` | argparse surface over `common.obs.dispatch` |
| `adapters.py` | per-engine request shape: exact length, cache defeat, server timings |
| `client.py` | streaming SSE, and the one place a wall-clock stamp is taken |
| `metrics.py` | arrival times to reported statistics; pure, no I/O |
| `workloads.py` | profile TOML to an ordered cell list |
| `prompting.py`, `prompts/` | length-controlled real-text prompts |
| `registry.py` | `schemas/bench_models.json`, validated |
| `conditions.py` | host and container facts, read from the running system |
| `runner.py` | health gate, warm-up discard, execution, reduction, assembly |
| `report.py` | envelope assembly, write, read |
| `delta.py` | verdicts against the 5% bar and the measured floor |
| `profiles/` | `quick` (~8 min) and `publish` (~45-60 min, all four services) |

## Traps

- **The suite has no make wrapper, deliberately.** A benchmark is a serialized GPU
  consumer (AGENTS.md: one at a time across every worktree) and a Make target would make
  it look like an ordinary one.
- **Pin the client off the service's cores.** `llama-cpu` holds `cpuset 0-15`. The
  measurement protocol in
  `docs/optimizations/ideation/rust-harness-optimization-candidates.md` says
  `taskset -c 0-7`, which is inside that range: doing so makes client and server contend
  and taints the measurement it was meant to stabilise. Use `8-15`, or narrow the
  service. Both cpusets are recorded and `compare` refuses if they differ between arms.
- **A delta without a measured noise floor is not a result.** Run `bench noise` first.
  Four consecutive runs of one unmodified binary on this workstation have been recorded
  spreading 44%.
- **`e2e_tok_s` is not a decode rate.** It includes prefill and exists only so figures
  recorded by the earlier tooling remain comparable.
- **A model absent from the registry cannot be benchmarked.** That is the point: a
  result whose architecture is unrecorded is how the current confusion arose.
- **The rust harness is not a target, deliberately.** Its optimization arms are
  properties of the runtime's internals, and `make ladder SIZE=quick MODEL=<dir>` already
  exposes them as first-class flags (`--dot`, `--batch`, `--threads`, `--single-thread`)
  measured in-process against the real checkpoint, writing `rung_duration_s` into the
  same `metrics.jsonl` this repository's dashboard already charts. Going over HTTP would
  add a serialization layer and a mutex on top of the thing being measured, and
  `schemas/harness_chat_endpoint.json` records that the endpoint refuses `stream: true`,
  so it could not report a prefill/decode split either. Asking for it by name returns an
  error that says so rather than a bare KeyError.
