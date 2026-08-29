# Benchmark suite verification runbook

**Entries: 0 verified, 0 blocked, 9 pending, of 9.**

Manual verification campaign for `python -m bench` on the workstation (RTX 5090, AMD
Ryzen 9950X). The suite passed 158 unit tests in the sandbox against synthetic fixtures
and a stub engine; none of them proves a number, because none of them touched a service.
This runbook is the campaign that closes that gap.

Every entry follows [../../../.claude/skills/verify-host/SKILL.md](../../../.claude/skills/verify-host/SKILL.md):
the acceptance criterion is stated before the command is run, and the pasted output is
the evidence. The protocol the commands implement is
[../../../.claude/skills/benchmarking/SKILL.md](../../../.claude/skills/benchmarking/SKILL.md).

## Order matters, and why

Run these in order. Three of them are preconditions for the rest rather than
measurements in their own right:

- **B1** proves the CLI reaches a service at all. Everything after it assumes so.
- **B2** is the noise floor. Until it exists, every delta in B6 through B9 is
  uninterpretable — the 5% regression threshold has no meaning underneath an unmeasured
  spread, and this workstation has been recorded spreading 44% across four consecutive
  runs of one unmodified binary
  ([../../optimizations/ideation/rust-harness-optimization-candidates.md](../../optimizations/ideation/rust-harness-optimization-candidates.md)).
- **B3** is the registry. `vllm-gpu` and `harness` have no registered model yet, so B5
  and B8 cannot run until someone adds the entries.

**One GPU consumer at a time** (AGENTS.md). These entries serialize against each other,
against training, and against `make quality-full`.

## Before any entry

```bash
make status
make health
```

A benchmark against a cold or unhealthy service is not a slow result, it is no result.
`bench run` gates on health itself, but confirm it first so a failure is attributable.

---

## B1 The CLI produces a schema-valid result against a real service

Implementation: `bench/cli.py`, `bench/runner.py`, `bench/client.py`, `bench/report.py`.

llama-gpu first, deliberately: it is the fastest target and the one that reports
server-side timings, so a wrong number is obvious rather than merely plausible.

```bash
python3 -m bench run --service llama-gpu --model <slug> --profile quick --json \
    > /tmp/b1.json
```

Expected signal: exit 0. Nothing but JSON on stdout (`.styles/PYTHON_OUTPUT_STYLE.md`
reserves it for machine output; the `[RUN]`/`[OK]` lines belong on stderr).
`report == "bench_result"`, `schema_version == 1`, and a non-`unknown` `git_sha`.
`phases.prefill.timings_source == "server"` — llama.cpp returns a `timings` object, so a
`client` value here means the field is not being parsed.
`phases.prefill.slope.r2 >= 0.95` on a quiet machine. Every `points[].metrics.ttft_ms.n`
equals the profile's `repeats` (3 for `quick`), which is what proves the warm-up was
discarded rather than counted.

```bash
python3 -c "
import json; d=json.load(open('/tmp/b1.json'))
print(d['report'], d['schema_version'], d['git_sha'])
print('timings_source', d['phases']['prefill']['timings_source'])
print('slope', d['phases']['prefill']['slope'])
print([(p['point'], p['metrics']['ttft_ms']['n']) for p in d['phases']['prefill']['points']])
"
```

Observed:

```
```

Status: PENDING

---

## B2 The noise floor for llama-gpu, and for llama-cpu

Implementation: `bench/commands/noise.py`, `bench/delta.py`.

```bash
python3 -m bench noise --service llama-gpu --model <slug> --profile quick
taskset -c 8-15 python3 -m bench noise --service llama-cpu --model <slug> --profile quick
```

**The `taskset` on the CPU arm is not optional and is not the value the rust-harness
document prescribes.** That document says `taskset -c 0-7`, but `llama-cpu` holds
`cpuset: "0-15"` in `docker-compose.inference.yaml`, so pinning the client into `0-7`
puts it on cores the server is using and taints the measurement it was meant to
stabilise. `8-15` is also inside the service's set on this box; if the floor comes back
wide, the next thing to try is narrowing the service to `0-11` for the run and pinning
the client to `12-15`.

Expected signal: exit 0, a `bench_noise` report written, and `worst_pct` reported. There
is no pass/fail threshold here — the floor is whatever it is, and its value is the
deliverable. What to do with it:

- `worst_pct < 5` — the 5% regression threshold is usable as written.
- `worst_pct >= 5` — the tool warns, and it is right to. A 5% delta on this workload is
  not resolvable; either raise `repeats`, fix the pinning, or quote deltas against the
  floor rather than against 5%.

Record both numbers here. Every later entry's verdicts depend on them, and a floor is
specific to a service and a profile.

Observed:

```
```

Status: PENDING

---

## B3 The registry covers every service to be benchmarked

Implementation: `bench/registry.py`, `schemas/bench_models.json`.

`schemas/bench_models.json` ships one entry, registered for `llama-cpu` and `llama-gpu`
only. `vllm-gpu` serves a Hugging Face checkpoint rather than a GGUF, and the harness
serves its own; neither has an entry, so `bench run --service vllm-gpu` fails at
`resolve()` with "registered for [...], not 'vllm-gpu'". That refusal is correct
behaviour — a result whose architecture is unrecorded is what this suite exists to stop —
but it blocks B5 and B8 until the entries exist.

Add one entry per served model, then:

```bash
python3 -m bench models
python3 -m bench models --json | python3 -c "import json,sys; json.load(sys.stdin)"
```

Expected signal: exit 0, and every service in `common/serving.py`'s `SERVICES` that you
intend to benchmark appears in at least one entry's `engines`. The loader validates on
read, so a `quantization` that disagrees with the model filename, an unknown
`architecture`, or an MoE entry missing `params_active_b` fails here rather than in a
published number.

Observed:

```
```

Status: PENDING

---

## B4 Prefill and decode are measurably different quantities

Implementation: `bench/metrics.py`, `bench/workloads.py`.

The claim this suite exists to support is that a single tokens/second figure hides two
phases. This entry is where that stops being an argument and becomes a measurement.

```bash
taskset -c 8-15 python3 -m bench run --service llama-cpu --model <slug> \
    --profile publish --json > /tmp/b4.json
```

Expected signal: on the same run, `decode_tok_s.median` and `e2e_tok_s.median` differ by
more than the B2 floor, and `decode_tok_s > e2e_tok_s` — end-to-end charges prefill to
the denominator, so it must sit below the decode rate it is often quoted as. TTFT rises
monotonically with prompt length across the sweep. `slope.intercept_ms > 0`: that
intercept is the fixed per-request cost, and it is the number that makes a single-point
prefill estimate wrong.

A large intercept relative to the smallest point's TTFT is itself a finding worth
recording — it says how badly the old single-point method would have misreported prefill
on this service.

Observed:

```
```

Status: PENDING

---

## B5 Concurrency is real, and vLLM behaves like a batching engine

Implementation: `bench/runner.py::_dispatch`.

Blocked on B3 (no registered vLLM model).

The runner originally ran a concurrency-16 point as sixteen serial requests and labelled
it 16. `tests/bench/test_runner.py::TestConcurrency` now asserts the requests overlap in
time against a stub, but only a real service shows the scheduler behaviour.

```bash
python3 -m bench run --service vllm-gpu --model <slug> --profile publish --json \
    > /tmp/b5.json
```

Expected signal: across the `[1, 4, 16]` sweep, aggregate throughput
(`decode_tok_s.median * concurrency`) rises with concurrency while per-request
`decode_tok_s.median` falls and `ttft_ms.median` rises. That shape is continuous
batching working. If per-request throughput is flat across the sweep, the requests are
not actually contending and the concurrency is still not real. Every point at
concurrency > 1 carries `includes_queue: true`.

Observed:

```
```

Status: PENDING

---

## B6 `NO_REPACK`, the A/B the code itself asks for

Implementation: `docker/llama-cpu/entrypoint.cpu.sh`, `bench/commands/run.py`.

`entrypoint.cpu.sh` carries the comment "Benchmark `NO_REPACK=false` before changing:
repacking is the main CPU throughput lever on AVX-512", and it has never been measured.
Weight repacking trades startup time for GEMM throughput, so the decode arm is where it
should show if anywhere.

```bash
taskset -c 8-15 python3 -m bench run --service llama-cpu --model <slug> \
    --profile publish --arm repack-off:NO_REPACK=true --arm repack-on:NO_REPACK=false
python3 -m bench compare <repack-off>.json <repack-on>.json --noise <b2-cpu-noise>.json
```

Note the polarity: `NO_REPACK=true` passes `--no-repack`, so it is the arm with
repacking **disabled**, which is today's default.

Expected signal: exit 0 from `compare` (1 means a regression outside the floor, which is
also a valid outcome to record). A verdict per metric. The result that matters is on
`decode_tok_s` at the decode point. Any of the five verdicts is a real answer —
`within_noise` would say the lever does not move this model on this box, which is worth
knowing and worth writing down rather than re-litigating later.

Also record the startup-time side: repacking is a load-time cost, so note how long
`wait_healthy` took on each arm (the `[OK] Health:` line reports it).

Observed:

```
```

Status: PENDING

---

## B7 `THREADS=12` against `THREADS=16`, to re-anchor the 26.2% claim

Implementation: `docker-compose.inference.yaml`, `bench/commands/run.py`.

[../../optimizations/README.md](../../optimizations/README.md) carries a 26.2%
improvement figure measured at `THREADS=12` on a service that now runs 16, by a script
no longer in the repository, with prefill folded into a decode-shaped number. It is
marked superseded. This entry is what replaces it.

```bash
taskset -c 8-15 python3 -m bench run --service llama-cpu --model <slug> \
    --profile publish --arm t16:THREADS=16 --arm t12:THREADS=12
python3 -m bench compare <t12>.json <t16>.json --noise <b2-cpu-noise>.json
```

Expected signal: a verdict on `decode_tok_s`, with both arms' medians recorded. This does
**not** reproduce the 26.2% figure and is not trying to: that number bundled a BIOS
change with everything else, and this isolates one variable. What it does give is a
current, phase-separated, reproducible baseline for the 16-core configuration the service
actually runs — which is what
[../../optimizations/README.md](../../optimizations/README.md) should cite instead.

The `taskset -c 8-15` pin gives the client 8 cores while the service holds 16, so the two
arms differ only in the server's thread count. Record `conditions.host.service_cpuset`
from each arm to confirm compose applied the override.

Observed:

```
```

Status: PENDING

---

## B8 The harness, and what a non-streaming engine can honestly report

Implementation: `bench/adapters.py::HarnessAdapter`, `bench/client.py::whole`.

Blocked on B3 (no registered harness model).

`schemas/harness_chat_endpoint.json` records that the harness serves only
`POST /v1/chat/completions` and refuses `stream: true` with a 400. One response means one
timestamp, so time to first token and the inter-token series do not exist for this
service and are reported absent rather than approximated.

```bash
python3 -m bench run --service harness --model <slug> --profile quick --json \
    > /tmp/b8.json
```

Expected signal: exit 0. `phases.decode.points[].metrics.streamed == false`,
`ttft_ms == null`, `tpot_ms == null`, `decode_tok_s == null`, and `e2e_tok_s` present with
a median near the 5.5–5.8 tok/s `common/serving.py` records. The absent fields are the
entry's point: a `null` here is the suite refusing to invent a phase split it could not
observe.

If this entry motivates phase separation for the harness, the work is exposing the chunk
streamer `schemas/harness_chat_endpoint.json` notes already exists in
`harness/src/stream.rs` — Rust work in another crate, not something the adapter can
supply.

Observed:

```
```

Status: PENDING

---

## B9 A comparison refuses what it should refuse

Implementation: `bench/delta.py::check_comparable`.

The guard is only worth having if it fires. Two results from different services, or at
different thread counts, or from different profiles, are not two arms of one experiment.

```bash
python3 -m bench compare <b1-llama-gpu>.json <b4-llama-cpu>.json
```

Expected signal: exit 2, and an error naming every mismatch at once —
`conditions.service`, `conditions.engine`, and the cpusets — rather than the first one.
A delta must not be printed. Then confirm the positive case still works: comparing an arm
against itself exits 0 with every metric `within_noise` against a floor from B2.

Observed:

```
```

Status: PENDING

---

## What this campaign does not cover

- **The huge-page arm.** Being verified separately by hand; the `HUGEPAGES=on|off`
  toggle exists in the entrypoint and compose and is an ordinary `--arm` once that
  settles.
- **Perplexity.** `scripts/benchmark/test_perplexity.sh` is the quality arm and is
  unchanged by this work. Any optimization that could change numerics — a KV cache type,
  a quantization — needs it alongside the speed measurement, because a faster wrong
  answer is not a win.
- **The blog rendering path.** `bench chart` and `bench promote` are not implemented, so
  no result from this campaign is committed or rendered yet. The results live under
  `/mnt/ai-data/benchmarks/bench/<run_id>/` in the meantime; note the run ids in the
  Observed blocks so they can be promoted later.
