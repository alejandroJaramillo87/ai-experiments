# Benchmark suite verification runbook

**Entries: 0 verified, 0 blocked, 9 pending, of 9.**

Manual verification campaign for `python -m bench` on the workstation (RTX 5090, AMD
Ryzen 9950X). The suite passes its unit tests in the sandbox against synthetic fixtures
and a stub engine; none of them proves a number, because none of them touched a service.
This runbook is the campaign that closes that gap.

The quality campaign recorded in [bench-qa.md](bench-qa.md) ran between the first draft of
this runbook and this one, and it changed what several entries can accept. Where an entry
below says an old criterion was wrong, that is why.

Three services are in scope: `llama-cpu`, `llama-gpu` and `vllm-gpu`. The rust harness
is not, and that is a decision rather than an omission -- see the closing section.

Every entry follows [../../../.claude/skills/verify-host/SKILL.md](../../../.claude/skills/verify-host/SKILL.md):
the acceptance criterion is stated before the command is run, and the pasted output is
the evidence. The protocol the commands implement is
[../../../.claude/skills/benchmarking/SKILL.md](../../../.claude/skills/benchmarking/SKILL.md).

## Order matters, and why

Run these in order. Three of them are preconditions for the rest rather than
measurements in their own right:

- **B1** proves the CLI reaches a service at all. Everything after it assumes so.
- **B2** is the noise floor. Until it exists, every delta in B6 through B8 is
  uninterpretable — the 5% regression threshold has no meaning underneath an unmeasured
  spread, and this workstation has been recorded spreading 44% across four consecutive
  runs of one unmodified binary
  ([../../optimizations/ideation/rust-harness-optimization-candidates.md](../../optimizations/ideation/rust-harness-optimization-candidates.md)).
- **B3** is the registry. `vllm-gpu` has no registered model yet, so B5 and B9 cannot run
  until someone adds the entry.

B2's floor is per workload, not per profile name. B6, B7 and B9 each need a floor measured
at the profile they run, because `bench compare` refuses one whose workload digest does not
match the arms'.

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
Every `points[].metrics.ttft_ms.n` equals the profile's `repeats` (3 for `quick`), which
is what proves the warm-up was discarded rather than counted.

**Do not accept on `slope.r2` here** (BQ-F05). `quick` sweeps two prefill lengths, and a least
squares line through two points passes through both: `r2` is 1.0 by construction and says
nothing about the fit. `bench/metrics.py::prefill_slope` now returns `None` below three
points, so `slope` is absent from a `quick` run entirely and its presence would be the
finding. The slope criterion belongs to B4, which runs `publish`.

Read `slope.prompt_tokens_source` when the slope is present. `observed` means the fit used
the server's own token counts; `estimated` means it fell back to the four-characters-per-
token approximation that sliced the prompt, and a slope fitted through an estimated axis
carries that approximation's error.

```bash
python3 -c "
import json; d=json.load(open('/tmp/b1.json'))
print(d['report'], d['schema_version'], d['git_sha'])
print('timings_source', d['phases']['prefill']['timings_source'])
print('slope', d['phases']['prefill'].get('slope'))  # None on quick: two points
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
specific to a service and a workload.

**Measure the floor at the profile the comparison will use.** B6 and B7 run `publish`, so
a `quick` floor cannot grade them: `bench compare` refuses a floor whose
`run.workload_hash` does not match the arms', because the two profiles share point keys
and a stale floor would land on exactly the keys a regression is graded against. Run
`bench noise --profile publish` for those entries, and expect the wall clock to match a
`publish` run.

The floor document carries the pair it was measured from (BQ-F44) -- both arms in full, their
conditions, and a `coverage` block. Check `coverage.floor_metrics` against
`coverage.arm_metrics`: a floor narrower than the workload was measured over cells that
failed, and `compare` refuses it rather than applying it, because the metrics it does not
carry grade as `inconclusive`, which does not fail.

Observed:

```
```

Status: PENDING

---

## B3 The registry covers every service to be benchmarked

Implementation: `bench/registry.py`.

`MODELS` in `bench/registry.py` ships one entry, registered for `llama-cpu` and `llama-gpu`
only. `vllm-gpu` serves a Hugging Face checkpoint rather than a GGUF and has no entry, so
`bench run --service vllm-gpu` fails at `resolve()` with "registered for [...], not
'vllm-gpu'". That refusal is correct behaviour — a result whose architecture is
unrecorded is what this suite exists to stop — but it blocks B5 until the entry exists.

Add one entry per served model, then:

```bash
python3 -m bench models
python3 -m bench models --json | python3 -c "import json,sys; json.load(sys.stdin)"
```

Expected signal: exit 0, and each of `llama-cpu`, `llama-gpu` and `vllm-gpu` appears in
at least one entry's `engines`, in the JSON form as well as the human one (BQ-F29). The loader
validates on read, so a `quantization` that disagrees with the model filename, an unknown
`architecture`, or an MoE entry missing `params_active_b` fails here rather than in a
published number.

```bash
python3 -m bench models --json | python3 -c "
import json, sys
registry = json.load(sys.stdin)
print(sorted({engine for spec in registry.values() for engine in spec['engines']}))
"
```

The registry is also checked against the running service at the start of every `bench run`
and `bench noise`: a container serving a model the registry does not claim is refused with
exit 2 rather than measured. The training commands swap models on these same services and
skip their restore under `--keep-serving`, so the disagreement is reachable without any
fault.

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
the denominator, so it sits below the decode rate it is often quoted as. TTFT rises
monotonically with prompt length across the sweep. `slope.intercept_ms > 0`: that
intercept is the fixed per-request cost, and it is the number that makes a single-point
prefill estimate wrong. `publish` sweeps enough prefill lengths for `slope.r2 >= 0.95` to
mean something; read it here rather than at B1.

**The separation is conditional, not a property of the measurement** (BQ-F20). `decode_tok_s`
exceeds `e2e_tok_s` exactly when time to first token exceeds the mean time per output
token, which is expected on this hardware and is not guaranteed by arithmetic. A run where
they converge is a finding about the service -- prefill has become cheap relative to
decode -- and must be recorded as one, not read as a broken tool. Record both medians and
the TTFT median so the condition can be checked rather than assumed.

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
(`e2e_tok_s.median * concurrency`) rises with concurrency while per-request
`e2e_tok_s.median` falls and `ttft_ms.median` rises. That shape is continuous batching
working. If per-request throughput is flat across the sweep, the requests are not actually
contending and the concurrency is still not real. Every point at concurrency > 1 carries
`includes_queue: true`.

**Accept on `e2e_tok_s`, not `decode_tok_s`** (BQ-F54). vLLM merges queued deltas into one
streamed chunk under load, which is what it is in this stack to do, and a client cannot
distinguish a chunk carrying four tokens from four chunks it never received. The
reduction refuses `decode_tok_s` and `tpot_ms` for any point where that happened, so on
the one service with a concurrency sweep those two fields are expected to be `null`
exactly where the sweep is most interesting. `e2e_tok_s` spans submission to last
arrival, and both endpoints are invariant under packing.

Record `tokens_per_chunk_mean_max` and `arrivals_incomplete` per point (BQ-F55). They are the
evidence that packing, rather than a lost cell, is why the per-token rates are absent --
`bench compare` reads the same two fields to say so.

```bash
python3 -c "
import json; d=json.load(open('/tmp/b5.json'))
for point in d['phases']['decode']['points']:
    m = point['metrics']
    print(point['point'], m['concurrency'], m['e2e_tok_s']['median'],
          m['decode_tok_s'], m['tokens_per_chunk_mean_max'], m['arrivals_incomplete'])
"
```

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

`NO_REPACK` appeared in no compose file until this campaign (BQ-F52), so the variable never
crossed into the container and both arms of this entry ran the same service. It is
interpolated in `docker-compose.inference.yaml` now, and `bench compare` refuses a pair
whose `conditions.env_knobs` and `conditions.server_argv` are identical while the arm
names differ -- the check that would have caught it.

```bash
taskset -c 8-15 python3 -m bench noise --service llama-cpu --model <slug> \
    --profile publish
taskset -c 8-15 python3 -m bench run --service llama-cpu --model <slug> \
    --profile publish --arm repack-off:NO_REPACK=true --arm repack-on:NO_REPACK=false
python3 -m bench compare <repack-off>.json <repack-on>.json --noise <publish-cpu-noise>.json
```

Note the polarity: `NO_REPACK=true` passes `--no-repack`, so it is the arm with
repacking **disabled**, which is today's default.

The floor is measured at `publish`, not reused from B2's `quick` run: `compare` refuses a
floor whose workload digest does not match the arms'.

Both arms are recreated, whether or not they carry an override, so neither measures a
service that has been up longer than the other. Add `--interleave` to alternate the arms
at the repeat level instead of running each arm's whole profile in turn. It is not the
default because it costs a container recreate and a full warm-up pass per arm per repeat
-- roughly 2 to 2.5 times the wall clock on `publish` -- so reach for it when a result is
contested or when the two arms' medians land close to the floor. Every report records
`run.arm_sequence`, and `compare` warns when a differenced pair was blocked.

Expected signal: exit 0 from `compare` (1 means a regression outside the floor, which is
also a valid outcome to record). A verdict per metric. The result that matters is on
`decode_tok_s` at the decode point. Any of the five verdicts is a real answer —
`within_noise` would say the lever does not move this model on this box, which is worth
knowing and worth writing down rather than re-litigating later.

Also record the startup-time side: repacking is a load-time cost, so note how long
`wait_healthy` took on each arm (the `[OK] Health:` line reports it).

The run leaves the service configured as its last arm set it and prints the recreate that
clears it. Run that before the next entry, or the next measurement is taken under this
one's arm.

Observed:

```
```

Status: PENDING

---

## B7 `THREADS=12` against `THREADS=16`, to re-anchor the 26.2% claim

Implementation: `docker-compose.inference.yaml`, `bench/commands/run.py`.

[../../bench/results.md](../../bench/results.md) carries a 26.2%
improvement figure measured at `THREADS=12` on a service that now runs 16, by a script
no longer in the repository, with prefill folded into a decode-shaped number. It is
marked superseded. This entry is what replaces it.

`THREADS` was a literal in `docker-compose.inference.yaml` until this campaign (BQ-F52), so
`--arm t12:THREADS=12` set a variable in the client's environment that the container never
read: both arms ran 16 threads and the comparison reported a clean null on an A/B that
measured nothing. The compose file interpolates it now, and `THREADS_BATCH` beside it --
`--threads` is the decode lever and `--threads-batch` the prompt-processing one, so an arm
that means to move both says so (`--arm t12:THREADS=12,THREADS_BATCH=12`).

```bash
taskset -c 8-15 python3 -m bench run --service llama-cpu --model <slug> \
    --profile publish --arm t16:THREADS=16 --arm t12:THREADS=12
python3 -m bench compare <t12>.json <t16>.json --noise <publish-cpu-noise>.json
```

Reuse B6's `publish` floor rather than B2's `quick` one.

Expected signal: a verdict on `decode_tok_s`, with both arms' medians recorded. This does
**not** reproduce the 26.2% figure and is not trying to: that number bundled a BIOS
change with everything else, and this isolates one variable. What it does give is a
current, phase-separated, reproducible baseline for the 16-core configuration the service
actually runs — which is what
[../../optimizations/README.md](../../optimizations/README.md) should cite instead.

The `taskset -c 8-15` pin gives the client 8 cores while the service holds 16, so the two
arms differ only in the server's thread count. Record `conditions.host.threads` and
`conditions.server_argv` from each arm: those are read from the running container rather
than from the invocation, so they are the proof that compose applied the override. A
comparison whose two arms agree on both is refused with exit 2 and the message saying the
A/B measured nothing -- which is the correct outcome if the interpolation ever regresses.

Observed:

```
```

Status: PENDING

---


## B8 A comparison refuses what it should refuse

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

Four more refusals were added after the unit-level campaign, each for a way of publishing
a meaningless delta that used to exit 0. Confirm each fires against real artifacts, since
the sandbox only ever showed them synthetic ones:

| Refusal | How to reach it on the host |
|---------|------------------------------|
| Different pairs | Compare B6's `repack-off` against B7's `t12`. Both carry a `run.pair_id`, and they differ. |
| A partial run | Interrupt a `publish` run partway (Ctrl-C is not it — stop the service instead, so the health gate fails on the second arm), then compare the arm that was written. |
| A workload that moved | Compare B6's arms against B2's `quick` floor. |
| An A/B that measured nothing | Run `--arm a:NOT_A_KNOB=1 --arm b:NOT_A_KNOB=2`. The variable reaches no service, so both arms' `conditions.env_knobs` and `conditions.server_argv` are identical. |

Each is exit 2 with a named cause. The last is the one that matters most: it is the check
that would have caught the compose defect B6 and B7 were written against.

Observed:

```
```

Status: PENDING

---

## B9 vLLM reports its own timings, including the one a client cannot derive

Implementation: `docker/vllm-gpu/entrypoint.sh`, `docker-compose.inference.yaml`,
`bench/adapters.py::VllmAdapter`.

Every vLLM phase reports `timings_source: client`, because the adapter had no server
timings to parse (BQ-F19). At the pinned version the engine can attach per-request timings to the
same final usage frame the adapter already reads: `time_to_first_token_ms`,
`generation_time_ms`, `inter_token_latency_ms` and `queue_time_ms`. Queue time is the
quantity this suite documents as impossible to derive client-side, and it is what lets the
concurrency sweep report a queue-free latency beside the queued one.

**This changes the measurement conditions, so it is stated rather than slipped in.** The
timings ride on the engine statistics that `--disable-log-stats` turns off, and vLLM
refuses the two flags together. The service therefore runs with statistics logging on for
this entry, which costs engine-side accounting on every request. Both knobs are
interpolated in `docker-compose.inference.yaml` now, so this is an arm rather than an edit:

```bash
python3 -m bench run --service vllm-gpu --model <slug> --profile quick \
    --arm metrics-off \
    --arm metrics-on:DISABLE_LOG_STATS=false,ENABLE_PER_REQUEST_METRICS=true \
    --json > /tmp/b9.json
```

Expected signal: exit 0, and a `bench_pair` document whose two arms differ in
`phases.*.timings_source` — `client` for `metrics-off`, `server` for `metrics-on`. On the
`metrics-on` arm, `conditions.server_argv` contains `--enable-per-request-metrics` and does
not contain `--disable-log-stats`.

```bash
python3 -c "
import json; d=json.load(open('/tmp/b9.json'))
for name, arm in d['arms'].items():
    print(name, {p: b['timings_source'] for p, b in arm['phases'].items()})
    print(' ', [a for a in arm['conditions']['server_argv'] if 'stats' in a or 'metrics' in a])
"
```

The second half of this entry is the cost: compare the two arms and record whether turning
statistics on moved anything.

```bash
python3 -m bench noise --service vllm-gpu --model <slug> --profile quick
python3 -m bench compare <metrics-off>.json <metrics-on>.json --noise <vllm-noise>.json
```

Any of the five verdicts is a real answer. `within_noise` says the accounting is free at
this workload and the timings can stay on for every later vLLM entry; a regression says
they are a measurement instrument to be switched on deliberately and switched off before
publishing a throughput number.

Observed:

```
```

Status: PENDING

---

## What this campaign does not cover

- **The rust harness.** Deliberately not a target. Its optimization arms -- the SIMD
  panel kernels, matvec threading, prefill batching, the widen hoist -- are properties of
  the runtime's internals, and `make ladder SIZE=quick MODEL=<dir>` already exposes them
  as first-class flags (`--dot`, `--batch`, `--threads`, `--single-thread`) measured
  in-process against the real checkpoint, writing `rung_duration_s` into the same
  `metrics.jsonl` the dashboard charts. An HTTP benchmark would add a serialization layer
  and a mutex on top of the thing being measured, and
  [../../../schemas/harness_chat_endpoint.json](../../../schemas/harness_chat_endpoint.json)
  records that its endpoint refuses `stream: true` outright, so it could not report a
  prefill/decode split either. `make ladder` is the harness's benchmark.
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
