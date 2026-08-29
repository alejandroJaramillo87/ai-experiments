---
name: benchmarking
description: The reproducible benchmark protocol for this workstation — which entry points to run, warm-up policy, how results are recorded and compared, and what counts as a regression. Use before reporting any latency/throughput/memory number.
---
# Benchmarking protocol

Reference for producing comparable performance numbers on the RTX 5090 / Ryzen 9950X
stack. Follow `.styles/TEST_OUTPUT_STYLE.md` for all output.

> Sandboxed execution: agents cannot reach the host, its GPU, or the running services.
> Every command in this skill is executed by the human on the workstation. The agent
> prepares the exact command line (per this protocol), waits for the human to report the
> output, and interprets the reported numbers. Preconditions (`make status`,
> `make health`) are likewise confirmed via human-reported output.

## Entry points

There is no `make benchmark` target — a benchmark is a serialized GPU consumer
(AGENTS.md), and a Make target would make it look like an ordinary one. Run the CLI
directly against a running service:

- `python3 -m bench run --service <name> --model <slug> [--profile quick|publish]`
- `python3 -m bench noise --service <name> --model <slug>` — measure the noise floor
- `python3 -m bench compare <baseline.json> <candidate.json> --noise <noise.json>`
- `python3 -m bench models` — the registry, and a validation check on it
- Perplexity/quality: `./scripts/benchmark/test_perplexity.sh`

Services are named as `common/serving.py` names them: `llama-cpu`, `llama-gpu`,
`vllm-gpu`, `harness`. Models are slugs from `schemas/bench_models.json`.

Preconditions: the target service must be healthy first — check `make status` (and
`make health`) before starting. `bench run` gates on health itself and refuses to
proceed, because a benchmark against a cold or unhealthy service is not a slow result,
it is no result.

## What the tool does, and what you still have to do

Warm-up discard, repeat count, interleaving and the reported statistics are properties
of the profile, not of the operator's memory. `bench run` discards the first run of
every point, executes repeats round-robin over points rather than blocked, and reports
median, p95, p99, min, max and stdev. What remains yours:

1. **Hold the workload constant.** Same model, same profile, same service. `bench
   compare` refuses on a mismatch of service, engine, model, quantization,
   architecture, context length, prompt source, cpuset or profile hash.
2. **Measure the noise floor before trusting a delta.** See below.
3. **Pin the client off the service's cores.** `llama-cpu` holds `cpuset 0-15`; running
   the benchmark client inside that range makes client and server contend and taints the
   measurement. Use `taskset -c 8-15` for CPU arms, or narrow the service for the run.
   Both cpusets are recorded, and `compare` refuses if they differ between arms.

## Prefill and decode are different measurements

A single tokens/second figure moves when either phase moves and identifies neither. The
suite measures them separately and so should any claim made from it:

- **Prefill** — the prompt-length sweep at `max_tokens=1`. The reported
  `prefill_tok_s_slope` is the reciprocal slope of time-to-first-token against prompt
  length, which cancels the fixed per-request overhead that a single-point estimate
  charges to the model. `intercept_ms` is that overhead, and `r2` says whether the fit
  is worth quoting. Where the engine reports its own timings (llama.cpp does; vLLM and
  the harness do not) `prefill_tok_s_server` is authoritative and `timings_source` says
  which you are looking at.
- **Decode** — short prompt, long generation. `tpot_ms` and `decode_tok_s` exclude
  prefill. `e2e_tok_s` includes it and is retained only so figures recorded by the
  earlier tooling remain comparable; do not quote it as a decode rate.

Compare across engines on client-side numbers on both sides. Compare two arms of one
engine on server-side numbers when both have them.

## Dense and MoE are not benchmarked the same way

An MoE routes each token to a subset of experts chosen on content, so:

- Prompts are real text, never random token ids. Uniform vocabulary noise routes
  near-uniformly across experts — the best case for load balance, the worst for
  expert-weight locality, and wrong with no predictable sign.
- The decode arm draws three disjoint prompts and the spread between them is the
  routing sensitivity, a reported quantity rather than an unmeasured confound.
- Quote `params_active_b` alongside `params_total_b`. A 30B-A3B model is comparable to a
  dense 3B on arithmetic while carrying a dense 30B's residency; a claim that states only
  the total is wrong in both directions.

`schemas/bench_models.json` carries these facts and `bench run` stamps them into every
report. A model absent from the registry cannot be benchmarked, which is deliberate: a
result whose architecture is unrecorded is how the repository came to attribute an MoE
measurement to its CPU track without noticing.

## Recording and comparison

- Name each arm with `--arm <name>[:KEY=VALUE,...]`. An arm is a service configuration,
  not a code path, so `--arm off:HUGEPAGES=off` needs no new code.
- Two `--arm` flags interleave at the repeat level, so drift over the run cannot alias
  onto one arm. A container recreate costs 30–300s, which is why interleaving is per
  repeat and not per request.
- Full output and raw samples go to `/mnt/ai-data/benchmarks/bench/<run_id>/`; results
  are run data and do not belong in the worktree.
- Every report carries its own conditions — model, architecture, quantization, thread
  count, cpusets, huge-page counters, driver and CUDA versions, the service's environment
  and its actual argv. A number without them is not citable.

## Regression criterion

A change is a regression when the median is more than **5%** slower than the baseline
**and** the difference is outside the measured run-to-run spread. Both clauses bind.

The second one needs a number, so measure it: `bench noise` runs one unchanged
configuration twice, interleaved, and the per-metric difference between the two is the
floor — the largest difference this rig produces from having changed nothing. Four
consecutive runs of one unmodified binary on this workstation have been recorded
spreading 44% (`docs/optimizations/ideation/rust-harness-optimization-candidates.md`), so
a 5% threshold under an unmeasured floor reports noise as findings.

`bench compare` returns one of five verdicts per metric:

| Verdict | Meaning |
|---------|---------|
| `regression` | beyond 5% and outside the floor, slower |
| `improvement` | beyond 5% and outside the floor, faster |
| `below_threshold` | resolvable above the floor, but under the 5% reporting bar |
| `within_noise` | inside the floor: the rig cannot tell it from no change at all |
| `inconclusive` | beyond 5%, but no floor was measured, so resolvability is unknown |

`within_noise` and `inconclusive` are results and must be reported as such. Do not round
either to "no change" or to a win, and do not average a regression away against unrelated
wins.
