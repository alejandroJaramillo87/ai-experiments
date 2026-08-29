# Benchmark suite quality campaign

**Operational runbook. Executed in an agent sandbox rather than against the host, so it
needs no GPU, no Docker daemon, and no model data. Keep the commands runnable, and the
paths accurate — the documentation gates check this directory, so a stale reference here
fails the gate.**

**Entries: 9 verified, 0 blocked, 0 pending, of 9.**

The sandbox counterpart to [bench.md](bench.md). That runbook proves numbers against real
services and cannot run until the workstation is free; this one proves the reduction logic,
the wire decoding, and the failure paths, none of which need a service at all.

The division is what a property depends on. A claim about what llama.cpp returns belongs in
`bench.md`. A claim about what `python3 -m bench` does with a given stream of bytes belongs
here, because a container proves it as well as the RTX 5090 would.

The method is [../../../.claude/skills/bug-bash/SKILL.md](../../../.claude/skills/bug-bash/SKILL.md):
oracles before fixes, adversarial verification before remediation, and a pushed commit per
batch. Recording protocol and status vocabulary are defined in [../README.md](../README.md).
None of these entries is a GPU consumer, so the one-at-a-time rule does not apply.

## Table of contents

- [Scope and exclusions](#scope-and-exclusions)
- [Phase A: oracles](#phase-a-oracles)
- [Findings ledger](#findings-ledger)
- [What this campaign does not cover](#what-this-campaign-does-not-cover)

## Scope and exclusions

In scope: `bench/` and `tests/bench/`.

Excluded, and the reason for each:

- **`scripts/benchmark/test_perplexity.sh`.** The quality arm, unchanged by the work that
  produced `bench/`, and out of scope for a campaign against the speed path.
- **`common/serving.py`, `common/reports.py`, `common/output.py`.** Consumed by `bench/` but
  owned elsewhere and covered by `tests/common/`. A defect found there is filed against that
  owner rather than absorbed here.
- **Any claim requiring a running service.** Those are `bench.md` entries. A finding here
  that can only be settled on the workstation is recorded and routed there, not guessed at.
- **The engine versions.** llama.cpp is pinned at commit
  `e3546c7948e3af463d0b401e6421d5a4c2faf565` in
  [../../../docker/llama-cpu/Dockerfile.cpu](../../../docker/llama-cpu/Dockerfile.cpu) and
  [../../../docker/llama-gpu/Dockerfile.llama-gpu](../../../docker/llama-gpu/Dockerfile.llama-gpu);
  vLLM at `0.25.0` in
  [../../../docker/vllm-gpu/Dockerfile.vllm-gpu](../../../docker/vllm-gpu/Dockerfile.vllm-gpu).
  Both are behind their upstreams. Bumping either is a measured arm for
  [../../optimizations/roadmap.md](../../optimizations/roadmap.md), not campaign work, and
  every wire-format claim below is made against the pinned versions because those are what
  the containers run.

No prior campaign has run against this scope, so the exclusion list carries no settled
waivers to honour.

---

## BQ.1 The gate is green before anything changes

Implementation: [../../../make/quality.mk](../../../make/quality.mk).

The bug-bash ladder opens by confirming the project gate exits 0. A campaign that cannot
state its starting gate result has no baseline, and a fix cannot be distinguished from a
pre-existing failure.

```bash
$ make quality
```

Expected signal: exit 0. In a sandbox `uv lock --check` announces its own skip; every other
step runs.

Observed:

```
$ make quality
uv run --no-sync ruff format --check .
314 files already formatted
uv run --no-sync ruff check .
All checks passed!
uv run --no-sync mypy
bench/delta.py:81: error: Incompatible types in assignment ...  [assignment]
bench/runner.py:197: error: "object" has no attribute "get"  [attr-defined]
bench/runner.py:206: error: List comprehension has incompatible type ...  [misc]
bench/runner.py:206: error: Value of type "object" is not indexable  [index]
bench/runner.py:208: error: "object" has no attribute "get"  [attr-defined]
Found 5 errors in 2 files (checked 149 source files)
make: *** [make/quality.mk:110: quality] Error 1

(after the two fixes recorded as BQ-F17 and BQ-F18)

$ make quality
... ruff format, ruff check, mypy, uv lock --check, six documentation gates,
    shellcheck, and the rust-harness gate, all reported clean ...
GATE_EXIT=0
```

The gate did not exit 0 at the start of the campaign, which is the ladder's own
precondition. Five type errors stood against `bench/`, and the reference gate also
failed on a git-ignored path cited by `docs/agents/README.md`. Both are recorded as
findings and both were fixed before any other work, because a campaign cannot tell its
own fixes from a pre-existing failure on a red tree. The rust-harness gate is included
in the green run above.

Status: VERIFIED 2026-08-29

---

## BQ.2 The suite baseline, its skip count, and its coverage

Implementation: `tests/bench/`, [../../../pyproject.toml](../../../pyproject.toml).

A green suite is not a passing suite until the skip count is read
([../../../tests/README.md](../../../tests/README.md)). Coverage is recorded as a number
rather than enforced: `fail_under` is configured in `pyproject.toml` but no target invokes
coverage, so it gates nothing today.

```bash
$ uv run --no-sync pytest tests/bench -q
$ uv run --no-sync pytest tests/bench --cov=bench --cov-report=term-missing -q
```

Expected signal: every test passes, the skip count is zero, and the per-module coverage
table is recorded so later work has something to move.

Observed:

```
$ uv run --no-sync pytest tests/bench -q
153 passed in 1.40s

$ uv run --no-sync pytest tests/bench --cov=bench --cov-report=term-missing -q
TOTAL                          894    173    81%
FAIL Required test coverage of 83.7% not reached. Total coverage: 80.65%

  bench/client.py             61      8    87%   56, 95-105
  bench/conditions.py         87     29    67%   36-37, 58-59, 74-77, 86-90, 112-122, ...
  bench/commands/compare.py   49     33    33%   15-24, 61-109
  bench/commands/noise.py     55     33    40%   53-117
  bench/commands/run.py       74     42    43%   76-83, 94-150

(at the end of this phase)

$ uv run --no-sync pytest tests/bench -q
222 passed, 4 xfailed in 3.81s
$ uv run --no-sync coverage report --precision=2
TOTAL                          894    139  84.45%
```

The skip count is zero: no test in this directory is gated on hardware, so a green run
here is a complete run rather than a partial one. That is not true of the suite as a
whole, which reports 41 skipped and 26 deselected in this sandbox.

Coverage began below the configured floor and ends above it, though the floor gates
nothing: no target invokes coverage, so the number is recorded rather than enforced.
The four expected failures are the reproductions recorded under BQ.9, BQ-F05 and
BQ-F11.

Status: VERIFIED 2026-08-29

---

## BQ.3 The wire fixtures agree with the pinned engine source

Implementation: `tests/bench/fixtures/`, `bench/adapters.py::parse_usage`.

The fixtures the adapter tests assert against were written by hand and pinned to nothing.
Their module comment claims they are the shapes each engine emits, and no capture or source
citation supports that claim, so an engine changing its frame shape would fail no test.

Replacing them with fixtures derived from the pinned source is weaker evidence than a
capture and stronger than an assertion: source shows every branch, a capture shows the one
path a request happened to take. A capture at `bench.md` B1 confirms them later.

The two facts that decide whether `bench/client.py` decodes correctly at all:

- vLLM emits its terminal usage frame with an empty `choices` list, so nothing in it can be
  mistaken for a token arrival.
- llama.cpp sets the final frame content to the empty string in stream mode, because the
  text has already gone out in the last partial chunk.

Both hold at the pinned versions, and both are what the structural detection in
`bench/client.py::stream` depends on.

```bash
$ uv run --no-sync pytest tests/bench/test_adapters.py -q
```

Expected signal: the fixtures load from `tests/bench/fixtures/`, each carries a source
citation, and the adapter tests pass against them unchanged.

Observed:

```
$ uv run --no-sync pytest tests/bench/test_adapters.py -q
21 passed in 0.05s

$ ls tests/bench/fixtures/
llama_cpp_final.json  llama_cpp_partial.json  llama_cpp_withheld.json
provenance.json  vllm_chunk.json  vllm_final.json  vllm_final_with_metrics.json
```

Six fixtures, each traced in `provenance.json` to the serializer that emits it. The two
facts the client's structural detection depends on both hold at the pinned versions,
and reading the source rather than a capture surfaced a third that no capture would
have shown: vLLM can report server-side timings, including queue time, which the
adapter assumes it cannot. That is recorded as BQ-F19.

Status: VERIFIED 2026-08-29

---

## BQ.4 An arrival is stamped when it arrives

Implementation: `bench/client.py::stream`.

Eight tests drive the real client through a `BytesIO` standing in for the socket. A
`BytesIO` yields every line at once, so no test has ever asserted the property the whole
suite rests on: that a gap between frames on the wire becomes a gap of the same size in the
inter-token latency series.

```bash
$ uv run --no-sync pytest tests/bench/test_client_timing.py -q
```

Expected signal: injected inter-frame delays are recovered in the arrival series within a
stated tolerance, and the error branches that no fixture reaches today are exercised: a 5xx
response raising a transient error, a 4xx raising a permanent one, and a connection failure
raising a transient one.

Observed:

```
$ uv run --no-sync pytest tests/bench/test_client_timing.py -q
15 passed, 2 xfailed in 2.24s

(three consecutive runs: 2.24s, 2.24s, 2.23s)
```

Injected gaps are recovered over both fakes, including through a real socket, so
frames written close together are not coalescing into a single read. The three error
branches that no fixture reached before now have tests, and so does the chat-shaped
delta path.

Status: VERIFIED 2026-08-29

---

## BQ.5 The client's own timing resolution is measured

Implementation: `tests/bench/fakes.py`, `bench/client.py::stream`.

At concurrency 16 the client runs sixteen Python threads, each decoding a socket and parsing
JSON under the global interpreter lock. The arrival stamp is taken after that thread wins the
lock, so the client is itself an instrument with an error, and nothing has ever measured it.

This is the same principle the measurement campaign applies to the workstation, turned on the
tool: a delta smaller than the instrument's own resolution is not a small result.

Expected signal: a recorded number, not a threshold. The stamp error against a scripted
server at a known frame rate, at concurrency 1 and at the highest concurrency any profile
uses. A figure materially larger than the inter-token latencies being measured is a finding
in its own right.

Observed:

```
$ uv run --no-sync python tests/bench/resolution.py
=== bench client timing resolution ===

[INFO] Injected gap: 20.0 ms over 24 tokens per stream
[RUN] Concurrency 1: streaming
  [OK] Concurrency 1: median 0.17 ms, p95 0.24 ms, max 0.26 ms over 23 intervals
[RUN] Concurrency 4: streaming
  [OK] Concurrency 4: median 0.14 ms, p95 0.21 ms, max 0.28 ms over 92 intervals
[RUN] Concurrency 16: streaming
  [OK] Concurrency 16: median 0.12 ms, p95 0.21 ms, max 1.42 ms over 368 intervals

[INFO] Stamp error stays under a quarter of the interval
```

The hypothesis that prompted this entry is refuted, and the refutation is the result.
Sixteen threads contending for the interpreter lock do not move the median stamp
error, which stays near 0.15 ms at every level measured.

The tail does move: the worst single interval grows from 0.26 ms at one stream to
1.42 ms at sixteen, roughly five times. Read against llama-cpu, whose inter-token
interval is tens of milliseconds, that is immaterial. Read against llama-gpu at a few
milliseconds per token it is not, and it lands in the same p95 and max fields a
comparison reads. The figure to carry forward is therefore the tail, not the median.

This measures the client against a server that answers instantly from a script. It
does not and cannot separate the client's error from a real engine's own jitter.

Status: VERIFIED 2026-08-29

---

## BQ.6 Host facts degrade rather than raise

Implementation: `bench/conditions.py::collect`, `bench/conditions.py::container`.

`collect` runs after all measurement, inside `bench/runner.py::assemble`, and `assemble` sits
outside the error handler in `bench/commands/run.py`. Anything raising there discards a
completed run, so every path through `conditions.py` that meets a missing binary, a
non-zero exit, a timeout, or unparseable output must return an absent field rather than an
exception.

```bash
$ uv run --no-sync pytest tests/bench/test_conditions.py -q
```

Expected signal: every degradation path yields a well-formed mapping with the unavailable
fields absent or null. The one path that does raise is recorded as a finding rather than
asserted as correct.

Observed:

```
$ uv run --no-sync pytest tests/bench/test_conditions.py -q
16 passed, 1 xfailed in 0.07s
```

Every path that meets a missing binary, a non-zero exit, a timeout, an operating
system error, or unparseable output yields an absent field. The one path that raises
is JSON that parses but is not a mapping, recorded as BQ-F11 and reproduced as an
expected failure.

Writing these tests corrected a finding rather than confirming it. The first version
of the two-query test stubbed every subprocess to fail, which makes the GPU helper
return early and never reach its second invocation, so the test measured one call and
contradicted BQ-F12. The finding was right and the test was wrong: the second spawn is
reached only on a host where the first query succeeds.

Status: VERIFIED 2026-08-29

---

## BQ.7 The reduction holds over generated input

Implementation: `bench/metrics.py`, `bench/delta.py`, `bench/workloads.py`.

The pure modules are tested against hand-chosen examples, which prove the cases the author
thought of. Properties hold over inputs nobody chose.

Four properties: every statistics block satisfies the ordering its field names imply; a
result compared against itself produces no regression at any threshold; `bench/delta.py::classify`
is total, mapping every input to exactly one verdict; and `bench/metrics.py::prefill_slope`
recovers an exact slope from synthetic linear input.

Seeded rather than random, so a failure is reproducible from the recorded seed.

```bash
$ uv run --no-sync pytest tests/bench/test_properties.py -q
```

Expected signal: every property holds across its generated cases, and any failure is
reported with the seed that produced it.

Observed:

```
$ uv run --no-sync pytest tests/bench/test_properties.py -q
29 passed, 1 xfailed in 0.30s
```

One property failed on first run, and it was the property that was wrong. The claim
that the decode rate always exceeds the end-to-end rate is not a law: algebra makes it
equivalent to time-to-first-token exceeding the mean time per output token, which is
true of this hardware and not of the statement. It is now asserted as the
biconditional it is, with both branches exercised, and BQ-F20 records that `bench.md`
entry B4 states the unconditional form as an acceptance criterion.

Status: VERIFIED 2026-08-29

---

## BQ.8 The mutation baseline, taken before any fix

Implementation: `tests/bench/mutate.py`.

Coverage says the tests executed a line. It does not say they would notice the line being
wrong. For two modules whose entire purpose is to avoid reporting a number that is not true,
that distinction is the whole question.

The baseline is taken against the tree as it stands. Taking it after a remediation batch
would measure a repaired tree and could not say what the new tests were worth.

```bash
$ uv run --no-sync python tests/bench/mutate.py
```

Expected signal: a killed-versus-survived count per module. There is no pass mark; the number
is the deliverable, and each survivor is a finding about the tests rather than about the code.

Observed:

```
$ uv run --no-sync python tests/bench/mutate.py

(baseline, against the suite as it stood at the start of this phase)
bench/metrics.py: 61 of 84 killed (72.6%)
bench/delta.py:    9 of 14 killed (64.3%)
Overall: 70 of 98 killed (71.4%)

(after the modules added in this phase)
bench/metrics.py: 80 of 84 killed (95.2%)
bench/delta.py:   11 of 14 killed (78.6%)
Overall: 91 of 98 killed (92.9%)
```

The baseline is the first number, taken before any test written here. The second is
what the phase was worth: nineteen mutants that the suite previously accepted now fail
it.

The gaps the baseline exposed were concentrated rather than scattered. Nothing pinned
the two-token boundary that decides whether the prefill arm reports a decode rate it
has no basis for. Nothing pinned the quantile constants, so the published 95th and
99th percentiles could have been any order statistic. Nothing pinned the
goodness-of-fit arithmetic behind the only defensible client-side prefill figure.

Seven survivors remain and none is a test gap:

- Two centre a covariance on plus-mean instead of minus-mean. The deviations sum to
  zero, so the two expressions are equal. No test can distinguish them.
- Three sit behind guards their own preconditions exclude: a signed comparison reached
  only when the magnitude already exceeds five, and two default arguments for empty
  collections that the callers refuse to produce.
- Two require a variance strictly between zero and one square millisecond across a
  prompt-length sweep, which is not a measurement this suite can take.

A mutation score is a measure of the tests. Reporting an equivalent mutant as a gap
would make it a measure of nothing, so each survivor above was checked by hand.

Status: VERIFIED 2026-08-29

---

## BQ.9 The withheld-frame defect is reproduced

Implementation: `bench/metrics.py::reduce_samples`, `bench/client.py::stream`.

llama.cpp does not send a partial response for a token that leaves an incomplete UTF-8
sequence: the branch containing the only call that emits one is skipped entirely, and the
next token's frame carries both tokens' text. A token held back for a pending stop-string
match reaches the same outcome through an empty text field.

Either way the arrival count falls below the completion-token count, one inter-token
interval silently spans two token generations, and the decode rate divides a token count by
a span that does not cover those tokens.

The prompt corpus makes this reachable rather than theoretical.
[../../../scripts/benchmark/datasets/wikitext-2-raw/](../../../scripts/benchmark/datasets/wikitext-2-raw/)
is 0.107 percent non-ASCII, chiefly en dashes, em dashes, and accented characters, and the
generation is prompted with that prose. Infrequent is the harder case: it produces occasional
large latency outliers indistinguishable from scheduling jitter.

The reproduction ships as a strict expected failure, so the suite stays green while the
defect is recorded executably. Strictness is explicit because the suite does not configure
it globally, and a non-strict expected failure that starts passing would go unnoticed.

```bash
$ uv run --no-sync pytest tests/bench/test_client_timing.py -q -rx
```

Expected signal: the reproduction is reported as an expected failure naming BQ-F01. Its fix
belongs to the remediation batch, not to this phase.

Observed:

```
$ uv run --no-sync pytest tests/bench/test_client_timing.py -q -rx
XFAIL TestWithheldFrames::test_a_leading_withhold_does_not_overstate_the_decode_rate
XFAIL TestWithheldFrames::test_a_leading_withhold_does_not_overstate_time_to_first_token
15 passed, 2 xfailed in 2.24s
```

Reproduced, and narrower than first filed. Working the arithmetic showed the effect
depends on where in the generation the frame is withheld, and the two cases are
different defects:

- **Interior.** Every aggregate stays correct. The span still covers every interval
  and the token count is still right, so the decode rate, the mean time per output
  token and time to first token are all unaffected. What breaks is the distribution:
  one interval spans two token generations and enters the series as a genuine latency
  outlier. On an eight-token generation at twenty milliseconds, an interval of forty
  appears where none occurred.
- **At either end.** The span loses one interval while the token count does not, so
  the decode rate reads high by one interval's worth and the mean time per token reads
  low. On the same generation that is 29.2 tokens per second against a true 25.0, an
  overstatement of 16.7 percent. A withheld leading frame also puts time to first
  token one interval late, 140 milliseconds against 100.

Both reproductions are strict expected failures, so the suite stays green and a fix
that lands without removing them fails the gate rather than passing silently.

Status: VERIFIED 2026-08-29

---

## Findings ledger

Campaign-local identifiers, following the rust-harness precedent: an identifier in
[../bugs/known-bugs.md](../bugs/known-bugs.md) is for a defect a host runbook step will hit,
and a finding is promoted there only once it is shown to surface at a `bench.md` entry.

Severity is proposed, not settled. Every row below was found by reading and none has an
independent verdict, so under the rubric each is `reasoning only` and cannot be graded above
low until the verification phase records a verdict.

| ID | Location | Proposed | Defect |
|----|----------|----------|--------|
| BQ-F01 | `bench/metrics.py`, `bench/client.py` | high | A withheld frame costs one arrival while the reported generation length does not change. Withheld at either end of the generation, the decode rate reads high by one interval's worth (29.2 against 25.0 tokens per second on the reproduction) and a leading withhold also puts time to first token one interval late. Withheld in the interior, every aggregate stays correct and one fabricated interval enters the latency distribution as an outlier. Filed first as one critical defect covering both; the arithmetic under BQ.9 shows they are two, and the interior case never moves a rate. |
| BQ-F02 | `bench/metrics.py` | high | The field naming the worst observed chunk packing is computed as a mean of per-sample means, so a single densely packed chunk is diluted below the threshold that would flag the series as approximate. |
| BQ-F03 | `bench/delta.py` | critical | `compare` never checks that the supplied noise floor was measured against the same workload as the arms. `bench.md` B6 and B7 instruct feeding a `quick` floor to a `publish` comparison, where every lookup misses and every metric silently reports as inconclusive while appearing to have been checked against a floor. The bare-metric fallback in the lookup is unreachable. |
| BQ-F04 | `bench/delta.py` | high | Observed prompt and completion token counts are statistics blocks inside the metrics mapping, so they are differenced and graded as latencies. A run that tokenized more prompt than its baseline reports a regression. |
| BQ-F05 | `bench/metrics.py` | high | The coefficient of determination is exactly 1.0 for any two-point fit. The `quick` profile sweeps two prefill lengths, and `bench.md` B1 accepts a value at or above 0.95, so that check passes by construction on a broken instrument. |
| BQ-F06 | `bench/runner.py` | high | One rejected sample discards its entire concurrency cell; at the sixteen-way vLLM point, fifteen good results go with it. |
| BQ-F07 | `bench/runner.py`, `bench/commands/run.py` | high | Samples are written once at the end, and `assemble` sits outside the error handler. Any failure not derived from the serving error type, late in a run of up to an hour, leaves nothing on disk. |
| BQ-F08 | `bench/runner.py::write_samples` | medium | The samples file carries bare performance-counter values with no wall-clock anchor and no cell ordinal, so the interleaving the design cites as its evidence cannot be checked from the artifact. |
| BQ-F09 | `bench/client.py` | medium | A frame that fails to parse is skipped silently, removing an arrival and inflating the intervals on either side of it. Nothing counts them or reports them. |
| BQ-F10 | `bench/adapters.py::parse_usage` | medium | The integer conversion of the usage fields is unguarded. A non-numeric value raises through `bench/runner.py::_dispatch`, which handles only the serving error type, ending the run. |
| BQ-F11 | `bench/conditions.py::container` | medium | Successfully parsing JSON that is not a mapping makes the subsequent lookup raise, uncaught here and uncaught in `collect`. |
| BQ-F12 | `bench/conditions.py` | medium | Two separate GPU query subprocesses plus a container inspection run after all measurement is complete and on the run's critical path, each under its own timeout. The second query is issued from inside the first helper's return expression, so it is reached only when the first succeeds; a host without the tool pays for one call, and the workstation pays for both. |
| BQ-F13 | `bench/conditions.py::collect` | low | The service key is passed where a container name is expected, and the parameter that would carry a resolved name is unused at both call sites. |
| BQ-F14 | `tests/bench/test_runner.py` | low | Three unit tests call `assemble`, which invokes host binaries on whatever machine runs the suite. Machine-dependent, and nothing asserts the result. |
| BQ-F15 | `bench/client.py` | low | The event-stream prefix constant requires a space the specification makes optional. |
| BQ-F19 | `bench/adapters.py`, `bench/README.md` | high | The vLLM adapter does not override the server-timings parser, so every vLLM phase is labelled client-sourced and the package README states the engine reports no server timings. At the pinned version the engine can attach per-request timings to the same final usage frame the adapter already parses, carrying time to first token, generation time, mean inter-token latency and queue time. Queue time is the one quantity this suite documents as impossible to derive client-side, and it is what would let the concurrency sweep report a queue-free latency. Reaching it needs a launch flag the compose file does not pass, which is the same gap that leaves every other vLLM knob unreachable. |
| BQ-F20 | `docs/verification/bench/bench.md` | medium | Entry B4 accepts on the decode rate exceeding the end-to-end rate. The separation is conditional, not a property of the measurement: it holds exactly when time to first token exceeds the mean time per output token. It is expected to hold on this hardware, so the criterion is an assumption about the machine stated as an invariant, and a run that failed it would be read as a broken tool rather than as the finding it would be. |
| BQ-F16 | documentation | low | `pyproject.toml` omits `bench` from the isort first-party list while including it in the type-checker file list; `tests/README.md` omits `tests/bench/` from its layout table; `../README.md` records nine entries for `bench.md`, which has eight; `../../testing/quality-gate.md` says ten steps and lists twelve; `../../README.md` claims a gate exemption for this directory that was removed on 2026-08-08, and still directs readers to two retired scripts; `../README.md` described `bench.md` as covering four serving services where that runbook scopes itself to three. Every item in this row is corrected in this batch. |
| BQ-F17 | `bench/runner.py`, `bench/delta.py` | high | The type checker reported five errors against `bench/`, so `make quality` did not exit 0 on the pushed branch. The repository carries no continuous integration, and the gate is invoked by hand, which is how a red gate reached the remote. Fixed in this batch by annotation alone, with no change to behavior. Closed. |
| BQ-F18 | `scripts/docs/check_doc_refs.py` | low | `docs/agents/README.md` cites a git-ignored per-checkout settings file in the sentence that explains the file is git-ignored, so a fresh clone failed the reference gate. Fixed in this batch by recording it among the audited absences. Closed. |

## What this campaign does not cover

- **Numbers.** No entry here produces a latency, a throughput, or a noise floor. Those are
  `bench.md` and require the workstation.
- **The remediation.** Findings above are recorded and, where a reproduction is possible in
  a sandbox, reproduced. Fixing them is the next batch, verified first.
- **Amendments to `bench.md`.** BQ-F03 and BQ-F05 both name acceptance criteria in that
  runbook that are wrong today. The corrections describe verified fixes, so they follow the
  remediation rather than lead it.
