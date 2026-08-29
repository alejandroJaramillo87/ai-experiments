# Benchmark suite quality campaign

**Operational runbook. Executed in an agent sandbox rather than against the host, so it
needs no GPU, no Docker daemon, and no model data. Keep the commands runnable, and the
paths accurate — the documentation gates check this directory, so a stale reference here
fails the gate.**

**Entries: 10 verified, 0 blocked, 0 pending, of 10.**

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
- [Phase B: verification](#phase-b-verification)
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

No prior campaign has run against this scope, so this campaign's own ledger is the whole
exclusion list. Every defect already filed below as BQ-F01 through BQ-F23 is out of scope for
any further finder: it is recorded, and re-filing it wastes a verification slot on a claim
that already has one. A finder that cannot read this section reports its absence rather than
assuming the set is empty.

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
31 passed, 1 xfailed in 0.26s
```

This block first recorded `29 passed`. Re-running the same command against the same
unchanged tree during the verification phase produced 31, so the number was written from a
run taken before the last two properties were added and never refreshed. Corrected here and
filed as BQ-F26; the audit that caught it re-ran every command in this document, and this was
the only observation that did not reproduce.

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

## Phase B: verification

## BQ.10 Every finding faces an independent verifier

[../../../.claude/skills/bug-bash/SKILL.md](../../../.claude/skills/bug-bash/SKILL.md) sets the
budget by severity: two recorded verdicts for a critical or high finding and a third on
disagreement, one for a medium, none for a low. Zero verdicts is `unverified`, never
confirmed, so every row in the ledger above entered this phase carrying no standing at all.

Verifiers were given the claim and told to refute it, one finding each, with no sight of any
other verifier's reasoning. The verdict lines below are recorded verbatim, refutations
included: a campaign that deletes what it disproved teaches the next campaign nothing, and the
same claim comes back.

```
uv run --no-sync pytest tests/bench -q -rxX
python3 -m bench compare <baseline> <candidate> --noise <floor>
```

Observed: 24 verdicts across 17 findings. The full text of each is in the ledger's Verdict
column. What the phase established, in order of how much it changed:

- **Nine of seventeen findings were refuted as filed.** In seven of those the mechanism held
  and the description did not: wrong arithmetic, an invented consequence, a severity resting
  on a false universal, or a cited file that does not contain the sentence attributed to it.
  A refuted finding whose mechanism survives is re-stated and re-verified, not quietly kept.
- **One refutation is worse news than the finding it refuted.** BQ-F03 was filed as "every
  floor lookup misses, so every metric reports inconclusive". Both verifiers refuted it and
  both found the same stronger chain: `quick` and `publish` share the point key
  `prefill/p128/g1/c1`, and `prefill/slope` is shared by construction, so a mismatched floor
  is partly *applied* rather than wholly ignored. A verifier reproduced a real +40% time to
  first token regression printing as `within_noise` against a foreign floor, with exit 0 and
  an empty warnings list.
- **One upstream premise was settled directly rather than deferred to the host.** A verifier
  fetched llama.cpp's server context translation unit at the pinned commit through the proxy
  and read `process_token` itself. The result corrected the finding: the pending stop-string
  branch does *not* skip the send, and is unreachable from this suite anyway because
  `bench/adapters.py::build_request` sends no `stop` field. The reachable branch is the
  incomplete UTF-8 one, where no frame arrives at all.
- **One finding cited the wrong file.** BQ-F19 attributed to `../../../bench/README.md` a claim
  that vLLM reports no server timings. That file says no such thing; the sentence is in
  `../../../bench/adapters.py`. The same finding asserted the suite documents queue time as
  impossible to derive client-side, and no such sentence exists anywhere in the tree.

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
| BQ-F21 | `bench/commands/compare.py::cmd` | medium | The three report reads that open a comparison sit outside any handler. A missing file, a truncated document, or one that fails envelope validation leaves the command with a traceback rather than a diagnosed error and an exit code. The reader is proven to raise; nothing proves its only caller survives it. |
| BQ-F22 | `bench/commands/compare.py`, `bench/commands/noise.py`, `bench/__main__.py` | medium | No test imports any of the three. The noise floor is the mechanism every `within_noise` verdict rests on, and the command that measures it has never been executed by the suite; it is also the second call site of the assembly function, so a defect reachable only from there is invisible. |
| BQ-F23 | `tests/bench/fixtures/` | low | Two of the seven wire fixtures written in the previous batch, the llama.cpp partial frame and the vLLM final frame carrying per-request metrics, are loaded by no test. A fixture nothing reads asserts nothing, and the second is the shape BQ-F19 turns on. |
| BQ-F24 | `bench/commands/run.py::cmd`, `bench/delta.py::check_comparable` | critical | The A/B arms are not interleaved. `cmd` runs one arm's entire profile to completion, writes it, then recreates the service and runs the next, so drift over the run aliases wholly onto the second arm -- the exact failure the design exists to prevent, and which the subcommand's own description promises it prevents: "With two --arm flags the arms are interleaved at the repeat level, so drift over the run cannot alias onto one of them." Interleaving is implemented at the repeat level within one arm, never between arms. The guard that would catch it is defeated by the same gap: the comparison warns when the baseline's `run.order` is not `interleaved`, and that field is copied from the profile, which says `interleaved` in both profiles on disk regardless of how the arms were sequenced. So the one warning that exists can never fire. Surfaced by a verifier working BQ-F08 and confirmed by reading; filed here rather than folded into that finding because it is a different defect in a different file. |
| BQ-F25 | `bench/delta.py::floor_from_pair` | medium | The floor builder differences every metric in the mapping and takes the largest absolute change as `worst_pct`, with none of the filtering the comparison command applies to its own table. An observed token count, which is not a performance metric, can therefore set the published worst spread. `bench.md` B2 reads `worst_pct` as an acceptance signal and `bench/commands/noise.py` warns on it against the five percent threshold, so a tokenization artifact can declare the rig unable to resolve a real change. Surfaced by a verifier working BQ-F04, whose own claim it refuted. |
| BQ-F26 | `docs/verification/bench/bench-qa.md` | low | The observed block for BQ.7 recorded `29 passed, 1 xfailed` for a command that produces `31 passed, 1 xfailed` against the same unchanged tree: a figure captured before the last two properties were written and never refreshed. Recorded evidence that does not reproduce is worse than none, because the entry is marked verified on the strength of it. Every command in this document was re-run during verification; this was the only one that did not reproduce. Corrected in this batch. |
| BQ-F27 | `bench/registry.py::_validate` | medium | The guard that refuses a registry entry whose stated quantization contradicts its filename is skipped for the only model registered. The shared filename parser accepts a suffix only if it starts with `Q` or names a float type, so every IQ-family and TQ-family llama.cpp quantization parses as unknown and the check treats unknown as nothing to verify. The shipped entry is an IQ4_XS file, so the guard is inert on exactly the case it exists for, and the unchecked value flows into the report and into the fields a comparison refuses to cross. The suite covers the guard with a `Q4_K_M` path, proving it on a shape the registry does not contain. |
| BQ-F28 | `bench/registry.py::resolve` | high | Resolving a model checks only that a hand-maintained list in the same file names the service. Nothing reads the model file, and nothing compares the registry's claim against what the service is actually serving, at resolve time or after. The report then carries both: `conditions.model` from the registry and `conditions.env_knobs` plus `server_argv` from the running container, with nothing reconciling them and no warning when they disagree. The comparison refuses on the asserted fields and ignores the observed ones. The precondition is reachable, because the training commands swap models on these same services. This is the defect the package's own conditions module names as its reason to exist: a condition recorded from configuration rather than from the running system. |
| BQ-F29 | `bench/commands/models.py` | medium | The machine-readable listing cannot answer the question the human listing answers. The JSON branch dumps the model block a report carries, whose key set is fixed elsewhere and contains no `engines`, so a consumer cannot tell which services a model resolves for and must scrape the human lines from standard error instead. `bench.md` B3 states its acceptance in terms of `engines` while piping the JSON form. |
| BQ-F30 | `bench/registry.py::_validate` | medium | Validation checks field presence and two enumerations and nothing about type, shape, or internal consistency. A bare string for the engine list becomes a per-character tuple that the listing prints one letter at a time; an empty list makes an entry advertised but unresolvable for every service; an engine name that is not a service is accepted, though a test exists to assert otherwise; and active parameters exceeding total, active experts exceeding total, a negative parameter count and a string context length all reach the report unchallenged. |
| BQ-F31 | `bench/commands/models.py::cmd` | low | The command catches only the registry error type, so a registry entry with a wrong-typed field raises out of the loader as a type error, past the handler, and reaches the user as a traceback with no exit code from the command — which the Python output style names among the things to avoid. The same handler shape appears in the run and noise commands. |
| BQ-F32 | `bench/cli.py` | low | The top-level description, the first thing the help output prints, names the rust harness among the benchmarked services. The same parser's service choices refuse it, deliberately and with a documented reason. |
| BQ-F33 | `bench/workloads.py::build_prompt`, `bench/commands/noise.py` | high | The noise floor is measured against a warm prefix cache. The prompt nonce is built from the profile seed, the cell index and the slot, all deterministic, and the noise command runs both arms in one process against one live service with no recreate between them, because the arm helper returns early on an empty environment. The second arm therefore replays the first arm's prompts byte for byte -- 76 of 76 identical in the reproduction -- and vLLM's prefix caching is on by default and not switchable per request, which the adapter names as the single reason the runner varies the prefix at all. The floor is the difference between a cold arm and a cached one, and every `within_noise` verdict rests on it. |
| BQ-F34 | `bench/workloads.py::load`, `bench/delta.py` | high | No profile hash is computed anywhere in the tree, though the module docstring says a workload is reproducible from one and the benchmarking skill lists a profile hash among the mismatches a comparison refuses. What the comparison actually blocks on is the profile's declared name, a string the file asserts about itself. Two profiles with different sweeps and the same declared name compare with no warning; unmatched points are dropped in silence, while the prefill slope is keyed on the phase alone and is differenced across two fits through different sweeps. |
| BQ-F35 | `bench/workloads.py::Profile.cells` | medium | The ordering field selects round-robin only on an exact string match and is never enumerated, so a typo, a capitalisation or a synonym silently falls through to the blocked ordering the protocol exists to forbid. Mutating the selector to accept any string beginning with the right letter leaves the whole suite green. The comparison warns only on the baseline arm's recorded order, never the candidate's, and a warning is not a refusal. |
| BQ-F36 | `bench/workloads.py::points` | medium | An empty concurrency list, or an empty decode generation list, is a value rather than an absence, so it passes the presence guard and produces a report with one arm missing. The refusal message for the missing key directs the operator to write a list even if it is a single entry, and an empty list is a list. The run exits 0 with a schema-valid document carrying half the measurement. |
| BQ-F37 | `bench/workloads.py::load` | medium | Per-service overrides are resolved with fallback lookups on both the service key and the field name, so a misspelling of either matches nothing and silently falls back to the base sweep. The shipped override exists because an eight-thousand-token prefill on CPU takes minutes; a typo in it widens that sweep inside a serialized measurement window with no error and no warning. An override naming a service that does not exist is accepted as dead configuration. |
| BQ-F38 | `bench/workloads.py`, `bench/runner.py::reduce` | medium | The character-per-token constant carries a comment saying it is only used to slice a candidate and that the length reaching the report is always the server's own count. It is not: the estimate is written as each point's identity and is the x-axis the prefill slope is fitted through, while the server's own count is reduced beside it and never used. The slope is therefore a rate in estimated tokens while the server-sourced prefill rate in the same document is in real ones, and both are graded as throughput. |
| BQ-F39 | `bench/workloads.py::load` | medium | Nine required fields are bare subscripts and the only read error caught is a missing file, so every other malformation -- a missing key, malformed TOML, a scalar where a list belongs -- escapes the command's handler and reaches the user as a traceback with exit 1, which the CLI documents as failure or regression rather than invalid usage. A string where a repeat count belongs raises nothing at all. |
| BQ-F40 | `bench/workloads.py::load` | medium | No field is range- or type-checked, so three plausible values load cleanly and then raise uncaught from inside the run loop -- after the health gate has passed and, for an A/B arm, after a container recreate that on a thirty-billion-parameter model is a full reload. A repeat count of zero loads, produces only warm-up cells, and the run command misdiagnoses the empty result as every cell having failed. Negative counts and a scalar where the decode prompt length belongs reach the report as degenerate point identities. |
| BQ-F41 | `bench/profiles/publish.toml`, `bench/README.md` | low | Both state the publish profile covers four services. The concurrency table twenty-five lines below enumerates three, the adapter map has three, and a fourth would be refused. Not covered by the documentation row already in this ledger, which named different files. |
| BQ-F42 | `bench/workloads.py::build_prompt` | low | Two paths return a prompt that is only the nonce, against a comment promising a short region is repeated rather than silently yielding a shorter prompt: an empty region repeats to nothing, and a target small enough that the nonce fills the budget makes the length guard compare against a negative number. Reachable only through the unvalidated prompt length above, so filed low. |
| BQ-F43 | `bench/commands/run.py::apply_arm` | high | An arm recreates the service with its environment applied and nothing ever restores it. There is no restore call, no `finally`, and no closing hint, though three training commands in this repository do all three and the serving module's docstring carries the manual one-liner for exactly this case. The service is left in the last-attempted arm's configuration, including when that arm failed and produced no result, and every later consumer inherits it silently. Running `bench.md` B7 as written leaves the CPU service at twelve threads, the configuration the optimizations index marks superseded, until a human recreates it. |
| BQ-F44 | `bench/commands/noise.py::cmd` | high | The floor document is assembled by hand from a header plus four fields, dropping the errors list, the conditions and both arms' phases. A pair in which cells failed therefore yields a floor covering fewer metrics than the workload, with a worst spread taken over the survivors, and nothing in the artifact says so. Reproduced: an arm losing eight of nine cells wrote a floor of four entries where a clean run writes fifteen, worst spread zero, exit 0 -- and feeding that floor to a comparison returned a green all-clear on a genuine thirty-three percent decode regression. The health gate is also called once for the pair, so a service degrading between the two arms is never re-checked. |
| BQ-F45 | `bench/commands/compare.py::cmd` | high | The command never asserts that it compared anything. Points absent from the baseline are skipped, the regression count is summed over whatever survived, and zero regressions prints the same green line whether every metric passed or none was examined. Reproduced two ways: a baseline that lost its decode phase against a candidate fifty percent slower on decode printed no decode line and exited 0, and two noise documents compared to each other graded exactly zero metrics and still printed no regression. `bench.md` B6 reads that exit code as its acceptance signal. |
| BQ-F46 | `bench/commands/compare.py::cmd` | high | A presentation flag decides the exit status. The headline filter is applied before both the table and the regression count, so a metric outside that set is graded correctly against the measured floor and then discarded. Inter-token latency is not in the headline set, though the CLI names it as half the package's purpose. Reproduced: the same two files and the same floor give exit 0 without the flag and exit 1 with it, on a forty percent inter-token regression the tool had already classified. Neither B6 nor B7 passes the flag. |
| BQ-F47 | `bench/commands/compare.py::cmd` | medium | The report reader validates the envelope and never reads the type discriminator, so any envelope-conformant document is accepted as a noise floor. A result file passed there yields an empty floor and every metric grades as unmeasured -- and because the missing-floor warning is guarded on the argument being absent rather than useless, naming the wrong file is quieter than naming no file at all. Distinct from the workload-matching finding, which is inside the comparison, and from the unguarded read, which is about a failure: here the read succeeds on a valid document of the wrong kind. |
| BQ-F48 | `bench/commands/run.py::cmd` | medium | The JSON dump sits inside the per-arm loop, so a two-arm run emits two top-level documents back to back on standard output against a flag documented as writing the report. The stream does not parse. The package distinguishes line-delimited output explicitly elsewhere, so concatenation is not the declared format, and the runbook establishes the pipe-into-a-parser pattern for the sibling command. |
| BQ-F49 | `bench/commands/run.py` | medium | The output root is knowable at parse time and is never checked until the first write, after the service has been reconfigured, the health gate passed and the whole workload run. It defaults to a path this repository records as absent anywhere but the workstation, and the comment beside it acknowledges as much by offering an override. Filed separately from the handler-narrowness finding because it fires on the happy path where nothing else went wrong, and a widened handler would diagnose it but not prevent it; fold the two together if triage judges the overlap too close. |
| BQ-F50 | `bench/commands/run.py`, `bench/delta.py` | medium | The identifier binding two arms of a pair has writers and no reader. It is absent from the fields a comparison refuses to cross, so when the second arm fails the first arm's file remains on disk carrying a pair identifier whose sibling does not exist, indistinguishable from a completed pair. Reproduced: that orphan compared against a different pair's arm of the same name graded fifteen metrics with no warning and exited 0. |
| BQ-F51 | `bench/commands/run.py::apply_arm` | medium | The recreate is skipped when an arm carries no overrides, so a bare-named arm measures a service that has been up for an unknown time -- warm page cache, settled clocks, and whatever configuration a previous run left behind -- while its partner measures a container recreated seconds earlier. The benchmarking skill documents the overrides as optional, so the invocation is advertised. Nothing in either artifact records that one arm was recreated and one was not. |
| BQ-F52 | `docker-compose.inference.yaml`, `docs/verification/bench/bench.md` | critical | Two of the host runbook's A/B entries measure nothing. An arm applies its overrides as the environment of the compose subprocess, which reaches a container only through interpolation. `THREADS` is written as the literal `16` in the compose file, so the thread-count entry's twelve-thread arm never reaches the service and both arms run at sixteen; `NO_REPACK` appears in no compose file at all, so both arms of the repack entry run the entrypoint default. Only the huge-page knob is interpolated and therefore only that one arm is real. The entry that would catch this is the same entry that cannot: it instructs the operator to confirm the override applied by reading the recorded cpuset, which says nothing about a thread count. The runbook would produce two clean-looking comparisons of a service against itself, and the twenty-six percent figure the thread entry exists to re-anchor would be re-anchored to a null result. Surfaced by a verifier working BQ-F43, whose own claim it refuted. |

### Verdicts

Severity below is the settled grade, not the proposed one. Where they differ, the verification
phase moved it and the reason is in the restatements underneath.

| ID | Verdicts | Outcome | Settled |
|----|----------|---------|---------|
| BQ-F01 | 1 refuted, 1 confirmed, 1 refuted | mechanism reproduced three times; the wording was wrong three times | high |
| BQ-F02 | 2 refuted | flag fires correctly on every reachable input; naming defect only | low |
| BQ-F03 | 2 refuted | gap real, stated consequence false, true chain is worse | re-file, see below |
| BQ-F04 | 2 refuted | one field pinned by construction, the other invisible without a flag | low |
| BQ-F05 | 2 confirmed | confirmed | high |
| BQ-F06 | 1 confirmed, 2 refuted | discard is the deliberate and correct choice | refuted |
| BQ-F07 | 2 confirmed | confirmed with a reproduction; the subprocess half of the mechanism is wrong | medium |
| BQ-F08 | 1 refuted | line order and the stamps both recover the interleaving | low |
| BQ-F09 | 1 refuted | intervals merge, they do not inflate; no reachable path | low |
| BQ-F10 | 1 refuted | no pinned engine can emit a non-numeric token count | low |
| BQ-F11 | 1 refuted | every route from a real `docker inspect` is already intercepted | low |
| BQ-F12 | 1 refuted | costs no reported number; the second query is forced by the tool | low |
| BQ-F13, BQ-F14, BQ-F15, BQ-F23 | 0, by policy | recorded without verification | low |
| BQ-F19 | 2 refuted on the documentary half, 1 confirmed on the upstream half | see below | high |
| BQ-F20 | 1 confirmed | confirmed | medium |
| BQ-F21 | 1 confirmed | confirmed, weaker than filed: an exit code is set | medium |
| BQ-F22 | 1 refuted | the premise is false; the modules are imported | low |
| BQ-F24 | 1 confirmed, 1 refuted | core confirmed and broader than filed; two legs of the filing wrong | high |
| BQ-F25, BQ-F26 | 0 | filed during verification, unverified | pending |

Nine findings were refuted as filed. In seven the mechanism survived and the description did
not, which is a different outcome from a finding that was simply wrong, and the restatements
below are what a remediator works from — not the original rows.

**BQ-F03 must be re-filed before it is fixed.** It was filed as "every lookup misses, so every
metric reports inconclusive". Both verifiers refuted that and both found the same worse chain
independently: `quick` and `publish` share the point key `prefill/p128/g1/c1`, and
`prefill/slope` is shared by construction, so a mismatched floor is partly *applied*. One
verifier reproduced a genuine +40% regression in time to first token printing as `within_noise`
against a foreign floor, exit 0, warnings empty. The filed severity was right and its stated
reason was wrong, which is the most dangerous shape a finding can have: a fix scoped to the
filed text would harden a path that does not fail.

**BQ-F19 splits cleanly and the halves disagree.** The documentary half is wrong as filed — it
attributes to `../../../bench/README.md` a claim that file does not make (the sentence is in
`../../../bench/adapters.py`), and asserts the suite documents queue time as underivable
client-side, which no file in the tree says. The upstream half was expected to need the
workstation and did not: a verifier fetched vLLM at the pinned tag and read
`PerRequestTimingMetrics` with its `queue_time_ms` field, attached to the streamed completion
response under `--enable-per-request-metrics`. The same reading found what the finding missed —
that flag is rejected when combined with `--disable-log-stats`, which
`../../../docker/vllm-gpu/entrypoint.sh` passes by default. Reaching the timings is two changes,
and the second re-enables statistics logging on a benchmark target, which is a
measurement-conditions decision rather than a flag flip.

**BQ-F01's engine premise was settled from source, not from this campaign's own fixtures.** Two
verifiers independently fetched llama.cpp at the pinned commit and read `process_token`. Both
found the filing's paraphrase wrong on one branch: a pending stop-string match does not skip the
send, it sends an empty string, and that branch is unreachable here because
`../../../bench/adapters.py` sends no stop field. The reachable branch is incomplete UTF-8,
where no frame is sent at all — so no change to the client can recover the arrival, and the
defect has to be met in the reduction. The filed figures were also wrong: they are the
arithmetic for a forty millisecond inter-token gap, and the reproduction they cite uses twenty.

**BQ-F01 took three verifiers to reach an accurate sentence, and the third refuted the
correction.** Every verifier reproduced the mechanism and both strict expected failures, and
two independently read the engine source. What kept failing was the description. The filed
figures were the arithmetic for a forty millisecond gap against a reproduction that uses
twenty. The correction offered to the tie-breaker then asserted that the mean time per output
token reads high at the ends; it reads low — 17.3 against a true 20.2 milliseconds, because the
span loses a gap from the numerator while the denominator stays at the server's token count —
which is what this document already said two hundred lines above, and what the correction
contradicted. The accurate statement, which is what remediation works from:

> At either end of the generation the decode rate reads high and the mean time per output token
> reads low. A leading withhold additionally puts time to first token one interval late; a
> trailing withhold additionally inflates the end-to-end rate. In the interior every aggregate
> is correct and one fabricated interval enters the latency series. Nothing rejects the sample,
> though `itl_approximate` and the chunk-packing field do fire, so the report is not silent.
> Recovery is impossible in `../../../bench/client.py` because the reachable engine branch sends
> no frame at all; detection fits in the client, the runner, or the reduction.

Severity settles at high rather than critical on two grounds the verifier named: the packing
flag already makes the condition visible in the report to a reader who looks, and reachability
is unproven here — the corpus statistic this campaign cited is about the *prompt* text, while
the trigger is a *generated* token whose bytes end mid-character. Raising it needs a host run.

**BQ-F24 was confirmed on its core and wrong on two legs.** Both verifiers confirmed the arms
run sequentially and that the promise of interleaving is false, and the second found the
overclaim repeated in two more places this campaign had not looked: the benchmarking skill,
which invents a design rationale for machinery that does not exist, and the noise command's own
description. But the filing said the comparison's ordering warning "can never fire", and that is
false — a profile with `order = "blocked"` loads unvalidated and does fire it, which the
property suite already exercises. The true statement is narrower: the warning cannot detect the
condition its own message describes, because arm-level sequencing is never recorded anywhere.
Severity settles at high rather than critical because every dependent runbook entry is still
pending, so no published number is invalidated — the defect is to be fixed before those entries
run, not after they have misled anyone.

**Two findings were refuted by verifiers who then found a real defect next to them.** BQ-F08's
verifier found that the A/B arms are not interleaved at all, which is BQ-F24 and is graver than
anything else in this ledger. BQ-F04's verifier found that the floor builder grades observed
token counts into `worst_pct`, which is BQ-F25.

**One refutation was of a premise this campaign took on trust.** BQ-F22 asserted that the
compare and noise command modules are imported by no test. They are: the parser test imports
`../../../bench/cli.py`, which imports both. The claim came from a survey that was not checked
before it was filed.

### Verdicts, the discovery top-up

| ID | Verdicts | Outcome | Settled |
|----|----------|---------|---------|
| BQ-F27 | 1 confirmed | confirmed; the fix belongs to the registry, not the shared parser | medium |
| BQ-F28 | 2 confirmed | confirmed, and reachable without any fault | high |
| BQ-F30 | 1 confirmed | confirmed; five of seven examples are already caught by the suite | medium |
| BQ-F33 | 1 confirmed | confirmed; scoped to vLLM, and the floor is too wide rather than too narrow | high |
| BQ-F34 | 2 confirmed | confirmed, with a fabricated verdict reproduced twice | high |
| BQ-F35, BQ-F36, BQ-F37 | 1 confirmed each | confirmed | medium |
| BQ-F43 | 2 refuted | the stated consequence is impossible; a worse one is not | refuted |
| BQ-F44, BQ-F45, BQ-F46 | 2 confirmed each | confirmed twice, both from documents the normal build path produces | high |
| BQ-F25 | 1 confirmed | confirmed; needs asymmetric cell loss to fire, and clearing five percent is unproven | medium |
| BQ-F29 | 1 confirmed | confirmed; the sharper claim that B3 cannot pass is refuted | medium |
| BQ-F47 | 1 confirmed | confirmed; naming the wrong file prints strictly less than naming none | medium |
| BQ-F48 | 1 confirmed | confirmed but latent: no caller in the tree combines two arms with the flag | low |
| BQ-F49 | 1 confirmed | confirmed, and distinct from the handler finding rather than a duplicate | medium |
| BQ-F50 | 1 confirmed | confirmed; blocking on the pair identifier would not break the runbook's control | medium |
| BQ-F51 | 1 confirmed | confirmed; the bias runs warm-arm-faster | medium |
| BQ-F38 | 1 confirmed | confirmed; corrupts absolute claims only, and cancels exactly in a delta | medium |
| BQ-F39, BQ-F40 | 1 confirmed each | confirmed; a hand-edited profile is the designed input, not an exotic one | medium |
| BQ-F52 | 0 | filed during this round by two verifiers refuting another claim | critical, unverified |

The prefill slope is reported in the wrong unit. The four-character-per-token estimate reaches
the report as each point's length and as the slope's x-axis, while the server's own count is
reduced beside it and used for nothing. The error is a pure rescale of the x-axis, so the
intercept and the goodness of fit are untouched and an A/B delta cancels it exactly -- verified
at three different scale factors, all giving the same delta to six decimal places. What it
corrupts is the absolute figure, which is the one the optimizations index publishes and the one
the benchmarking skill blesses for cross-engine comparison. Direction: too high whenever the
corpus runs longer than four characters per token, which this one does. The size needs the
served tokenizer and is a host question.

The confirmation rate is the difference between the two rounds, and the reason is evidence
rather than judgement. The first round's findings were filed from reading and nine of seventeen
were refuted as filed. The top-up's were filed with reproductions that had been executed, and
almost none were refuted -- the verifiers spent their effort correcting severity and scope
instead. Where a top-up finding did lose ground it was on an escalation rather than a
mechanism: the machine-readable model listing really does omit the field, but the runbook entry
that reads it also runs the human form, so that entry can pass after all.

What the round established beyond the individual rows:

**The refutation of BQ-F43 produced BQ-F52, which is the gravest finding in this campaign.** The
claim was that an arm's environment leaks and leaves the CPU service on twelve threads. It
cannot: `../../../docker-compose.inference.yaml` writes `THREADS` as a literal, so the compose
subprocess environment an arm sets never reaches the container, and the repack knob appears in
no compose file at all. Only the huge-page knob is interpolated. So two of the host runbook's
A/B entries would compare a service against itself and report it cleanly, and the entry meant
to re-anchor a twenty-six percent figure would re-anchor it to nothing. Both verifiers reached
this independently while refuting the finding they were given.

**BQ-F28's precondition is a documented flag, not a crash.** Both verifiers found that the
training commands restore in a `finally`, which was the counter-argument -- and both then found
that `--keep-serving` skips that restore deliberately on a successful run, and that `bench`'s
own arm mechanism accepts a model-path override that recreates the service on a different
model. One ran `check_comparable` on two reports whose containers had loaded different model
files and got no refusal and no warning. The fix is a string comparison over data already
collected: the registry's own comment says its paths are the in-container paths that match what
the compose file passes.

**BQ-F45's reachability was the open question and it is closed.** The refusal list inspects
conditions and the profile name and never the phases, while the reduction drops a phase whose
cells all failed. So an ordinary report missing its decode arm passes as comparable with zero
warnings, and a candidate fifty percent slower on decode prints no decode line and exits 0. No
hand-built document is needed.

**BQ-F33 is narrower and better characterised than filed.** It is vLLM-only: llama.cpp sends a
cache-defeating flag per request, and its server cache is per-slot rather than a hash-keyed
pool, so a replay from the start of the sequence cannot hit. The filed causal clause was also
wrong -- the noise command does not call the arm helper at all, so there is no early return to
blame. And the direction matters: caching can only make the second arm faster, so the floor
comes out too wide, which suppresses real findings rather than fabricating them.

## What this campaign does not cover

- **Numbers.** No entry here produces a latency, a throughput, or a noise floor. Those are
  `bench.md` and require the workstation.
- **The remediation.** Findings above are recorded and, where a reproduction is possible in
  a sandbox, reproduced. Fixing them is the next batch, verified first.
- **Amendments to `bench.md`.** BQ-F03 and BQ-F05 both name acceptance criteria in that
  runbook that are wrong today. The corrections describe verified fixes, so they follow the
  remediation rather than lead it.
