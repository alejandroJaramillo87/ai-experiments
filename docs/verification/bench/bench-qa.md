# Benchmark suite quality campaign

**Operational runbook. Executed in an agent sandbox rather than against the host, so it
needs no GPU, no Docker daemon, and no model data. Keep the commands runnable, and the
paths accurate — the documentation gates check this directory, so a stale reference here
fails the gate.**

**Entries: 14 verified, 0 blocked, 0 pending, of 14.**

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
exclusion list. Every defect already filed in the ledger below is out of scope for
any further finder: it is recorded, and re-filing it wastes a verification slot on a claim
that already has one. Read the table rather than any range quoted in prose: this sentence first named BQ-F01 through BQ-F23 and was stale before the campaign that wrote it ended.
A finder that cannot read this section reports its absence rather than assuming the set is empty.

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

### The oracle figures are pre-remediation

Every observed block under Phase A records what the command produced **before** any fix
landed, which is what made those entries a baseline. Remediation has since changed the tree,
so several no longer reproduce, and an entry marked verified on a figure that no longer
reproduces is the defect this campaign filed against itself as BQ-F26. Rather than rewrite
observed evidence, both readings are kept:

| Entry | Recorded, pre-remediation | Reproduces today |
|-------|---------------------------|------------------|
| BQ.2 | `222 passed, 4 xfailed`; coverage 84.45% | `269 passed, 1 xfailed`; coverage 89% |
| BQ.3 | `21 passed` | unchanged |
| BQ.4 | `15 passed, 2 xfailed` | `20 passed`; the two expected failures were fixed and their tests rewritten |
| BQ.6 | `16 passed, 1 xfailed` | unchanged; the remaining expected failure is BQ-F11, whose package has not run |
| BQ.7 | `29 passed, 1 xfailed`, corrected to 31 | `35 passed` |
| BQ.8 | 91 of 98 mutants killed | unmeasured: both mutated modules changed, and the harness cannot run while a remediator holds them |
| BQ.9 | two expected failures quoted verbatim | those tests no longer exist; the defect they recorded is fixed |

Three of the four strict expected failures have flipped and lost their decorators, which is
what they were written strict for. The fourth belongs to a package that has not been worked.

### One commit in this batch was red

`beb9c12` claims in its own message to have been committed green. It was not. The gate and
the suite were both run and both passed immediately before it, but a remediator was still
writing, and the test rewrite for BQ-F01 landed between the gate and the commit. That commit
therefore asserts a property its own reduction module does not yet have -- the property
arrives one commit later -- and six tests fail at it. The tree is not bisectable across this
batch, and the red-before evidence for that finding is not what the commit sequence shows,
though it was genuinely observed and is recorded above.

The lesson is narrow and worth keeping: running the gate before committing does not protect
against a writer that is still running. Wait for the agent, then gate, then commit.

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

## BQ.11 Every arm knob the CLI advertises reaches the container

An arm applies its overrides as the environment of the `docker compose` subprocess, and
that environment reaches a service only through interpolation in the compose file. A knob
written there as a literal is inert: both arms run one configuration and the comparison
reports a clean null on an A/B that measured nothing. BQ-F52 is exactly that, and it was
found only after two host runbook entries had been written against knobs that could not
move.

The check is textual on purpose. A loader that resolved interpolations would erase the
distinction under test.

```
uv run --no-sync pytest tests/bench/test_arm_knobs.py -q
```

Observed: `6 passed`. The pinned set is `HUGEPAGES`, `NO_REPACK`, `THREADS` and
`THREADS_BATCH` on `llama-cpu`, and `DISABLE_LOG_STATS` with
`ENABLE_PER_REQUEST_METRICS` on `vllm-gpu`. `NO_REPACK` appeared in no compose file and
`THREADS` was a literal; both are interpolated now, and adding a knob to `bench run`'s
documented set without adding it to the compose file fails here.

Status: VERIFIED 2026-08-30

---

## BQ.12 A run that collapses leaves what it measured

`bench run` buffered every sample in memory and flushed once at the end, and `assemble`
sat outside the error handler which caught only `ServingError` -- a type already handled
per cell, so it could never lose a run. Anything else arriving late in a run of up to an
hour left the process with a traceback and the disk with nothing.

The reproduction is a client that fails after eight cells with an exception the suite does
not anticipate.

```
uv run --no-sync pytest tests/bench/test_run_command.py -q -k "UnexpectedFailure or Interleaved"
```

Observed before the fix: the exception propagated out of `cmd`, no exit code, no
`samples.jsonl`, no `result.json`. After: `samples.jsonl` carries the five measured cells
of the first eight, `result.json` records `run.status == "partial"` and the exception in
`errors`, and the command exits 1. An interleaved pair whose third pass fails writes both
arms rather than discarding two completed repeats, and marks both partial -- a pair cut
short leaves neither arm covering the profile, whichever arm's pass failed.

Status: VERIFIED 2026-08-30

---

## BQ.13 The noise floor carries the pair it was measured from

The floor is the mechanism every `within_noise` verdict rests on, and it was assembled by
hand from a header plus four fields -- dropping the errors, the conditions, and both arms'
phases. An arm that lost most of its cells produced a narrower floor with a smaller worst
spread, identical in shape to a clean run.

```
uv run --no-sync pytest tests/bench/test_noise_command.py -q
```

Observed: `13 passed`. The reproduction is an arm refused at two of three points. Before
the fix it wrote `worst spread 0.0% across 3 metrics` against a clean run's 12, exit 0,
with nothing in the artifact recording the loss. After, the document carries both arms in
full, their conditions, the errors tagged by arm, and a coverage block; the command warns
`this floor covers 3 of 12 metrics ... (a: 0, a-prime: 8)`; and `bench compare` refuses
such a floor rather than applying it, because the metrics it does not carry grade as
`inconclusive`, which does not fail. The health gate also runs once per arm now rather
than once for the pair.

Status: VERIFIED 2026-08-30

---

## BQ.14 The oracles after remediation

The mutation harness and the coverage figure are the campaign's own instruments, and both
were unmeasured once the modules they scored had changed. Neither can run while a
remediator holds `bench/metrics.py` or `bench/delta.py`, because
`tests/bench/mutate.py` overwrites them in place.

```
uv run --no-sync pytest tests/bench -q
uv run --no-sync pytest tests/bench --cov=bench --cov-report=term-missing -q
python3 tests/bench/mutate.py
```

Observed: `409 passed`, zero expected failures -- the campaign's last strict reproduction
was cleared with its package. Coverage `96%` against the 89% recorded at BQ.2, with
`bench/metrics.py` and `bench/report.py` at 100%. Mutation `115 of 115 killed (100.0%)`
across both scored modules, against `91 of 98` at BQ.8; the seven survivors that baseline
recorded are gone, and the mutant count rose because both modules grew guards that are
themselves mutable.

A perfect mutation score is a statement about the harness's operator set, not about the
tests being complete: it scores two modules of sixteen, and no mutation of
`bench/commands/` is generated at all.

Status: VERIFIED 2026-08-30

---

## Findings ledger

Moved to [bench-qa-findings.md](bench-qa-findings.md), which carries every finding this
campaign raised, its settled severity, its disposition, and the verdicts it drew. The
entries above cite those rows by identifier.

## What this campaign does not cover

- **Numbers.** No entry here produces a latency, a throughput, or a noise floor. Those are
  `bench.md` and require the workstation.
- **The remediation.** Findings above are recorded and, where a reproduction is possible in
  a sandbox, reproduced. Fixing them is the next batch, verified first.
- **Amendments to `bench.md`.** BQ-F03 and BQ-F05 both name acceptance criteria in that
  runbook that are wrong today. The corrections describe verified fixes, so they follow the
  remediation rather than lead it.
