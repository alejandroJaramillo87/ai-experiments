# Measurement method

What makes a benchmark number citable, independent of which optimization produced it or when.

The protocol this repository runs — entry points, warm-up policy, the prefill and decode
split, and the regression criterion — is owned by
[../../.claude/skills/benchmarking/SKILL.md](../../.claude/skills/benchmarking/SKILL.md).
This document owns the reasoning underneath it: what a delta is a delta of, how many of them
each optimization needs, and why a number without its reference state cannot be quoted.

## A measurement is of a change, not of a layer

"Benchmark the BIOS" is not an operation that can be performed. The operation that can be
performed is "measure the delta from BIOS change X, with every other layer held at state S",
and the result is valid only at S.

This is why the archived 26.2 percent figure cannot be re-anchored. It was measured at 12
threads and `cpuset 0-11`; the service now runs 16 threads and `cpuset 0-15`. The number is
the answer to a question about a configuration that no longer exists.
[results.md](results.md) records that figure and the three further grounds on which it is
disqualified.

Every delta therefore names its reference state, and the reference state is a committed
artifact rather than a description.

## Two deltas per optimization

Each optimization is measured twice, and the two numbers differ.

- **Baseline delta.** One knob changed from the reference, everything else at reference.
  Answers what the optimization adds to nothing.
- **Ablation delta.** One knob turned off from the final configuration, everything else at
  final. Answers what the optimization adds to everything.

The ablation delta is the publishable number, because it is measured at the configuration
that actually runs. The gap between the two is the interaction, quantified. An optimization
worth 12 percent from the reference and 4 percent from the final configuration has had most
of its benefit captured by something else in the stack, and identifying that something is a
result in its own right.

`bench compare` performs both; only the reference file differs.

## A number carries its conditions or it is not a result

The reference state is not prose in a commit message. Every report `bench run` writes stamps
its own conditions — the model and its architecture, the quantization, the thread count, the
cpusets, the huge-page counters, the driver and CUDA versions, and the service's environment
and actual argv — and `bench compare` refuses a pair whose conditions disagree rather than
reporting a delta across the mismatch.

That refusal is the mechanism this document describes, enforced. An older result that
predates it cannot be repaired after the fact, because the conditions were never captured;
it can only be re-measured. [results.md](results.md) is the worked example.

## Related documentation

- [../../bench/README.md](../../bench/README.md) — the suite that produces the numbers, its
  module layout and its traps
- [../optimizations/roadmap.md](../optimizations/roadmap.md) — the order in which this
  repository's optimizations are measured, and why that order
- [../verification/bench/bench.md](../verification/bench/bench.md) — the host runbook that
  proves the suite itself measures what it claims
