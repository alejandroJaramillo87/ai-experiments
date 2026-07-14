---
description: Check preconditions, run the benchmark suite for a service, and return the results table
argument-hint: <service>
agent: bench-runner
allowed-tools: Read, Grep, Glob
---
Service under test: $ARGUMENTS

Sandboxed execution: this session has no access to the host or its services, so health
cannot be checked and benchmarks cannot be run from here. First ask the human to run
`make status` on the host and paste the output; if the service is not healthy, stop and
report that instead of benchmarking.

Then delegate to the bench-runner subagent, which follows the `benchmarking` skill's
protocol: prepare the exact benchmark command for the human to run (discard warm-up,
multiple samples at a fixed workload), wait for the reported output, and return a results
table with median + spread and the exact command used. This is the closing artifact of a
performance-relevant task.
