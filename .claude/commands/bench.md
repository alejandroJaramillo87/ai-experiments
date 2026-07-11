---
description: Check preconditions, run the benchmark suite for a service, and return the results table
argument-hint: <service>
agent: bench-runner
allowed-tools: Bash(make status), Read, Grep, Glob
---
Service under test: $ARGUMENTS

Service health before benchmarking:

!`make status`

Delegate to the bench-runner subagent. If the service above is not healthy, stop and report
that instead of benchmarking. Otherwise follow the `benchmarking` skill's protocol: discard
warm-up, take multiple samples at a fixed workload, and return a results table with median +
spread and the exact command used. This is the closing artifact of a performance-relevant
task.
