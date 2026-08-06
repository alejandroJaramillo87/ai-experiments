---
description: Check preconditions, run the benchmark suite for a service, and return the results table
argument-hint: <service>
allowed-tools: Read, Grep, Glob, Task
---
Service under test: $ARGUMENTS

Read `.claude/skills/benchmarking/SKILL.md` first; it is the protocol this command executes.

Sandboxed execution: this session has no access to the host or its services, so health cannot
be checked and benchmarks cannot be run from here. **You own the handoff to the human — do not
delegate it.** A subagent has no channel to the human: it cannot emit a command, pause, and
read the reply, so a subagent asked to do this can only report failure or invent output. Keep
every request-and-wait step in this conversation.

1. Preconditions. Ask the human to run `make status` **and** `make health` on the workstation
   and paste both outputs. `make status` is the compose-level summary; `make health` is the
   detailed per-service check, and the benchmarking skill requires both. If the service under
   test is not healthy, stop and report that instead of benchmarking — a benchmark against an
   unhealthy service is not a slow result, it is no result.
2. Prepare the command. Following the benchmarking protocol, write the exact benchmark
   invocation for `$ARGUMENTS`: fixed workload, warm-up discarded, multiple samples. State what
   output would constitute a valid run.
3. Hand it over and stop. Print the command, then wait for the human to run it and paste the
   output back. Never claim to have run it; never fabricate a number.
4. Analyse. Once the human has pasted real output, you may delegate the reduction to the
   bench-runner subagent — it receives the pasted text and returns median plus spread. It must
   never be asked to obtain the output itself.
5. Report. Return the results table with median + spread, the exact command used, and the
   health evidence from step 1. This is the closing artifact of a performance-relevant task.

Comparative claims must compare at matched quantization and matched context length. If the
run is compared against a stored baseline, name the baseline and its date.
