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
2. Establish the noise floor. Unless a recent floor for this service and profile is already in
   hand, the first command you hand over is
   `python3 -m bench noise --service $ARGUMENTS --model <slug> --profile <profile>`.
   Without it a delta cannot be told from the machine's own variation, and on this
   workstation that variation has been measured at up to 44%.
3. Prepare the run. Write the exact invocation:
   `python3 -m bench run --service $ARGUMENTS --model <slug> --profile quick|publish`,
   plus `--arm name:KEY=VALUE` for each arm of an A/B. Name the model slug from
   `schemas/bench_models.json` — a model absent from the registry cannot be benchmarked, and
   a result whose architecture is unrecorded is not publishable. For a CPU service, state the
   `taskset -c 8-15` prefix and why: `llama-cpu` holds `cpuset 0-15`, so a client pinned
   inside that range contends with the server it is measuring. State what output would
   constitute a valid run: a schema-valid result file, `n` equal to the profile's repeats,
   and non-null medians.
4. Hand it over and stop. Print the commands, then wait for the human to run them and paste
   the output back. Never claim to have run them; never fabricate a number.
5. Reduce. The tool already reports median and spread, so there is nothing to reduce by hand.
   For an A/B, the next command is
   `python3 -m bench compare <baseline.json> <candidate.json> --noise <noise.json>`.
   You may delegate interpretation of pasted output to the bench-runner subagent; it must
   never be asked to obtain the output itself.
6. Report. Return the results table, the exact commands used, and the health evidence from
   step 1. Quote prefill and decode separately — `decode_tok_s` excludes prefill,
   `e2e_tok_s` does not, and only the former is a decode rate. For an MoE, quote
   `params_active_b` alongside `params_total_b`. This is the closing artifact of a
   performance-relevant task.

Comparative claims must compare at matched quantization and matched context length; `bench
compare` refuses the mismatch rather than reporting a delta across it. If the run is compared
against a stored baseline, name the baseline and its date.

Report `within_noise` and `inconclusive` verdicts as the results they are. A delta the rig
cannot resolve is not a small win, and a delta above 5% with no measured floor is not a win
at all — it is an unanswered question about the measurement.
