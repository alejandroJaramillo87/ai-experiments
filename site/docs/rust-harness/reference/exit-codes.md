---
title: Exit codes
---

# Exit codes

Every binary in the workspace uses the same three codes.

| Code | Meaning | Written to |
|------|---------|------------|
| 0 | Success, or `--help` and `--version` answered | help and version to stdout |
| 1 | The run started and failed, or a check the binary exists to make did not pass | an `[ERROR]` line on stderr, often with a `Solution:` and `Command:` line beneath it |
| 2 | Usage error: an unknown flag, a missing required flag, a flag without the flag it requires, or a value that does not parse | the error and the usage line on stderr |

## Per binary

| Binary | Exits 1 when |
|--------|--------------|
| `harness` | The model fails to load, an interp artifact fails its geometry or identity check, or the task, chat session, or server fails. Under `--status`, when any reported precondition would stop the run |
| `checkpoint-info` | The checkpoint cannot be read. With `--check`, also when the backend's verdict is unsupported; without `--check` an unsupported verdict is reported and exits 0 |
| `activation-capture`, `attention-capture` | The model fails to load, a layer or head is out of range, the dataset cannot be read, or the output cannot be written |
| `ladder` | Any rung fails, or no rung had a reference artifact and nothing was measured |
| `layer-diff` | The model or reference fails to load, or any row exceeds a tolerance |
| `long-context-golden`, `tiny-model` | The destination cannot be written |

A usage error exits before any run record is written. A failure after a run id exists writes a
`fail` event carrying the exit code and a `cmd.end` with the same code; see
[Run records](../observability/run-records.md).

## Status streams

Status lines go to stderr with a bracketed tag: `[INFO]`, `[OK]`, `[RUN]`, `[WARN]`,
`[SKIP]`, `[ERROR]`. The harness's answer, `--help`, `--version`, and `checkpoint-info`'s
report go to stdout. Redirecting stdout therefore captures the product of a run and not its
commentary.
