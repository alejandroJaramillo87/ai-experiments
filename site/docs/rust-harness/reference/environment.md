---
title: Environment variables
---

# Environment variables

The binaries read four environment variables. None is required.

| Variable | Read by | Effect | Default |
|----------|---------|--------|---------|
| `OBS_LOG_DIR` | `harness`, `ladder`, `activation-capture`, `attention-capture`, `layer-diff` | The root the run records are written under; the binaries write into its `harness` subdirectory | `/mnt/ai-data/logs` |
| `OBS_RUN_ID` | The same five | When set by a parent process, recorded as `parent_run_id` on the run's `cmd.start` event; the run still mints its own id | unset |
| `NO_COLOR` | Every binary | When set to anything, the empty string included, status lines carry no color escapes | unset |
| `HARNESS_PYTHON` | The workspace's own test suite | The Python interpreter the cross-language contract tests run the Python exporters and readers with; searched before `.venv/bin/python3` in the repository and `python3` on the path. No binary reads it at run time | unset |

Setting `OBS_LOG_DIR` is the one of these an ordinary run is likely to need:

```bash
export OBS_LOG_DIR=/tmp/harness-demo/obs
```

If the root cannot be created or written, recording degrades to a silent no-op and the run
proceeds. `harness --status` reports which case applies as its last line.

Color is display only. A status tag such as `[OK]` or `[WARN]` is colored when stderr is a
terminal and `NO_COLOR` is unset, and plain otherwise; the color is on the bracketed tag alone,
never on the message. The plain output at a terminal under `NO_COLOR` and the output through a
pipe are byte-identical, so a `grep` over redirected output matches the same lines a person
reads.

## Files read from the working directory

These are not environment variables but they configure a run in the same way, by presence:

| File | Read by | Effect |
|------|---------|--------|
| `permissions.yaml` | `harness` in task mode | The tool permission rules; absent means every call asks. See [permissions.yaml](permissions-yaml.md) |
| `HARNESS.md` | `harness` in task and chat mode | Persistent memory, rendered as a leading turn of the prompt and never written to the transcript |
| `session.jsonl` | `harness` in task and chat mode | The transcript, appended and resumed |
| `session.kv` | `harness` in task and chat mode | The KV snapshot, restored at start and written on clean exit |

The working directory is the directory the harness was started in.
