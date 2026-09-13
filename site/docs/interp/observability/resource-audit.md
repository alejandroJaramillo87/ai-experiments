---
title: Resource audit
---

# Resource audit

The resource audit is the finest-grained record a verb can leave: every file it opened, every
module it imported, and every process it spawned, as debug-level events in the run's event
file. It exists for the forensic question, what did this run touch, and it is off by default
because answering it grows the event file with every file the interpreter opens.

## Turning it on

```bash
OBS_TRACE=1 make probe-train ARGS="--capture /mnt/ai-data/interp/activations/refusal-v1 --name refusal-v1"
```

`OBS_TRACE=1` does two things. It installs the audit hook, and it raises the log level to
`debug` unless `OBS_LOG_LEVEL` is set explicitly, because the audit records are debug-level
and would otherwise be filtered out of the file the flag was set to populate. The hook is an
interpreter audit hook, registered once for the life of the process and consulted on every
event, so a run without the flag pays one attribute check per event and nothing more.

## What it records

| Event | Fields | Meaning |
|-------|--------|---------|
| `open` | `path`, `mode` | A file opened; `mode` distinguishes a read from a write |
| `import` | `module` | A module imported, including the lazy imports a verb's handler performs |
| `spawn` | `argv0`, `argv` | A subprocess started, with its full argument list |

The records go to the event file only, never to the console. They are joined to the rest of
the run by the run id like every other event, so a question such as which files a run wrote
is one `jq` filter:

```bash
jq -r 'select(.event == "open" and (.mode | test("w|a"))) | .path' \
    "$OBS_LOG_DIR"/interp/R-20260806-142530-a3f19c.jsonl | sort -u
```

## What it costs

Every open, import, and spawn becomes a serialized record, so a verb that touches many files,
a harvest writing thousands of shards or a capture reading a dataset, produces an event file
proportional to that count. The audit changes nothing the verb computes; it observes the
interpreter and does not sit in any data path.
