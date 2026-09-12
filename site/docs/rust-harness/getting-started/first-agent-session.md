---
title: First agent session
---

# First agent session

A task run is an agent session: the model is given six tools, calls them by replying with a
JSON object, and is answered with the tool's result until it replies with text instead. This
page runs one such session with the permission prompt on, then with a rules file. It assumes
`BIN` and `MODEL` from [Build](build.md).

Against the synthetic checkpoint the model never produces a well-formed tool call, so the
session ends after the first reply. The mechanics below are what a real checkpoint exercises;
the commands are the same.

## The tools

| Tool | Arguments | Permission target |
|------|-----------|-------------------|
| `read` | `path` | the normalized path |
| `write` | `path`, `content` | the normalized path |
| `edit` | `path`, `old`, `new` | the normalized path |
| `bash` | `command` | the command line |
| `glob` | `pattern` | the pattern |
| `todo` | `items` | none |

The working root of every tool is the directory the harness was started in. The `bash` tool
runs with that directory as its working directory and is not otherwise confined.

## Run with the permission prompt

With no rules file present, every tool call asks on the terminal before it runs:

```bash
"$BIN/harness" --model "$MODEL" --max-tokens 64 "List the files in this directory"
```

A call prints `allow bash ls? [y/N]` to stderr and reads one line from stdin. Anything other
than `y` or `yes` denies it, and a denied call is reported to the model as denied rather than
ending the session.

## Run with a rules file

The harness reads `permissions.yaml` from the working directory at startup. Each rule names a
tool, optionally a pattern over the permission target, and an action; the first matching rule
wins and no match means ask. Write one that allows reads and globs, asks about the rest, and
denies writes outside a scratch subdirectory:

```bash
cat > permissions.yaml <<'EOF'
rules:
  - tool: read
    action: allow
  - tool: glob
    action: allow
  - tool: write
    pattern: "scratch/**"
    action: allow
  - tool: write
    action: deny
  - tool: edit
    action: deny
  - tool: bash
    action: ask
EOF
"$BIN/harness" --model "$MODEL" --max-tokens 64 "List the files in this directory"
```

The file's format, the pattern language, and the two limits of what a rule can confine are on
the [permissions.yaml reference page](../reference/permissions-yaml.md). A misspelled key or an
unknown tool name is refused at startup rather than ignored.

## Persistent memory

If a file named `HARNESS.md` exists in the working directory, its contents are read once at
startup and rendered as a leading turn of the prompt. It is never written to the transcript,
so it is the place for standing instructions rather than for history:

```bash
printf '# Memory\n\nPrefer short answers. Never modify files under vendor/.\n' > HARNESS.md
```

## Context and compaction

Two flags bound a long session. `--context <n>` caps the token budget below the checkpoint's
trained window, and `--compact-at <f>` (default 0.8) is the fraction of that budget at which
the harness asks the model to summarize the middle of the conversation and splices the summary
in. Compaction is recorded in `session.jsonl` as its own record naming the replaced range, so
the raw history survives on disk. See [Agent session](../guides/agent-session.md) for the
snapshot and restart behavior that goes with it.
