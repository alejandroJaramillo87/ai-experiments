---
title: First agent session
---

# First agent session

A task run is an agent session: the model is given six tools, calls them by replying with a
JSON object, and is answered with the tool's result until it replies with text instead. This
page runs one such session with the permission prompt on, then with a rules file, and shows
the transcript a completed session leaves. It assumes `BIN` and `MODEL` from [Build](build.md)
and an empty working directory. The agent loop needs an instruction-tuned checkpoint; the
outputs below are from `gemma-4-E2B-it`.

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

Tool-call formation is grammar-constrained on the task path: a model turn that opens with `{`
is held to exactly one call, so a second call, a fabricated result, or prose around a call
in one turn cannot reach the loop.

## Run with the permission prompt

With no rules file present, every tool call asks on the terminal before it runs:

```bash
"$BIN/harness" --model "$MODEL" --text-only --max-tokens 64 "List the files in this directory"
```

A call prints `allow bash ls? [y/N]` to stderr and reads one line from stdin. Anything other
than `y` or `yes` denies it, and a denied call is reported to the model as denied rather than
ending the session. A call that fails identically three times in a row aborts the run with
exit 1: `aborting: tool call read main.py failed identically 3 times`.

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
"$BIN/harness" --model "$MODEL" --text-only --max-tokens 64 "List the files in this directory"
```

The file's format, the pattern language, and the two limits of what a rule can confine are on
the [permissions.yaml reference page](../reference/permissions-yaml.md). A misspelled key or an
unknown tool name is refused at startup rather than ignored.

## A completed session

A rules file that allows every tool lets a session run to its end without a prompt. This one
writes a file and reads it back:

```bash
rm -rf /tmp/harness-demo/agent && mkdir -p /tmp/harness-demo/agent && cd /tmp/harness-demo/agent
printf 'rules:\n  - tool: "*"\n    action: allow\n' > permissions.yaml
"$BIN/harness" --model "$MODEL" --text-only \
    "Create a file named hello.txt containing the word hello, then read it back"
cat session.jsonl
cat hello.txt
```

```text
[INFO] Model load: 3.61 s
[INFO] Prefill: 261 tokens in 11.85 s (22 tok/s)
[INFO] Prefill: 18 tokens in 0.87 s (21 tok/s)
[INFO] Prefill: 11 tokens in 0.55 s (20 tok/s)
hello
[INFO] Session snapshot: 333 cached tokens at /tmp/harness-demo/agent/session.kv
```

```text
{"role":"user","content":"Create a file named hello.txt containing the word hello, then read it back"}
{"role":"model","content":"{\"tool\": \"write\", \"args\": {\"path\": \"hello.txt\", \"content\": \"hello\"}}"}
{"role":"tool","content":"wrote 5 bytes to hello.txt"}
{"role":"model","content":"{\"tool\": \"read\", \"args\": {\"path\": \"hello.txt\"}}"}
{"role":"tool","content":"hello"}
{"role":"model","content":"hello"}
hello
```

This is what a good session looks like: three model turns, one `write` call, one `read` call,
and one plain-text answer, each a whole turn with nothing beside it. Both `tool` entries were
written by the harness, not the model, and the last line is `hello.txt` on disk. The three
`Prefill:` lines are the three turns; the first carries the tool contract and the task, and
the later two only the tool results, because the KV cache held the rest.

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
the raw history survives on disk. See [Running an agent session](../guides/agent-session.md)
for the snapshot and restart behavior that goes with it.
