---
title: Running an agent session
---

# Running an agent session

The harness has three modes, chosen by the last argument: a positional task runs the agent
loop once, `--chat` opens a REPL, and `--serve <port>` answers HTTP requests. This page covers
what each does with a conversation over time: the transcript, the KV snapshot, compaction, and
the differences between the modes. It assumes `BIN` and `MODEL` from
[Build](../getting-started/build.md).

## Task mode

```bash
"$BIN/harness" --model "$MODEL" --max-tokens 64 "Summarize what is in this directory"
```

The agent loop renders the tool contract, the optional `HARNESS.md` memory, and the task as a
prompt, then alternates model turns and tool results until the model replies with text, which
is printed to stdout. A tool call the rules deny is answered with `permission denied` and the
loop continues; one the rules mark `ask` is put to the terminal. The loop is bounded at 64
steps. [First agent session](../getting-started/first-agent-session.md) covers the tools and
the rules file.

## Chat mode

```bash
"$BIN/harness" --model "$MODEL" --chat --max-tokens 64
```

Chat is a REPL over the same generator with no tools and no permissions. Each input line is a
user turn; the reply streams to stdout. Two lines are commands: `/quit` ends the session, and
`/compact` forces a compaction now. End of input also ends the session. Chat can be driven
from a pipe, which is how the workspace's own tests run it:

```bash
printf 'hello\n/quit\n' | "$BIN/harness" --model "$MODEL" --chat --max-tokens 8 --temperature 0
```

## The transcript

Task and chat append every turn to `session.jsonl` in the working directory, one JSON object
per line with a `role` and `content`, and both read it back at startup. Starting the harness
twice in the same directory continues the same conversation. To start a fresh one, start in a
fresh directory or remove the file. A crash mid-write leaves at most one torn last line, which
is dropped on read.

## The KV snapshot

On clean exit, task and chat save the KV cache to `session.kv` beside the transcript; on the
next start in the same directory the cache is restored and only the new turn is prefilled:

```text
[INFO] Session restored: 1178 cached tokens from /tmp/harness-demo/session.kv
```

The snapshot carries a fingerprint of the weights it was computed under. A snapshot from a
different checkpoint, or from a different version of the format, is refused loudly. A
snapshot whose tokens no longer match the transcript, because the transcript was edited or
compacted, degrades to a re-prefill from the point of divergence rather than failing.
Continuation logits after a restore are bit-identical to an unbroken run.

## Compaction

`--context <n>` caps the token budget below the checkpoint's trained window;
`max_position_embeddings` from the checkpoint is both the default and the maximum. When the
rendered prompt crosses `--compact-at` (default 0.8) of that budget, the harness asks the model
to summarize the middle of the conversation and replaces it with the summary. The replacement
is recorded in `session.jsonl` as a compaction record naming the replaced range, and the raw
turns stay in the file; readers fold the record in. The `Prefill:` status line reports what a
compaction cost, because a splice shortens the cached prefix and the harness re-prefills only
the summary and the tail.

```bash
"$BIN/harness" --model "$MODEL" --context 512 --compact-at 0.5 --chat --max-tokens 32
```

The synthetic checkpoint's sliding window is four tokens, so its rewind horizon is short and
the status lines about rebuilding appear early; the mechanism is the same on the real window.

## Serve mode

```bash
"$BIN/harness" --model "$MODEL" --serve 8005 --max-tokens 64
```

```text
[INFO] Serving on http://127.0.0.1:8005 (model /home/user/ai-experiments/rust-harness/fixtures/tiny-gemma4)
```

Two paths are served. `GET /v1/health` answers 200 with a JSON body once the model is loaded;
every client checks the status code and nothing else. `POST /v1/chat/completions` takes a JSON
body with `messages` (each a `role` of `system`, `user`, or `assistant` and a `content`) and
optionally `model`, `temperature`, `top_p`, `top_k`, `max_tokens`, and `seed`, which override
the command-line values for that request. It answers one whole completion in the OpenAI
shape: `id`, `object` of `chat.completion`, `created`, `model`, one choice with an assistant
message and a `finish_reason`, and a `usage` object with prompt, completion, and total tokens.
`stream: true` is refused rather than answered whole.

```bash
curl -s http://127.0.0.1:8005/v1/health
curl -s http://127.0.0.1:8005/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

Serve mode writes no transcript and no snapshot, and restores none: a server's requests are
unrelated callers. `--serve-addr <ip>` changes the bind address, which only a container needs
to set. `--interp-log <dir>` appends one record per completed request, covered under
[Loading a probe for live monitoring](probe-monitoring.md).

## What every mode shares

A probe, a control vector, the flight recorder, and the trace are installed on the model before
the mode starts, so all three modes observe and steer identically. Every mode writes the run
records under `${OBS_LOG_DIR}/harness/` described in [Run records](../observability/run-records.md).
