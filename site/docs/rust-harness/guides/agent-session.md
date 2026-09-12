---
title: Running an agent session
---

# Running an agent session

The harness has three modes, chosen by the last argument: a positional task runs the agent
loop once, `--chat` opens a REPL, and `--serve <port>` answers HTTP requests. This page covers
what each does with a conversation over time: the transcript, the KV snapshot, compaction, and
the differences between the modes. It assumes `BIN` and `MODEL` from
[Build](../getting-started/build.md) and an empty working directory; the outputs are from
`gemma-4-E2B-it` except where a block names `gemma-4-E4B`.

## Task mode

```bash
"$BIN/harness" --model "$MODEL" --text-only --max-tokens 64 "Summarize what is in this directory"
```

The agent loop renders the tool contract, the optional `HARNESS.md` memory, and the task as a
prompt, then alternates model turns and tool results until the model replies with text, which
is printed to stdout. A tool call the rules deny is answered with `permission denied` and the
loop continues; one the rules mark `ask` is put to the terminal. The loop is bounded at 64
steps. [First agent session](../getting-started/first-agent-session.md) covers the tools, the
rules file, and what a completed transcript looks like.

## Chat mode

```bash
"$BIN/harness" --model "$MODEL" --text-only --chat --max-tokens 128
```

Chat is a REPL over the same generator with no tools, no permissions, and no tool contract in
the prompt, so a turn prefills a few dozen tokens where a task prefills a few hundred. Each
input line is a user turn; the reply streams to stdout. Two lines are commands: `/quit` ends
the session, and `/compact` forces a compaction now. End of input also ends the session, which
is what makes a chat scriptable from a pipe. Three turns, the first stating two facts and the
other two asking for one each back:

```bash
printf '%s\n' \
    "My name is Alejandro and my workstation has an RTX 5090." \
    "Which GPU did I say my workstation has?" \
    "Reply with just my name." \
  | "$BIN/harness" --model "$MODEL" --text-only --chat --max-tokens 128
```

```text
[INFO] Model load: 3.73 s
[INFO] Prefill: 25 tokens in 1.00 s (25 tok/s)
That sounds like a very powerful workstation, Alejandro! The **RTX 5090** is a high-end, next-generation graphics card, suggesting you're set up for demanding tasks like:

* **High-end Gaming:** Running the most demanding titles at ultra settings with high frame rates.
* **3D Rendering & Modeling:** Accelerating workflows in software like Blender, Maya, or 3ds Max.
* **AI/Machine Learning:** Handling complex deep learning models due to its massive parallel processing power.
* **Video Editing & Content Creation:** Smooth, real-time playback and rendering in
[INFO] Prefill: 19 tokens in 0.84 s (23 tok/s)
You said your workstation has an **RTX 5090**.
[INFO] Prefill: 15 tokens in 0.68 s (22 tok/s)
Alejandro
[INFO] Chat ended: 3 model turns
[INFO] Session snapshot: 210 cached tokens at /tmp/harness-demo/session.kv
```

The second and third replies name facts stated only in the first turn. The first reply ran to
the 128-token budget and is cut mid-sentence; the other two stopped at the tokenizer's
end-of-sequence token, which is how the instruction-tuned checkpoint ends a turn. Each
`Prefill:` line counts the new turn alone, not the transcript, because the KV cache held the
rest.

Chat mode is also the frame for establishing an interpretability effect, since a task run
prepends about 250 tokens of tool contract that a probe or a control vector never saw; see
[Loading a probe for live monitoring](probe-monitoring.md).

## The transcript

Task and chat append every turn to `session.jsonl` in the working directory, one JSON object
per line with a `role` and `content`, and both read it back at startup. The three-turn chat
above leaves six lines alternating `user` and `model`. Starting the harness twice in the same
directory continues the same conversation. To start a fresh one, start in a fresh directory or
remove the file. A crash mid-write leaves at most one torn last line, which is dropped on read.

A directory that already holds a transcript is the most common source of a run that looks
wrong and is not: the new prompt is answered in the old conversation's context, and the only
sign is the `Session restored` line and a prefill count far above the prompt's length. See
[Troubleshooting](../troubleshooting.md).

## The KV snapshot

On clean exit, task and chat save the KV cache to `session.kv` beside the transcript; on the
next start in the same directory the cache is restored and only the new turn is prefilled.
Two task runs in one directory on `gemma-4-E4B`:

```bash
"$BIN/harness" --model "$MODEL" --text-only --trace-chrome turn1.json \
    "Summarize the Unix philosophy in three sentences"
"$BIN/harness" --model "$MODEL" --text-only --trace-chrome turn2.json \
    "Now compress that summary to one sentence"
```

```text
[INFO] Session snapshot: 234 cached tokens at /tmp/harness-demo/session.kv

[INFO] Session restored: 234 cached tokens from /tmp/harness-demo/session.kv
[INFO] Model load: 5.96 s
[INFO] Prefill: 17 tokens in 1.45 s (12 tok/s)
[INFO] Session snapshot: 507 cached tokens at /tmp/harness-demo/session.kv
```

The second run restored 234 tokens and prefilled 17, the new user turn. The trace is the
evidence at the span level; from `rust-harness`, `make trace-check TRACE=turn2.json` prints,
among its table:

```text
[INFO] session.sync args {"cached_len": "234"}
[INFO] first model.forward args {"len": "17", "start": "234"}
```

The prefill forward started at position 234 with 17 tokens, not at 0 with 251. Prefill cost
scales with the delta; attention cost scales with the context, so the second turn's steps each
attend over 234 more keys.

The snapshot carries a fingerprint of the weights it was computed under. A snapshot from a
different checkpoint is refused with
`[WARN] Session snapshot ... discarded: snapshot was written for different model weights` and
the run starts with an empty cache; one from a different version of the format is refused the
same way. A snapshot whose tokens no longer match the transcript, because the transcript was
edited or compacted, degrades to a re-prefill from the point of divergence rather than failing.
Continuation logits after a restore are bit-identical to an unbroken run.

## Compaction

`--context <n>` caps the token budget below the checkpoint's trained window;
`max_position_embeddings` from the checkpoint is both the default and the maximum, and a
value above it fails the run naming both figures. When the rendered prompt crosses
`--compact-at` (default 0.8) of that budget, the harness asks the model to summarize the middle
of the conversation and replaces it with the summary. The replacement is recorded in
`session.jsonl` as a compaction record naming the replaced range, and the raw turns stay in the
file; readers fold the record in.

A compaction rewrites the middle of the transcript, so the new prompt shares only its head with
the cache, and on the real 512-token sliding window that rewind falls below the cache's floor:
the harness clears and re-prefills rather than rewinding, and the `Prefill:` line says so. Sixty
short turns on `gemma-4-E4B` with the budget capped at 2048 and the threshold at 0.4:

```bash
python3 -c 'w="matter energy motion field charge wave particle heat light sound".split()
for i in range(60): print(f"turn {i+1}: "+" ".join(w[(i+j)%len(w)] for j in range(30)))' \
  | "$BIN/harness" --model "$MODEL" --text-only --chat --context 2048 --compact-at 0.4 \
      --max-tokens 8 > replies.txt 2> run.log
grep 'Rewind horizon' run.log
jq -r .role session.jsonl | sort | uniq -c
grep 'rebuilt from 0' run.log
```

```text
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
      4 compaction
     60 model
     60 user
[INFO] Prefill: 724 tokens in 66.46 s (11 tok/s); rebuilt from 0: a rewind to 2 fell below the window floor at 656
[INFO] Prefill: 168 tokens in 13.92 s (12 tok/s); rebuilt from 0: a rewind to 2 fell below the window floor at 603
[INFO] Prefill: 751 tokens in 69.36 s (11 tok/s); rebuilt from 0: a rewind to 2 fell below the window floor at 683
[INFO] Prefill: 168 tokens in 13.92 s (12 tok/s); rebuilt from 0: a rewind to 2 fell below the window floor at 630
```

Four compactions in sixty turns, and two rebuilds per compaction: the summarization prompt,
which is the whole transcript, and then the compacted prompt. The line names the prefix the
rewind wanted and the floor it fell below. A rewind past the floor is a slower path, not an
error; a run that failed here would mean the floor was being enforced as a refusal.

## Serve mode

```bash
"$BIN/harness" --model "$MODEL" --text-only --serve 8005 --max-tokens 64
```

```text
[INFO] Model load: 3.71 s
[INFO] Serving on http://127.0.0.1:8005 (model /mnt/ai-data/models/hf/gemma-4-E2B-it)
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
curl -s http://127.0.0.1:8005/v1/chat/completions -H 'content-type: application/json' \
    -d '{"messages":[{"role":"user","content":"Name three primes."}],"temperature":0,"max_tokens":64}'
```

Two greedy requests for that prompt on `gemma-4-E2B-it` returned identical text,
`Here are three prime numbers:\n\n1. **2**\n2. **3**\n3. **5**`, with `usage` of 13 prompt,
29 completion, and 42 total tokens, in about 3 s each. Each request renders its whole message
list and the session truncates to the common prefix, so one request cannot contaminate the
next.

Three things to size a client by. Decode runs at about 10 tokens per second on E2B-it and 5.5
on E4B, so a 256-token reply takes 25 to 45 s and a client's default timeout of two minutes
covers about 660 tokens on E4B; pass an explicit timeout. The server prints `Model load:` once
and no per-request status line, since a status line per request on a long-lived process is
unbounded output. And a served E4B holds the full 4.05 GiB KV cache for the life of the
process; `--context` is the lever, and a serving deployment likely wants it set.

Serve mode writes no transcript and no snapshot, and restores none, whatever the directory
holds: a server's requests are unrelated callers. `--serve-addr <ip>` changes the bind address,
which only a container needs to set. `--interp-log <dir>` appends one record per completed
request, covered under [Loading a probe for live monitoring](probe-monitoring.md).

## What every mode shares

A probe, a control vector, the flight recorder, and the trace are installed on the model before
the mode starts, so all three modes observe and steer identically. Every mode writes the run
records under `${OBS_LOG_DIR}/harness/` described in [Run records](../observability/run-records.md).
