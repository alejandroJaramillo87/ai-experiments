---
title: Flight recorder
---

# Flight recorder

`--record <path>` wraps the model in a recorder that appends every forward call to a JSONL
file: the input positions and the full output logits, and, above the model, every token the
sampler chose. A recording can be replayed as a scripted model, so a run made against the real
checkpoint on one machine can be debugged on another with no GPU and no checkpoint, and two
recordings of the same prompt through two implementations diff to the first divergent
position. This page assumes `BIN` and `MODEL` from [Build](../getting-started/build.md).

## Record a run

```bash
"$BIN/harness" --model "$MODEL" --record flight.jsonl --max-tokens 4 --temperature 0 "Say hello"
```

The recorder sits at the model boundary; the session, sampler, and agent loop see the same
model with or without it, and the values it records are the ones the run used.

## What the file holds

The first line is a header with the vocabulary size and the sampling parameters:

```text
{"vocab_size":288,"sampling":{"temperature":0.0,"top_p":0.95,"top_k":null,"seed":0}}
```

Then one line per forward call, carrying the cache position the call started at, the token ids
it consumed, and the logits it produced; and one line per sampled token, naming the token, its
position, and the sampler's cumulative draw count.

The selection lines are what make a replay a replay. Logits alone would reproduce the
distribution and then draw again; with the header's parameters a reader rebuilds the sampler
that made the choices, and with the draw count a replay resumes mid-stream rather than
restarting.

## Size

A logit row is the whole vocabulary. Against the synthetic checkpoint's 288-token vocabulary
the four-token run above is 22 KB; against the real 262144-token vocabulary each decoded token
adds about 2 MB of JSON. Record short runs, or a prompt of interest rather than a session.

## Replay

The workspace replays a recording as a scripted model in its own tests. From `rust-harness`,
the target takes a recording by path and states the selections it replays:

```bash
cd rust-harness
make replay-check RECORDING=/tmp/harness-demo/flight.jsonl
```

The target refuses to run without `RECORDING` rather than skipping.

## Bisecting a divergence

The canonical numerics defect is a logit stream that diverges from a reference at some
position. Record the same prompt through the harness and through the reference, diff the two
files line by line, and the first divergent forward call is the one to inspect. Locating the
call is this tool's half; finding the cause is [rr debugging](rr-debugging.md), which runs that
call backwards from the divergent value to the write that produced it.
