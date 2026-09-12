---
title: Flight recorder
---

# Flight recorder

`--record <path>` wraps the model in a recorder that appends every forward call to a JSONL
file: the input positions and the full output logits, and, above the model, every token the
sampler chose. A recording can be replayed as a scripted model, so a run made against the real
checkpoint on one machine can be debugged on another with no checkpoint, and a recording can
be audited position by position against `transformers`. This page assumes `BIN` and `MODEL`
from [Build](../getting-started/build.md) and an empty working directory.

## Record a run

Record a sampled run, not a greedy one. At temperature 0 the sampler draws nothing, so a
recording of a greedy run holds logits and no selections, and a replay of it cannot be told
from a recomputation:

```bash
"$BIN/harness" --model "$MODEL" --text-only --record flight.jsonl \
    --temperature 0.8 --top-k 40 --seed 1 "Reply with the single word: ready"
```

The recorder sits at the model boundary; the session, sampler, and agent loop see the same
model with or without it, and the values it records are the ones the run used.

## What the file holds

The first line is a header with the vocabulary size and the sampling parameters as the run
resolved them, `top_p` at its default included:

```text
{"vocab_size":262144,"sampling":{"temperature":0.8,"top_p":0.95,"top_k":40,"seed":1}}
```

Then one line per forward call, carrying the cache position the call started at, the token ids
it consumed, and the logits it produced; and one line per sampled token, naming the token, its
position, and the sampler's cumulative draw count. A run that generates 256 tokens writes 257
forward records, the extra one being the prefill call, which produces logits and draws nothing.

The selection lines are what make a replay a replay. Logits alone would reproduce the
distribution and then draw again; with the header's parameters a reader rebuilds the sampler
that made the choices, and with the draw count, which rises by exactly one per selection, a
replay resumes mid-stream rather than restarting.

## Size

A logit row is the whole vocabulary. Against the 262144-token vocabulary each forward call
adds about 2 MB of JSON, so a 256-token reply is a file of about half a gigabyte. Record short
runs, or a prompt of interest rather than a session.

## Replay

The workspace replays a recording as a scripted model in its own tests. From `rust-harness`,
the target takes a recording by path and states the selections it replays:

```bash
cd rust-harness
make replay-check RECORDING=/tmp/harness-demo/flight.jsonl
```

```text
[INFO] sampling: SamplingProvenance { temperature: 0.8, top_p: 0.95, top_k: Some(40), seed: 1 }
[INFO] script: 272 positions, vocab 262144
[INFO] selections: 256 at positions 16..=271, draws 1..=256
test a_flight_recording_replays_as_a_script_and_states_its_selections ... ok
```

The three lines are the sampling parameters as recorded, the script's position count and
vocabulary width, and the selection count with its position and draw ranges; the test asserts
that the draw count rises by one per selection and that every selection was drawn from a
scripted row. The target refuses to run without `RECORDING` rather than skipping. The
recording is too large to move, so the replay runs on the machine that made it.

## Auditing a sampled reply against transformers

A reply from a base checkpoint is fluent whether or not the forward pass is right, and a
sampled reply visits tokens no greedy reference would, so reading it decides nothing. The
recording holds every logit row and every token the sampler chose, and
`rust-harness/fixtures/audit_recording.py` feeds the recorded token sequence to
`transformers` once, teacher-forced, and compares its logits with the harness's at every
recorded position under the ladder's `L2` tolerances. For every token the sampler drew it
also reports that token's rank and probability under `transformers`. This needs torch and the
checkpoint, so it runs from the repository on the machine holding both:

```bash
cd "$REPO"
uv run python rust-harness/fixtures/audit_recording.py \
    --model "$MODEL" --record /tmp/harness-demo/flight.jsonl \
    --out /tmp/harness-demo/audit.json --show 5
```

On a 134-token `gemma-4-E4B` run at the default sampling:

```text
[OK] Recording: 134 tokens, 115 logit rows, 115 selections, sampling {'temperature': 0.8, 'top_p': 0.95, 'top_k': None, 'seed': 0}
[RUN] Forward pass: 134 positions, teacher-forced
[OK] Distributions: 115 positions, mean kl 0.000000 (<= 0.001), worst kl 0.000000 (<= 0.005) at position 19, argmax agreement 1.0000 (>= 0.98), worst top-50 overlap 1.0000 (>= 0.98)
[OK] Chosen tokens: 115, 114 within the reference's top-50, worst rank 54 at position 133, last token '<eos>' rank 2 p=0.3074
  position  133 rank     54 p=0.0024 '复制'
  position   30 rank     21 p=0.0060 ' _'
  position   29 rank     18 p=0.0062 'def'
  position   21 rank     17 p=0.0091 ' program'
  position   79 rank     10 p=0.0089 ' g'
```

`[OK] Distributions:` says the two engines agree at every position inside the tolerances; the
JSON report put the mean KL at 6.3e-11 nats. `[OK] Chosen tokens:` lists the lowest-ranked
draws. A token at rank 54 with probability 0.0024 is a draw that temperature 0.8 permits, one
in four hundred, and the reference offers it too; a token the reference gives no weight to is
a decoder fault. `[ERROR] Distributions:` names the worst position and exits 1, and is a
numerics finding for the [ladder](../guides/ladder.md) rather than a sampling one.

## Bisecting a divergence

The canonical numerics defect is a logit stream that diverges from a reference at some
position. The audit above locates the position; two recordings of the same prompt through two
builds of the harness diff to the first divergent forward call. Locating the call is this
tool's half; finding the cause is [rr debugging](rr-debugging.md), which runs that call
backwards from the divergent value to the write that produced it.
