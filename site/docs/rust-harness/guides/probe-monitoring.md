---
title: Loading a probe for live monitoring
---

# Loading a probe for live monitoring

`--probe` loads a linear probe and scores it at every position of its layer while the model
runs, inside the forward pass. In a task or chat run the flagged positions are written at the
end; under `--serve` the scores are attached to each request's record. This page assumes `BIN`
and `MODEL` from [Build](../getting-started/build.md), and a probe made for that checkpoint.

## Get a probe that fits

A probe is fitted to one checkpoint: its width is the residual width, its layer is one decoder
block, and its sidecar records the sha256 of the weights it was trained on. The harness checks
all three before loading. The committed fixture probe under `rust-harness/fixtures/` is a
format fixture for the exporter's tests and fits no runnable checkpoint.

The shortest path to a probe for `$MODEL` is
[Producing a probe and a control vector](../../formats/producing-a-probe-and-control-vector.md),
which takes about a minute on the CPU and leaves `probe.safetensors` and `probe.json` in a
directory. The commands below assume `PROBE` names that image:

```bash
export PROBE=/tmp/harness-demo/probes/demo/probe.safetensors
```

The `.json` sidecar beside it is required; it carries the layer, the site, and the decision
threshold.

## Check the fit before running

```bash
"$BIN/harness" --model "$MODEL" --probe "$PROBE" --status --deep
```

```text
[OK] probe: demo, layer 1, site LayerOutput, width 16
[OK] probe geometry: width 16 and layer 1 fit the model
[OK] probe identity: matches the served weights, sha256 0df22ea2...
```

`--deep` hashes the whole checkpoint to run the identity line; without it the geometry lines
still run. A probe for a different model fails here rather than after the load.

## Task and chat runs

```bash
"$BIN/harness" --model "$MODEL" --probe "$PROBE" --max-tokens 6 --temperature 0 "Say hello"
```

```text
[INFO] Probe demo: layer 1
[INFO] Prefill: 1170 tokens in 0.85 s (1372 tok/s)
[INFO] Probe demo: 1176 positions scored, 1161 flagged (threshold 0.5269914), flags at /tmp/harness-demo/probe-flags.jsonl
```

Every position of the run is scored, prompt and generation alike. Positions whose score
crosses the sidecar's threshold are written to `probe-flags.jsonl` in the working directory, or
to `--probe-out <path>`, one JSON object per position:

```text
{"layer":1,"position":0,"probe":"demo","score":0.5835134387016296,"threshold":0.5269914269447327}
```

Against the synthetic checkpoint nearly every position flags, because the probe was trained on
forty examples of a random model. The mechanics are what the example shows.

## Which text a position produced

`--alignment-out <path>` writes a second artifact mapping every position of the run to the byte
span of text it produced, the forward pass that computed it, and the probe score taken there.
It is off by default because it makes the model record every pass:

```bash
"$BIN/harness" --model "$MODEL" --probe "$PROBE" --alignment-out alignment.jsonl --max-tokens 6 --temperature 0 "Say hello"
```

The file opens with a header carrying the run id, the full rendered prompt text, the probe name
and layer, and then holds one row per position with `byte_start`, `byte_end`, `role`
(`content` or `scaffold`, the latter for chat-template markers), `pass`, `phase`, `token_id`,
and, for positions of the probe's layer, `score`. Concatenating every row's span in position
order reproduces the header's text exactly. The header's `run_id` is the same id that names
the run's event file under `${OBS_LOG_DIR}/harness/`, so the two join.

## Under the server

A served run has no end at which to write flags, so the scores go to the per-request log
instead. Without `--interp-log` they are computed and discarded, and the harness warns at
startup.

```bash
"$BIN/harness" --model "$MODEL" --probe "$PROBE" --serve 18098 --interp-log ilog --max-tokens 16 &
sleep 3
curl -s http://127.0.0.1:18098/v1/chat/completions -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hi"}]}'
kill %1
```

Each completed request appends one record to `<dir>/harness/YYYY-MM-DD.jsonl` under the
`--interp-log` directory, carrying the request messages, the reply, usage, latency, and a
`probe` object with the probe's name, layer, site, and threshold, the count of positions scored,
the maximum score, and the list of flagged positions with their scores. This file is what the
Python probe monitor aggregates.
