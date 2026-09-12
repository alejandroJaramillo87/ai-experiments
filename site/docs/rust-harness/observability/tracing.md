---
title: Tracing
---

# Tracing

`--trace-chrome <path>` writes a Chrome trace event file of the run's span timeline, loadable
in Perfetto. No subscriber is installed without the flag, so spans cost nothing in an ordinary
run. The spans and their fields form a stable vocabulary shared with the repository's Python
observability tooling. This page assumes `BIN` and `MODEL` from
[Build](../getting-started/build.md) and an empty working directory.

## Write a trace

```bash
"$BIN/harness" --model "$MODEL" --text-only --trace-chrome trace.json "Reply with the single word: ready"
```

Open the file at `https://ui.perfetto.dev` with **Open trace file**. Each span is a slice on
the main thread's track; worker threads carry no spans, because a kernel's span is opened on
the caller before the parallel region and closed after the join, so it measures the whole
region.

## Depth

`--trace-level` selects how deep the trace goes and requires `--trace-chrome`:

| Level | Adds | Spans for a 42-layer run of 257 forward calls |
|-------|------|-----------------------------------------------|
| `info` | The session-level timeline: load, encode, each forward, each sync and step, each sample | about a thousand |
| `debug` (default) | One span per layer and per block of the forward pass and of the load | about 44 thousand more |
| `trace` | One span per kernel call: each projection, the attention loop, the LM head rows | about 100 thousand more |

Each span costs a few microseconds on the calling thread against kernels that take
milliseconds, so no level is measurable in the wall clock; the file grows from a few megabytes
at `info` to a few tens at `trace`. Every field is an integer or a boolean already in scope.

## The spans

| Span | Level | Key fields |
|------|-------|------------|
| `model.load` | info | `text_only` |
| `checkpoint.read`, `checkpoint.parse`, `model.from_checkpoint`, `checkpoint.fingerprint` | debug | sequential children of `model.load` |
| `model.load.layer` | debug | `layer`, `owns_kv`; one per layer read |
| `tokenizer.encode` | info | bytes, token count |
| `model.forward` | info | `start` (cached length), `len` (tokens in the call), `groups` (prefill groups the call split into) |
| `model.prelude` | debug | `n`, `ple` |
| `model.layer` | debug | `layer`, `n`, `kv_source`, `owns_kv` |
| `model.layer.attn` | debug | `owns_kv`, `kv_len` |
| `model.layer.mlp` | debug | `ffn_dim` |
| `model.layer.ple` | debug | `ple_dim`; only on a checkpoint with per-layer embeddings |
| `model.lm_head` | debug | `rows`, `n` |
| `kernel.matmul` | trace | `out_dim`, `in_dim`, `n`, `threads`; one per projection |
| `kernel.attention` | trace | `n`, `kv_len`, `heads`; once per layer |
| `kernel.lm_head` | trace | `rows`, `in_dim`, `threads` |
| `session.sync` | info | delta length in tokens, cached length |
| `session.step` | info | position, chosen token |
| `sampler.sample` | info | temperature, candidate count |
| `grammar.advance` | info | state, allowed-token count; present only when a grammar constrains decoding |
| `tool.exec` | info | tool name, duration, output bytes; present only in an agent run that called a tool |
| `agent.fork` | info | parent cached length, task delta length |

Nesting is contextual: no span names its parent, and the tree is what the timeline shows.
There is no `prefill` or `decode` span. `session.sync` fires once per generation and is the
prefill; `session.step` fires once per decoded token. The `prefill` and `decode` names are
phases in the run records, a separate mechanism that is present on every run whether or not a
trace was asked for; see [Run records](run-records.md). `grammar.advance` and `tool.exec` are
absent from a plain text completion, and their absence there is not a finding.

## Per-name totals

In Perfetto's query view, this SQL is the table of where the time went:

```sql
select name, count(*), sum(dur) from slice group by name order by 3 desc
```

From the terminal, the workspace's own checker prints the same table, asserts the count
relations between spans, and reads the sync arguments out of the file:

```bash
cd rust-harness
make trace-check TRACE=/tmp/harness-demo/trace.json
```

A 256-token task run on `gemma-4-E4B` at the default level:

```text
[INFO] session.step                256 calls     45.07s
[INFO] model.forward               257 calls     45.00s
[INFO] model.layer               10794 calls     39.71s
[INFO] model.layer.mlp           10794 calls     29.10s
[INFO] model.layer.attn          10794 calls      8.19s
[INFO] model.load                    1 calls      5.48s
[INFO] model.from_checkpoint         1 calls      5.48s
[INFO] model.lm_head               257 calls      5.03s
[INFO] model.load.layer             42 calls      2.80s
[INFO] model.layer.ple           10794 calls      2.34s
[INFO] sampler.sample              256 calls      1.29s
[INFO] session.sync                  1 calls      1.21s
[INFO] model.prelude               257 calls      0.24s
[INFO] checkpoint.read               1 calls      0.00s
[INFO] checkpoint.fingerprint        1 calls      0.00s
[INFO] tokenizer.encode              1 calls      0.00s
[INFO] checkpoint.parse              1 calls      0.00s
[INFO] forwards exceed steps by one: the run stopped on its token budget
[INFO] session.sync args {"cached_len": "0"}
[INFO] first model.forward args {"len": "16", "start": "0"}
[INFO] layers 42, forwards 257, model.layer / model.forward = 42.0
[INFO] kernel spans absent: a debug-level trace
test a_chrome_trace_from_a_real_run_has_the_documented_span_shape ... ok
```

The table is where the time goes: of 39.71 s inside the layers, the MLP takes 29.10 s,
attention 8.19 s, and the per-layer embeddings 2.34 s, with the LM head another 5.03 s per
run and the sampler under 3 percent of the forward pass. The lines beneath it are the shape
checks: 257 forwards against 256 steps is the prefill call, `model.layer` is exactly 42 times
the forward count, and the sync arguments name where the prefill started. On a multi-turn chat
the checker prints one `session.sync` line per turn with its `cached_len` and the prefill's
`start` and `len`, and a count of how many turns stopped on the end-of-sequence token against
how many ran to the budget. Without `TRACE` the test skips, so the quality gate is not gated
on a trace.

## Determinism

A trace changes no value. The workspace asserts on every test run that the logits are
bit-identical with and without a subscriber, so a trace of a divergent run is a trace of the
same divergence.
