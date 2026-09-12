---
title: Tracing
---

# Tracing

`--trace-chrome <path>` writes a Chrome trace event file of the run's span timeline, loadable
in Perfetto. No subscriber is installed without the flag, so spans cost nothing in an ordinary
run. The spans and their fields form a stable vocabulary shared with the repository's Python
observability tooling. This page assumes `BIN` and `MODEL` from
[Build](../getting-started/build.md).

## Write a trace

```bash
"$BIN/harness" --model "$MODEL" --trace-chrome trace.json --max-tokens 4 --temperature 0 "Say hello"
```

Open the file at `https://ui.perfetto.dev` with **Open trace file**. Each span is a slice on
the main thread's track; worker threads carry no spans, because a kernel's span is opened on
the caller before the parallel region and closed after the join, so it measures the whole
region.

## Depth

`--trace-level` selects how deep the trace goes and requires `--trace-chrome`:

| Level | Adds | Spans for a 42-layer run of about 115 forward calls |
|-------|------|-----------------------------------------------------|
| `info` | The session-level timeline: load, encode, each forward, each sync and step, each sample | hundreds |
| `debug` (default) | One span per layer and per block of the forward pass and of the load | about 20 thousand more |
| `trace` | One span per kernel call: each projection, the attention loop, the LM head rows | about 45 thousand more |

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
trace was asked for; see [Run records](run-records.md).

## Per-name totals

In Perfetto's query view, this SQL is the table of where the time went:

```sql
select name, count(*), sum(dur) from slice group by name order by 3 desc
```

From the terminal, the workspace's own checker prints the same table and asserts the count
relations between spans, such as one `model.layer` per layer per prefill group:

```bash
cd rust-harness
make trace-check TRACE=/tmp/harness-demo/trace.json
```

## Determinism

A trace changes no value. The workspace asserts on every test run that the logits are
bit-identical with and without a subscriber, so a trace of a divergent run is a trace of the
same divergence.
