# Live run events and viewer

Design for live visibility into a harness run in progress: the harness emits
run-state events as JSONL through the existing observability contract, and
the parent repository's observability tooling gains a small serve mode plus
an HTML view that polls the event file. The same view pointed at a finished
run replays it, so the feature doubles as a post-hoc run inspector.

**Partly built, and the unbuilt half is the live one.** The record stream this
document proposed exists: the U campaign landed phase spans, per-tool header
events and a `metrics.jsonl` that the snapshot dashboard already renders. What
remains a design is the serve mode and the polling page under
[Viewer](#viewer) — watching a run *as it happens*, rather than reading it
afterwards. That half was never blocked on RH.8 and RH.11 in the end; it is
unbuilt because nothing has yet needed it.

Where this document proposed a shape and the implementation took a different
one, the implemented shape is what is recorded below.
[observability.md](observability.md) is the reference for what is built;
this document keeps only the design that is not.

## Table of contents

- [Purpose](#purpose)
- [Event stream](#event-stream)
- [Viewer](#viewer)
- [Non-goals](#non-goals)
- [Sequencing](#sequencing)

## Purpose

While a run executes, the terminal shows almost nothing. An agent task
prints its answer to stdout once the task ends and a chat turn prints its
reply once the turn ends — neither streams tokens, because
`harness/src/stream.rs` is not wired to an entry point — and the probe flag
summary reaches stderr at the same late moment.

Part of that gap has since closed at the terminal. The run header states the
resolved configuration before the work starts, and a completed operation
reports what it cost: `Model load: 53.50 s`, then `Prefill: 234 tokens in
1.90 s (123 tok/s)` on every turn, with a clause naming the rewind the
sliding window refused when a compaction forced a rebuild. Those are summaries after the fact, which
is the distinction that keeps this document alive — they say what a step took
once it is over, and say nothing while it runs.

So what is still unshown is the state of a run *as it evolves*: KV cache fill
against the budget (how close is compaction), rates while the work is
happening rather than after, which tool call is executing, subagent forks in
flight. On the workstation, where a
real-model run may take minutes, that live view is the difference between
watching a scrolling terminal and watching the experiment: a stuck tool
call, a runaway prefill, or a probe firing storm becomes visible during the
run instead of in the post-mortem.

The valuable artifact is the machine-readable status stream, not an
interface. The harness binary stays a line-oriented Unix program; the
viewer is a separate consumer that can evolve independently or be skipped
entirely. This design does not replace the existing debugging stack:
Perfetto remains the trace viewer (`--trace-chrome`), the flight recorder
remains the replay mechanism (`--record`), and rr remains the debugger for
the CPU backend.

## Event stream

`obs/src/lib.rs` writes an event file at
`${OBS_LOG_DIR}/harness/<run_id>.jsonl` in the shared obs contract (fields
`ts`, `level`, `run_id`, `cli`, `command`, `pid`, `event`, `msg`, plus
per-event extras), bracketed by `cmd.start` and `cmd.end`. Live visibility
extends the same file with events emitted during the run — no new file, no
new transport, and every existing reader (runs catalog, retention pruner)
continues to work unchanged because unknown `event` values pass through
stateless readers.

### What the file carries today

Five binaries write it. Beyond the `cmd.start`/`fail`/`cmd.end` bracket, each
run emits one per-tool header event naming its resolved configuration, and a
`phase.start`/`phase.end` pair around each timed operation.
`schemas/obs_record.json` is the contract; `observability.md` is the reference.

**The phase grain landed as a pair, not as one discrete event.** This document
proposed a single `phase` record carrying `prefill`, `decode`, or `tool` with a
tool name. What shipped is `phase.start` and `phase.end`, both written by the
Python and the Rust stacks, with the end record carrying `duration_s` rounded
to three decimals. A pair is what lets a reader see a phase that is still open
— which is exactly what a live viewer needs and a single end-of-phase record
cannot give it. The names actually written:

| Phase | Written by |
|-------|------------|
| `model.load` | all five instrumented binaries |
| `prefill`, `decode` | the harness, one pair per turn or served request |
| `rung.L0` through `rung.L4` | the ladder |

**There is no `tool` phase**, and its absence is a decision rather than an
oversight: `harness/src/tools.rs` already opens a `tool.exec` tracing span,
which `--trace-chrome` renders, and duplicating it into the obs file would put
the same measurement in two contracts.

### Still proposed

These have no implementation and no contract entry. The grain is a design
sketch, to be settled against a real run rather than in advance:

| Event | Kind | Payload sketch |
|-------|------|----------------|
| `progress` | periodic | tokens processed, cache fill vs budget, tokens/second |
| `compaction` | discrete | replaced range, summary length in tokens |
| `snapshot` | discrete | `saved` or `restored`, cached token count |
| `fork` / `join` | discrete | subagent identifier |
| `probe` | periodic | flag count since the previous record |

`progress` is the one with a partial answer already: `metrics.jsonl` carries a
per-token `tokens_per_s` and `rss_gb` series for the harness and a per-rung
series for the ladder. What it does not carry is cache fill against the budget,
which is the number that says how close a run is to compaction.

Two rules keep the stream cheap and safe:

- **Bounded cadence.** Periodic events are emitted at most once per second
  or once per N tokens, never per token. A 128k-token prefill contributes
  on the order of a hundred `progress` lines, not 128,000.
- **Torn-line discipline.** Each record is written and flushed as one line,
  the same discipline `session.jsonl` uses, so a polling reader never
  observes a partial record. Obs failures degrade silently to a no-op,
  matching the existing behavior in environments without the log root.

## Viewer

The viewer lives in the parent repository's observability tooling, not in
this crate. A static HTML page opened from disk cannot poll a local file
(browsers block `fetch` on `file://` URLs), so the live view requires a
minimal serve mode:

- A stdlib `http.server`-based command in the Python obs tooling serves the
  HTML page and one endpoint, `/events?run=<id>&since=<n>`, which returns
  the JSONL records after line `n` for the given run under `OBS_LOG_DIR`.
- The page polls at roughly 1 Hz and renders the run state: a cache-fill
  bar with the compaction threshold marked, current phase, rate sparkline,
  and the discrete event timeline. Polling a JSONL file at this rate is
  adequate for events arriving a few per second; no websockets are needed.
- Replay falls out for free: the same page pointed at a finished run's file
  renders the full timeline of any historical run.

## Non-goals

- **No per-token events.** The stream is a status feed, not a trace; token
  and layer granularity belong to `--trace-chrome`.
- **No websockets or async server.** Poll a file at 1 Hz.
- **No TUI in the harness binary.** The binary keeps line-oriented output;
  interactive rendering is the viewer's job.
- **No change to the transcript or stderr output.** Captured session
  transcripts stay clean text.

## Sequencing

Steps 1 and 2 are done, by a different route than planned: the U campaign
landed the record stream against the tiny fixture without waiting on RH.8 and
RH.11, and `harness/tests/obs_contract.rs` and `runtime/tests/ladder_obs.rs`
assert the shapes against the obs contract. What is left:

1. Decide which of the events under [Still proposed](#still-proposed) are worth
   emitting, from a real workstation run rather than in advance. The `progress`
   record is the one with an established need and a partial answer already.
2. The parent-repository viewer and serve mode land as one commit, testable in
   the sandbox by replaying a recorded event file. The snapshot dashboard
   (`report --run`) already renders a finished harness run, so this step is the
   polling endpoint and the live page, not a viewer from nothing.
