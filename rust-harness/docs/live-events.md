# Live run events and viewer

Design for live visibility into a harness run in progress: the harness emits
run-state events as JSONL through the existing observability contract, and
the parent repository's observability tooling gains a small serve mode plus
an HTML view that polls the event file. The same view pointed at a finished
run replays it, so the feature doubles as a post-hoc run inspector.

This is a design awaiting host verification. Implementation is blocked on
the workstation chat and agent runs (runbook entries RH.11 and RH.12), which
decide which events are worth watching and what cadences real-model rates
justify. No code changes accompany this document.

## Table of contents

- [Purpose](#purpose)
- [Event stream](#event-stream)
- [Viewer](#viewer)
- [Non-goals](#non-goals)
- [Sequencing](#sequencing)

## Purpose

While a run executes, the terminal shows interleaved stderr: streamed
tokens, status lines, probe flag summaries. No current tool shows the state
of the run as it evolves — KV cache fill against the budget (how close is
compaction), prefill and decode rates, which tool call is executing,
subagent forks in flight. On the workstation, where a real-model run may
take minutes, that live view is the difference between watching a scrolling
terminal and watching the experiment: a stuck tool call, a runaway prefill,
or a probe firing storm becomes visible during the run instead of in the
post-mortem.

The valuable artifact is the machine-readable status stream, not an
interface. The harness binary stays a line-oriented Unix program; the
viewer is a separate consumer that can evolve independently or be skipped
entirely. This design does not replace the existing debugging stack:
Perfetto remains the trace viewer (`--trace-chrome`), the flight recorder
remains the replay mechanism (`--record`), and rr remains the debugger for
the CPU backend.

## Event stream

`harness/src/obs.rs` already writes an event file at
`${OBS_LOG_DIR}/harness/<run_id>.jsonl` in the shared obs contract (fields
`ts`, `level`, `run_id`, `cli`, `command`, `pid`, `event`, `msg`, plus
per-event extras), bracketed by `cmd.start` and `cmd.end`. Live visibility
extends the same file with events emitted during the run — no new file, no
new transport, and every existing reader (runs catalog, retention pruner)
continues to work unchanged because unknown `event` values pass through
stateless readers.

Proposed catalog, final grain to be decided after RH.11 and RH.12:

| Event | Kind | Payload sketch |
|-------|------|----------------|
| `phase` | discrete | `prefill`, `decode`, or `tool` with the tool name |
| `progress` | periodic | tokens processed, cache fill vs budget, tokens/second |
| `compaction` | discrete | replaced range, summary length in tokens |
| `snapshot` | discrete | `saved` or `restored`, cached token count |
| `fork` / `join` | discrete | subagent identifier |
| `probe` | periodic | flag count since the previous record |

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
  granularity belongs to `--trace-chrome`.
- **No websockets or async server.** Poll a file at 1 Hz.
- **No TUI in the harness binary.** The binary keeps line-oriented output;
  interactive rendering is the viewer's job.
- **No change to the transcript or stderr output.** Captured session
  transcripts stay clean text.

## Sequencing

1. Workstation runs RH.11 and RH.12 establish which events matter and what
   real-model rates look like; observed pain points scope the catalog.
2. Harness emission lands as one commit, sandbox-testable end to end: the
   tiny fixture model exercises every event type, and Tier 1 tests assert
   the record shapes against the obs contract.
3. The parent-repository viewer and serve mode land as a second commit,
   testable in the sandbox by replaying a recorded event file.
