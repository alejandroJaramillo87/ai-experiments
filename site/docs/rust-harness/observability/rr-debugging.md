---
title: Debugging under rr
---

# Debugging under rr

rr records a run once and replays it deterministically inside gdb, with reverse execution: stop
at a wrong value and run backwards to the write that produced it. The CPU backend was written
to be recordable this way, and this is the reason the runtime holds the decode path free of
threads and wall-clock reads. rr itself is a Linux tool that needs a supported processor and
kernel performance counters; it is workstation tooling, not something the binary ships.

## Why the backend is recordable

- The sampler is seeded and the decode path reads no clock, so a run is a function of its
  inputs.
- The one form of threading is a row split inside the matrix kernels that reorders no
  arithmetic, so the logits are bit-identical at every width. `--single-thread` sets the width
  to one; rr serializes threads regardless, so the flag is for the reader's sanity rather than
  the result.
- Every value of interest is reachable in owned source: the residual stream, the KV cache,
  and the logits are ordinary Rust data behind a `Model` trait, so a watchpoint can be set on
  any of them.

## Requirements

- rr, installed from the distribution's package or from its releases.
- gdb, which rr replays through.
- `kernel.perf_event_paranoid` at 1 or lower for a non-root user.
- A build with debug info at release optimization. The workspace defines the
  `release-with-debug` Cargo profile for this.

## Record and replay

The workspace Makefile wraps the two steps. From `rust-harness`:

```bash
cd rust-harness
make debug-record ARGS='--model fixtures/tiny-gemma4 --max-tokens 4 --temperature 0 "Say hello"'
make debug-replay
```

The first target builds the harness with the `release-with-debug` profile and runs it under
`rr record` with the given arguments. The second opens the most recent recording in gdb with
`rr replay`. Any harness flags go in `ARGS`; a real checkpoint with `--text-only` works the
same way, at the real checkpoint's memory cost.

## From a divergent logit to its cause

The sequence, once [the flight recorder](flight-recorder.md) has located the first divergent
forward call at some position:

1. Record the failing run under rr with the same flags.
2. Inside the replay, break at the forward pass for the divergent position, run to it, then
   set a hardware watchpoint on the divergent value and run backwards:

```text
(rr) break cpu.rs:forward_one if position == <divergent position>
(rr) continue
(rr) watch -l logits[<token>]
(rr) reverse-continue
```

The replay stops at the write that last changed the value, inside the layer loop, and every
value in scope there is the one the original run held. The same sequence one level deeper
watches a single residual-stream element inside the layer loop and reverse-continues to the
attention or MLP write that produced it.

## Scope

rr records CPU processes only; it cannot record GPU ioctls, which is one of the two reasons the
CPU backend exists as a reference. Decode speed under the backend is bandwidth-bound and low
against the real checkpoint, so a recording of a short prompt is what is practical; the point
is that a wrong value at one position is a finite number of instructions away from its cause.
