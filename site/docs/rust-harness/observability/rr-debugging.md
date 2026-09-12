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
  to one.
- Every value of interest is reachable in owned source: the residual stream, the KV cache,
  and the logits are ordinary Rust data behind a `Model` trait, so a watchpoint can be set on
  any of them.

## Requirements

- rr 5.9.0 or later. rr identifies the processor from a fixed table rather than probing it,
  and an unlisted part is a hard failure before any harness code runs:
  `[FATAL ...compute_cpu_microarch()] ... AMD CPU type 0x40f40 (ext family 0xb) unknown`.
  The rr that Ubuntu 24.04 packages, 5.7.0, does not list the Ryzen 9950X; the 5.9.0 release
  does, and ships as a `.deb` on rr's GitHub releases page. `rr --version` must print 5.9.0 or
  later before anything below is worth running.
- gdb, which rr replays through.
- `kernel.perf_event_paranoid` at 1 or lower for a non-root user:
  `sudo sysctl kernel.perf_event_paranoid=1`.
- A build with debug info at release optimization. The workspace defines the
  `release-with-debug` Cargo profile for this.
- `--single-thread` on the harness. rr checks whether the processor's SpecLockMap optimization
  is enabled and records reliably with it enabled only for a single thread; rr 5.9.0 warns and
  continues, and later versions refuse. To record the threaded backend, run rr's
  `zen_workaround.py` as root first, which does not persist across a reboot, then drop the
  flag.

## Record and replay

The workspace Makefile wraps the two steps. From `rust-harness`:

```bash
cd rust-harness
make debug-record ARGS='--model fixtures/tiny-gemma4 --single-thread "Reply with the single word: ready"'
make debug-replay
```

The first target builds the harness with the `release-with-debug` profile and runs it under
`rr record` with the given arguments. The second opens the most recent recording in gdb with
`rr replay`. Any harness flags go in `ARGS`; a real checkpoint with `--text-only` works the
same way, at the real checkpoint's memory cost and with a load of several seconds under rr
before the first forward pass. The example records the workspace's committed synthetic
checkpoint, `fixtures/tiny-gemma4`, because it loads in milliseconds and the mechanics of a
replay do not depend on the weights; its output is noise.

## From a divergent logit to its cause

Once [the flight recorder](flight-recorder.md) has located the first divergent forward call,
the replay runs from that call's logit write backwards. The sequence, recorded on the tiny
checkpoint:

1. Find the line that fills the logits. It is the `fill_dots(` call in the tied-head
   projection of `runtime/src/model/cpu.rs`, and it moves whenever the file is edited above
   it, so recompute it from the checkout the recording was built from. Two lines match; the
   breakpoint is the one whose `&mut logits` argument follows it:

```bash
grep -n 'fill_dots($' runtime/src/model/cpu.rs
```

2. Inside the replay, break at that line, run to it, set a hardware watchpoint on the first
   logit's heap cell, run forward to the write, and run backwards to it:

```text
(rr) break cpu.rs:3242
(rr) continue
(rr) watch -l logits.buf.inner.ptr.pointer.pointer[0]
(rr) continue
(rr) reverse-continue
```

The forward `continue` catches the write so that `reverse-continue` has a live write to
travel back from. Both stops land in `fill_run_blocked` at the `*dst = post(g)` store inside
the tied-head projection, with the same stack above it: `fill_dots`, `forward_batch`,
`prefill_groups`, `prefill`, `Session::sync`, `Session::generate`. At the reverse stop gdb is
standing before the store it stood after a moment earlier, and every local in scope holds the
value the original run held. The same sequence one level deeper watches a single
residual-stream element inside the layer loop and reverse-continues to the attention or MLP
write that produced it.

Three details of that sequence matter:

- **The watchpoint names the heap cell, not the `Vec`.** `logits` is a `Vec<f32>`, a struct of
  pointer, capacity, and length, and gdb subscripts only C arrays, so `watch -l logits[0]`
  fails with `Cannot subscript non-array type`. The field path reaches the buffer through the
  pinned toolchain's `RawVec` layout; a toolchain bump that renames a field shows up as
  `There is no member named ...`, at which point `print logits` shows the new path.
- **The breakpoint is a line, not a function.** `logits` is a local of the projection loop,
  not of `forward_one`, so a breakpoint on the function name has nothing to watch. gdb reports
  the stop one or two lines after the number given, because the call's first instruction
  belongs to an argument on the next line.
- **Locals may be optimized out.** `release-with-debug` inherits `release`. If a local prints
  `<optimized out>`, rebuild the recording arm with
  `CARGO_PROFILE_RELEASE_WITH_DEBUG_OPT_LEVEL=1`.

gdb prints repeated `warning: (Internal error: pc ... in read in CU, but not in symtab.)`
lines on Rust binaries; they have no effect on the session.

## Scope

rr records CPU processes only; it cannot record GPU ioctls, which is one of the two reasons the
CPU backend exists as a reference. Decode speed under the backend is bandwidth-bound and low
against the real checkpoint, so a recording of a short prompt is what is practical; the point
is that a wrong value at one position is a finite number of instructions away from its cause.
For where the time goes rather than where a value came from, see [Profiling](profiling.md).
