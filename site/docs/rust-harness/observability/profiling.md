---
title: Profiling
---

# Profiling

A flame graph of a harness run attributes CPU cycles to functions, which is how a change to
the forward pass is shown to have moved the time rather than only the numbers. It needs
`perf` and `cargo-flamegraph`, both workstation tooling, and a real checkpoint, since a
profile of the synthetic fixture measures the load path and nothing else. This page assumes
the repository checked out and `MODEL` from [Build](../getting-started/build.md).

## Requirements

- `perf`, from the distribution's kernel tools package, and
  `kernel.perf_event_paranoid` at 1 or lower for a non-root user.
- `cargo-flamegraph`, installed with the lockfile pinned, because an unpinned install resolves
  a dependency that needs a newer toolchain than the workspace pins:

```bash
cargo install flamegraph --locked
```

- `kernel.kptr_restrict` at 0 to resolve kernel frames. With it set, kernel time appears as
  unresolved `[k] 0xffffffff...` addresses, which on a harness run is about 8 percent of
  cycles that the profile cannot name.

## Profile a run

The run starts from `rust-harness`, because cargo builds there, so clear the session files
first: a stale `session.kv` in the workspace root restores a conversation into the profiled
run, and a profile is the one run where nobody reads the text. `make clean` removes the
session files, the previous profile and the gdb history, and leaves the build output.

```bash
cd rust-harness
make clean
sysctl kernel.perf_event_paranoid kernel.kptr_restrict
cargo flamegraph -F 97 --profile release-with-debug --bin harness -- \
    --model "$MODEL" --text-only "Reply with the single word: ready"
ls -la flamegraph.svg perf.data
perf report -i perf.data --no-children --sort sym --stdio 2>/dev/null | head -n 20
```

**Pass `-F 97`.** At perf's default sampling frequency a run over the real checkpoint lost
48 percent of its samples and wrote a 45.9 GB `perf.data`; a profile at that loss rate is not
trustworthy and the file is a disk hazard. The `make flame` target in the workspace runs the
same command without arguments and without the frequency flag, so the direct invocation is the
one to use against a real checkpoint.

## What the profile shows

The top of `perf report` on `gemma-4-E4B`:

```text
# Overhead  Command  Shared Object         Symbol
    38.65%  harness  harness               [.] runtime::tensor::Weight::row_dot_block
     6.10%  harness  harness               [.] <alloc::vec::Vec<T> as alloc::vec::spec_from_iter::SpecFromIter<T,I>>::from_iter
     3.14%  harness  harness               [.] runtime::model::cpu::fill_panel_widened
     2.89%  harness  [unknown]             [k] 0xffffffffade83d4b
     1.89%  harness  harness               [.] core::slice::sort::stable::quicksort::quicksort
     1.58%  harness  harness               [.] runtime::model::cpu::prefill_groups
     1.20%  harness  libm.so.6             [.] __expm1f
     0.81%  harness  libm.so.6             [.] __tanhf
     0.72%  harness  harness               [.] runtime::model::cpu::matmul_panel
```

The forward pass's matrix-vector path dominates, as it should: `row_dot_block` alone is
38.65 percent, and with `fill_panel_widened` and `matmul_panel` the matvec path is about
42.5 percent of cycles, with the bf16-to-f32 widening inlined at the leaf. Tokenization does
not appear. A profile in which sampling, tokenization, or I/O is near the top is the finding.
The call graph reads
`row_dot_block <- fill_panel_blocked <- matmul_panel <- forward_batch <- prefill_groups <- prefill <- step`.

`flamegraph.svg` is the same data as an interactive graph; open it in a browser. Where a
single value came from, rather than where the time went, is the question for
[Debugging under rr](rr-debugging.md).
