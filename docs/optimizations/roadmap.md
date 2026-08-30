# Optimization measurement roadmap

The order in which to measure every optimization in this repository, and the reasoning
behind that order. The campaign starts from a firmware reset and ends with a table stating
what each layer is worth.

This document owns the ordering only. [os/os-optimizations.md](os/os-optimizations.md) owns
the OS settings, [bios/bios-optimizations.md](bios/bios-optimizations.md) owns the BIOS
settings, [ideation/](ideation/) owns the unscheduled menus, and
[../verification/bench/bench.md](../verification/bench/bench.md) owns the acceptance
evidence. Nothing here restates a setting value that one of those files already owns.

## Table of contents

- [Method](#method)
- [The reference configuration](#the-reference-configuration)
- [Stage status](#stage-status)
- [M0 Reset the firmware and pin the reference](#m0-reset-the-firmware-and-pin-the-reference)
- [M1 Calibrate the instrument](#m1-calibrate-the-instrument)
- [M2 Positive controls](#m2-positive-controls)
- [M3 Container runtime arms](#m3-container-runtime-arms)
- [M4 Shims](#m4-shims)
- [M5 Container build arms](#m5-container-build-arms)
- [M6 BIOS bundles](#m6-bios-bundles)
- [M7 Re-validation and ablation](#m7-re-validation-and-ablation)
- [M8 GPU track](#m8-gpu-track)
- [Session discipline](#session-discipline)
- [Publication mapping](#publication-mapping)
- [See also](#see-also)

## Method

The general rules — that a delta is of a change and not of a layer, that each
optimization is measured twice, and that a result carries its reference state — are
owned by [../bench/method.md](../bench/method.md). What follows is what those rules
imply for the order of this campaign, which is a separate question and the one this
document exists to answer.

### Order by redo cost

The stages below are not ordered bottom-up through the stack. They are ordered by what it
costs to change one arm, cheapest first, so that when an interaction forces a re-run, the
re-run is cheap.

| Layer | Cost per arm swap | Interleavable by `bench run` |
|-------|-------------------|------------------------------|
| Container runtime environment | 30 to 300 seconds (compose recreate) | Yes |
| Shims (`LD_PRELOAD`) | 30 to 300 seconds (compose recreate) | Yes |
| OS runtime (sysctl, cpuset, IRQ affinity) | Seconds | Yes, with a script between arms |
| Container build arguments | 5 to 30 minutes (image rebuild) | No |
| OS boot parameters (`isolcpus`, huge page reservation) | Reboot, about 1 minute | No |
| BIOS | Reboot plus manual entry, about 5 minutes | No |

Re-running a thread sweep after a BIOS change costs 20 minutes. Re-running a BIOS sweep
after a thread-count change costs a weekend. The asymmetry sets the order.

Two hard dependency edges override the cost ordering. The huge page pool must be reserved
before the wrapper can be measured, and CPU topology must be known before any cpuset arm
means anything. Both are resolved in M0.

### The boot state is the scheduling unit

A reboot costs operator attention, not just wall-clock time. Loop over boot states, not over
layers: enter a boot state once, drain every measurement that state can serve, and do not
leave until the queue for that state is empty.

Each stage below therefore lists its arms as a queue rather than a sequence, and the GPU
track (M8) is explicitly drained inside whichever boot state is current rather than
scheduled as a block at the end.

### Positive controls

Before trusting the rig to report a 5 percent delta, prove it reports a delta already known
to be large. Two are available, one per track, and both are run in M2. A rig that cannot see
a known effect is not ready to measure an unknown one.

## The reference configuration

The reference is the state every baseline delta in M2 through M6 is measured against. It
advances exactly once, at the end of M1, and never again. Advancing it later would
invalidate every delta measured before the advance.

The rule for inclusion: a setting belongs in the reference if it is required for the
hardware to function correctly, and belongs in an arm if it is intended to make the hardware
faster.

**BIOS.** Factory defaults, plus the following.

| Setting | Value | Why it is in the reference |
|---------|-------|----------------------------|
| Above 4G Decoding | Enabled | Prerequisite for Resizable BAR |
| Resizable BAR | Enabled | Required for full 32GB VRAM addressing |
| PCIe slot link speed | Gen5 | Pinned explicitly so the arm is reproducible rather than negotiated |
| Initial display output | IGD | Dedicates the RTX 5090 to compute; a functional requirement |
| EXPO/XMP | **Disabled** | Measured as an arm in M2, not assumed |

Disabling EXPO is the one deliberate departure from the setup guide. Memory bandwidth is the
dominant term in decode, so EXPO is plausibly the single largest x86 decode lever available.
Placing it in the reference would make it permanently unmeasurable.

**OS.** Stock installation, plus swap disabled.

Swap is in the reference rather than in an arm because paging a 16GB model set is a
correctness hazard on a host with a 96GB container limit, not a performance trade. An arm
with swap enabled would measure the swap subsystem.

No sysctl tuning, no huge page reservation, default CPU governor, and default transparent
huge page settings are in the reference. Each is an arm.

**Container.** The values committed in `docker-compose.inference.yaml`, with `HUGEPAGES=off`.

**Shims.** None.

## Stage status

| Stage | What | Boot states | Status |
|-------|------|-------------|--------|
| M0 | Firmware reset, topology facts, reference pinned | 1 | not started |
| M1 | Noise floor and instrument calibration | 2 | not started |
| M2 | Positive controls | 2 | not started |
| M3 | Container runtime arms, coarse | 0 (shares M2's) | not started |
| M4 | Shims, including the extended huge page wrapper | 1 | not started |
| M5 | Container build arms | 0 (shares M4's) | not started |
| M6 | BIOS bundles | 3 to 4 | not started |
| M7 | Re-validation and ablation | 0 (shares M6's final) | not started |
| M8 | GPU track | 0 (drained inside each state above) | not started |

Prerequisite: entries B1 and B3 of
[../verification/bench/bench.md](../verification/bench/bench.md) must pass before M1 begins.
B1 proves the CLI reaches a service; B3 registers the models. Entry B2 is not a separate
prerequisite because M1 subsumes it.

## M0 Reset the firmware and pin the reference

Flash the current BIOS release, load optimized defaults, and apply only the reference
settings above. This discards the accumulated settings behind the archived measurements,
which is the point: their provenance is not recoverable, so they are a liability rather than
a starting position.

Resolve the three Tier 0 unknowns from
[ideation/os-optimization-candidates.md](ideation/os-optimization-candidates.md) in the same
sitting. Each is one command, and each gates a later stage.

| Unknown | Gates | Resolution |
|---------|-------|------------|
| Huge page size and pool size actually configured | M4 | `grep -r . /sys/kernel/mm/hugepages/*/nr_hugepages` and `grep Huge /proc/meminfo` |
| CPU topology behind `cpuset 0-15` | M3, M4, M6 | `lscpu -e=CPU,CORE,SOCKET,NODE,L3` |
| Active cpufreq driver | M3 | `cat /sys/devices/system/cpu/amd_pstate/status` and the `scaling_driver` file |

The topology answer is the load-bearing one. [README.md](README.md) and
[os/os-optimizations.md](os/os-optimizations.md) both describe cores 0-15 as the first CCX.
On a 9950X, Linux typically enumerates 0-15 as the 16 physical cores across both CCDs, with
16-31 as SMT siblings. If that holds here, the pinned container spans two CCDs and the L3 is
split, which changes the meaning of every thread-count and cpuset arm in M3 and makes the
CCD-aware shim in M4 worth building.

Whichever document contradicts the host is the bug, per the compose-and-Dockerfile rule in
`AGENTS.md`. Correct it before proceeding.

Pin the reference as a committed artifact: an `.env` fragment, the BIOS settings written
down, and a `bench` profile. Every result file from M1 onward records it in `conditions`.

**Acceptance:** all three unknowns answered and recorded; contradicting documents corrected;
`make health` green on all three services at the reference configuration.

## M1 Calibrate the instrument

The deliverable of this stage is a resolution, not a speed. A 5 percent regression threshold
sitting beneath an unmeasured spread is not a threshold, and this host has been recorded
spreading 44 percent across four consecutive runs of one unmodified binary
([ideation/rust-harness-optimization-candidates.md](ideation/rust-harness-optimization-candidates.md)).
Until the floor is known, every delta in M2 through M7 is uninterpretable.

Measure the floor three times, once after each step, so the progression itself is the result.

```bash
python3 -m bench noise --service llama-cpu --model <slug> --profile quick
```

**Step 1, reference floor.** Run `bench noise` at the M0 reference with the client pinned
off the service cpuset. Pinning matters: the rust-harness protocol prescribes `taskset -c
0-7`, but `llama-cpu` holds `cpuset 0-15`, so pinning the client into that range makes client
and server contend for the same cores and taints the measurement it was meant to stabilize.
Pin the client to cores outside the service cpuset.

**Step 2, runtime quieting.** These require no reboot and are applied together, because each
individually moves variance by less than the floor can resolve.

| Change | Mechanism | Target |
|--------|-----------|--------|
| `irqbalance` stopped and disabled | systemd | Variance |
| All IRQs moved off the inference cores | `/proc/irq/*/smp_affinity` | Variance |
| `kernel.numa_balancing=0` | sysctl | Variance |
| `vm.min_free_kbytes` raised to 1 to 2GB | sysctl | Avoids direct reclaim stalls mid-run |
| `vm.compaction_proactiveness=0` | sysctl | Stops background compaction during a run |

The IRQ work also settles an open question:
[os/os-optimizations.md](os/os-optimizations.md) records GPU IRQ affinity as IMPLEMENTED via
`/usr/local/bin/set-gpu-irq-affinity.sh`, and that file is not in the repository. Determine
whether the assignment exists on the host and whether it survives uptime with `irqbalance`
running.

**Step 3, boot quieting.** One reboot, adding to the kernel command line:

```
isolcpus=domain,managed_irq,0-15 nohz_full=0-15 rcu_nocbs=0-15
```

`nohz_full` is the load-bearing one. Each timer tick is an interrupt that evicts L1 and
pollutes the TLB, and decode is a tight loop across 16 threads for tens of milliseconds per
token. `isolcpus` removes the cores from the load balancer and `rcu_nocbs` moves RCU callback
processing to housekeeping cores.

A runtime alternative exists and is worth recording either way: cgroup v2 exclusive cpuset
partitions (`cpuset.cpus.partition=root`) provide scheduler isolation without a reboot. It is
less complete than `isolcpus` and composes with Docker's cpuset, so it is the fallback if the
boot parameters prove disruptive to the rest of the host.

**Acceptance:** three recorded floors. The step 3 floor becomes the threshold every later
stage compares against, and it is stated in every subsequent result.

**The reference advances here, once.** Steps 2 and 3 are conditions under which everything
else is measured, not optimizations claimed as wins. They are recorded in `conditions` on
every result file from this point.

## M2 Positive controls

Two arms, one per track, both with a large expected effect. If the rig cannot resolve these
cleanly against the M1 floor, the rig is the problem and no further stage is worth running.

| Arm | Change | Mechanism | Expected |
|-----|--------|-----------|----------|
| EXPO | Disabled versus enabled at DDR5-6000 | BIOS, one reboot pair | Large decode delta; decode is bandwidth-bound |
| Clock lock | GPU clocks unlocked versus locked | `nvidia-smi -lgc`, runtime | Large regression when locked; recorded at 38 percent |

The clock-lock arm is a control precisely because the answer is already known and negative.
Reproducing a known 38 percent regression validates the GPU measurement path at zero risk of
a surprising result.

EXPO leaves the reference at the end of this stage: enabled for all subsequent work, with the
delta recorded. This is the one settings change permitted outside M1 that does not reset the
campaign, because it is measured before it is adopted.

**Acceptance:** both deltas exceed the M1 floor by a wide margin and carry the expected sign.

## M3 Container runtime arms

The cheapest layer, swept first, and swept coarsely. The goal is a *representative* workload
shape rather than an optimal one: PBO's value in M6 depends on how many cores are loaded, and
the huge page wrapper's value in M4 depends on the working set. Both need the shape settled,
neither needs it optimal.

Every knob below is already read by `docker/llama-cpu/entrypoint.cpu.sh` and needs no code.

| Arm | Change | Phase | Notes |
|-----|--------|-------|-------|
| Threads | `THREADS` swept independently of `THREADS_BATCH` | Both | `THREADS` drives decode, `THREADS_BATCH` drives prefill; sweeping them together has never separated the two |
| CCD width | `cpuset` narrowed to one CCD's physical cores versus spanning both | Decode | Depends on the M0 topology answer; a bandwidth-bound decode may prefer eight cores sharing one L3 over 16 split across the Infinity Fabric |
| SMT | Inference cores versus their SMT siblings included | Both | Requires the M0 topology map to identify siblings |
| Repacking | `NO_REPACK` true versus false | Prefill | The entrypoint comment calls repacking the main CPU throughput lever on AVX-512, and it has never been measured |
| KV quantization | `CACHE_TYPE_K` and `CACHE_TYPE_V` at `q8_0` versus `f16` | Decode | Halves KV cache bandwidth; pair with `scripts/benchmark/test_perplexity.sh`, since this arm can change numerics |
| Batch shape | `BATCH_SIZE` and `UBATCH_SIZE` | Prefill | |
| Flash attention | `FLASH_ATTN` | Both | |
| Unified KV | `KV_UNIFIED` | Decode | Also determines whether the KV cache is one buffer or several, which M4 needs |

`CACHE_TYPE_K`/`CACHE_TYPE_V` is the only arm in this stage that can change output quality.
Run the perplexity script on both arms; a speed win paid for in numerics is not a win.

**Diagnosis worth running once, here.** Measure effective frequency separately per phase with
`perf stat -e cycles,ref-cycles`. Prefill is dense AVX-512 and may pull enough current to drop
clocks; decode is memory-bound and mostly stalls, so it should hold boost. Zen 5 implements
AVX-512 on a native 512-bit datapath and is far better behaved here than earlier Intel parts,
so this may be a null result. If it is not, it means the M6 BIOS bundle is largely a prefill
story, and this suite is the first tooling in the repository capable of showing that.

Two further diagnostics, one run each: `perf mem` for the load-latency distribution, and
`perf c2c` for false sharing across the 16 threads. With two CCDs and a shared KV cache,
cross-CCD cache-line transfer is plausible and entirely unmeasured.

**Acceptance:** a recorded thread count, cpuset and batch shape, each with its delta and
spread, adopted as the M3 working configuration for M4 through M6.

## M4 Shims

Depends on M0 (pool size, topology) and M3 (thread and cpuset shape).

**The huge page arm design.** Reserve the 1GB pool in *both* arms and toggle only the
`LD_PRELOAD`. Reserving 1GB pages requires a boot-time parameter, because runtime allocation
of 1GB pages is unreliable once memory fragments. Making pool reservation the difference
between arms would make the arm a boot-state pair and would confound the wrapper's effect
with the effect of removing that memory from the page cache. Reserving in both arms leaves
exactly one variable.

| Arm | Change | New code | Notes |
|-----|--------|----------|-------|
| Weights on 1GB pages | `HUGEPAGES=on` versus `off` | None | The arm the entrypoint defect made impossible until now |
| KV cache on 2MB pages | Wrapper extended to cover the KV cache allocation | Yes | Separate pool, so weights and KV stop competing; 2MB granularity tracks a context-length sweep without the rounding waste 1GB pages impose |
| Compute buffer on 2MB pages | Wrapper extended to cover ggml scratch | Yes | Prefill-targeted; scales with `BATCH_SIZE` and is on 4KB pages today |
| Page-size ladder | `select_page_size()` plus `round_up(size, page_size)` with a 1GB to 2MB to plain fallback | Yes | Replaces the hardcoded 1GB path and improves the existing weights arm |
| CCD-aware thread pinning | `pthread_create` intercepted, workers assigned to specific cores in CCD order | Yes | Removes thread migration entirely and tests the M3 CCD hypothesis directly without depending on llama.cpp's own handling |

The thread-affinity shim is the highest-value new code in the campaign. It is the same shape
as the existing wrapper, it makes "eight cores on one CCD" a first-class arm, and it can
optionally set `SCHED_FIFO` on workers. Gate the real-time option behind its own environment
variable and leave it off by default; a spinning real-time thread can make the host
unresponsive.

Before writing the KV extension, determine whether llama.cpp allocates the KV cache as one
buffer or several, and how that interacts with `KV_UNIFIED`. One known hazard: glibc's
`posix_memalign` over-allocates and then trims with a partial `munmap`, which cannot be
performed at an arbitrary offset on a hugetlb mapping. File-backed mappings are never trimmed,
so this hazard is new relative to the existing weights path.

Also run `llama.cpp`'s own `--numa` modes as a zero-code arm before building the shim. If they
already deliver the placement, the shim's scope narrows.

**Instrumentation.** `bench run --perf` scoped to the measurement window supplies
`dTLB-load-misses` and a page-table-walk event. The rust-harness ideation document names those
counters as the prerequisite for the huge page claim: without them the A/B shows the outcome
and not the mechanism.

**Acceptance:** the huge page arm produces a delta with TLB counters that corroborate its
direction, and the perplexity script confirms unchanged numerics.

## M5 Container build arms

Expensive to swap and therefore late, but drained inside M4's boot state.

| Arm | Change | Phase |
|-----|--------|-------|
| AOCL-BLIS | `GGML_BLAS=ON` with the AMD BLAS backend; currently OFF | Prefill only |
| AOCL-LibMem | AMD's optimized `memcpy`/`memset` via `LD_PRELOAD` | Both |
| Compiler flags | Current `-march=znver5` settings versus alternatives | Both |
| Allocator | tcmalloc and mimalloc | Decode |

The allocator arm re-opens a closed question deliberately. jemalloc is recorded at a 6.4
percent regression in [os/os-optimizations.md](os/os-optimizations.md), but that measurement
carries the same provenance problem as the 26.2 percent figure and was taken against a
baseline of roughly 13.5 tokens/second, which does not correspond to any current
configuration. The recorded conclusion may well hold; it is not currently anchored.

BLAS is a prefill-only bet. It replaces the batched matmul path and does nothing for the
single-token decode loop, so a decode-only comparison will report a null and be misread as a
failure. Report it against the prefill slope.

**Acceptance:** each arm reported against the phase it can plausibly affect, with a stated
null where the phase is unaffected.

## M6 BIOS bundles

Last, because it is the most expensive to swap, and now measured at a representative upper
stack rather than a naive one.

Bundle correlated knobs rather than isolating each one. Memory frequency and FCLK interact by
construction; PBO and the curve optimizer are one power story. Isolating them would multiply
reboots for deltas that cannot be interpreted separately anyway.

| Bundle | Contents | Phase |
|--------|----------|-------|
| Memory | FCLK 2100 MHz, UCLK=MEMCLK, power down disabled, memory context restore disabled | Decode |
| Power and boost | PBO limits, PBO scalar, boost clock override, curve optimizer, curve shaper | Prefill, per the M3 frequency diagnosis |
| Latency | Global C-states disabled, ECO mode disabled | Variance and TTFT |
| SMT | SMT disabled versus enabled | Both; interacts with the M3 cpuset finding |

The SMT bundle is placed here rather than in M3 because it is a firmware setting, but its
result is only interpretable against M3's cpuset and thread findings. Run it late in the
stage, after the other bundles have settled.

Motherboard profile slots, if the board provides them, reduce a bundle swap from full manual
re-entry to a single selection. Determine whether the X870E Aorus Elite WiFi exposes them
before starting; the answer changes how many bundles fit in a session.

**Acceptance:** each bundle's delta recorded against the M5 configuration, with reboot count
and session boundaries noted.

## M7 Re-validation and ablation

**Re-validation.** Re-run the M3 sweep at the final configuration. If the thread optimum moved
after the M6 power bundle landed, that is an interaction, it is the most interesting result
the campaign can produce, and it cost 20 minutes to find. If it did not move, the campaign has
earned the right to state that the deltas compose, which is a claim almost no published
benchmark supports.

**Ablation.** From the final configuration, turn each optimization off one at a time and
measure. This produces the table the whole campaign exists to fill, and it is the honest
denominator: every figure is measured at the configuration that ships.

Ablation covers, at minimum: EXPO, the huge page wrapper, core isolation, CCD pinning, thread
count, `NO_REPACK`, KV quantization, and each M6 bundle.

**Acceptance:** one promoted baseline per optimization, each carrying both its baseline delta
and its ablation delta, and a stated interaction where the two disagree by more than the M1
floor. The promotion path and its committed location are owned by `bench promote`, which is
not yet built; until it is, promoted results are written by hand into this directory tree
under the same envelope.

## M8 GPU track

Not a stage in sequence. The GPU queue is drained inside whichever boot state is current,
after that state's CPU queue, because GPU arms mostly do not depend on CPU-side OS tuning.

| Arm | Change | Mechanism |
|-----|--------|-----------|
| Kernel modules | Open versus proprietary NVIDIA modules | Module install, reboot |
| PCIe ASPM | `pcie_aspm=off` | Boot parameter |
| vLLM memory | `GPU_MEMORY_UTILIZATION` | Runtime |
| vLLM batching | `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS` | Runtime |
| vLLM KV dtype | `KV_CACHE_DTYPE` at fp8 versus auto | Runtime; pair with perplexity |
| vLLM attention | `ATTENTION_BACKEND` | Runtime |
| CUDA graphs | Default versus `--enforce-eager` | Requires an entrypoint argument |
| Chunked prefill | Enabled versus disabled | Requires an entrypoint argument |
| llama-gpu split | `SPLIT_MODE`, `NO_KV_OFFLOAD` | Runtime |

`docker/vllm-gpu/entrypoint.sh` already reads every environment variable in the runtime rows
above, but `docker-compose.inference.yaml` does not pass them, so they currently fall back to
defaults and cannot be varied. Adding them to the compose environment block is a prerequisite
for this track, in the same shape as the `HUGEPAGES` addition for `llama-cpu`.

The vLLM concurrency sweep is the one place in the campaign where continuous batching is
actually exercised. `bench` profiles already carry `[1, 4, 16]` for `vllm-gpu` against `[1]`
for the llama.cpp services.

## Session discipline

- **One GPU consumer at a time**, per `AGENTS.md`. CPU and GPU arms serialize within a boot
  state; they do not overlap.
- **Thermal drift aliases onto whatever is measured last.** `bench` interleaves repeats
  round-robin within one comparison, which handles drift inside a comparison but not across
  comparisons in a long session. Record session boundaries in the run notes, and do not
  compare a result from the start of a session against one from its fourth hour without
  re-running one of them.
- **A benchmark against an unhealthy service is not a slow result, it is no result.** Run
  `make status` and `make health` before each session; `bench run` gates on health, but a
  prior check makes a failure attributable.
- **Record the reboot count per stage.** It is the campaign's real cost metric and the only
  input that improves the next estimate.

## Publication mapping

The measurement order and the publication order differ, and that is intentional. Measurement
order follows redo cost; publication order follows pedagogy. The mapping holds because M0 pins
a named reference that every post cites.

| Stage | Post |
|-------|------|
| M1 | Benchmarking methodology; the floor progression is the payoff |
| M3, M5 | Inference containers series |
| M4 | Huge pages series |
| M6 | x86 BIOS |
| M1, M2 | x86 OS |
| M8 | GPU optimizations |
| M7 | Synthesis: what each layer was worth |

The synthesis post is the one the current outline lacks, and it carries the ablation table,
the baseline-versus-ablation gap, and the optimizations that turned out to be worth nothing.
Negative results are the rarest content in this genre and the repository already holds two of
them.

## See also

- [README.md](README.md) — validated optimizations and their measurements
- [../bench/method.md](../bench/method.md) — what makes a measurement citable, the rules
  this campaign's ordering serves
- [../bench/results.md](../bench/results.md) — the recorded measurements, and what each
  is still worth
- [../verification/bench/bench.md](../verification/bench/bench.md) — the acceptance runbook
  for the benchmark suite itself
- [../../bench/README.md](../../bench/README.md) — the suite that produces every number here
- [../../.claude/skills/benchmarking/SKILL.md](../../.claude/skills/benchmarking/SKILL.md) —
  the protocol the suite implements
- [../../.claude/skills/verify-host/SKILL.md](../../.claude/skills/verify-host/SKILL.md) —
  the handshake every host-side acceptance check follows
