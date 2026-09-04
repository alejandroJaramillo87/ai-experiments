# OS optimization candidates

**Parked proposals — recorded 2026-07-30. Items graduate into
[../os/os-optimizations.md](../os/os-optimizations.md) once measured; this document is
edited only as they do.**

A recorded menu of host-level OS and GPU optimization candidates for the
AMD Ryzen 9950X + RTX 5090 workstation. Nothing in
this document is implemented. Every item requires host-side verification run
by the human, with before/after numbers recorded per the repository
benchmarking protocol; candidates graduate into
[os-optimizations.md](../os/os-optimizations.md) with its status vocabulary
(IMPLEMENTED, TESTED - NOT BENEFICIAL) once measured. The implemented set
and the validated negative results (jemalloc, GPU clock locking) live in
that document and are not repeated here.

## Table of contents

- [Tier 0: verify before optimizing](#tier-0-verify-before-optimizing)
- [Tier A: likely wins](#tier-a-likely-wins)
- [Tier B: measure-worthy](#tier-b-measure-worthy)
- [Tier C: examined, nothing to do](#tier-c-examined-nothing-to-do)
- [Sequencing](#sequencing)

## Tier 0: verify before optimizing

Three facts about the current host are undetermined or contradicted in the
existing documentation. Each is one command; the answers gate Tier A.

**Huge-page configuration contradiction.**
[os-optimizations.md](../os/os-optimizations.md) records 80 x 1 GB pages
(`hugepages-1048576kB/nr_hugepages = 80`, wrapper allocating with
`MAP_HUGE_1GB`), while [../README.md](../README.md) records
`vm.nr_hugepages = 46080` (2 MB pages, 90 GB) and the 2 MB wrapper. Both
cannot describe the same host. Decide with:

```bash
grep -r . /sys/kernel/mm/hugepages/*/nr_hugepages
grep Huge /proc/meminfo
```

Whichever document loses is the documentation bug and gets corrected to
match the host.

**CPU topology behind cpuset "0-15".**
[os-optimizations.md](../os/os-optimizations.md) describes cores 0-15 as "first
CCX". On a 9950X, Linux typically enumerates cores 0-15 as the 16 physical
cores across both CCDs, with 16-31 as SMT siblings — in which case the
pinned container spans two CCDs rather than isolating one. Decide with:

```bash
lscpu -e=CPU,CORE,SOCKET,NODE,L3
```

The L3 column shows the CCD split. The answer determines whether CCD-aware
pinning (Tier A) is a change or already the accidental status quo.

**Active cpufreq driver.**
The performance-governor setting predates the maturity of the `amd_pstate`
driver. If the host is on `acpi-cpufreq`, the governor is a coarser
mechanism than the hardware supports; `amd_pstate` in active mode with the
performance energy-performance preference is the current recommendation for
Zen 5. Decide with:

```bash
cat /sys/devices/system/cpu/amd_pstate/status
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver
```

## Tier A: likely wins

**CCD-aware pinning.** Cross-CCD cache traffic is the classic Zen latency
tax: threads sharing an L3 communicate through it, threads on different
CCDs communicate through the Infinity Fabric. If Tier 0 shows cpuset
"0-15" spans both CCDs, pinning the inference threads to one CCD's
physical cores is a candidate latency and consistency win — and may
retroactively explain the validated finding that 12 dedicated cores
outperform 16 for a single model instance. Also applies to the rust
harness's planned threaded matvec
(`docs/optimizations/ideation/rust-harness-optimization-candidates.md`).

**Full core isolation.** The cpuset keeps containers off the housekeeping
cores but does not keep kernel threads, timers, or system daemons off the
inference cores. The remaining rungs: `nohz_full` and `rcu_nocbs` kernel
parameters for the inference cores, systemd `CPUAffinity` confining the
system slice to the housekeeping cores, and disabling `irqbalance` so the
existing GPU IRQ affinity assignment persists — `irqbalance` rewrites
`smp_affinity` at runtime, so first verify whether the current assignment
survives an hour of uptime:

```bash
for irq in $(grep -E "nvidia" /proc/interrupts | cut -d: -f1); do
    cat /proc/irq/$irq/smp_affinity; done
```

The target metric is consistency: the BIOS work already demonstrated a 59%
reduction in token-rate standard deviation, and isolation attacks the same
jitter sources.

**Open GPU kernel modules.** As of driver series 580.x, the open NVIDIA
kernel modules are the recommended default for Blackwell GPUs, and new
features land there first. Check which flavor the host runs:

```bash
modinfo nvidia | grep -i license
```

If proprietary, switching to the open modules is a low-risk,
forward-aligned change.

## Tier B: measure-worthy

**CPU mitigations.** `mitigations=off` recovers a few percent of syscall
and context-switch overhead on a single-user workstation. Recorded here as
considered and declined by default: agent sandboxes execute untrusted
tool output on this host, so weakening speculative-execution mitigations
trades real security posture for a small win. Revisit only if profiling
shows unusual syscall overhead.

**PCIe ASPM.** `pcie_aspm=off` (or the performance profile) removes link
power-state exit latency on the GPU path. Targets latency consistency
rather than throughput; cheap to test with the standard benchmark.

**Jitter trio.** `nmi_watchdog=0`, disabling the audit subsystem, and
disabling timer migration each shave sub-percent noise; collectively they
are visible in standard deviation rather than mean throughput. Same
consistency rationale as core isolation.

**sched_ext BPF schedulers.** `sched_ext` is mainline since kernel 6.12
and the host runs 6.18; schedulers in the scx family can outperform EEVDF
for pinned latency-sensitive workloads. Explicitly experimental: a
tinkering-budget item, not a production recommendation.

**GPU compute mode EXCLUSIVE_PROCESS.** `nvidia-smi -c EXCLUSIVE_PROCESS`
would make the repository's one-GPU-consumer rule kernel-enforced rather
than conventional. Caveat: it prevents two services from mapping the GPU
at once, so it only fits an operational pattern where exactly one GPU
service is up at a time; the current compose setup maps the GPU into
multiple service definitions.

## Tier C: examined, nothing to do

- **NUMA tuning** — single-node topology; `vm.zone_reclaim_mode = 0` is
  already set and sufficient.
- **CUDA MPS** — serves multi-process GPU sharing; the repository rule is
  one GPU consumer at a time.
- **UVM tunables** — relevant to VRAM oversubscription, which the serving
  configuration avoids by design.
- **GPU clock locking** — validated 38% regression; stays rejected.
- **jemalloc** — validated 6.4% regression for the latency workload; stays
  rejected.
- **GSP firmware and PAT driver options** — defaults on driver series
  580.x with the RTX 5090.

## Sequencing

1. Resolve the three Tier 0 checks first; they are single commands, they
   gate Tier A, and one of them ends in a documentation fix either way.
2. Implement candidates one at a time, each with before/after benchmark
   numbers from the standard suite, recorded verbatim.
3. Promote each measured candidate into
   [os-optimizations.md](../os/os-optimizations.md) under its status vocabulary,
   including negative results.
