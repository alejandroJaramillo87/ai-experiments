# Hardware stress test verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 0 verified, 0 blocked, 6 pending, of 6.**

One-time firmware stability gate for the BIOS undervolt and overclock stack
documented in
[bios-optimizations.md](../optimizations/bios/bios-optimizations.md):
Curve Optimizer -15, Curve Shaper offsets, CPU boost clock override
+200 MHz, and FCLK 2100 MHz. That configuration was validated in 2025
against bursty chatbot inference — one subsystem loaded at a time, minutes
per session. The workloads now on the roadmap (multi-day training runs,
long harness sessions) hold the CPU and GPU at sustained load
simultaneously, a harsher envelope for power delivery and case thermals
than the original validation ever exercised. A marginal undervolt fails
silently: rare WHEA machine-check errors, bit flips, or a crash at hour 30
of a training run. This phase soaks the machine at worst-case combined
load and confirms the firmware profile is stable before any long run
trusts it.

Scope: this is a one-time gate, not a recurring benchmark. Re-run the
phase only after a BIOS settings change (any Curve Optimizer, Curve
Shaper, boost, FCLK, or memory timing adjustment). It is distinct from
the `hw_smoke` pytest tier, which checks the software environment, not
electrical stability. All entries are host-side: the human runs each
command on the workstation and records output per the
[recording protocol](README.md#recording-protocol). This phase is
independent of the software verification phases and may run any time
after its own preconditions are met, but it must be VERIFIED before the
first long training run (phase 2 of the training tier onward).

## Table of contents

- [HS.0 Preconditions and baseline](#hs0-preconditions-and-baseline)
- [HS.1 CPU sustained AVX-512 soak](#hs1-cpu-sustained-avx-512-soak)
- [HS.2 GPU sustained soak](#hs2-gpu-sustained-soak)
- [HS.3 Combined mixed-load soak](#hs3-combined-mixed-load-soak)
- [HS.4 Memory soak](#hs4-memory-soak)
- [HS.5 Post-soak sweep and disposition](#hs5-post-soak-sweep-and-disposition)
- [Failure protocol](#failure-protocol)

## HS.0 Preconditions and baseline

Implementation: the `status` target in the root `Makefile` and the
healthchecks in `docker-compose.inference.yaml`.

No other GPU consumer may be active for the duration of the phase:
inference services down, no benchmark in flight.

Command:

```bash
$ make status
$ journalctl -k -b | grep -iE "whea|mce|hardware error"
$ nvidia-smi -q -d ECC | grep -A4 "Aggregate"
```

Expected signal: `make status` shows all services stopped; the journal
sweep prints nothing (a clean baseline — any pre-existing WHEA or
machine-check line must be investigated before soaking); the ECC query
reports zero aggregate errors (or `N/A` on consumer firmware, which is
acceptable — Xid sweeps in later entries cover the GPU error path).

Install the stressors. `stress-ng` comes from the distribution archive;
`gpu-burn` builds from source and needs the Blackwell architecture flag;
Prime95 (`mprime`) downloads from mersenne.org. CoreCycler, referenced by
the BIOS guide for per-core Curve Optimizer validation, is a
Windows-only alternative and is not required here.

```bash
$ sudo apt-get install stress-ng
$ git clone https://github.com/wilicc/gpu-burn && cd gpu-burn
$ make COMPUTE=120   # sm_120, Blackwell
```

Observed:

```
```

Status: PENDING

## HS.1 CPU sustained AVX-512 soak

Implementation: none in this repository; the entry runs external stressors
(mprime, stress-ng) and reads `journalctl` and `sensors`.

All-core AVX-512 load for 60 minutes. Prime95 small FFTs with AVX-512
enabled is the harshest widely available stressor for this path;
`stress-ng --matrix` with forced AVX is the packaged fallback.

Command:

```bash
$ ./mprime -t   # Torture Test: Small FFTs, AVX-512 enabled, all 32 workers
# or, packaged fallback:
$ stress-ng --matrix 32 --matrix-method prod --timeout 60m --metrics
$ journalctl -k -b | grep -iE "whea|mce|hardware error"
$ sensors | grep -i tctl
```

Expected signal: 60 minutes with zero worker errors and no thermal
shutdown; the journal sweep prints nothing; record the sustained
all-core clock (`grep MHz /proc/cpuinfo` mid-run) and peak package
temperature in the Observed block.

Observed:

```
```

Status: PENDING

## HS.2 GPU sustained soak

Implementation: none in this repository; the entry runs the external
gpu-burn stressor and reads `journalctl` and `nvidia-smi`.

Sustained CUDA load for 60 minutes.

Command:

```bash
$ ./gpu_burn 3600
$ journalctl -k -b | grep -iE "xid"
$ nvidia-smi -q -d ECC | grep -A4 "Aggregate"
```

Expected signal: `gpu_burn` finishes with `OK` and zero errors; no Xid
lines in the journal; no ECC error increase over the HS.0 baseline;
record sustained clocks and peak temperature from `nvidia-smi` in the
Observed block.

Observed:

```
```

Status: PENDING

## HS.3 Combined mixed-load soak

Implementation: none in this repository; the entry runs the external
gpu-burn and mprime (or stress-ng) stressors and reads `journalctl`.

The actual gate. CPU and GPU stressors run simultaneously for a minimum
of three hours, so the power supply rails and case heat-soak see the
worst case a training run produces. Marginal undervolts commonly survive
a 20-minute test and fail after the case reaches thermal equilibrium;
the three-hour floor exists to get past that point.

Command:

```bash
$ ./gpu_burn 10800 &
$ ./mprime -t     # or the stress-ng fallback with --timeout 180m
$ journalctl -k -b | grep -iE "whea|mce|xid|hardware error"
```

Expected signal: both stressors run the full duration and finish
error-free; the journal sweep prints nothing; record peak CPU and GPU
temperatures, sustained clocks under combined load, and wall power if a
meter is available.

Observed:

```
```

Status: PENDING

## HS.4 Memory soak

Implementation: none in this repository; the entry runs the external
stress-ng stressor and reads `journalctl`.

EXPO DDR5-6000 with FCLK 2100 MHz and Memory Context Restore disabled is
itself an overclock and gets its own pass. Recommended rather than
blocking: HS.1 and HS.3 already exercise the memory controller heavily,
but this entry isolates it.

Command:

```bash
$ stress-ng --vm 4 --vm-bytes 100G --vm-method all --verify --timeout 60m
$ journalctl -k -b | grep -iE "whea|mce|hardware error"
```

Expected signal: zero verification failures; the journal sweep prints
nothing.

Observed:

```
```

Status: PENDING

## HS.5 Post-soak sweep and disposition

Implementation: none in this repository; the entry reads `journalctl` only.

Command:

```bash
$ journalctl -k -b | grep -iE "whea|mce|xid|hardware error"
```

Expected signal: nothing printed. With HS.1 through HS.3 VERIFIED and
this sweep clean, the BIOS profile is trusted for long runs; record the
disposition here and treat any later WHEA, machine-check, or Xid event
during real workloads as a firmware regression, not a code bug, until
proven otherwise.

Observed:

```
```

Status: PENDING

## Failure protocol

Any computation error, WHEA or machine-check line, Xid event, or crash
during the phase:

1. Back Curve Optimizer toward zero one step (-15 to -10 to -5 to 0),
   leaving all other settings in place.
2. Re-run HS.3 (the combined soak). Repeat until it passes clean.
3. Re-run the CPU benchmark suite and record the performance cost of the
   reduced offset against the validated 26.2 percent figure in
   [bios-optimizations.md](../optimizations/bios/bios-optimizations.md).
4. If errors persist at Curve Optimizer 0, remove the boost clock
   override next, then re-validate FCLK 2100 MHz; update the BIOS guide
   to match whatever configuration finally passes.
