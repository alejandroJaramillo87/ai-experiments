# Hardware tier verification

The runbook for the workstation's firmware stability gate: the BIOS undervolt and
overclock stack under sustained load, executed once and re-executed when the settings
change.

Operational: the operator's pasted output is the evidence, and the page's progress line
says how far along it is.

## Runbooks

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [hardware-stress.md](hardware-stress.md) | One-time firmware stability gate for the BIOS undervolt and overclock stack documented in [bios-optimizations.md](../../optimizations/bios/bios-optimizations.md): Curve Optimizer -15, Curve Shaper offsets, CPU boost clock override +200 MHz, and FCLK 2100 MHz. |

<!-- inventory:pages:end -->

## Related documentation

- [../README.md](../README.md) — the campaign and the recording protocol
- [../../bios/README.md](../../bios/README.md) — the settings under test
- [../../hardware/README.md](../../hardware/README.md) — the build they run on
