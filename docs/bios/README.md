# BIOS configuration

The UEFI firmware setup of the workstation's Gigabyte X870E Aorus Elite WiFi board under
the AMD Ryzen 9950X and RTX 5090: the update procedure, POST-code diagnostics and the
EXPO memory profile, chosen for stability rather than speed.

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [bios-configuration.md](bios-configuration.md) | BIOS setup and configuration guide for the AMD Ryzen 9950X + RTX 5090 AI engineering workstation, focusing on system stability, memory optimization, and diagnostic capabilities for sustained AI workloads on the Gigabyte X870E Aorus Elite WiFi motherboard. |

<!-- inventory:pages:end -->

The settings that trade stability for inference speed are a different document,
[../optimizations/bios/README.md](../optimizations/bios/README.md), and the firmware
stability gate that exercises them is the runbook in
[../verification/hardware/README.md](../verification/hardware/README.md).

## Related documentation

- [../hardware/README.md](../hardware/README.md) — the components this firmware configures
- [../os/README.md](../os/README.md) — the operating system installed once the board posts
