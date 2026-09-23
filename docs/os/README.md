# Operating system

Configuring Ubuntu 24.04 LTS for AI work on the AMD Ryzen 9950X + RTX 5090 hardware. The
walkthrough runs from installation through the GPU stack, Docker and the Python
environment; one directory or page per subsystem owns the detail.

## Directories

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [gpu-stack/](gpu-stack/README.md) | The NVIDIA software stack under every GPU workload on this workstation: the driver, the CUDA toolkit and cuDNN, one page each, with an overview of how the three layers fit together and how the installation is verified. |
| [python/](python/README.md) | The uv-managed Python environment: how it is installed, the annotated reference for every dependency pin, and how the PyTorch CUDA build is changed. |

<!-- inventory:dirs:end -->

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [backup_and_recovery.md](backup_and_recovery.md) | Imaging the OS drive with Clonezilla and restoring it, with the data drive kept out of every operation. |
| [storage-configuration.md](storage-configuration.md) | Dual-SSD architecture configuration for AI workstation with dedicated OS and data drives optimized for model loading and inference workloads. |
| [workstation-setup.md](workstation-setup.md) | Configuring Ubuntu 24.04 LTS for AI work on the AMD Ryzen 9950X + RTX 5090 hardware. |

<!-- inventory:pages:end -->

Read the walkthrough first and follow its links out: it references the automation under
`scripts/setup/` and `scripts/update/`, which are reviewed before they are run.

## Related documentation

- [../bios/README.md](../bios/README.md) — the firmware the OS boots from
- [../sandbox/README.md](../sandbox/README.md) — how the services are contained once
  Docker is installed
- [../optimizations/os/README.md](../optimizations/os/README.md) — the host tuning applied
  after setup
