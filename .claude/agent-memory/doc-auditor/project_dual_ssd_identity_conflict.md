---
name: dual-ssd-identity-conflict
description: RESOLVED 2026-08-02 by host lsblk — nvme1n1 is the 990 PRO 2TB AI data drive, nvme0n1 is the 990 EVO Plus 1TB OS drive
metadata:
  type: project
---

Resolved 2026-08-02 from `lsblk -o NAME,SIZE,MODEL,FSTYPE,LABEL,MOUNTPOINT` and
`blkid -L AIENGINEERING` run on the workstation:

- `nvme1n1` — Samsung SSD 990 **PRO** 2TB — AI data drive. `nvme1n1p2` is 1.8T ext4,
  `LABEL=AIENGINEERING`, mounted at `/mnt/ai-data`.
- `nvme0n1` — Samsung SSD 990 **EVO Plus** 1TB (931.5G) — OS drive. `nvme0n1p2` is
  930.5G ext4 at `/`; `nvme0n1p1` is the 1G EFI partition at `/boot/efi`.

`README.md:205` and `docs/hardware/README.md:159-169` were correct. Wrong and since
corrected: `docs/os/storage-configuration.md` (device nodes right, drive models swapped),
`docs/os/README.md` (assignment fully inverted), and `docs/os/backup_and_recovery.md`,
which called `/dev/nvme1n1` the OS drive and named it as a restore *target* — that would
have overwritten `/mnt/ai-data`.

**Why:** No repo file recorded the physical layout, so three documents drifted into three
incompatible positions and the tree could not adjudicate. The resolution came from host
output, which is the only authoritative source for block-device identity.

**How to apply:** Treat the mapping above as settled; do not re-derive it from prose. Still
resolve the AI drive by `blkid -L AIENGINEERING` rather than a node path in any script or
command — node numbering is not stable across reboots, which is how this drift began.
PCIe generation and sequential-throughput figures for these drives remain **unverified**
against this machine and conflict between documents; a 990 PRO is a PCIe 4.0 device, so
neither the 12,400 MB/s nor the 5,000 MB/s claim should be repeated as measured fact.
Related: [[doc-audit-storage-tree-sources]].
