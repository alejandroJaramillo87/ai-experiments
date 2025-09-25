# Complete System Backup and Recovery with Clonezilla

## Goal
Create a complete backup of your Ubuntu 24.04 system (nvme1n1) on a 1TB USB drive that can be restored at the BIOS level, even when the OS and UEFI boot are corrupted.

## What You Need
- 1TB USB external drive (for backup storage)
- Separate USB drive for Clonezilla Live (8GB+ recommended)
- Ubuntu 24.04 system with nvme1n1 SSD to backup

## Step 1: Prepare USB Drives

### Backup Storage USB (1TB)
```bash
# Find your USB drives
lsblk

# Format the 1TB USB drive (replace /dev/sdX with your 1TB USB drive)
# WARNING: This erases everything on the USB drive
sudo fdisk /dev/sdX
# Press: n, p, Enter, Enter, Enter, w

# Format as ext4
sudo mkfs.ext4 /dev/sdX1 -L "CLONEZILLA_BACKUP"
```

### Clonezilla Live USB
1. Download Clonezilla Live ISO from: https://clonezilla.org/downloads.php
2. Create bootable USB with `dd` or balenaEtcher:
```bash
# Using dd (replace /dev/sdY with your Clonezilla USB drive)
sudo dd if=clonezilla-live-*.iso of=/dev/sdY bs=4M status=progress
sync
```

## Step 2: Create System Backup

### Boot into Clonezilla Live
1. Insert both USB drives (1TB backup + Clonezilla Live)
2. Restart computer and enter BIOS/UEFI (F2/F12/DEL during startup)
3. Set Clonezilla Live USB as first boot device
4. Boot into Clonezilla Live environment

### Navigate Clonezilla Menus
1. **Boot Screen**: Choose "Clonezilla live"
2. **Language**: Select "English" (or your preference)
3. **Keymap**: Choose "Keep default"
4. **Start**: Select "Start Clonezilla"

### Configure Backup Operation
1. **Mode**: Choose "device-image" (work with disk/partition to/from image)
2. **Experience Level**: Select "Beginner mode"
3. **Repository**: Choose "local_dev" (use local device for image repository)

### Mount Backup Drive
1. Wait for drives to be detected
2. Select your 1TB USB drive (likely /dev/sdb or /dev/sdc)
3. Choose the ext4 partition on your backup USB drive
4. Press Enter to mount

### Create Disk Image
1. **Operation**: Select "savedisk" (save local disk as an image)
2. **Image Name**: Enter descriptive name like `nvme1n1_backup_2025-01-09`
3. **Source Disk**: Select "/dev/nvme1n1" (your OS drive)
4. **Compression**: Keep default settings (good balance of speed/size)
5. **Verification**: Enable image verification (recommended)

### Start Backup Process
1. Review all settings carefully
2. Confirm to start backup process
3. Wait 30-60 minutes for completion
4. Verify success message and image integrity check

## Step 3: System Recovery Options

### Method 1: Normal System Recovery
*When your system boots but needs restoration*

```bash
# Boot from Clonezilla Live USB
# Follow menu navigation as above, but choose:
# - "device-image" → "Beginner mode" → "local_dev"
# - Mount your backup USB drive
# - Choose "restoredisk" (restore disk from image)
# - Select your backup image
# - Target disk: /dev/nvme1n1
# - Confirm and start restoration (15-30 minutes)
```

### Method 2: Emergency BIOS-Level Recovery
*When OS won't boot, UEFI is corrupted, or complete system failure*

1. **Boot from Clonezilla Live USB**
   - Enter BIOS/UEFI settings during startup
   - Set Clonezilla Live USB as primary boot device
   - Clonezilla runs independently of your corrupted system

2. **Connect Backup USB**
   - Insert your 1TB USB with the backup image
   - Clonezilla will detect both drives automatically

3. **Restore Process**
   - Choose "device-image" → "Beginner mode" → "local_dev"
   - Mount your backup USB drive
   - Select "restoredisk" (restore disk from image)
   - Choose your backup image
   - Target: "/dev/nvme1n1" (your SSD)
   - Confirm restoration

4. **Automatic UEFI Recovery**
   - Clonezilla restores exact partition table and ESP
   - UEFI boot loaders and variables are reconstructed
   - No manual boot repair needed

5. **Complete Recovery**
   - Remove USB drives and reboot
   - System boots normally with all data intact

## Step 4: Verification and Maintenance

### Verify Successful Backup
```bash
# After backup creation, Clonezilla shows:
# - Image creation status: "Completed successfully"
# - Image verification: "Image integrity check passed"
# - Backup location and size information
```

### Regular Backup Schedule
```bash
# Create new backup before risky operations:
# 1. Boot Clonezilla Live USB
# 2. Follow backup steps with new image name
# 3. Keep multiple backup images if storage allows
```

### Image Management
- Each backup image takes ~300-500GB (compressed)
- Name images with dates for easy identification
- Keep at least 2-3 recent backups if space allows
- Oldest images can be deleted to free space

## Quick Reference Commands

### Backup Creation Timeline
- Boot into Clonezilla Live: 2-3 minutes
- Menu navigation and setup: 5 minutes  
- Actual backup process: 30-60 minutes
- **Total time**: 40-70 minutes

### Restoration Timeline
- Boot into Clonezilla Live: 2-3 minutes
- Menu navigation and setup: 5 minutes
- Actual restore process: 15-30 minutes
- **Total time**: 25-40 minutes

### Drive Identification Help
```bash
# In Clonezilla, drives appear as:
# /dev/nvme1n1 - Your 1TB NVMe SSD (source/target)
# /dev/sdb or /dev/sdc - Your 1TB backup USB drive
# Always verify drive sizes before proceeding
```

## Important Notes

- **UEFI Restoration**: Clonezilla preserves exact boot configuration
- **System State**: Everything is restored exactly as backed up
- **Hardware Independence**: Clonezilla Live runs regardless of system state
- **Data Safety**: Always verify drive selection before starting operations
- **Multiple Backups**: Keep recent backups; delete old ones when space needed

## Emergency Recovery Capability

This method works even when:
- UEFI boot partition is completely destroyed
- Partition table is corrupted or wiped
- Operating system won't start at all
- Hard drive is completely reformatted
- System has been infected with malware

The Clonezilla Live environment operates independently of your installed system, making it ideal for emergency recovery situations.