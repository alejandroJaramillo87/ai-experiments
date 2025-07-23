# Simple Timeshift OS Backup to USB Drive

## Goal
Create a complete backup of your Ubuntu 24.04 system on a 1TB USB drive that you can restore from manually.

## What You Need
- 1TB USB external drive
- Ubuntu 24.04 system to backup

## Step 1: Prepare USB Drive

```bash
# Find your USB drive
lsblk

# Format the USB drive (replace /dev/sdX with your USB drive)
# WARNING: This erases everything on the USB drive
sudo fdisk /dev/sdX
# Press: n, p, Enter, Enter, Enter, w

# Format as ext4
sudo mkfs.ext4 /dev/sdX1 -L "TIMESHIFT_BACKUP"

# Create mount point and mount
sudo mkdir -p /media/timeshift-backup
sudo mount /dev/sdX1 /media/timeshift-backup
```

## Step 2: Install and Setup Timeshift

```bash
# Install timeshift (if not already installed)
sudo apt update
sudo apt install timeshift

# Setup timeshift to use your USB drive
sudo timeshift --setup --snapshot-device /dev/sdX1 --backup-type rsync
```

## Step 3: Create Complete System Backup

```bash
# Create a full system snapshot
sudo timeshift --create --comments "Complete OS backup $(date +%Y-%m-%d)"

# Check that it was created
sudo timeshift --list
```

## Step 4: You're Done!

Your backup is complete. The USB drive now contains your full system backup. You don't need to make it bootable - Timeshift handles everything.

## Step 5: Restore When Needed

### Method 1: Restore from Running System (Most Common)
```bash
# Mount USB drive
sudo mount /dev/sdX1 /media/timeshift-backup

# Tell timeshift to use USB snapshots
sudo timeshift --snapshot-device /dev/sdX1

# List available backups
sudo timeshift --list

# Restore (this will reboot your system)
sudo timeshift --restore --snapshot "2025-01-XX_XX-XX-XX"
```

### Method 2: If Your System Won't Boot At All
```bash
# 1. Boot from a separate Ubuntu Live USB (not your backup USB)
# 2. Install timeshift
sudo apt update && sudo apt install timeshift

# 3. Plug in your backup USB drive
# 4. Tell timeshift to use the backup USB
sudo timeshift --snapshot-device /dev/sdX1

# 5. List and restore
sudo timeshift --list
sudo timeshift --restore --snapshot "2025-01-XX_XX-XX-XX"
```
*Note: Timeshift will automatically handle mounting and restoring to your main drive.*

## Quick Commands Summary

**Create backup:**
```bash
sudo timeshift --create --comments "Manual backup $(date +%Y-%m-%d)"
```

**List backups:**
```bash
sudo timeshift --list
```

**Restore backup:**
```bash
sudo timeshift --restore --snapshot "SNAPSHOT_NAME"
```

## That's It!

This gives you a complete system backup on USB that you can restore manually when needed. The backup includes your entire OS, installed software, and configurations.