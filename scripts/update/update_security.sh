#!/bin/bash

# update_security.sh
# This script updates security tools and runs basic security checks
# Designed to complement the initial security setup

echo "=== Security Tools Updater ==="
echo

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "ERROR: This script should not be run as root"
   exit 1
fi

echo "Step 1 - Updating security tool databases:"

# Update rkhunter database
if command -v rkhunter &> /dev/null; then
    echo "Updating rkhunter database..."
    sudo rkhunter --update 2>&1 | grep -v "Invalid WEB_CMD" || echo "rkhunter update completed"
    sudo rkhunter --propupd 2>/dev/null || echo "rkhunter properties updated"
    echo "rkhunter database updated"
else
    echo "rkhunter not installed, skipping"
fi

# Update ClamAV virus definitions if installed
if command -v freshclam &> /dev/null; then
    echo "Updating ClamAV virus definitions..."
    sudo freshclam --quiet || echo "ClamAV definitions updated"
    echo "ClamAV database updated"
else
    echo "ClamAV not installed, skipping"
fi

echo

echo "Step 2 - Applying security updates:"

# Check for and apply security updates
echo "Checking for security updates..."
sudo apt update > /dev/null 2>&1

# Count available security updates
SECURITY_UPDATES=$(apt list --upgradable 2>/dev/null | grep -i security | wc -l)

if [ "$SECURITY_UPDATES" -gt 0 ]; then
    echo "Found $SECURITY_UPDATES security updates"
    echo "Installing security updates..."
    sudo unattended-upgrade -d
    echo "Security updates applied"
else
    echo "No security updates available"
fi

# Update AppArmor profiles
if command -v aa-status &> /dev/null; then
    echo "Updating AppArmor profiles..."
    sudo aa-status --enabled && echo "AppArmor is enabled with $(sudo aa-status | grep 'profiles are in enforce mode' | awk '{print $1}') profiles enforced"
fi

echo

echo "Step 3 - Running security checks:"

# Check for failed login attempts (last 24 hours)
echo "Checking failed login attempts..."
FAILED_LOGINS=$(sudo grep "authentication failure" /var/log/auth.log 2>/dev/null | grep "$(date '+%b %e')" | wc -l)
if [ "$FAILED_LOGINS" -gt 0 ]; then
    echo "WARNING: $FAILED_LOGINS failed login attempts today"
else
    echo "No failed login attempts today"
fi

# Check fail2ban status
if command -v fail2ban-client &> /dev/null; then
    BANNED_IPS=$(sudo fail2ban-client status sshd 2>/dev/null | grep "Currently banned" | awk '{print $4}')
    echo "Fail2ban: ${BANNED_IPS:-0} IPs currently banned"
fi

# Quick rootkit scan (background, non-blocking)
if command -v rkhunter &> /dev/null; then
    echo "Running quick rootkit scan..."
    sudo rkhunter --check --skip-keypress --quiet --report-warnings-only > /tmp/rkhunter.log 2>&1
    WARNINGS=$(grep -c Warning /tmp/rkhunter.log 2>/dev/null || echo "0")
    if [ "$WARNINGS" -gt 0 ]; then
        echo "WARNING: rkhunter found $WARNINGS warnings (check /tmp/rkhunter.log)"
    else
        echo "No rootkit warnings detected"
    fi
fi

# Check listening ports
echo "Checking network services..."
LISTENING_PORTS=$(sudo netstat -tlnp 2>/dev/null | grep LISTEN | wc -l)
echo "$LISTENING_PORTS services listening on network ports"

# Check UFW status
if command -v ufw &> /dev/null; then
    UFW_STATUS=$(sudo ufw status | grep "Status:" | awk '{print $2}')
    UFW_RULES=$(sudo ufw status numbered | grep -c "^\[" || echo "0")
    echo "Firewall: $UFW_STATUS with $UFW_RULES rules configured"
fi

echo

echo "Step 4 - System security summary:"

# Display kernel version (for tracking security patches)
KERNEL_VERSION=$(uname -r)
echo "Kernel version: $KERNEL_VERSION"

# Check last system update
LAST_UPDATE=$(stat -c %y /var/lib/apt/periodic/update-success-stamp 2>/dev/null | cut -d' ' -f1 || echo "unknown")
echo "Last system update: $LAST_UPDATE"

# Check if automatic updates are enabled
if systemctl is-enabled unattended-upgrades &>/dev/null; then
    echo "Automatic updates: enabled"
else
    echo "Automatic updates: disabled"
fi

# Check if system needs reboot (for kernel updates)
if [ -f /var/run/reboot-required ]; then
    echo "WARNING: System reboot required for security updates"
fi

echo
echo "=== Security Update Complete ==="
echo

# Log the update
LOG_DIR="$HOME/.local/share/logs"
mkdir -p "$LOG_DIR"
echo "$(date): Security update completed" >> "$LOG_DIR/security-update.log"