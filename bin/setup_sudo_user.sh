#!/bin/bash

# Passwordless sudo setup script for AI engineering workstation
# Run this script as root or with sudo privileges

set -e  # Exit on any error

# Get the current user (or specify username as argument)
TARGET_USER=${1:-$SUDO_USER}

if [ -z "$TARGET_USER" ]; then
    echo "Error: Could not determine target user"
    echo "Usage: sudo $0 [username]"
    exit 1
fi

echo "Setting up passwordless sudo for user: $TARGET_USER"

# Add user to sudo group (if not already there)
usermod -aG sudo "$TARGET_USER"

# Add user to other useful groups for AI/development work
usermod -aG docker "$TARGET_USER" 2>/dev/null || echo "Docker group not found (install Docker later)"
usermod -aG video "$TARGET_USER"    # For GPU access
usermod -aG render "$TARGET_USER"   # For GPU rendering access

# Create sudoers file for passwordless sudo
SUDOERS_FILE="/etc/sudoers.d/99-${TARGET_USER}-nopasswd"

echo "Creating sudoers file: $SUDOERS_FILE"
cat > "$SUDOERS_FILE" << EOF
# Passwordless sudo for $TARGET_USER
$TARGET_USER ALL=(ALL) NOPASSWD: ALL
EOF

# Set proper permissions on sudoers file
chmod 440 "$SUDOERS_FILE"

# Verify the sudoers file syntax
if visudo -cf "$SUDOERS_FILE"; then
    echo "✓ Sudoers file created successfully"
else
    echo "✗ Error in sudoers file syntax"
    rm -f "$SUDOERS_FILE"
    exit 1
fi

echo "✓ Setup complete for user: $TARGET_USER"
echo "✓ User added to groups: sudo, video, render"
echo "✓ Passwordless sudo enabled"
echo ""
echo "The user will need to log out and log back in for group changes to take effect."