#!/bin/bash

# Ubuntu 24.04 AI Workstation Security Hardening Script
# Designed for AI engineering workflows with RTX 5090 + AMD 9950
# Balances security with development flexibility

set -e

echo "🔒 Starting Ubuntu AI Workstation Security Hardening..."
echo "This script will secure your system while preserving AI development workflows"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root. Run as your regular user with sudo access."
   exit 1
fi

# 1. UPDATE SYSTEM AND ENABLE AUTOMATIC UPDATES
print_status "Setting up automatic security updates..."
sudo apt update && sudo apt upgrade -y

# Install unattended-upgrades
sudo apt install -y unattended-upgrades apt-listchanges

# Configure automatic updates
sudo dpkg-reconfigure -plow unattended-upgrades

# Configure unattended-upgrades for security updates only
sudo tee /etc/apt/apt.conf.d/20auto-upgrades > /dev/null <<EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Download-Upgradeable-Packages "1";
EOF

# Configure what gets updated automatically
sudo tee -a /etc/apt/apt.conf.d/50unattended-upgrades > /dev/null <<EOF
// Only install security updates automatically
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};

// Automatically reboot if required (important for kernel updates)
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "03:00";

// Send email notifications (configure your email)
// Unattended-Upgrade::Mail "your-email@example.com";
EOF

# 2. CONFIGURE UFW FIREWALL FOR AI WORKSTATION
print_status "Configuring UFW firewall for AI development..."
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (change port if you use custom SSH port)
sudo ufw allow 22/tcp

# Allow common AI/ML development ports
sudo ufw allow 8888/tcp comment "Jupyter Notebook"
sudo ufw allow 8080/tcp comment "Development servers"
sudo ufw allow 3000/tcp comment "React/Node dev servers"
sudo ufw allow 5000/tcp comment "Flask dev servers"
sudo ufw allow 8000/tcp comment "Django/Python dev servers"
sudo ufw allow 6006/tcp comment "TensorBoard"
sudo ufw allow 7860/tcp comment "Gradio apps"
sudo ufw allow 8501/tcp comment "Streamlit apps"

# Allow Docker-related ports (if using Docker containers)
sudo ufw allow from 172.17.0.0/16 comment "Docker containers"

# Enable firewall
sudo ufw --force enable



# 4. INSTALL AND CONFIGURE FAIL2BAN
print_status "Installing and configuring Fail2Ban..."
sudo apt install -y fail2ban

# Configure fail2ban for SSH and common attacks
sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[ssh]
enabled = true
port = 22
filter = sshd
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600

[nginx-http-auth]
enabled = false

[nginx-noscript]
enabled = false

[nginx-badbots]
enabled = false

[apache-auth]
enabled = false

[apache-badbots]
enabled = false
EOF

sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# 5. CONFIGURE SYSTEM AUDITING
print_status "Setting up system auditing..."
sudo apt install -y auditd audispd-plugins

# Basic audit rules for security monitoring
sudo tee /etc/audit/rules.d/audit.rules > /dev/null <<EOF
# Monitor file access
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/sudoers -p wa -k identity

# Monitor authentication
-w /var/log/auth.log -p wa -k authentication
-w /var/log/faillog -p wa -k authentication

# Monitor system calls
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time-change

# Monitor network configuration
-a always,exit -F arch=b64 -S sethostname -S setdomainname -k system-locale
-a always,exit -F arch=b32 -S sethostname -S setdomainname -k system-locale

# Monitor sudo usage
-w /var/log/sudo.log -p wa -k actions
EOF

sudo systemctl enable auditd
sudo systemctl start auditd

# 6. SECURE KERNEL PARAMETERS
print_status "Configuring secure kernel parameters..."
sudo tee /etc/sysctl.d/99-security.conf > /dev/null <<EOF

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK SECURITY PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

# IP Spoofing Protection (Reverse Path Filtering)
# Prevents attackers from sending packets with fake source IP addresses
# Value 1 = Strict mode: drop packets if return path differs from incoming path
# Value 0 = Disabled (vulnerable to IP spoofing attacks)
net.ipv4.conf.all.rp_filter = 1          # Apply to all network interfaces
net.ipv4.conf.default.rp_filter = 1      # Apply to future interfaces by default

# ICMP Redirect Protection
# ICMP redirects can be used by attackers to hijack network traffic
# by convincing your system to route traffic through malicious hosts
# Value 0 = Ignore ICMP redirect messages (secure)
# Value 1 = Accept ICMP redirects (vulnerable to man-in-the-middle attacks)
net.ipv4.conf.all.accept_redirects = 0      # Ignore IPv4 redirects on all interfaces
net.ipv6.conf.all.accept_redirects = 0      # Ignore IPv6 redirects on all interfaces  
net.ipv4.conf.default.accept_redirects = 0  # Default for new IPv4 interfaces
net.ipv6.conf.default.accept_redirects = 0  # Default for new IPv6 interfaces

# Prevent Sending ICMP Redirects
# Your system shouldn't send ICMP redirects to other hosts
# as this can leak network topology information to attackers
# Value 0 = Don't send ICMP redirects (secure)
# Value 1 = Send ICMP redirects (information disclosure risk)
net.ipv4.conf.all.send_redirects = 0        # Never send redirects from any interface
net.ipv4.conf.default.send_redirects = 0    # Default for new interfaces

# Source Route Protection
# Source routing allows the sender to specify the route packets should take
# This can be abused to bypass firewall rules and routing policies
# Value 0 = Reject source-routed packets (secure)
# Value 1 = Accept source-routed packets (vulnerable to routing attacks)
net.ipv4.conf.all.accept_source_route = 0      # Block IPv4 source routing
net.ipv6.conf.all.accept_source_route = 0      # Block IPv6 source routing
net.ipv4.conf.default.accept_source_route = 0  # Default for new IPv4 interfaces
net.ipv6.conf.default.accept_source_route = 0  # Default for new IPv6 interfaces

# Log Martian Packets (Impossible Addresses)
# "Martian" packets have source addresses that shouldn't exist on your network
# Examples: private IPs from internet, localhost from external sources
# Value 1 = Log these suspicious packets for security monitoring
# Value 0 = Silently drop without logging
net.ipv4.conf.all.log_martians = 1          # Log suspicious packets on all interfaces
net.ipv4.conf.default.log_martians = 1      # Log on future interfaces

# ICMP Echo (Ping) Response Policy
# Responding to pings can reveal that your system exists to network scanners
# Value 0 = Respond to ping requests (normal behavior, but reveals presence)
# Value 1 = Ignore ping requests (stealth mode, breaks some network diagnostics)
# Set to 0 here to maintain normal network functionality for AI development
net.ipv4.icmp_echo_ignore_all = 0

# Ignore Bogus ICMP Error Messages
# Malformed ICMP errors can be used in various network attacks
# Value 1 = Ignore malformed ICMP error messages (secure)
# Value 0 = Process all ICMP errors (vulnerable to ICMP-based attacks)
net.ipv4.icmp_ignore_bogus_error_responses = 1

# ═══════════════════════════════════════════════════════════════════════════════
# TCP SYN FLOOD PROTECTION
# ═══════════════════════════════════════════════════════════════════════════════

# SYN Flood Attack Protection
# SYN floods overwhelm servers by sending many connection requests without completing them
# SYN cookies allow the server to handle connections without storing state
# Value 1 = Enable SYN cookies when under attack (essential protection)
# Value 0 = Disable SYN cookies (vulnerable to SYN flood attacks)
net.ipv4.tcp_syncookies = 1

# Maximum SYN Backlog Queue Size
# How many half-open connections can be queued before SYN cookies activate
# Higher values = can handle more legitimate simultaneous connections
# Lower values = faster activation of SYN cookie protection
# 2048 is a good balance for AI workstations that might run web services
net.ipv4.tcp_max_syn_backlog = 2048

# SYN-ACK Retry Attempts
# How many times to retry sending SYN-ACK packets for incoming connections
# Lower values = faster detection of SYN flood attacks
# Higher values = more tolerance for legitimate slow connections
# 2 retries provides good protection while maintaining compatibility
net.ipv4.tcp_synack_retries = 2

# SYN Retry Attempts (Outgoing Connections)
# How many times to retry SYN packets when making outgoing connections
# 5 retries provides good reliability for AI workstation's outbound connections
# (downloading models, connecting to remote services, etc.)
net.ipv4.tcp_syn_retries = 5

# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY AND PROCESS SECURITY (Critical for AI Workloads)
# ═══════════════════════════════════════════════════════════════════════════════

# Address Space Layout Randomization (ASLR)
# Randomizes memory layout to prevent buffer overflow exploits
# Value 2 = Full randomization (strongest protection)
# Value 1 = Partial randomization  
# Value 0 = No randomization (vulnerable to memory-based attacks)
# Essential for AI workstations handling large datasets and complex computations
kernel.randomize_va_space = 2

# Kernel Pointer Restriction
# Prevents unprivileged users from reading kernel memory addresses
# Value 1 = Hide kernel pointers from unprivileged users (secure)
# Value 0 = Allow reading kernel pointers (information disclosure risk)
# Important for AI workstations running untrusted ML code or models
kernel.kptr_restrict = 1

# Kernel Message Restriction
# Controls who can read kernel debug messages (dmesg)
# Value 1 = Only privileged users can read kernel messages
# Value 0 = All users can read kernel messages (information disclosure)
# Protects against information leakage from kernel logs
kernel.dmesg_restrict = 1

# Process Tracing (Debugging) Restrictions
# Controls which processes can be debugged by unprivileged users
# Value 1 = Only allow tracing child processes (secure)
# Value 0 = Allow tracing any process owned by same user (less secure)
# Prevents malicious code from debugging other processes to steal data
kernel.yama.ptrace_scope = 1

# ═══════════════════════════════════════════════════════════════════════════════
# CORE DUMP SECURITY
# ═══════════════════════════════════════════════════════════════════════════════

# SUID Program Core Dump Control
# SUID programs run with elevated privileges and their core dumps could contain sensitive data
# Value 0 = Disable core dumps for SUID programs (secure)
# Value 1 = Allow core dumps for SUID programs (data leakage risk)
fs.suid_dumpable = 0

# Core Dump File Location
# Redirect all core dumps to /dev/null (discard them)
# This prevents sensitive data from AI applications being written to disk
# Default would be to write core dumps to current directory
# For AI workstations, core dumps could contain model weights or training data
kernel.core_pattern = /dev/null

EOF

# Apply all the security settings immediately
# -p flag loads settings from the specified file
sudo sysctl -p /etc/sysctl.d/99-security.conf

# 7. CONFIGURE APPARMOR (already enabled by default in Ubuntu 24.04)
print_status "Ensuring AppArmor is active..."
sudo systemctl enable apparmor
sudo systemctl start apparmor

# Install additional AppArmor profiles
sudo apt install -y apparmor-profiles apparmor-utils

# 8. SECURE SHARED MEMORY
print_status "Securing shared memory..."
if ! grep -q "tmpfs /run/shm" /etc/fstab; then
    echo "tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0" | sudo tee -a /etc/fstab
fi

# 9. INSTALL SECURITY TOOLS
print_status "Installing security monitoring tools..."
sudo apt install -y \
    rkhunter \
    chkrootkit \
    lynis \
    aide \
    logwatch \
    tiger

# Configure rkhunter
# sudo rkhunter --update
# sudo rkhunter --propupd

# 10. CONFIGURE SECURE PACKAGE MANAGEMENT
print_status "Securing package management..."
# Ensure only signed packages are installed
sudo tee /etc/apt/apt.conf.d/99-security > /dev/null <<EOF
APT::Get::AllowUnauthenticated "false";
APT::Get::Assume-Yes "false";
APT::Install-Recommends "false";
APT::Install-Suggests "false";
EOF

# 11. CONFIGURE AUTOMATIC SECURITY SCANS
print_status "Setting up automated security scans..."
sudo tee /etc/cron.daily/security-scan > /dev/null <<'EOF'
#!/bin/bash
# Daily security scan script

# Run rkhunter scan
/usr/bin/rkhunter --check --skip-keypress --report-warnings-only

# Run chkrootkit
/usr/sbin/chkrootkit

# Check for failed login attempts
/usr/bin/lastb | head -20

# Check listening ports
/bin/netstat -tuln
EOF

sudo chmod +x /etc/cron.daily/security-scan

# # 12. SETUP NVIDIA DRIVER SECURITY (for RTX 5090)
# print_status "Configuring NVIDIA driver security..."
# # Ensure NVIDIA persistence daemon runs securely
# sudo tee /etc/systemd/system/nvidia-persistenced.service.d/override.conf > /dev/null <<EOF
# [Service]
# ExecStart=
# ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --no-persistence-mode --verbose
# EOF

# 13. CONFIGURE LIMITS FOR AI WORKLOADS
print_status "Setting up resource limits for AI workloads..."
sudo tee /etc/security/limits.d/99-ai-limits.conf > /dev/null <<EOF
# Resource limits for AI workstation
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited
EOF

# 14. BACKUP CONFIGURATION
print_status "Creating configuration backup..."
sudo mkdir -p /root/security-backup
sudo cp /etc/fail2ban/jail.conf /root/security-backup/
sudo cp /etc/ufw/before.rules /root/security-backup/

# 15. FINAL SYSTEM CONFIGURATION
print_status "Applying final configurations..."
# Restart services
sudo systemctl restart fail2ban
sudo systemctl restart ufw

# Set up automatic updates check
sudo systemctl enable unattended-upgrades

print_status "Security hardening complete!"
echo ""
echo "🎉 Your AI workstation is now secured with the following features:"
echo "✅ Automatic security updates enabled"
echo "✅ UFW firewall configured for AI development"
echo "✅ System auditing enabled"
echo "✅ Kernel security parameters optimized"
echo "✅ AppArmor profiles active"
echo "✅ Security monitoring tools installed"
echo "✅ Daily security scans configured"
echo ""
echo "🔧 Post-installation recommendations:"
echo "1. Reboot your system to apply all changes: sudo reboot"
echo "2. Configure email notifications in /etc/apt/apt.conf.d/50unattended-upgrades"
echo "3. Review firewall rules: sudo ufw status numbered"
echo "4. Check fail2ban status: sudo fail2ban-client status"
echo "5. Monitor logs: tail -f /var/log/auth.log"
echo "6. Run security scan: sudo lynis audit system"
echo ""
echo "🚨 AI Development Notes:"
echo "- Jupyter ports (8888) and development servers are allowed"
echo "- Docker container communication is permitted"
echo "- GPU workloads have appropriate resource limits"
echo "- System will auto-reboot at 3 AM for kernel updates"
echo ""
echo "Your system maintains full AI development capability while being secure!"