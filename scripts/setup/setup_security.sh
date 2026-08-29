#!/bin/bash
#
# setup_security.sh -- harden the workstation without getting in the way of AI
# development.
#
# Six independent areas, each idempotent: unattended security updates, a UFW
# policy, fail2ban, auditd, kernel parameters, and resource limits for large
# model loads.
#
# The firewall is deliberately narrow. Every inference service in
# docker-compose.inference.yaml binds 127.0.0.1, so opening development ports to
# the network would grant access to nothing while widening the attack surface.
# Only SSH and the Docker bridge are allowed in.
#
# Runs as a regular user with sudo rights, not as root.

set -euo pipefail

DRY_RUN=false

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }
print_skip()  { printf "${BLUE}[SKIP]${NC} %s\n" "$*"; }

usage() {
    cat <<'EOF'
Usage: setup_security.sh [OPTIONS]

Harden the workstation: unattended security updates, a UFW policy, fail2ban,
auditd, kernel security parameters, and AI resource limits.

Options:
  --dry-run        Print every action without executing it
  --no-color       Disable ANSI color
  --help           Show this message

Run as a regular user with sudo rights, not as root. The firewall allows only
SSH and the Docker bridge inbound; the inference services bind 127.0.0.1 and
need no rule. Reboot afterwards to apply every change.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=true; shift ;;
        --no-color) RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''; shift ;;
        --help|-h)  usage; exit 0 ;;
        *)          print_error "Unknown option: $1"; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${NO_COLOR:-}" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

$DRY_RUN && print_info "Dry run: no changes will be made"

run() {
    if $DRY_RUN; then
        printf "[DRY-RUN] %s\n" "$*"
    else
        print_run "$*"
        "$@"
    fi
}

# Writing a root-owned file needs a pipe into tee, which run() cannot express
# because it takes argv. Every config below is written whole rather than
# appended, so a second run replaces its own output instead of duplicating it.
write_root_file() {
    local path="$1" content="$2"
    if $DRY_RUN; then
        printf "[DRY-RUN] %s\n" "write $path"
    else
        print_run "write $path"
        printf '%s\n' "$content" | sudo tee "$path" >/dev/null
    fi
}

# Install packages one at a time. A single apt call under set -e means one
# package missing from the archive silently skips every step below it.
install_packages() {
    local pkg
    for pkg in "$@"; do
        if run sudo apt-get install --yes "$pkg"; then
            print_ok "$pkg"
        else
            print_warn "$pkg: not available in this archive; skipped"
        fi
    done
}

if [[ $EUID -eq 0 ]]; then
    print_error "This script must not run as root"
    echo "  Command: ./scripts/setup/setup_security.sh"
    exit 1
fi

echo "=== Security hardening ==="
echo

echo "=== Automatic security updates ==="
run sudo apt-get update
install_packages unattended-upgrades apt-listchanges

write_root_file /etc/apt/apt.conf.d/20auto-upgrades \
'APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
APT::Periodic::Download-Upgradeable-Packages "1";'

# A dedicated file rather than an append to the distro's 50unattended-upgrades:
# appending duplicates the whole block on every re-run, and the higher number
# means these settings win without editing a file apt also owns.
write_root_file /etc/apt/apt.conf.d/51ai-workstation \
'// Security pockets only. Feature updates are applied deliberately, because an
// unattended CUDA or driver bump breaks inference without warning.
Unattended-Upgrade::Allowed-Origins {
    "${distro_id}:${distro_codename}-security";
    "${distro_id}ESMApps:${distro_codename}-apps-security";
    "${distro_id}ESM:${distro_codename}-infra-security";
};

// Kernel updates need a reboot to take effect; 03:00 is outside working hours.
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "03:00";'

run sudo systemctl enable unattended-upgrades
print_ok "Unattended security updates configured"
echo

echo "=== Firewall ==="
install_packages ufw
run sudo ufw --force reset
run sudo ufw default deny incoming
run sudo ufw default allow outgoing
run sudo ufw allow 22/tcp comment "SSH"
# The Docker bridge, so containers can reach host services. No development
# ports: llama.cpp, vLLM, Open WebUI, and ComfyUI all bind 127.0.0.1 in
# docker-compose.inference.yaml and are unreachable from the network by design.
run sudo ufw allow from 172.17.0.0/16 comment "Docker containers"
run sudo ufw --force enable
print_ok "Firewall enabled: SSH and the Docker bridge only"
echo

echo "=== Intrusion prevention ==="
install_packages fail2ban

# The jail is named sshd to match the distro's own definition; a second jail
# under a different name would run the same filter twice over one log.
write_root_file /etc/fail2ban/jail.local \
'[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3
backend = systemd

[sshd]
enabled = true
port = 22
maxretry = 3'

run sudo systemctl enable fail2ban
run sudo systemctl restart fail2ban
print_ok "fail2ban configured for SSH"
echo

echo "=== System auditing ==="
install_packages auditd audispd-plugins

write_root_file /etc/audit/rules.d/audit.rules \
'# Account and privilege files
-w /etc/passwd -p wa -k identity
-w /etc/group -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/sudoers -p wa -k identity
-w /etc/sudoers.d -p wa -k identity

# Authentication records
-w /var/log/auth.log -p wa -k authentication
-w /var/log/faillog -p wa -k authentication

# Clock changes, which can be used to confuse log correlation
-a always,exit -F arch=b64 -S adjtimex -S settimeofday -k time-change
-a always,exit -F arch=b32 -S adjtimex -S settimeofday -S stime -k time-change

# Host identity changes
-a always,exit -F arch=b64 -S sethostname -S setdomainname -k system-locale
-a always,exit -F arch=b32 -S sethostname -S setdomainname -k system-locale'

run sudo systemctl enable auditd
run sudo systemctl restart auditd
print_ok "Audit rules installed"
echo

echo "=== Kernel parameters ==="
write_root_file /etc/sysctl.d/99-security.conf \
'# Network
net.ipv4.conf.all.rp_filter = 1                # Reverse-path filter: drop spoofed source addresses
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.accept_redirects = 0         # ICMP redirects can reroute traffic through a hostile host
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv4.conf.all.send_redirects = 0           # Sending them leaks local routing topology
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0      # Source routing bypasses firewall path assumptions
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0
net.ipv4.conf.all.log_martians = 1             # Log impossible source addresses
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1

# Ping is answered deliberately: dropping it breaks routine network diagnostics
# on a machine that is not internet-facing, and hides nothing from a scanner.
net.ipv4.icmp_echo_ignore_all = 0

# TCP SYN flood handling
net.ipv4.tcp_syncookies = 1                    # Serve connections without holding half-open state
net.ipv4.tcp_max_syn_backlog = 2048
net.ipv4.tcp_synack_retries = 2
net.ipv4.tcp_syn_retries = 5

# Memory and process
kernel.randomize_va_space = 2                  # Full ASLR
kernel.kptr_restrict = 1                       # Hide kernel addresses from unprivileged readers
kernel.dmesg_restrict = 1
kernel.yama.ptrace_scope = 1                   # Restrict ptrace to descendants

# Core dumps
fs.suid_dumpable = 0
# Discarded rather than written: a dump from an inference process contains model
# weights and whatever was in the prompt, which is not wanted on disk.
kernel.core_pattern = /dev/null'

run sudo sysctl -p /etc/sysctl.d/99-security.conf
print_ok "Kernel parameters applied"
echo

echo "=== Mandatory access control ==="
install_packages apparmor-profiles apparmor-utils
run sudo systemctl enable apparmor
run sudo systemctl start apparmor
print_ok "AppArmor active"
echo

echo "=== Shared memory ==="
if grep -q "tmpfs /run/shm" /etc/fstab; then
    print_skip "/run/shm entry already in fstab"
elif $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "append /run/shm entry to /etc/fstab"
else
    echo "tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0" | sudo tee -a /etc/fstab >/dev/null
    print_ok "/run/shm mounted noexec,nosuid at next boot"
fi
echo

echo "=== Security monitoring tools ==="
install_packages rkhunter chkrootkit lynis aide logwatch tiger

write_root_file /etc/cron.daily/security-scan \
'#!/bin/bash
# Daily security scan. Output reaches root via cron mail.

/usr/bin/rkhunter --check --skip-keypress --report-warnings-only
/usr/sbin/chkrootkit
/usr/bin/lastb | head -20
/usr/bin/ss -tuln'

run sudo chmod +x /etc/cron.daily/security-scan
print_ok "Daily security scan installed"
echo

echo "=== Package management policy ==="
write_root_file /etc/apt/apt.conf.d/99-security \
'APT::Get::AllowUnauthenticated "false";
APT::Get::Assume-Yes "false";
APT::Install-Recommends "false";
APT::Install-Suggests "false";'
print_ok "Only signed packages, no recommends"
echo

echo "=== Resource limits for AI workloads ==="
# memlock unlimited is what lets a model be locked into RAM rather than paged;
# the file descriptor and process ceilings cover many-shard model loads.
write_root_file /etc/security/limits.d/99-ai-limits.conf \
'* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited'
print_ok "Resource limits installed"
echo

echo "=== Configuration backup ==="
# The distro defaults, kept so a later diff can show what this script changed.
run sudo mkdir -p /root/security-backup
for original in /etc/fail2ban/jail.conf /etc/ufw/before.rules; do
    if [[ -f "$original" ]]; then
        run sudo cp "$original" /root/security-backup/
        print_ok "$(basename "$original")"
    else
        print_skip "$(basename "$original") (not present)"
    fi
done
echo

echo "=== Complete ==="
if $DRY_RUN; then
    print_info "No changes were made"
    exit 0
fi

print_ok "Security hardening applied"
echo "  Reboot to apply every change: sudo reboot"
echo
echo "Verify:"
echo "  sudo ufw status numbered"
echo "  sudo fail2ban-client status sshd"
echo "  sudo sysctl -a --pattern 'kernel.(randomize_va_space|kptr_restrict)'"
echo "  sudo lynis audit system"
