#!/bin/bash
#
# setup_sudo_user.sh -- grant a user passwordless sudo and the group membership
# the AI stack needs (docker for the daemon socket, video and render for GPU
# device nodes).
#
# Must run as root, because it writes /etc/sudoers.d. The sudoers file is
# validated in a temporary location and only then installed: a malformed file
# under /etc/sudoers.d breaks sudo for every user on the machine, so it must
# never be live and invalid, not even briefly.

set -euo pipefail

RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
BLUE='\033[0;34m' CYAN='\033[0;36m' NC='\033[0m'

print_ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
print_warn()  { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }
print_error() { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
print_info()  { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
print_run()   { printf "${BLUE}[RUN]${NC} %s\n" "$*"; }

usage() {
    cat <<'EOF'
Usage: setup_sudo_user.sh [OPTIONS] [USERNAME]

Grant USERNAME passwordless sudo and add it to the docker, video, and render
groups. USERNAME defaults to $SUDO_USER, the account that invoked sudo.

Options:
  --no-color       Disable ANSI color
  --help           Show this message

Must be run as root: sudo ./scripts/setup/setup_sudo_user.sh
Group changes take effect at the user's next login.
EOF
}

TARGET_USER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-color) RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''; shift ;;
        --help|-h)  usage; exit 0 ;;
        -*)         print_error "Unknown option: $1"; usage >&2; exit 2 ;;
        *)          TARGET_USER="$1"; shift ;;
    esac
done

if [[ -n "${NO_COLOR:-}" ]]; then
    RED='' GREEN='' YELLOW='' BLUE='' CYAN='' NC=''
fi

if [[ $EUID -ne 0 ]]; then
    print_error "This script must run as root"
    echo "  Context: it writes /etc/sudoers.d and changes group membership"
    echo "  Command: sudo $0 ${TARGET_USER:-<username>}"
    exit 1
fi

TARGET_USER="${TARGET_USER:-${SUDO_USER:-}}"

if [[ -z "$TARGET_USER" ]]; then
    print_error "Could not determine the target user"
    echo "  Context: SUDO_USER is unset, which happens under a root login shell"
    echo "  Solution: name the user explicitly"
    echo "  Command: $0 <username>"
    exit 1
fi

if ! id -u "$TARGET_USER" >/dev/null 2>&1; then
    print_error "No such user: $TARGET_USER"
    echo "  Command: getent passwd | cut -d: -f1"
    exit 1
fi

echo "=== Configuring sudo and groups for $TARGET_USER ==="

# sudo is required; the rest are best-effort. docker exists only after
# setup_docker.sh has run, and render is absent on hosts with no DRM devices.
print_run "usermod -aG sudo $TARGET_USER"
usermod -aG sudo "$TARGET_USER"
print_ok "Group sudo"

for group in docker video render; do
    if ! getent group "$group" >/dev/null; then
        print_warn "Group $group does not exist; skipped"
        echo "  Context: docker appears once setup_docker.sh has run"
        continue
    fi
    usermod -aG "$group" "$TARGET_USER"
    print_ok "Group $group"
done
echo

echo "=== Installing the sudoers rule ==="

SUDOERS_FILE="/etc/sudoers.d/99-${TARGET_USER}-nopasswd"
STAGED=$(mktemp)
trap 'rm -f "$STAGED"' EXIT

cat > "$STAGED" <<EOF
# Passwordless sudo for $TARGET_USER
$TARGET_USER ALL=(ALL) NOPASSWD: ALL
EOF

# Validate before installing, never after. A rejected file is discarded while
# the live /etc/sudoers.d is still untouched.
if ! visudo -cf "$STAGED"; then
    print_error "Generated sudoers rule failed validation; nothing installed"
    exit 1
fi

install -m 440 -o root -g root "$STAGED" "$SUDOERS_FILE"
print_ok "Sudoers rule installed: $SUDOERS_FILE"
echo

print_ok "Setup complete for $TARGET_USER"
print_info "Log out and back in for the group changes to take effect"
