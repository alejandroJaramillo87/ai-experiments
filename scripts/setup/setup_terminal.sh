#!/bin/bash
#
# setup_terminal.sh -- install the interactive shell environment: zsh, Oh My
# Zsh with three plugins, Tilix, and the CLI tools the other scripts assume.
#
# Re-running is safe. The generated ~/.zshrc carries a marker line; if it is
# already there the file is left alone, which is what keeps the CUDA exports
# appended by setup_cuda.sh from being erased by a second run of this script.
# A pre-existing unmanaged ~/.zshrc is backed up before being replaced.

set -euo pipefail

ZSHRC_FILE="$HOME/.zshrc"
ZSHRC_MARKER="# managed by scripts/setup/setup_terminal.sh"
ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"

# USER is not exported by every non-login shell, so it cannot be relied on
# under set -u; id(1) is always right.
TARGET_USER="${USER:-$(id -un)}"

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
Usage: setup_terminal.sh [OPTIONS]

Install zsh, Oh My Zsh, Tilix, and the development CLI tools, then generate a
~/.zshrc for AI work and make zsh the login shell.

Options:
  --dry-run        Print every action without executing it
  --no-color       Disable ANSI color
  --help           Show this message

An existing unmanaged ~/.zshrc is backed up to ~/.zshrc.bak-<timestamp> before
being replaced. Log out and back in for the shell change to take effect.
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

echo "=== Installing packages ==="
run sudo apt-get update
run sudo apt-get install --yes \
    zsh \
    tilix \
    autojump \
    git \
    curl \
    wget \
    build-essential \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    python3-pip \
    python3-venv \
    nodejs \
    npm \
    htop \
    btop \
    nvtop \
    tree \
    fzf \
    ripgrep \
    fd-find \
    bat \
    neofetch \
    jq \
    unzip \
    vim \
    neovim \
    tmux \
    screen
print_ok "Packages installed"
echo

echo "=== Installing Oh My Zsh ==="
if [[ -d "$HOME/.oh-my-zsh" ]]; then
    print_skip "Oh My Zsh already installed"
elif $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "install Oh My Zsh (unattended)"
else
    print_run "install Oh My Zsh"
    RUNZSH=no CHSH=no sh -c \
        "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" \
        "" --unattended
    print_ok "Oh My Zsh installed"
fi
echo

echo "=== Installing Oh My Zsh plugins ==="
# Cloning over an existing checkout fails, and under set -e that would abort
# every step below it. A present directory is the success case on a re-run.
install_plugin() {
    local name="$1" url="$2"
    local dest="$ZSH_CUSTOM/plugins/$name"

    if [[ -d "$dest" ]]; then
        print_skip "$name (already present)"
        return
    fi
    run git clone --depth 1 "$url" "$dest"
    print_ok "$name"
}

run mkdir -p "$ZSH_CUSTOM/plugins"
install_plugin zsh-autosuggestions https://github.com/zsh-users/zsh-autosuggestions
install_plugin zsh-syntax-highlighting https://github.com/zsh-users/zsh-syntax-highlighting.git
install_plugin zsh-completions https://github.com/zsh-users/zsh-completions
echo

echo "=== Generating $ZSHRC_FILE ==="
if [[ -f "$ZSHRC_FILE" ]] && grep -qF "$ZSHRC_MARKER" "$ZSHRC_FILE"; then
    print_skip "$ZSHRC_FILE already managed by this script"
    echo "  Context: regenerating would discard later edits, including the CUDA exports"
    echo "  Solution: delete the file and re-run to regenerate it"
elif $DRY_RUN; then
    printf "[DRY-RUN] %s\n" "generate $ZSHRC_FILE (backing up any existing file)"
else
    if [[ -f "$ZSHRC_FILE" ]]; then
        BACKUP="$ZSHRC_FILE.bak-$(date +%Y%m%d%H%M%S)"
        run cp "$ZSHRC_FILE" "$BACKUP"
        print_warn "Existing .zshrc backed up to $(basename "$BACKUP")"
    fi

    # bat and fd-find install their binaries as batcat and fdfind on Debian and
    # Ubuntu: the unsuffixed names belong to unrelated packages in the archive.
    cat > "$ZSHRC_FILE" <<EOF
$ZSHRC_MARKER
export ZSH="\$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"

plugins=(
    git
    docker
    docker-compose
    python
    pip
    virtualenv
    zsh-autosuggestions
    zsh-syntax-highlighting
    zsh-completions
    autojump
    fzf
    colored-man-pages
    command-not-found
)

source \$ZSH/oh-my-zsh.sh

export EDITOR='nvim'
export BROWSER='firefox'

# CUDA toolkit. setup_cuda.sh appends the same exports; duplicates are harmless.
export CUDA_HOME=/usr/local/cuda
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH

alias cat='batcat'
alias find='fdfind'
alias grep='rg'
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'
alias sysmon='btop'
alias pyserver='python3 -m http.server'
alias jlab='jupyter lab --no-browser'
alias jnb='jupyter notebook --no-browser'
alias dockerps='docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"'
alias dockerimg='docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"'
alias cls='clear'
alias c='clear'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias mkdir='mkdir -p'
# ss, not netstat: net-tools is not installed by default on Ubuntu 24.04.
alias ports='ss -tuln'
alias meminfo='free -h'
alias cpuinfo='lscpu'
alias diskinfo='df -h'
alias myip='curl ipinfo.io/ip'

function gpu_memory() {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits \\
        | awk '{print "GPU memory: " \$1 "MB / " \$2 "MB (" int(\$1/\$2*100) "%)"}'
}

export FZF_DEFAULT_COMMAND='fdfind --type f --hidden --follow --exclude .git'
export FZF_CTRL_T_COMMAND="\$FZF_DEFAULT_COMMAND"
export FZF_ALT_C_COMMAND='fdfind --type d --hidden --follow --exclude .git'
EOF
    print_ok "$ZSHRC_FILE generated"
fi
echo

echo "=== Setting the terminal and login shell ==="
run sudo update-alternatives --install /usr/bin/x-terminal-emulator \
    x-terminal-emulator /usr/bin/tilix 50
print_ok "Tilix registered as an x-terminal-emulator alternative"

if [[ "${SHELL:-}" == *zsh ]]; then
    print_skip "Login shell is already zsh"
elif ! ZSH_BIN=$(command -v zsh); then
    # Reachable under --dry-run, where the install step above did not run.
    print_warn "zsh is not on PATH; login shell left unchanged"
    echo "  Command: sudo chsh -s \"\$(command -v zsh)\" $TARGET_USER"
else
    run sudo chsh -s "$ZSH_BIN" "$TARGET_USER"
    print_ok "Login shell set to zsh"
fi
echo

echo "=== Complete ==="
if $DRY_RUN; then
    print_info "No changes were made"
else
    print_ok "Terminal environment configured"
    print_info "Log out and back in to start using zsh"
fi
