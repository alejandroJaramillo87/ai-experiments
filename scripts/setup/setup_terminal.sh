#!/bin/bash

# AI Engineering Zsh Setup Script
# For Ubuntu 24.04 with focus on AI/ML development

set -e  # Exit on any error

echo "🚀 Setting up AI Engineering Environment..."
echo "Installing zsh, Oh My Zsh, and AI development tools"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
echo "🛠️  Installing essential packages..."
sudo apt install -y \
    zsh \
    tilix \
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

# Install Oh My Zsh
echo "🎨 Installing Oh My Zsh..."
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
fi

# Install useful Oh My Zsh plugins
echo "🔌 Installing Oh My Zsh plugins..."
ZSH_CUSTOM=${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}

# zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM}/plugins/zsh-autosuggestions

# zsh-syntax-highlighting
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting

# zsh-completions
git clone https://github.com/zsh-users/zsh-completions ${ZSH_CUSTOM}/plugins/zsh-completions

# autojump
sudo apt install -y autojump

# Create .zshrc configuration
echo "⚙️  Configuring .zshrc..."
cat > $HOME/.zshrc << 'EOF'
# Oh My Zsh configuration
export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="robbyrussell"

# Plugins
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

source $ZSH/oh-my-zsh.sh

# User configuration
export EDITOR='nvim'
export BROWSER='firefox'

# AI/ML Environment Variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Aliases for AI Engineering
alias cat='bat'
alias find='fd'
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
alias ports='netstat -tuln'
alias meminfo='free -h'
alias cpuinfo='lscpu'
alias diskinfo='df -h'
alias myip='curl ipinfo.io/ip'

function gpu_memory() {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{print "GPU Memory: " $1 "MB / " $2 "MB (" int($1/$2*100) "%)"}'
}

# FZF configuration
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"
export FZF_ALT_C_COMMAND='fd --type d --hidden --follow --exclude .git'
EOF

# Set up Tilix as default terminal
echo "🖥️  Configuring Tilix..."
sudo update-alternatives --install /usr/bin/x-terminal-emulator x-terminal-emulator /usr/bin/tilix 50

# Change default shell to zsh
echo "🐚 Changing default shell to zsh..."
sudo chsh -s $(which zsh) $USER

echo "✅ Setup complete!"
echo ""
echo "🎉 Your AI engineering environment is ready!"
echo ""
echo "Next steps:"
echo "1. Log out and log back in to start using zsh"
echo "2. Open Tilix and enjoy your new shell"
echo "3. All your AI development aliases are ready to use"
echo ""
echo "Useful aliases:"
echo "  gpu/gpuwatch - Monitor GPU usage"
echo "  sysmon - System monitor (btop)"
echo "  jlab - Start Jupyter Lab"
echo "  c/cls - Clear screen"
echo ""