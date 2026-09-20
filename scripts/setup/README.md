# scripts/setup

One-time workstation provisioning: the NVIDIA driver, CUDA, cuDNN, Docker with the
container toolkit, the data SSD, security hardening, the sudo user and the terminal. The
install paths need the host, sudo and the network; every script answers `--help` without
touching anything, which is the one path `tests/setup/` holds to the output contract.

Run them in the order `docs/os/README.md` gives. The quality gate afterwards proves the
tooling, not the hardware; `make quality-full` is what proves the stack these scripts
installed.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `setup_cuda.sh` | Install the CUDA toolkit from NVIDIA's apt repository and put nvcc on the shell PATH. |
| `setup_cudnn.sh` | Install the cuDNN runtime matching the installed CUDA toolkit. |
| `setup_data_ssd.sh` | Provision the AI data drive at /mnt/ai-data. |
| `setup_docker.sh` | Install Docker CE and the NVIDIA Container Toolkit, the two halves of GPU-capable containers on this workstation. |
| `setup_nvidia.sh` | Purge any installed NVIDIA driver and install the branch this workstation is qualified on. |
| `setup_security.sh` | Harden the workstation without getting in the way of AI development. |
| `setup_sudo_user.sh` | Grant a user passwordless sudo and the group membership the AI stack needs (docker for the daemon socket, video and render for GPU device nodes). |
| `setup_terminal.sh` | Install the interactive shell environment: zsh, Oh My Zsh with three plugins, Tilix, and the CLI tools the other scripts assume. |

<!-- inventory:scripts:end -->

## Where the real docs live

- [../../docs/os/README.md](../../docs/os/README.md) — the installation these scripts perform
- [../../docs/os/gpu-stack/README.md](../../docs/os/gpu-stack/README.md) — driver, CUDA and cuDNN as three layers
- [../update/README.md](../update/README.md) — the update scripts that follow these
