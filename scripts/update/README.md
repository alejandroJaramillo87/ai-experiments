# System Update Scripts

Simple, focused scripts for updating Ubuntu AI development environment components.

## Overview

These scripts update specific components of the AI development stack:
- `update_ubuntu.sh` - System packages and kernel
- `update_nvidia.sh` - NVIDIA GPU driver (will reboot)
- `update_cuda.sh` - CUDA toolkit
- `update_cudnn.sh` - cuDNN library

## Usage

### Basic Usage (Update to Latest)

Run scripts in this order:
```bash
./update_ubuntu.sh      # System packages
./update_nvidia.sh      # GPU driver (will reboot)
# After reboot:
./update_cuda.sh        # CUDA toolkit
./update_cudnn.sh       # cuDNN library
```

### Specific Version Installation

Use environment variables to specify versions:

```bash
# Install specific NVIDIA driver version
NVIDIA_DRIVER_VERSION=nvidia-driver-560 ./update_nvidia.sh

# Install specific CUDA toolkit version
CUDA_VERSION=cuda-toolkit-12-7 ./update_cuda.sh

# Install specific cuDNN version
CUDNN_VERSION=cudnn9-cuda-11 ./update_cudnn.sh
```

### Force Update with GUI Running

Not recommended, but possible for NVIDIA driver:
```bash
FORCE_UPDATE=yes ./update_nvidia.sh
```

## Important Notes

### Before Running Updates

1. **Stop all containers**: `docker compose down`
2. **Save your work**: Driver updates will reboot the system
3. **Switch to TTY**: Press Ctrl+Alt+F2 for text console (recommended for driver updates)

### NVIDIA Driver Updates

- **Always run from TTY** (Ctrl+Alt+F2) when possible
- Driver updates will **automatically reboot** your system
- If issues occur after reboot:
  - Boot with previous kernel (hold Shift during boot)
  - The update script will show the rollback command before rebooting

### Post-Update Verification

After all updates complete:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check cuDNN
dpkg -l | grep cudnn

# Test GPU availability in Python
python3 -c 'import torch; print("CUDA available:", torch.cuda.is_available())'
```

## Troubleshooting

### Driver Update Failed

If the system fails to boot after driver update:
1. Boot with previous kernel (GRUB menu)
2. Check the driver version that was installed
3. Reinstall previous version or remove problematic driver

### CUDA/cuDNN Compatibility

After CUDA updates, you may need to:
- Reinstall PyTorch/TensorFlow for the new CUDA version
- Update environment variables in `.bashrc` or `.zshrc`
- Verify library paths are correct

### Repository Issues

If you get repository errors:
- Scripts automatically configure NVIDIA repositories
- Ubuntu version is detected automatically
- Keyring installation is handled by the scripts

## Script Details

### update_ubuntu.sh
- Updates system packages
- Performs distribution upgrade
- Installs latest kernel headers
- Updates Docker NVIDIA runtime if present
- Checks kernel count for safety

### update_nvidia.sh
- Updates NVIDIA GPU driver
- Checks for GUI sessions (safety)
- Provides rollback instructions
- Automatically reboots system
- Default: nvidia-driver-570-server-open

### update_cuda.sh
- Updates CUDA toolkit
- Configures NVIDIA repository
- Verifies nvcc after installation
- No reboot required
- Default: cuda-toolkit-12-8

### update_cudnn.sh
- Updates cuDNN library
- Manages CUDA keyring
- Verifies installation
- No reboot required
- Default: cudnn9-cuda-12

## Environment Variables

| Variable | Script | Purpose | Example |
|----------|--------|---------|---------|
| NVIDIA_DRIVER_VERSION | update_nvidia.sh | Specify driver package | nvidia-driver-560 |
| FORCE_UPDATE | update_nvidia.sh | Override GUI check | yes |
| CUDA_VERSION | update_cuda.sh | Specify CUDA package | cuda-toolkit-12-7 |
| CUDNN_VERSION | update_cudnn.sh | Specify cuDNN package | cudnn9-cuda-11 |

## Safety Features

- Kernel count checking before updates
- GUI session detection for driver updates
- Automatic keyring and repository configuration
- Error handling with clear exit messages
- Version verification after installation

## Best Practices

1. Always run updates from TTY for driver changes
2. Stop Docker containers before updating
3. Keep one working kernel as fallback
4. Test GPU functionality after updates
5. Document working driver/CUDA combinations

## Support

For issues with these scripts:
1. Check script output for specific error messages
2. Verify Ubuntu version compatibility
3. Ensure internet connectivity for repository access
4. Consult NVIDIA documentation for version compatibility