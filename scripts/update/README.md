# scripts/update

Updaters for the workstation stack: Ubuntu, the NVIDIA driver, CUDA, cuDNN, Docker, the
security tools and the uv-managed Python environment. None runs in a sandbox, since each
needs sudo and the network; `make update` wraps `update_python.sh` and the rest are run by
hand in the order below.

## Scripts

<!-- inventory:scripts:start -->

| Script | Purpose |
|--------|---------|
| `update_cuda.sh` | This script automatically checks for and applies updates for the currently installed CUDA toolkit package. |
| `update_cudnn.sh` | This script automatically checks for and applies updates for the cuDNN package matching your installed CUDA version. |
| `update_docker.sh` | This script updates Docker CE, Docker Compose, and NVIDIA Container Toolkit. |
| `update_nvidia.sh` | This script automatically checks for and applies updates for the installed nvidia-driver-*-open package, whichever branch that is. |
| `update_python.sh` | Updates the uv-managed Python environment and verifies the PyTorch CUDA stack. |
| `update_security.sh` | This script updates security tools and runs basic security checks. |
| `update_ubuntu.sh` | Comprehensive Ubuntu update script for AI workstation. |

<!-- inventory:scripts:end -->

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
NVIDIA_DRIVER_VERSION=nvidia-driver-580-server-open ./update_nvidia.sh

# Install specific CUDA toolkit version
CUDA_VERSION=cuda-toolkit-13-0 ./update_cuda.sh

# Install specific cuDNN version
CUDNN_VERSION=cudnn9-cuda-13 ./update_cudnn.sh
```

### Force Update with GUI Running

Not recommended, but possible for NVIDIA driver:
```bash
FORCE_UPDATE=yes ./update_nvidia.sh
```

## Important Notes

### Before Running Updates

1. **Stop all containers**: `docker compose -f docker-compose.inference.yaml down`
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
- Target: the installed `nvidia-driver-*-open` package, read from dpkg; the
  script refuses to guess if none is installed

### update_cuda.sh
- Updates CUDA toolkit
- Configures NVIDIA repository
- Verifies nvcc after installation
- No reboot required
- Target: derived from `nvcc` (`cuda-toolkit-<major>-<minor>`), falling back to
  `cuda-toolkit-13-0`

### update_cudnn.sh
- Updates cuDNN library
- Manages CUDA keyring
- Verifies installation
- No reboot required
- Target: derived from the CUDA major version `nvcc` reports
  (`cudnn9-cuda-<major>`), falling back to `cudnn9-cuda-13`

## Environment Variables

| Variable | Script | Purpose | Example |
|----------|--------|---------|---------|
| NVIDIA_DRIVER_VERSION | update_nvidia.sh | Specify driver package | nvidia-driver-580-server-open |
| FORCE_UPDATE | update_nvidia.sh | Override GUI check | yes |
| CUDA_VERSION | update_cuda.sh | Specify CUDA package | cuda-toolkit-13-0 |
| CUDNN_VERSION | update_cudnn.sh | Specify cuDNN package | cudnn9-cuda-13 |

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