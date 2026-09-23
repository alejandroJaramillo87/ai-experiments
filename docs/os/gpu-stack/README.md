# GPU stack

The NVIDIA software stack under every GPU workload on this workstation: the driver, the
CUDA toolkit and cuDNN, one page each, with an overview of how the three layers fit
together and how the installation is verified.

## Pages

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [cuda.md](cuda.md) | Understanding CUDA and how it transforms GPUs into AI acceleration engines. |
| [cudnn.md](cudnn.md) | Understanding cuDNN and how it optimizes neural network operations for maximum performance. |
| [gpu-stack.md](gpu-stack.md) | Understanding the NVIDIA GPU software stack that powers our AI experiments workstation. |
| [nvidia-drivers.md](nvidia-drivers.md) | Understanding NVIDIA GPU drivers and their role in AI computing. |

<!-- inventory:pages:end -->

Start with the overview, then the layer whose version or failure you are looking at.

## Related documentation

- [../README.md](../README.md) — the OS walkthrough this stack is installed in
- [../../optimizations/README.md](../../optimizations/README.md) — what is tuned on top
  of it
