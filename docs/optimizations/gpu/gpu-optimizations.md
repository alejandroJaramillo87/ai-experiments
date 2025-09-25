# GPU Optimizations for RTX 5090

GPU optimizations for NVIDIA RTX 5090 (Blackwell architecture) for AI inference workloads.

## Executive Summary

GPU optimizations focus on three key areas:
1. **Host system configuration** - NVIDIA driver settings and GPU state management
2. **Container runtime environment** - CUDA and runtime optimizations
3. **Architecture-specific tuning** - Blackwell SM 12.0 optimizations

## Host System GPU Configuration

For OS-level GPU optimizations including:
- NVIDIA driver persistence mode
- GPU power and clock management
- Compute mode configuration
- Kernel module parameters
- IRQ affinity tuning
- CUDA system environment
- Multi-Process Service (MPS)

**See: [`docs/optimizations/os/os-optimizations.md`](../os/os-optimizations.md#gpu-optimizations)**

### Quick Verification

Verify GPU configuration after applying OS optimizations:

```bash
# Check all GPU settings
nvidia-smi -q | grep -E "Persistence Mode|Power Limit|Compute Mode|PCIe Generation"

# Verify PCIe bandwidth
nvidia-smi -q | grep -A 4 "GPU Link Info"
# Should show: PCIe Generation: 5, Link Width: x16

# Check memory and clocks
nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw --format=csv
```

**Note:** BIOS PCIe settings required - see [`docs/optimizations/bios/bios-optimizations.md`](../bios/bios-optimizations.md#gpu-performance-configuration)

## Container Runtime Optimizations

### 1. CUDA Memory Pool Configuration

Optimize memory allocation with persistent pools.

**Environment Variables for Dockerfiles:**
```dockerfile
# CUDA memory pool settings
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_PATH=/tmp/cuda_cache
ENV CUDA_CACHE_MAXSIZE=2147483648  # 2GB kernel cache

# PyTorch memory pool for Blackwell 32GB VRAM
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.95,max_split_size_mb:2048

# CUDA allocator configuration
ENV CUDA_ALLOCATOR_BACKEND=cudaMallocAsync
ENV CUDA_MALLOC_ASYNC_POOLS=1
```

### 2. CUDA Compilation Cache

Reduce JIT compilation overhead.

```dockerfile
# Compilation cache settings
ENV CUDA_CACHE_DISABLE=0
ENV CUDA_CACHE_PATH=/tmp/cuda_cache
ENV CUDA_FORCE_PTX_JIT=0  # Disable PTX JIT, use compiled kernels
ENV CUDA_CACHE_MAXSIZE=2147483648  # 2GB cache limit
```

### 3. GPU Boost and Clock Management

Ensure consistent performance without thermal throttling.

```dockerfile
# Disable auto-boost for consistent clocks
ENV CUDA_AUTO_BOOST=0

# Force maximum clocks during inference
ENV CUDA_CLOCK_CONTROL=1
ENV CUDA_MAX_CLOCK=1
```

### 4. Multi-Process Service (MPS)

Enable for better GPU utilization with multiple containers.

```dockerfile
# MPS configuration for multi-container GPU sharing
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
ENV CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
```

### 5. Tensor Core Optimization

Leverage Blackwell's 5th generation tensor cores.

```dockerfile
# Enable tensor core usage
ENV CUDA_TENSOR_CORES=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8  # Optimal workspace for tensor cores
ENV CUDNN_TENSOR_OPS=1
```

## Blackwell Architecture Specific

### 1. SM 12.0 Compiler Flags

Optimize for Blackwell compute capability.

```dockerfile
# Build flags for Blackwell
ENV CUDA_ARCHITECTURES=120
ENV TORCH_CUDA_ARCH_LIST="12.0"
ENV CUDAFLAGS="-O3 --use_fast_math -arch=sm_120 --extra-device-vectorization"
ENV NVCC_FLAGS="-gencode arch=compute_120,code=sm_120"
```

### 2. FP8 Support

Enable FP8 operations for improved throughput.

```dockerfile
# FP8 configuration for Blackwell
ENV CUDA_FP8_E4M3=1
ENV CUDA_FP8_E5M2=1
ENV TRANSFORMER_ENGINE_FP8=1
```

### 3. Memory Access Patterns

Optimize for Blackwell's memory hierarchy.

```dockerfile
# L2 cache optimization (96MB on RTX 5090)
ENV CUDA_L2_PERSISTING_SIZE=100663296  # Use full 96MB L2

# Optimize memory access patterns
ENV CUDA_DEVICE_MAX_CONNECTIONS=32  # Concurrent kernel connections
ENV CUDA_COPY_SPLIT_THRESHOLD=256   # MB threshold for split copies
```

## Docker Compose Enhancements

### 1. GPU Service Configuration

```yaml
services:
  llama-gpu:
    # ... existing config ...
    environment:
      # GPU optimization environment
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
      - NVIDIA_REQUIRE_CUDA="cuda>=12.8"
      # Compute mode
      - CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100
      - CUDA_DEVICE_ORDER=FASTEST_FIRST
    tmpfs:
      - /tmp:size=4G,mode=1777  # CUDA cache and temp files
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu, compute, utility]
              # Note: Do NOT set compute-mode=exclusive - causes performance degradation
```

### 2. vLLM Specific Settings

```yaml
services:
  vllm-gpu:
    environment:
      # vLLM GPU optimizations
      - VLLM_CUDA_GRAPHS=1        # Enable CUDA graphs
      - VLLM_USE_V1=1             # Use optimized v1 engine
      - VLLM_ENABLE_PREFIX_CACHING=1
      - VLLM_FP8_E4M3=1           # FP8 support
      # NCCL settings for future multi-GPU
      - NCCL_P2P_DISABLE=0
      - NCCL_SHM_DISABLE=0
      - NCCL_CUMEM_ENABLE=1
```

## Dockerfile Optimizations

### Dockerfile.llama-gpu Additions

```dockerfile
# Runtime optimizations section
ENV CUDA_CACHE_MAXSIZE=2147483648
ENV CUDA_AUTO_BOOST=0
ENV CUDA_TENSOR_CORES=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV CUDA_L2_PERSISTING_SIZE=100663296
ENV CUDA_DEVICE_MAX_CONNECTIONS=32
ENV CUDA_ALLOCATOR_BACKEND=cudaMallocAsync

# Blackwell specific
ENV CUDA_FP8_E4M3=1
ENV TRANSFORMER_ENGINE_FP8=1
```

### Dockerfile.vllm-gpu Additions

```dockerfile
# vLLM specific GPU optimizations
ENV VLLM_CUDA_GRAPHS=1
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
ENV VLLM_USE_V1=1
ENV VLLM_FP8_E4M3=1
ENV VLLM_FP8_E5M2=1

# Memory pool optimization
ENV VLLM_PREEMPTION_MODE=swap
ENV VLLM_SWAP_SPACE_GB=4
ENV VLLM_GPU_MEMORY_UTILIZATION=0.95
```

## Host Setup Script

**IMPORTANT: Based on RTX 5090 benchmarking, clock locking causes 38% performance degradation.**

The actual optimized script at `/scripts/optimizations/optimize-gpu.sh`:

```bash
#!/bin/bash
set -e

echo "Starting GPU optimizations for RTX 5090 AI inference..."

# Enable persistence mode (reduces CUDA init time)
echo "Enabling NVIDIA persistence mode..."
nvidia-smi -pm 1

# Set maximum power limit (600W for RTX 5090)
echo "Setting power limit to 600W..."
nvidia-smi -pl 600

# WARNING: Do NOT lock clocks - causes 38% performance loss on RTX 5090
# Clock locking disabled based on benchmarks (287 → 178 tokens/sec)
# nvidia-smi -lgc 2400,2550  # DISABLED - degrades performance
# nvidia-smi -lmc 3002       # DISABLED - degrades performance

# Set compute mode to DEFAULT (EXCLUSIVE_PROCESS also degrades performance)
echo "Setting compute mode to DEFAULT..."
nvidia-smi -c DEFAULT

# Set GPU IRQ affinity to cores 24-31
echo "Setting GPU IRQ affinity..."
GPU_IRQS=$(cat /proc/interrupts | grep -E "nvidia|gpu" | awk '{print $1}' | tr -d ':')
if [ -n "$GPU_IRQS" ]; then
    for irq in $GPU_IRQS; do
        echo f0000000 > /proc/irq/$irq/smp_affinity 2>/dev/null || true
    done
    echo "GPU IRQs pinned to cores 24-31"
else
    echo "No GPU IRQs found to pin"
fi

# Create CUDA cache directory
if [ ! -d /tmp/cuda_cache ]; then
    mkdir -p /tmp/cuda_cache
    chmod 1777 /tmp/cuda_cache
    echo "Created CUDA cache directory"
fi

echo "GPU optimization complete!"

# Verify settings
echo "=== Current GPU Settings ==="
nvidia-smi --query-gpu=persistence_mode,power.limit,clocks.gr,clocks.mem,compute_mode --format=csv
echo ""
echo "=== PCIe Status ==="
nvidia-smi -q | grep -A 4 "GPU Link Info"
```

**Key Findings from Benchmarking:**
- Clock locking: 38% performance loss (287 → 178 tokens/sec)
- EXCLUSIVE_PROCESS mode: Performance degradation
- DEFAULT compute mode: Optimal performance
- Power typically draws ~437W (not always 600W) based on workload

## Monitoring and Verification

### Check GPU Settings

```bash
# Full GPU status
nvidia-smi -q

# Power and thermal status
nvidia-smi --query-gpu=power.draw,temperature.gpu,clocks.gr,clocks.mem --format=csv -l 1

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# Process compute mode
nvidia-smi --query-gpu=compute_mode --format=csv
```

### Container GPU Verification

```bash
# Check GPU access in container
docker exec llama-gpu nvidia-smi

# Verify CUDA environment
docker exec llama-gpu bash -c 'env | grep CUDA'

# Check tensor core usage
docker exec llama-gpu bash -c 'nvidia-smi -q | grep "Tensor Cores"'
```

## Performance Impact

Measured and expected improvements with optimizations:

| Optimization | Impact | Metric | Status |
|--------------|--------|--------|--------|
| Persistence Mode | -200ms | CUDA init time | ✓ Verified |
| Clock Locking | **-38%** | **Performance degradation** | ✗ Disabled |
| DEFAULT Compute Mode | Optimal | No degradation | ✓ Verified |
| Memory Pools | -30% | Allocation overhead | Pending |
| Tensor Cores | +40% | GEMM operations | Pending |
| FP8 Support | +100% | Throughput (model dependent) | Pending |
| CUDA Graphs | -20% | Kernel launch overhead | Pending |
| L2 Cache Config | +10% | Memory bandwidth utilization | Pending |

**Benchmark Results (llama.cpp with Qwen2.5-32B Q4):**
- Baseline: 287.37 tokens/sec
- With clock locking: 178.71 tokens/sec (-38%)
- Optimal (no locking, DEFAULT mode): 287.46 tokens/sec

## Troubleshooting

### Common Issues

1. **Clock Throttling**
```bash
# Check for throttling reasons
nvidia-smi -q | grep -A 10 "Clocks Throttle Reasons"
```

2. **Memory Fragmentation**
```bash
# Monitor memory fragmentation
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

3. **PCIe Bandwidth**
```bash
# Verify PCIe performance
nvidia-smi -q | grep -E "PCIe|Link"
```

## Implementation Priority

1. **High Priority** (Verified gains)
   - Enable persistence mode ✓
   - Set power limits (600W) ✓
   - Use DEFAULT compute mode ✓
   - Configure CUDA memory pools (Phase 1)

2. **Medium Priority** (Expected improvements)
   - Tensor core optimization (Phase 2)
   - CUDA compilation cache
   - Blackwell architecture tuning (Phase 3)

3. **Low Priority** (Model dependent)
   - FP8 enablement (Phase 4)
   - MPS configuration
   - L2 cache fine-tuning

**DO NOT IMPLEMENT:**
- Clock locking (causes 38% performance loss)
- EXCLUSIVE_PROCESS mode (degrades performance)

## Next Steps

1. Run setup-gpu.sh script after system boot
2. Update Dockerfiles with environment variables
3. Test inference performance with benchmarks
4. Monitor GPU metrics during production workloads
5. Adjust memory pool sizes based on model requirements

## References

- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [vLLM Performance Tuning Guide](https://docs.vllm.ai/en/latest/serving/performance.html)
- [llama.cpp CUDA Backend](https://github.com/ggerganov/llama.cpp/blob/master/docs/backend/CUDA.md)

---

*Last Updated: 2025-09-23*