# AI Inference Parameter Optimization

Comprehensive guide to runtime parameter optimization for llama.cpp and vLLM inference engines on the AMD Ryzen 9950X + RTX 5090 AI workstation.

## Optimization Philosophy

The project implements two distinct optimization strategies based on inference engine capabilities:

### llama.cpp (CPU & GPU) - Latency Optimization
**Goal**: Minimize response time for single requests (interactive chatbot experience)

- **Single request focus**: Optimize for fastest first-token and total response time
- **Memory efficiency**: Aggressive memory locking and GPU memory usage
- **Thread allocation**: Match physical core count for maximum single-thread performance
- **Cache optimization**: Direct memory access patterns, no memory mapping

### vLLM (GPU) - Throughput Optimization
**Goal**: Maximize concurrent request handling (API server workload)

- **Concurrent requests**: Handle multiple simultaneous inference requests
- **Batch processing**: Large batch sizes for GPU utilization efficiency
- **Memory pooling**: Intelligent memory allocation for multiple sequences
- **Scheduling**: Advanced request scheduling for optimal resource usage

## Current Implementation Status

| Service | Engine | Port | Status | Optimization Target |
|---------|--------|------|--------|-------------------|
| llama-cpu | llama.cpp | 8001 | Production | Latency (CPU) |
| llama-gpu | llama.cpp | 8004 | Production | Latency (GPU) |
| vllm-gpu | vLLM | 8005 | Waiting CUDA 13 | Throughput (GPU) |

## Version Status (September 23, 2025)

- **llama.cpp**: Latest release b6556 (September 23, 2025) - Current
- **vLLM**: Latest stable v0.10.2 (September 2025) - Awaiting CUDA 13.0 support

## Key Parameter Categories

### Memory Management
- **Context size**: Balance between capability and memory usage
- **Batch sizes**: Optimize for target workload (latency vs throughput)
- **Memory locking**: Prevent swapping for consistent performance
- **GPU memory utilization**: Maximize VRAM usage without OOM

### Threading and Parallelism
- **CPU threads**: Match hardware capabilities (12 cores available)
- **GPU layers**: Full offload vs hybrid CPU/GPU processing
- **HTTP threads**: Balance API responsiveness with compute resources
- **Parallel sequences**: Single vs multiple concurrent requests

### Hardware-Specific Optimizations
- **RTX 5090 features**: FlashAttention, FP8 quantization, tensor cores
- **AMD Zen 5 features**: AVX-512, cache locality, NUMA awareness
- **Memory hierarchy**: L1/L2/L3 cache optimization strategies

## Performance Metrics

### Latency Optimization (llama.cpp)
- **First-token latency**: Time to first response token
- **Tokens per second**: Single-request throughput
- **Response consistency**: Latency variance across requests
- **Memory efficiency**: RAM/VRAM usage per request

### Throughput Optimization (vLLM)
- **Concurrent requests**: Maximum simultaneous requests
- **Total throughput**: Aggregate tokens per second across all requests
- **Queue management**: Request scheduling efficiency
- **Resource utilization**: GPU/memory usage under load

## Documentation Structure

- **llama-cpp-parameters.md** - Current and missing parameters for latency optimization
- **vllm-parameters.md** - Throughput optimization and CUDA 13 considerations
- **parameter-optimization-guide.md** - Benchmarking and testing methodology

## Hardware Configuration

- **CPU**: AMD Ryzen 9950X (cores 0-11 allocated to inference)
- **GPU**: RTX 5090 32GB (CUDA 13.0.1, Blackwell architecture)
- **Memory**: 128GB+ system RAM with huge pages enabled
- **Storage**: Fast NVMe for model loading

---

*Parameter optimization documentation for the AMD Ryzen 9950X + RTX 5090 AI workstation as of September 2025. Focus on production performance for both interactive and batch inference workloads.*