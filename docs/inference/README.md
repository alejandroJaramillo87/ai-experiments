# AI Inference Documentation

Technical documentation for AI inference engines and their optimization on the AMD Ryzen 9950X + RTX 5090 workstation.

## Overview

This directory contains comprehensive documentation for understanding and optimizing AI inference engines used in local LLM deployment. The documentation covers both theoretical foundations and practical implementation details.

## Inference Engines

### Core Documentation
- **[engines/inference-engines.md](engines/inference-engines.md)** - Understanding what inference engines are and why they're needed
- **[engines/llama-cpp.md](engines/llama-cpp.md)** - Deep technical dive into llama.cpp architecture and optimizations
- **[engines/vllm.md](engines/vllm.md)** - High-throughput GPU inference with vLLM (pending CUDA 13 support)

### Parameter Tuning
- **[parameters/llama-cpp-parameters.md](parameters/llama-cpp-parameters.md)** - Tested llama.cpp parameters and findings
- **[parameters/vllm-parameters.md](parameters/vllm-parameters.md)** - vLLM configuration and throughput optimization
- **[parameters/optimization-guide.md](parameters/optimization-guide.md)** - Benchmarking methodology and results

### Performance Results
- **[benchmarks/performance-results.md](benchmarks/performance-results.md)** - Comprehensive benchmark results (pending)

## Current Implementation Status

| Service | Engine | Port | Status | Performance | Notes |
|---------|--------|------|--------|-------------|-------|
| llama-cpu | llama.cpp | 8001 | Active | 35.44 tok/s | Latency-optimized |
| llama-gpu | llama.cpp | 8004 | Active | 286.85 tok/s | GPU-accelerated |
| vllm-gpu | vLLM | 8005 | Pending | N/A | Waiting CUDA 13 |

## Architecture Summary

### Inference vs Training
Inference engines are specialized runtimes optimized for executing trained models, unlike training frameworks:
- No gradient computation or backpropagation
- Read-only weights with aggressive quantization
- Optimized for single/small batch processing
- Memory-efficient through techniques like paging

### Key Technologies

**Memory Management**:
- Memory mapping (mmap) for efficient model loading
- Memory locking (mlock) to prevent swapping
- Huge pages (2MB) for reduced TLB misses
- PagedAttention (vLLM) for dynamic KV cache allocation

**Hardware Acceleration**:
- CPU: AVX-512 VNNI for INT8 operations (Zen 5)
- GPU: CUDA kernels, tensor cores, Flash Attention
- Quantization: INT4/INT8 (llama.cpp), FP8 (vLLM)

**Scheduling Strategies**:
- Static batching (llama.cpp default)
- Continuous batching (vLLM) for dynamic request handling
- Chunked prefill for large context processing

## Quick Reference

### Choosing an Inference Engine

**Use llama.cpp when**:
- Single-user or few concurrent users
- Lowest latency is critical
- Running on CPU or edge devices
- Need aggressive quantization (GGUF)
- Memory constraints are tight

**Use vLLM when**:
- Serving many concurrent users
- Throughput more important than latency
- Have powerful datacenter GPUs
- Need dynamic batching
- Using standard HuggingFace models

### Performance Characteristics

**llama.cpp**:
- First token: 50-200ms
- Generation: 35 tok/s (CPU), 287 tok/s (GPU)
- Memory efficiency: 1.1x model size
- Quantization: Native GGUF support

**vLLM** (projected with CUDA 13):
- First token: 50-100ms
- Generation: 300-500 tok/s
- Memory efficiency: 1.3x model size
- Quantization: FP8, GPTQ, AWQ

## Hardware Configuration

- **CPU**: AMD Ryzen 9950X (32 cores, 12 dedicated to inference)
- **GPU**: RTX 5090 32GB VRAM (Blackwell architecture)
- **Memory**: 128GB DDR5-6000 with huge pages enabled
- **Storage**: NVMe SSD for model storage

---

*Last Updated: 2025-09-23*