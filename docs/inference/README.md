# Inference documentation

The inference engines running on this workstation (AMD Ryzen 9950X, RTX 5090), how they are
tuned, and what each parameter was measured to do.

## Overview

Two engines serve two different goals: llama.cpp for latency, vLLM for throughput. The
documents below cover what each engine does, then the parameters actually set in this
repository and the results that justify them. Container and compose details live in
[../sandbox/](../sandbox/); the engines' Dockerfiles are the source of truth for versions.

## Inference Engines

### Core Documentation
- **[engines/inference-engines.md](engines/inference-engines.md)** - Understanding what inference engines are and why they're needed
- **[engines/llama-cpp.md](engines/llama-cpp.md)** - Deep technical dive into llama.cpp architecture and optimizations
- **[engines/vllm.md](engines/vllm.md)** - High-throughput GPU inference with vLLM

### Parameter Tuning
- **[parameters/llama-cpp-parameters.md](parameters/llama-cpp-parameters.md)** - Tested llama.cpp parameters and findings
- **[parameters/vllm-parameters.md](parameters/vllm-parameters.md)** - vLLM configuration and throughput optimization

### Forward Looking
- **[future-optimizations.md](../optimizations/ideation/future-inference-optimizations.md)** - Parked survey of custom kernel and runtime work, with the conditions gating it

### Related Directories
- **[../rust-harness/README.md](../rust-harness/README.md)** - The `rust-harness` runtime and coding harness, documented in its own directory

## Current Implementation Status

| Service | Engine | Port | Status | Notes |
|---------|--------|------|--------|-------|
| llama-cpu | llama.cpp | 8001 | Active | Latency-optimized CPU inference |
| llama-gpu | llama.cpp | 8002 | Active | GPU-accelerated low-latency inference |
| vllm-gpu | vLLM | 8003 | Active | High-throughput batch serving |
| open-webui | — | 3000 | Active | Web interface, published as `127.0.0.1:3000:8080` |

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
- Latency-optimized for single-request inference (CPU and GPU)
- Memory efficiency: close to model size
- Quantization: Native GGUF support

**vLLM** (0.25.0 with CUDA 13):
- Throughput-optimized for concurrent request serving
- Quantization: FP8, GPTQ, AWQ

No benchmark has been run on this workstation; the tables above describe
qualitative characteristics only.

## Hardware Configuration

- **CPU**: AMD Ryzen 9950X (16 cores, 16 dedicated to inference)
- **GPU**: RTX 5090 32GB VRAM (Blackwell architecture)
- **Memory**: 128GB DDR5-6000 with huge pages enabled
- **Storage**: NVMe SSD for model storage
