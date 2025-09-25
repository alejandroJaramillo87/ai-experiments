# GPU Optimization Experiments for RTX 5090

## Overview

This document outlines experimental GPU optimizations specifically for the NVIDIA RTX 5090 (Blackwell architecture) with 32GB VRAM. These experiments complement the CPU/RAM optimizations and focus on maximizing inference performance for AI workloads.

## Hardware Context

- **GPU**: NVIDIA GeForce RTX 5090
- **Architecture**: Blackwell (Compute Capability 12.0)
- **VRAM**: 32GB GDDR7
- **CUDA**: 13.0 (system), 12.9 (PyTorch)
- **Driver**: 580.65.06
- **Power Limit**: 450W (configurable 300-450W)

## Experiment Categories

### 1. Memory Optimization Experiments

#### Experiment 1.1: Flash Attention 3 Implementation

**Objective**: Reduce memory usage and improve attention computation speed.

**Setup**:
```python
# Install Flash Attention
pip install flash-attn --no-build-isolation

# Test implementation
import torch
from flash_attn import flash_attn_func

def benchmark_attention(seq_len, batch_size, n_heads, d_head):
    q = torch.randn(batch_size, n_heads, seq_len, d_head).cuda().half()
    k = torch.randn(batch_size, n_heads, seq_len, d_head).cuda().half()
    v = torch.randn(batch_size, n_heads, seq_len, d_head).cuda().half()

    # Standard attention
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    output_standard = torch.matmul(attn, v)
    end.record()

    torch.cuda.synchronize()
    standard_time = start.elapsed_time(end)

    # Flash attention
    start.record()
    output_flash = flash_attn_func(q, k, v)
    end.record()

    torch.cuda.synchronize()
    flash_time = start.elapsed_time(end)

    return standard_time, flash_time
```

**Expected Results**:
- 2-4x speedup for long sequences
- 10-20 times memory reduction for attention layers
- Better scaling with sequence length

#### Experiment 1.2: VRAM Fragmentation Analysis

**Objective**: Understand and minimize memory fragmentation during inference.

**Implementation**:
```python
import torch
import gc

def analyze_memory_fragmentation():
    torch.cuda.empty_cache()

    # Initial state
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # Allocate and deallocate tensors of varying sizes
    tensors = []
    for size in [1, 10, 100, 500, 1000]:
        t = torch.randn(size, 1024, 1024).cuda()
        tensors.append(t)

    # Delete every other tensor
    for i in range(0, len(tensors), 2):
        del tensors[i]

    gc.collect()
    torch.cuda.empty_cache()

    # Check fragmentation
    print(f"After fragmentation:")
    print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # Try to allocate large tensor
    try:
        large = torch.randn(2000, 1024, 1024).cuda()
        print("Large allocation succeeded")
    except RuntimeError as e:
        print(f"Large allocation failed: {e}")
```

### 2. Quantization Experiments

#### Experiment 2.1: FP8 vs FP16 vs INT8 Comparison

**Objective**: Leverage RTX 5090's native FP8 support for optimal speed/quality tradeoff.

**Setup**:
```bash
# Install transformer-engine for FP8 support
pip install transformer-engine
```

**Implementation**:
```python
import torch
import transformer_engine.pytorch as te
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def compare_quantization_methods(model_name="meta-llama/Llama-2-7b-hf"):
    # Load base model
    model_fp16 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )

    # FP8 quantization (RTX 5090 native)
    model_fp8 = te.fp8_autocast(enabled=True)(model_fp16)

    # INT8 quantization
    model_int8 = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="cuda"
    )

    # Benchmark inference
    input_ids = torch.randint(0, 30000, (1, 512)).cuda()

    results = {}
    for name, model in [("FP16", model_fp16), ("FP8", model_fp8), ("INT8", model_int8)]:
        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(10):
                output = model(input_ids)

        torch.cuda.synchronize()
        end = time.perf_counter()

        results[name] = {
            "time": (end - start) / 10,
            "memory": torch.cuda.max_memory_allocated() / 1e9
        }

    return results
```

**Expected Results**:
- FP8: Approximately 1.5-2 times faster than FP16 with minimal quality loss
- INT8: Approximately 2-3 times faster but potential quality degradation
- Memory savings: FP8 (50%), INT8 (50-75%)

### 3. Inference Acceleration

#### Experiment 3.1: CUDA Graphs for Static Inference

**Objective**: Eliminate kernel launch overhead for fixed-shape inference.

**Implementation**:
```python
import torch

class CUDAGraphModel:
    def __init__(self, model, input_shape):
        self.model = model
        self.input_shape = input_shape

        # Warmup
        static_input = torch.randn(*input_shape).cuda()
        static_output = model(static_input)

        # Create graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = model(static_input)

        self.static_input = static_input

    def __call__(self, input_tensor):
        self.static_input.copy_(input_tensor)
        self.graph.replay()
        return self.static_output.clone()

# Benchmark
def benchmark_cuda_graphs(model, batch_size=1, seq_len=512):
    input_shape = (batch_size, seq_len)

    # Standard inference
    input_tensor = torch.randint(0, 30000, input_shape).cuda()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        output = model(input_tensor)
    end.record()

    torch.cuda.synchronize()
    standard_time = start.elapsed_time(end)

    # CUDA Graph inference
    graph_model = CUDAGraphModel(model, input_shape)

    start.record()
    for _ in range(100):
        output = graph_model(input_tensor)
    end.record()

    torch.cuda.synchronize()
    graph_time = start.elapsed_time(end)

    return standard_time, graph_time
```

**Expected Results**:
- 10-30% latency reduction for small batches
- Most effective for models with many small kernels
- No benefit for dynamic shapes

#### Experiment 3.2: Multi-Stream Parallel Inference

**Objective**: Maximize GPU utilization through concurrent inference streams.

**Implementation**:
```python
import torch
import torch.nn as nn
from torch.cuda import Stream
from queue import Queue
import threading

class MultiStreamInference:
    def __init__(self, model, n_streams=4):
        self.model = model
        self.n_streams = n_streams
        self.streams = [Stream() for _ in range(n_streams)]

    def process_batch(self, inputs):
        """Process multiple inputs in parallel streams"""
        results = [None] * len(inputs)

        for i, input_tensor in enumerate(inputs):
            stream_idx = i % self.n_streams
            stream = self.streams[stream_idx]

            with torch.cuda.stream(stream):
                results[i] = self.model(input_tensor)

        # Synchronize all streams
        for stream in self.streams:
            stream.synchronize()

        return results

# Benchmark multi-stream vs sequential
def benchmark_multi_stream(model, n_batches=16):
    inputs = [torch.randint(0, 30000, (1, 512)).cuda() for _ in range(n_batches)]

    # Sequential processing
    torch.cuda.synchronize()
    start = time.perf_counter()

    results_seq = []
    for inp in inputs:
        results_seq.append(model(inp))

    torch.cuda.synchronize()
    seq_time = time.perf_counter() - start

    # Multi-stream processing
    ms_inference = MultiStreamInference(model, n_streams=4)

    torch.cuda.synchronize()
    start = time.perf_counter()

    results_parallel = ms_inference.process_batch(inputs)

    torch.cuda.synchronize()
    parallel_time = time.perf_counter() - start

    return seq_time, parallel_time
```

### 4. Profiling and Tuning

#### Experiment 4.1: Power/Performance Optimization

**Objective**: Find optimal power limits for different workloads.

**Setup**:
```bash
# Set power limit (requires sudo)
sudo nvidia-smi -pl 300  # Set to 300W
sudo nvidia-smi -pl 450  # Set to 450W (max)

# Monitor power and performance
nvidia-smi dmon -s puc
```

**Measurement Script**:
```python
import subprocess
import torch
import time

def benchmark_at_power_limit(power_limit, model, input_data):
    # Set power limit
    subprocess.run(f"sudo nvidia-smi -pl {power_limit}", shell=True)
    time.sleep(5)  # Allow GPU to stabilize

    # Run benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(100):
            output = model(input_data)

    torch.cuda.synchronize()
    end = time.perf_counter()

    # Get power consumption
    result = subprocess.run(
        "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
        shell=True, capture_output=True, text=True
    )
    avg_power = float(result.stdout.strip())

    return {
        "time": end - start,
        "power": avg_power,
        "perf_per_watt": 100 / ((end - start) * avg_power)
    }

# Test different power limits
power_limits = [300, 350, 400, 450]
for pl in power_limits:
    results = benchmark_at_power_limit(pl, model, input_data)
    print(f"{pl}W: {results}")
```

#### Experiment 4.2: Tensor Core Utilization Analysis

**Objective**: Ensure maximum utilization of RTX 5090's Tensor Cores.

**Profiling with NCU**:
```bash
# Profile tensor core usage
ncu --set full \
    --metrics sm__ops_path_tensor_src_fp16_dst_fp16.sum,\
             sm__ops_path_tensor_src_fp16_dst_fp32.sum,\
             sm__ops_path_tensor_src_int8_dst_int32.sum \
    python inference_script.py

# Detailed kernel analysis
ncu --kernel-name gemm --launch-skip 10 --launch-count 1 \
    --section SpeedOfLight \
    python inference_script.py
```

**Python Analysis**:
```python
import torch
from torch.profiler import profile, ProfilerActivity

def analyze_tensor_core_usage(model, input_data):
    with profile(
        activities=[ProfilerActivity.CUDA],
        with_stack=True,
        with_modules=True,
        record_shapes=True
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                output = model(input_data)

    # Analyze tensor core operations
    tensor_core_ops = 0
    total_cuda_time = 0

    for event in prof.key_averages():
        if event.device_type == torch.DeviceType.CUDA:
            total_cuda_time += event.cuda_time_total
            # Check for tensor core operations (gemm, conv)
            if any(tc_op in event.key for tc_op in ['gemm', 'conv', 'bmm']):
                tensor_core_ops += event.cuda_time_total

    utilization = (tensor_core_ops / total_cuda_time) * 100 if total_cuda_time > 0 else 0

    print(f"Tensor Core Utilization: {utilization:.2f}%")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return utilization
```

### 5. Optimization Comparison

#### Experiment 5.1: vLLM vs TensorRT-LLM vs llama.cpp

**Objective**: Compare inference engines for deployment.

**Setup Script**:
```bash
#!/bin/bash

MODEL="meta-llama/Llama-2-7b-hf"
PROMPT="The future of AI is"

# vLLM
echo "Testing vLLM..."
python -m vllm.entrypoints.api_server \
    --model $MODEL \
    --dtype float16 \
    --max-model-len 2048 &
VLLM_PID=$!
sleep 30
curl -X POST "http://localhost:8000/generate" \
    -H "Content-Type: application/json" \
    -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": 100}" \
    -w "\nTime: %{time_total}s\n"
kill $VLLM_PID

# TensorRT-LLM (requires conversion)
echo "Testing TensorRT-LLM..."
trtllm-build --model_name llama \
    --hf_model_dir $MODEL \
    --output_dir ./trt_engines \
    --dtype float16

# llama.cpp
echo "Testing llama.cpp..."
./llama-bench -m llama-2-7b.gguf -p "$PROMPT" -n 100
```

## Implementation Priority

For maximum impact on your inference workloads:

1. **Flash Attention 3** - Immediate memory and speed benefits
2. **FP8 Quantization** - Unique to RTX 5090, significant speedup
3. **CUDA Graphs** - Low-effort latency reduction
4. **Multi-Stream Inference** - Better GPU utilization
5. **Power Optimization** - Balance performance and efficiency

## Monitoring and Validation

### Key Metrics to Track

1. **Performance Metrics**:
   - Tokens per second
   - First token latency (TTFT)
   - Inter-token latency
   - Batch processing throughput

2. **Resource Metrics**:
   - VRAM usage and fragmentation
   - GPU utilization percentage
   - Tensor Core utilization
   - Power consumption

3. **Quality Metrics**:
   - Perplexity comparison
   - Output consistency
   - Quantization error rates

### Monitoring Commands

```bash
# Real-time GPU monitoring
watch -n 0.5 nvidia-smi

# Detailed metrics
nvidia-smi dmon -s pucvmet

# Memory tracking
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv -l 1

# Process-specific monitoring
nvidia-smi pmon -i 0
```

## Expected Improvements

Based on RTX 5090 capabilities:

- **Memory Efficiency**: 50-70% reduction with Flash Attention and Quantization
- **Inference Speed**: 2-4 times improvement with FP8 and CUDA optimizations
- **Power Efficiency**: 20-30% better performance per watt with tuning
- **Throughput**: 3-5 times improvement with proper batching and streaming

## Next Steps

1. Start with Flash Attention 3 implementation
2. Benchmark baseline performance for comparison
3. Implement FP8 quantization for your specific models
4. Profile with NSight Systems for bottleneck identification
5. Deploy optimized configurations to Docker containers

## Notes

- RTX 5090's Blackwell architecture provides unique opportunities not available in previous generations
- FP8 support is native and should be prioritized over INT8 where quality matters
- The 32GB VRAM allows for larger batch sizes - experiment with dynamic batching
- CUDA 13.0 features are still emerging - check for updates regularly

---

*Last Updated: 2025-09-23*