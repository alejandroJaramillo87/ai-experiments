# Parameter Optimization Guide

Systematic methodology for benchmarking and optimizing inference parameters for llama.cpp and vLLM services on the AMD Ryzen 9950X + RTX 5090 workstation.

## Optimization Methodology

### Pre-Optimization Baseline

Before testing parameter changes, establish baseline performance:

```bash
# Run baseline benchmark
scripts/benchmark.py --service llama-cpu --label "baseline_cpu"
scripts/benchmark.py --service llama-gpu --label "baseline_gpu"
# vLLM when CUDA 13 available:
# scripts/benchmark.py --service vllm-gpu --label "baseline_vllm"
```

**Record baseline metrics**:
- Tokens per second (single request)
- First-token latency (ms)
- Memory usage (RAM/VRAM)
- CPU/GPU utilization
- Response consistency (standard deviation)

### Parameter Testing Framework

#### Test One Parameter at a Time
- **Isolation principle**: Change single parameter per test
- **Revert between tests**: Return to baseline configuration
- **Multiple runs**: Average results across 3-5 benchmark runs
- **Statistical significance**: Track variance and confidence intervals

#### Test Matrix Priority

**High Priority (Immediate Testing)**:
1. Cache optimizations (`--kv-split`, `--cache-reuse`)
2. GPU memory utilization (0.85 → 0.99 for vLLM)
3. Batch size variations
4. Thread count optimization

**Medium Priority (Secondary Testing)**:
1. Context size impact
2. HTTP thread optimization
3. Speculative decoding setup
4. Quantization strategies

**Low Priority (Experimental)**:
1. CPU affinity masks
2. Advanced scheduler tuning
3. Exotic optimization flags

## llama.cpp Parameter Testing

### CPU Service Optimization

#### Threading Tests
```bash
# Test different thread counts
for threads in 8 10 12 14; do
    # Update THREADS environment variable
    docker-compose exec llama-cpu env THREADS=$threads
    scripts/benchmark.py --service llama-cpu --label "threads_${threads}"
done
```

#### Batch Size Optimization
```bash
# Test batch size combinations
for batch in 1024 2048 4096; do
    for ubatch in 512 1024 2048; do
        # Update batch size configuration
        scripts/benchmark.py --service llama-cpu --label "batch_${batch}_${ubatch}"
    done
done
```

#### Cache Optimization Tests
```bash
# Test new cache parameters (requires server restart)
# Add to entrypoint.sh:
# --kv-split
# --cache-reuse
scripts/benchmark.py --service llama-cpu --label "cache_optimized"
```

### GPU Service Optimization

#### Context Size Impact
```bash
# Test different context sizes
for ctx in 16384 32768 65536 131072; do
    # Update ctx-size in Dockerfile.llama-gpu
    docker-compose build llama-gpu
    scripts/benchmark.py --service llama-gpu --label "ctx_${ctx}"
done
```

#### GPU Threading Tests
```bash
# Test different thread configurations
for threads in 1 2 4; do
    # Update threads parameter in Dockerfile.llama-gpu
    scripts/benchmark.py --service llama-gpu --label "gpu_threads_${threads}"
done
```

## vLLM Parameter Testing (Post CUDA 13)

### Memory Utilization Optimization
```bash
# Test GPU memory utilization
for mem_util in 0.85 0.90 0.95 0.99; do
    # Update --gpu-memory-utilization in Dockerfile.vllm-gpu
    scripts/benchmark.py --service vllm-gpu --label "mem_${mem_util}"
done
```

### Concurrent Request Testing
```bash
# Test concurrent request capacity
for max_seqs in 256 512 1024; do
    for batch_tokens in 32768 65536 131072; do
        # Update vLLM parameters
        scripts/benchmark.py --service vllm-gpu --label "seqs_${max_seqs}_tokens_${batch_tokens}" --concurrent-requests 10
    done
done
```

## Benchmarking Tools and Metrics

### Existing Benchmark Script

**scripts/benchmark.py** capabilities:
- Single request latency measurement
- Multiple test prompts (coding, chat, reasoning)
- JSON output for analysis
- Hardware utilization monitoring

### Enhanced Benchmarking Needs

#### Concurrent Request Testing
```bash
# Add concurrent testing capability
scripts/benchmark.py --concurrent-requests 5 --duration 60
```

#### Stress Testing
```bash
# Long-duration testing for stability
scripts/benchmark.py --duration 300 --ramp-up 30
```

#### Memory Profiling
```bash
# Monitor memory usage patterns
scripts/benchmark.py --profile-memory --output memory_profile.json
```

### Key Performance Metrics

#### Latency Optimization (llama.cpp)
- **First-token latency**: Time to first response token (target: <100ms)
- **Tokens per second**: Single request throughput (target: >200 t/s)
- **Total response time**: Complete response generation
- **Latency consistency**: Standard deviation (target: <20ms)

#### Throughput Optimization (vLLM)
- **Concurrent capacity**: Maximum simultaneous requests
- **Aggregate throughput**: Total tokens/second across all requests
- **Queue latency**: Time from request to processing start
- **Memory efficiency**: Requests per GB of VRAM

### Hardware Monitoring

#### CPU Metrics
```bash
# Monitor during benchmarks
htop
iostat -x 1
vmstat 1
```

#### GPU Metrics
```bash
# Monitor GPU utilization
nvidia-smi -l 1
nvidia-smi dmon -i 0 -s pmu
```

#### Memory Monitoring
```bash
# System memory
free -h
cat /proc/meminfo | grep Huge

# GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1
```

## Parameter Change Implementation

### Environment Variable Changes
```bash
# Temporary testing (no rebuild required)
docker-compose exec llama-cpu env NEW_PARAM=value command

# Persistent changes
# Update .env file or docker-compose.yaml environment section
```

### Dockerfile Parameter Changes
```bash
# Requires container rebuild
docker-compose build service-name
docker-compose up service-name

# Or specific service restart
docker-compose restart llama-gpu
```

### Configuration Testing Workflow
1. **Backup current configuration**
2. **Make single parameter change**
3. **Rebuild/restart affected services**
4. **Run benchmark suite**
5. **Record results**
6. **Revert to baseline**
7. **Repeat for next parameter**

## Optimization Decision Framework

### Performance Regression Detection
- **Threshold**: >5% performance decrease = revert change
- **Memory issues**: OOM errors or excessive swapping = revert
- **Stability problems**: Crashes or timeouts = revert

### Trade-off Analysis

#### Memory vs Performance
- **Higher memory usage**: Acceptable if <90% system utilization
- **Context size**: Larger contexts acceptable if latency impact <10%
- **Batch sizes**: Larger batches acceptable if latency impact <20%

#### Latency vs Throughput
- **llama.cpp**: Prioritize latency (single-request performance)
- **vLLM**: Prioritize throughput (concurrent request capacity)
- **Resource usage**: CPU/GPU utilization should be >80% under load

### Success Criteria

#### llama.cpp Optimization Success
- **Latency improvement**: >10% reduction in response time
- **Consistency improvement**: <15ms standard deviation
- **Resource efficiency**: Memory usage within limits

#### vLLM Optimization Success
- **Throughput improvement**: >20% increase in concurrent capacity
- **Queue efficiency**: <500ms average queue time
- **Memory utilization**: >95% VRAM usage without OOM

## Documentation and Tracking

### Results Documentation
```bash
# Create optimization log
echo "Date: $(date)" >> optimization_log.md
echo "Parameter: --kv-split" >> optimization_log.md
echo "Result: +15% performance" >> optimization_log.md
```

### Performance Database
- **JSON results**: Store in `benchmark_results/` directory
- **Configuration tracking**: Git commit for each significant change
- **Regression testing**: Re-run key benchmarks after system updates

### Rollback Procedures
- **Git revert**: For configuration file changes
- **Docker image tags**: Maintain known-good image versions
- **Configuration backup**: Store working configurations

---

*Systematic parameter optimization methodology for the AMD Ryzen 9950X + RTX 5090 AI workstation. Focus on measurable performance improvements with statistical validation.*