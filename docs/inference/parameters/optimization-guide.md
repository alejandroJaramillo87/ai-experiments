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

## llama.cpp Parameter Testing - Results

### CPU Service Optimization (Completed September 2025)

#### Batch Size Testing Results
**Tested on AMD Ryzen 9950X with 12 cores allocated:**
```bash
# Baseline (batch 2048): 35.44 tokens/sec ✓ OPTIMAL
# Batch 512: 34.79 tokens/sec (2% slower)
# Batch 4096: 34.95 tokens/sec (1.4% slower)
```

**Key Finding**: Batch size 2048 is optimal for CPU inference. Performance follows a curve where both smaller and larger batch sizes decrease throughput.

#### Invalid Parameters Discovered
```bash
# These parameters cause errors in llama.cpp server:
# --kv-split: "error: invalid argument: --kv-split"
# --cache-reuse: Not recognized
# --parallel: No effect on server (may be CLI-only)
```

#### Successful Configuration
```bash
# Optimal parameters for llama-cpu service:
BATCH_SIZE=2048
UBATCH_SIZE=2048
THREADS=12  # Match allocated CPU cores
--cont-batching  # Keep enabled
--mlock  # Memory locking essential
--no-warmup  # Skip warmup for faster startup
```

### GPU Service Optimization (Completed September 2025)

#### Batch Size Testing Results
**Tested on RTX 5090 with gpt-oss-20b model:**
```bash
# Optimal configuration found: batch 2048, ubatch 512
# Performance: 286.85 tokens/sec with 95% GPU utilization

# Easy testing with environment variables (no rebuild needed):
BATCH_SIZE=512 UBATCH_SIZE=512 docker-compose up llama-gpu
BATCH_SIZE=1024 UBATCH_SIZE=1024 docker-compose up llama-gpu
BATCH_SIZE=2048 UBATCH_SIZE=512 docker-compose up llama-gpu  # OPTIMAL
```

**Key Finding**: Batch size 2048 with ubatch 512 provides optimal GPU performance. The 4:1 ratio between batch and ubatch size works well for GPU memory access patterns.

#### Context Size Testing
```bash
# Test different context sizes using environment variables
CTX_SIZE=16384 docker-compose up llama-gpu
CTX_SIZE=32768 docker-compose up llama-gpu
CTX_SIZE=65536 docker-compose up llama-gpu  # Current default
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

### Environment Variable Changes (Recommended)
```bash
# Both llama-cpu and llama-gpu now use environment variables

# Test different configurations without rebuilding:
BATCH_SIZE=1024 docker-compose up llama-cpu
BATCH_SIZE=2048 UBATCH_SIZE=512 docker-compose up llama-gpu

# Persistent changes - add to docker-compose.yaml:
environment:
  - BATCH_SIZE=2048
  - UBATCH_SIZE=512
```

### Direct Container Testing
```bash
# Quick parameter testing
docker-compose exec llama-gpu env BATCH_SIZE=1024 /app/entrypoint.sh

# Monitor performance
docker-compose exec llama-gpu nvidia-smi
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

#### Memory vs Performance (Validated)
- **Higher memory usage**: Acceptable if <90% system utilization
- **Context size**: Larger contexts acceptable if latency impact <10%
- **Batch sizes**: **FINDING: Batch 2048 is optimal for both CPU and GPU**
  - CPU: 2048/2048 (unified batch) - 35.44 tok/s
  - GPU: 2048/512 (split batch) - 286.85 tok/s

#### Latency vs Throughput (Validated)
- **llama.cpp CPU**: Unified batch sizes (2048/2048) work best
- **llama.cpp GPU**: Split batch sizes (2048/512) optimize memory access
- **Key insight**: Batch size 2048 is universally optimal, ubatch varies by device
- **Resource usage**:
  - CPU utilization should be >80% under load
  - GPU utilization achieved 95% with optimal settings

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