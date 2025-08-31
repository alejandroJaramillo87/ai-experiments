# AI Workstation Benchmark Testing - Claude Context

## Hardware Specifications
- **GPU:** RTX 5090 (24GB VRAM)
- **CPU:** AMD Ryzen 9950X
- **RAM:** 128GB DDR5 
- **OS:** Ubuntu 24.04 LTS
- **Storage:** Samsung 990 Pro 2TB (available for models/datasets)

## Benchmark Directory Purpose
This directory contains performance testing and benchmarking utilities for AI model inference and training on the above hardware configuration.

## Primary Benchmarking Goals
1. **GPU vs CPU Performance:** Compare inference speeds between GPU (VRAM) and CPU (RAM) execution
2. **Concurrent Model Testing:** Benchmark running multiple models simultaneously (GPU + CPU)
3. **Memory Utilization:** Test VRAM and RAM usage patterns under different model loads
4. **Throughput Optimization:** Measure tokens/second, batch processing capabilities
5. **Framework Comparison:** Compare performance across different inference engines (vLLM, Ollama, Transformers, etc.)

## Repository Context
- **Working Repo:** https://github.com/alejandroJaramillo87/ai-workstation/tree/main/benchmark_tests
- **Focus:** Performance testing and optimization for dual GPU/CPU AI model deployment

## Typical Assistance Needed
- Benchmark script development and optimization
- Performance metrics collection and analysis
- Memory usage profiling and monitoring
- Comparative testing methodologies
- Hardware utilization measurement tools
- Results visualization and reporting

## Key Constraints
- Ubuntu 24.04 LTS environment
- Focus on inference performance over training benchmarks
- Must work with concurrent model deployment strategy
- Optimize for realistic production workloads

When asking for help, mention you're working in the benchmark directory of an AI workstation setup focused on performance testing of concurrent GPU/CPU AI model deployment.