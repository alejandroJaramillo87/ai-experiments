# EXPERIMENTAL: Memory Bandwidth Optimizations for CPU Inference

**STATUS: EXPERIMENTAL - Practical Implementation Focus**

**INSIGHT: CPU optimizations yielded minimal gains while memory optimizations provided 26% improvement, confirming memory bandwidth as the primary bottleneck**

## Table of Contents

- [Executive Summary](#executive-summary)
- [Technical Analysis: Why Memory Matters More](#technical-analysis-why-memory-matters-more)
  - [Bandwidth Mathematics](#bandwidth-mathematics)
  - [Why CPU Optimizations Had Minimal Impact](#why-cpu-optimizations-had-minimal-impact)
- [Experiment 1: Layer-Ahead Prefetch Wrapper](#experiment-1-layer-ahead-prefetch-wrapper)
- [Experiment 2: Mixed Huge Pages Strategy](#experiment-2-mixed-huge-pages-strategy)
- [Experiment 3: Cache Line Aligned Weight Packing](#experiment-3-cache-line-aligned-weight-packing)
- [Experiment 4: Memory Bandwidth Throttle Manager](#experiment-4-memory-bandwidth-throttle-manager)
- [Experiment 5: Weight Compression with AVX-512 Decompression](#experiment-5-weight-compression-with-avx-512-decompression)
- [Experiment 6: CCX-Aware Memory Allocation](#experiment-6-ccx-aware-memory-allocation)
- [Experiment 7: Asynchronous Double Buffering](#experiment-7-asynchronous-double-buffering)
- [Experiment 8: Smart KV Cache Manager](#experiment-8-smart-kv-cache-manager)
- [Experiment 9: VNNI Acceleration for Quantized Models](#experiment-9-vnni-acceleration-for-quantized-models)
- [Implementation Roadmap](#implementation-roadmap)
  - [Phase 1: Quick Wins](#phase-1-quick-wins-week-1)
  - [Phase 2: Medium Complexity](#phase-2-medium-complexity-week-2)
  - [Phase 3: Advanced](#phase-3-advanced-week-3-4)
  - [Phase 4: High-Impact Complex](#phase-4-high-impact-complex-week-4-6)
- [Combined Approach](#combined-approach)
  - [For FP32 Models](#for-fp32-models)
  - [For Quantized Models](#for-quantized-models-q8_0-q4_0)
- [Validation Methodology](#validation-methodology)
- [Risk Mitigation](#risk-mitigation)
- [Conclusion](#conclusion)

## Executive Summary

Based on benchmarking showing memory optimizations vastly outperformed CPU optimizations, this document presents 8 practical experiments to further address memory bandwidth constraints in CPU-based LLM inference. These are realistic, implementable solutions that avoid academic complexity.

**Key Finding**: System is memory bandwidth limited (96GB/s DDR5-6000) not compute limited
**Approach**: Focus on bandwidth efficiency, TLB optimization, and memory access patterns
**Expected Impact**: Additional 15-25% performance improvement possible
**Implementation Effort**: 3 days to 2 weeks per experiment

## Technical Analysis: Why Memory Matters More

### Bandwidth Mathematics

For a 30GB model generating tokens at 35 tok/s:
```
Model size: 30GB
Tokens/sec: 35
Theoretical bandwidth needed: 30GB × 35 = 1,050 GB/s
Available bandwidth: 96 GB/s (DDR5-6000)
Bandwidth ratio: 96/1050 = 9.1% efficiency
```

This shows we're using each byte ~11 times via cache, but still bandwidth limited.

### Why CPU Optimizations Had Minimal Impact

Your benchmarks revealed:
- **Before BIOS optimizations**: 28.09 tokens/sec
- **After CPU+Memory optimizations**: 35.44 tokens/sec
- **Memory optimizations (FCLK, timings)**: Contributed most of the 26% gain
- **CPU optimizations (boost, curve optimizer)**: Minimal contribution

This confirms the system has excess CPU cycles but insufficient memory bandwidth.

## Experiment 1: Layer-Ahead Prefetch Wrapper

**Effort**: 1 week | **Risk**: Low | **Expected Gain**: 8-12%

### Concept
Prefetch layer N+1 weights while computing layer N, hiding memory latency.

### Implementation
```cpp
// layer_prefetch_wrapper.cpp - LD_PRELOAD library

static int current_layer = 0;
static void* layer_weights[128];  // Pointers to each layer's weights

extern "C" void ggml_compute_forward(struct ggml_compute_params* params,
                                     struct ggml_tensor* tensor) {
    // Detect which layer we're processing
    int layer_id = detect_layer_id(tensor);

    if (layer_id != current_layer) {
        current_layer = layer_id;

        // Prefetch next layer's weights
        if (layer_id + 1 < num_layers) {
            void* next_weights = layer_weights[layer_id + 1];
            size_t weight_size = get_layer_size(layer_id + 1);

            // Software prefetch to L3 cache
            for (size_t i = 0; i < weight_size; i += 64) {
                _mm_prefetch((char*)next_weights + i, _MM_HINT_T2);
            }
        }
    }

    // Call original function
    original_ggml_compute_forward(params, tensor);
}
```

### Why This Works
- llama.cpp processes layers sequentially and predictably
- Prefetch happens during compute (overlapping operations)
- No complex pattern detection needed

## Experiment 2: Mixed Huge Pages Strategy

**Effort**: 3-4 days | **Risk**: Medium | **Expected Gain**: 5-8%

### Concept
Use different page sizes optimally based on access patterns:
- **1GB pages**: Model weights (sequential access)
- **2MB pages**: KV cache (strided access)
- **4KB pages**: Temporary buffers (short-lived)

### Implementation
```bash
# Kernel configuration
GRUB_CMDLINE_LINUX="hugepagesz=1G hugepages=32 hugepagesz=2M hugepages=1024"

# Mount points
mkdir -p /mnt/hugepages-1gb
mkdir -p /mnt/hugepages-2mb
mount -t hugetlbfs -o pagesize=1G none /mnt/hugepages-1gb
mount -t hugetlbfs -o pagesize=2M none /mnt/hugepages-2mb
```

```cpp
// Smart allocation based on data type
void* allocate_memory(size_t size, memory_type_t type) {
    switch(type) {
        case MEMORY_WEIGHTS:
            // Use 1GB pages for weights
            return mmap(NULL, size, PROT_READ|PROT_WRITE,
                       MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|MAP_HUGE_1GB, -1, 0);

        case MEMORY_KV_CACHE:
            // Use 2MB pages for KV cache
            return mmap(NULL, size, PROT_READ|PROT_WRITE,
                       MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|MAP_HUGE_2MB, -1, 0);

        default:
            // Regular malloc for temporary buffers
            return malloc(size);
    }
}
```

### TLB Impact
```
30GB model with different page sizes:
- 4KB pages: 7,864,320 pages → Constant TLB misses
- 2MB pages: 15,360 pages → Some TLB pressure
- 1GB pages: 30 pages → Fits in L1 TLB (8 entries for 1GB on Zen 5)
- Mixed: ~50 pages total → Minimal TLB pressure
```

## Experiment 3: Cache Line Aligned Weight Packing

**Effort**: 1 week | **Risk**: Low | **Expected Gain**: 5-10%

### Concept
Repack weights to optimize cache line usage and prevent split loads.

### Implementation
```cpp
// weight_repacker.cpp

struct AlignedTensor {
    void* data;
    size_t rows, cols;
    size_t stride;  // Padded for alignment
};

AlignedTensor* repack_weights(const float* original, size_t rows, size_t cols) {
    // Align to 64-byte cache lines
    const size_t cache_line = 64;
    const size_t floats_per_line = cache_line / sizeof(float);

    // Pad columns to cache line boundary
    size_t padded_cols = ((cols + floats_per_line - 1) / floats_per_line) * floats_per_line;

    // Allocate aligned memory
    void* aligned_data;
    posix_memalign(&aligned_data, cache_line, rows * padded_cols * sizeof(float));

    // Copy with padding
    float* dst = (float*)aligned_data;
    for (size_t r = 0; r < rows; r++) {
        memcpy(dst + r * padded_cols, original + r * cols, cols * sizeof(float));
        // Zero padding for vectorization
        memset(dst + r * padded_cols + cols, 0, (padded_cols - cols) * sizeof(float));
    }

    AlignedTensor* tensor = new AlignedTensor;
    tensor->data = aligned_data;
    tensor->rows = rows;
    tensor->cols = cols;
    tensor->stride = padded_cols;

    return tensor;
}
```

### Benefits
- No partial cache line loads
- Better prefetcher efficiency
- SIMD operations work on aligned data

## Experiment 4: Memory Bandwidth Throttle Manager

**Effort**: 4-5 days | **Risk**: Medium | **Expected Gain**: 3-7%

### Concept
Paradoxically, limiting bandwidth usage can increase throughput by reducing memory controller queuing.

### Implementation
```cpp
// bandwidth_manager.cpp

class BandwidthManager {
private:
    atomic<uint64_t> bytes_this_interval{0};
    uint64_t max_bytes_per_ms = 96 * 1024 * 1024;  // 96 GB/s = 96 MB/ms

public:
    void regulated_memcpy(void* dst, const void* src, size_t size) {
        const size_t chunk_size = 1024 * 1024;  // 1MB chunks

        for (size_t offset = 0; offset < size; offset += chunk_size) {
            size_t copy_size = min(chunk_size, size - offset);

            // Check bandwidth usage
            uint64_t current = bytes_this_interval.fetch_add(copy_size);
            if (current > max_bytes_per_ms * 0.8) {  // 80% threshold
                // Brief pause to let controller catch up
                _mm_pause();
                _mm_pause();
            }

            memcpy((char*)dst + offset, (char*)src + offset, copy_size);
        }
    }
};
```

### Why This Works
- Memory controllers have finite queue depth
- Oversaturation causes head-of-line blocking
- Strategic pauses allow queue drainage

## Experiment 5: Weight Compression with AVX-512 Decompression

**Effort**: 1-2 weeks | **Risk**: Medium | **Expected Gain**: 10-15%

### Concept
Store weights compressed, decompress on-the-fly using excess CPU cycles.

### Implementation
```cpp
// Simple delta encoding for weights
struct CompressedWeights {
    float base_value;
    int8_t deltas[];  // Delta from base, quantized to INT8
};

// AVX-512 decompression
void decompress_weights_avx512(const CompressedWeights* compressed,
                               float* output, size_t count) {
    __m512 base = _mm512_set1_ps(compressed->base_value);
    const float scale = 0.01f;  // Scaling factor for deltas
    __m512 scale_vec = _mm512_set1_ps(scale);

    for (size_t i = 0; i < count; i += 16) {
        // Load 16 INT8 deltas
        __m128i deltas_int8 = _mm_loadu_si128((__m128i*)&compressed->deltas[i]);

        // Convert INT8 to INT32
        __m512i deltas_int32 = _mm512_cvtepi8_epi32(deltas_int8);

        // Convert to float and scale
        __m512 deltas_float = _mm512_cvtepi32_ps(deltas_int32);
        deltas_float = _mm512_mul_ps(deltas_float, scale_vec);

        // Add base value
        __m512 result = _mm512_add_ps(base, deltas_float);

        // Store
        _mm512_storeu_ps(&output[i], result);
    }
}
```

### Bandwidth Reduction
- Original: 30GB at FP32
- Compressed: ~8GB (base + INT8 deltas)
- Bandwidth saved: 73%
- CPU overhead: Approximately 5% (we have excess)

## Experiment 6: CCX-Aware Memory Allocation

**Effort**: 3-4 days | **Risk**: Low | **Expected Gain**: 5-10%

### Concept
The Ryzen 9950X has 2 CCDs (8 cores each). Distribute memory across both to use both memory controllers.

### Implementation
```bash
# Check NUMA topology (even on single socket)
numactl --hardware

# Should show 2 nodes for 2 CCDs
# node 0: cores 0-7 (CCD0)
# node 1: cores 8-15 (CCD1)
```

```cpp
// Interleaved allocation across CCDs
void* allocate_interleaved(size_t size) {
    // Set memory policy to interleave across nodes
    unsigned long nodemask = 0x3;  // Nodes 0 and 1

    // Use mbind to interleave pages
    void* ptr = mmap(NULL, size, PROT_READ|PROT_WRITE,
                    MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

    mbind(ptr, size, MPOL_INTERLEAVE, &nodemask, 2, MPOL_MF_MOVE);

    return ptr;
}

// Or use numactl wrapper
// numactl --interleave=0,1 ./llama-server
```

### Benefits
- Uses both memory controllers
- Reduces contention on single controller
- Better bandwidth utilization

## Experiment 7: Asynchronous Double Buffering

**Effort**: 1 week | **Risk**: Medium | **Expected Gain**: 8-12%

### Concept
Use DMA to load next layer while computing current layer.

### Implementation
```cpp
class DoubleBuffer {
private:
    void* buffer_a;
    void* buffer_b;
    atomic<bool> swap_ready{false};
    thread prefetch_thread;

public:
    void start_prefetch(void* source, size_t size) {
        prefetch_thread = thread([this, source, size]() {
            // Use non-temporal copy to avoid cache pollution
            memcpy_nt(buffer_b, source, size);
            swap_ready = true;
        });
    }

    void* get_compute_buffer() {
        if (swap_ready) {
            swap(buffer_a, buffer_b);
            swap_ready = false;
        }
        return buffer_a;
    }

    void memcpy_nt(void* dst, const void* src, size_t size) {
        // Non-temporal stores bypass cache
        for (size_t i = 0; i < size; i += 64) {
            __m512i data = _mm512_loadu_si512((char*)src + i);
            _mm512_stream_si512((char*)dst + i, data);
        }
        _mm_sfence();
    }
};
```

### Timeline
```
Layer N:   [Compute....]
Layer N+1:      [Prefetch][Compute....]
Layer N+2:           [Prefetch][Compute....]
```

## Experiment 8: Smart KV Cache Manager

**Effort**: 5-6 days | **Risk**: Medium | **Expected Gain**: 5-8%

### Concept
Optimize KV cache layout and access patterns for better memory efficiency.

### Implementation
```cpp
class SmartKVCache {
private:
    // Ring buffer for better locality
    struct RingBuffer {
        float* data;
        size_t capacity;
        size_t head;
        size_t tail;
    };

    RingBuffer cache;

    // Compress older entries
    struct CompressedEntry {
        uint16_t values[];  // FP16 for older entries
    };

public:
    void add_kv(const float* k, const float* v, size_t size) {
        // Recent entries in full precision
        if (is_recent(position)) {
            memcpy(cache.data + position, k, size * sizeof(float));
        } else {
            // Older entries compressed to FP16
            compress_to_fp16(k, compressed_cache + position, size);
        }
    }

    void compress_to_fp16(const float* src, uint16_t* dst, size_t count) {
        for (size_t i = 0; i < count; i += 16) {
            __m512 vals = _mm512_loadu_ps(src + i);
            __m256i fp16 = _mm512_cvtps_ph(vals, _MM_FROUND_TO_NEAREST_INT);
            _mm256_storeu_si256((__m256i*)(dst + i), fp16);
        }
    }
};
```

### Benefits
- Ring buffer improves cache locality
- Compression reduces bandwidth for older cache entries
- Better TLB usage with contiguous allocation

## Experiment 9: VNNI Acceleration for Quantized Models

**Effort**: 2-3 weeks | **Risk**: High | **Expected Gain**: 2-4x for quantized models**

### Concept
Use AVX-512 VNNI instructions to process quantized weights directly without dequantization, dramatically reducing memory bandwidth requirements.

### How VNNI Helps Memory Bandwidth

Current llama.cpp flow wastes bandwidth:
```
INT8 weights (1 byte) → Dequantize → FP32 (4 bytes) → Compute
Memory traffic: 5 bytes per weight
```

With VNNI optimization:
```
INT8 weights (1 byte) → Direct INT8 compute → Scale result
Memory traffic: 1 byte per weight (80% reduction)
```

### Key Benefits for Memory-Constrained Systems
1. **4x reduction in cache footprint** - More weights fit in L2/L3
2. **75% less memory bandwidth** - Process INT8 directly
3. **Synergy with huge pages** - Fewer TLB entries needed for smaller data

### Implementation Overview
See `experimental-vnni-wrapper.md` for full details. Key points:
- LD_PRELOAD wrapper intercepts BLAS calls
- Detects quantized buffers and routes to VNNI kernels
- Uses `_mm512_dpbusd_epi32` for INT8×INT8 operations

### Why This Matters for Memory Bandwidth
Your benchmarks show the system is memory-limited at 35 tok/s. With VNNI:
- **Q8_0 models**: Use 8x less bandwidth than FP32 for weights
- **Q4_0 models**: Use 16x less bandwidth than FP32
- **Result**: More bandwidth available for KV cache and activations

### Expected Combined Gains
When combined with memory optimizations:
| Model Type | Current | With Mem Opts | +VNNI | Total Gain |
|------------|---------|---------------|--------|------------|
| FP32 | 35 tok/s | 44 tok/s | 44 tok/s | 25% |
| Q8_0 | 35 tok/s | 44 tok/s | 110-130 tok/s | 3-4x |
| Q4_0 | 35 tok/s | 44 tok/s | 90-110 tok/s | 2.5-3x |

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)
1. **1GB Huge Pages** - Kernel config only, code exists
2. **CCX-Aware Allocation** - Simple numactl wrapper
3. **Cache Line Alignment** - Minimal code changes

Expected gain: 10-15%

### Phase 2: Medium Complexity (Week 2)
4. **Layer-Ahead Prefetch** - LD_PRELOAD wrapper
5. **Double Buffering** - Thread management needed

Expected gain: 8-12%

### Phase 3: Advanced (Week 3-4)
6. **Weight Compression** - AVX-512 kernels
7. **Smart KV Cache** - Significant refactoring
8. **Bandwidth Throttling** - Requires tuning

Expected gain: 5-10%

### Phase 4: High-Impact Complex (Week 4-6)
9. **VNNI Acceleration** - For quantized models only
   - Highest potential gain (2-4x)
   - Most complex implementation
   - See `experimental-vnni-wrapper.md` for details

Expected gain: 2-4x for quantized models

## Combined Approach

### For FP32 Models
**Best combination**: 1GB Huge Pages + Layer Prefetch + CCX Interleaving
- Reduces TLB misses by 99%
- Overlaps memory access with compute
- Uses both memory controllers
- Can be implemented in 1 week
- Expected total gain: 20-25%

### For Quantized Models (Q8_0, Q4_0)
**Ultimate combination**: Memory Optimizations + VNNI Acceleration
1. **Week 1**: Implement memory optimizations (huge pages, prefetch, CCX)
2. **Week 2-4**: Add VNNI wrapper for quantized compute
3. **Result**:
   - Memory optimizations: 25% gain
   - VNNI acceleration: 2-4x additional gain
   - Total: 2.5-5x performance improvement

This combination attacks both bottlenecks:
- Memory optimizations improve bandwidth utilization
- VNNI reduces bandwidth requirements by 75-80%
- Together they can push Q8_0 models to 100+ tokens/second

## Validation Methodology

### Performance Metrics
```bash
# Monitor bandwidth utilization
pcm-memory 1  # Intel PCM tool works on AMD too

# Check TLB misses
perf stat -e dTLB-load-misses,dTLB-store-misses ./llama-server

# Measure memory latency
mlc --loaded_latency  # Intel MLC works on AMD

# Application metrics
time ./benchmark.py  # Your existing benchmark
```

### Success Criteria
1. Bandwidth utilization increases from 9% to 12%
2. TLB miss rate drops below 0.1%
3. Token/second improves by 15-25%
4. First-token latency reduces by 10%

## Risk Mitigation

### For Each Experiment
1. **Test in isolation first** - Measure individual impact
2. **Use feature flags** - Enable/disable via environment variables
3. **Maintain fallbacks** - Always have working baseline
4. **Profile extensively** - Use perf to validate assumptions

## Conclusion

Your benchmarks clearly show memory bandwidth as the bottleneck. These 9 experiments specifically target that constraint with practical, implementable solutions. Unlike the overly-academic memory prefetcher, these are all achievable with reasonable effort.

**For FP32 models**: The combination of 1GB huge pages, layer-ahead prefetch, and CCX-aware allocation could realistically achieve another 20-25% improvement on top of your current 35 tokens/second, potentially reaching 42-44 tokens/second.

**For quantized models**: The addition of VNNI acceleration (Experiment 9) is a game-changer. By processing INT8/INT4 directly without dequantization, you dramatically reduce memory bandwidth requirements. Combined with the memory optimizations, quantized models could reach 90-130 tokens/second - a 2.5-4x improvement.

The VNNI wrapper is more complex to implement but offers the highest potential gains for quantized inference, making it worth the 2-3 week investment if you primarily use quantized models.

---

*Note: These optimizations are specifically designed to be implementable by a skilled developer without requiring deep academic research. Start with the quick wins in Phase 1, then decide whether to pursue VNNI based on your model quantization strategy.*

*Last Updated: 2025-09-23*