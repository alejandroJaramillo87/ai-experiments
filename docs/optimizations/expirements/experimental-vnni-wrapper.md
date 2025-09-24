# EXPERIMENTAL: AVX-512 VNNI Acceleration Wrapper for llama.cpp

**STATUS: EXPERIMENTAL - Research and Development Phase**

**WARNING**: This is an experimental optimization that requires significant engineering effort and may not provide expected benefits for all models. Implementation complexity is high and requires deep understanding of both llama.cpp internals and x86 SIMD programming.

## Executive Summary

A novel LD_PRELOAD wrapper to accelerate llama.cpp's quantized inference using AMD Zen 5's AVX-512 VNNI (Vector Neural Network Instructions). This wrapper intercepts BLAS operations and rewrites them to use native INT8 computation, potentially achieving 2-4x speedup for quantized models.

**Expected Impact**: 2-4 times speedup for INT8/INT4 quantized models
**Engineering Effort**: 2-3 weeks
**Risk Level**: High
**Innovation Level**: First implementation for llama.cpp on Zen 5

## Technical Background

### The Quantization-Computation Gap

Current llama.cpp flow for quantized models:
```
INT8 weights → Dequantize to FP32 → FP32 GEMM → Result
```

Optimal flow with VNNI:
```
INT8 weights → Native INT8 GEMM with VNNI → Scale result
```

### Why VNNI Matters

AVX-512 VNNI provides instructions like `_mm512_dpbusd_epi32` that perform:
- 64 parallel INT8×INT8 multiplications
- Accumulation into INT32
- Single cycle execution

Current approach requires:
- Dequantization (3-4 cycles)
- FP32 multiplication (4 cycles)  
- FP32 accumulation (3 cycles)

Theoretical speedup: 10-12 cycles to 2-3 cycles

## Architecture Design

### Interception Strategy

```cpp
// vnni_wrapper.cpp - LD_PRELOAD library

extern "C" {
    // Intercept BLAS calls
    void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                     CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const int lda, const float *B, const int ldb,
                     const float beta, float *C, const int ldc) {
        
        // Check if inputs are from quantized data
        if (is_quantized_buffer(A) && is_quantized_buffer(B)) {
            // Route to VNNI implementation
            vnni_quantized_gemm(layout, TransA, TransB, M, N, K,
                               alpha, A, lda, B, ldb, beta, C, ldc);
            return;
        }
        
        // Fallback to original BLAS
        original_cblas_sgemm(layout, TransA, TransB, M, N, K,
                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
}
```

### VNNI Kernel Implementation

```cpp
void vnni_int8_gemm_kernel(const int8_t* A, const int8_t* B, 
                           int32_t* C, int M, int N, int K) {
    // Process 64 elements at a time (AVX-512 width)
    for (int m = 0; m < M; m += 4) {
        for (int n = 0; n < N; n += 16) {
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();
            
            for (int k = 0; k < K; k += 4) {
                // Load 4x4 tile from A
                __m128i a_tile = _mm_loadu_si128((__m128i*)&A[m*K + k]);
                
                // Load 4x16 tile from B  
                __m512i b0 = _mm512_loadu_si512(&B[(k+0)*N + n]);
                __m512i b1 = _mm512_loadu_si512(&B[(k+1)*N + n]);
                __m512i b2 = _mm512_loadu_si512(&B[(k+2)*N + n]);
                __m512i b3 = _mm512_loadu_si512(&B[(k+3)*N + n]);
                
                // Broadcast and multiply-accumulate
                acc0 = _mm512_dpbusd_epi32(acc0, 
                         _mm512_set1_epi32(((int32_t*)&a_tile)[0]), b0);
                acc1 = _mm512_dpbusd_epi32(acc1,
                         _mm512_set1_epi32(((int32_t*)&a_tile)[1]), b1);
                acc2 = _mm512_dpbusd_epi32(acc2,
                         _mm512_set1_epi32(((int32_t*)&a_tile)[2]), b2);
                acc3 = _mm512_dpbusd_epi32(acc3,
                         _mm512_set1_epi32(((int32_t*)&a_tile)[3]), b3);
            }
            
            // Store results
            _mm512_storeu_si512(&C[(m+0)*N + n], acc0);
            _mm512_storeu_si512(&C[(m+1)*N + n], acc1);
            _mm512_storeu_si512(&C[(m+2)*N + n], acc2);
            _mm512_storeu_si512(&C[(m+3)*N + n], acc3);
        }
    }
}
```

## Implementation Roadmap

### Phase 1: Proof of Concept (Week 1)
1. Create basic LD_PRELOAD wrapper
2. Intercept cblas_sgemm calls
3. Implement simple VNNI kernel for aligned sizes
4. Test with synthetic benchmarks

### Phase 2: Quantization Support (Week 2)
1. Detect llama.cpp quantized buffers
2. Implement dequantization for Q4_0, Q8_0 formats
3. Handle scale factors correctly
4. Validate numerical accuracy

### Phase 3: Production Hardening (Week 3)
1. Handle all matrix sizes (edge cases)
2. Implement fallback paths
3. Add runtime CPU detection
4. Performance profiling and optimization

## Integration with Existing Setup

### Dockerfile Modifications

```dockerfile
# Add to builder stage
COPY docker/vnni_wrapper.cpp /tmp/
RUN g++-14 -shared -fPIC -O3 -mavx512vnni -Wall \
    -o /tmp/vnni_wrapper.so /tmp/vnni_wrapper.cpp -ldl

# Copy to runtime
COPY --from=builder /tmp/vnni_wrapper.so /app/
```

### Entrypoint Modifications

```bash
# In entrypoint.sh
if [[ "$ENABLE_VNNI" == "true" ]]; then
    export LD_PRELOAD="${LD_PRELOAD:+$LD_PRELOAD:}/app/vnni_wrapper.so"
    echo "VNNI acceleration enabled (EXPERIMENTAL)"
fi
```

## Performance Measurement

### Benchmarking Methodology

```bash
# Baseline (without VNNI)
time ./server --model q8_0_model.gguf --prompt "..." --n-predict 100

# With VNNI wrapper
ENABLE_VNNI=true time ./server --model q8_0_model.gguf --prompt "..." --n-predict 100

# Measure:
# - Tokens per second
# - First token latency
# - Memory bandwidth utilization
```

### Expected Results by Quantization Type

| Quantization | Current (tok/s) | With VNNI | Speedup |
|--------------|-----------------|-----------|---------|
| Q8_0         | ~10             | ~30-40    | 3-4x    |
| Q4_0         | ~15             | ~35-45    | 2.5-3x  |
| Q4_K_M       | ~12             | ~25-30    | 2-2.5x  |
| F16          | ~8              | ~8        | 1x (no benefit) |

## Risk Assessment

### Technical Risks

1. **Quantization Format Complexity**
   - llama.cpp uses custom formats (Q4_K, Q5_K, etc.)
   - May need format-specific kernels
   - Risk: 30% of formats may not benefit

2. **Numerical Accuracy**
   - INT8 accumulation may overflow
   - Scaling factors must be precise
   - Mitigation: Extensive validation suite

3. **Integration Complexity**
   - llama.cpp internals may change
   - BLAS calls might be inlined
   - Mitigation: Multiple interception points

### Performance Risks

1. **Memory Bandwidth Bound**
   - If already bandwidth limited, compute improvements won't help
   - Mitigation: Combine with huge pages optimization

2. **Overhead from Detection**
   - Checking if buffers are quantized adds overhead
   - Mitigation: Cache detection results

## Alternative Approaches Considered

### Direct llama.cpp Modification
- Pros: Cleaner integration
- Cons: Maintenance burden, upstream unlikely to accept

### Custom BLAS Library
- Pros: Full control
- Cons: Massive effort, compatibility issues

### Using Intel MKL
- Pros: Well-tested
- Cons: Not optimized for llama.cpp formats, licensing

## Success Criteria

1. **Functional**: Bit-exact results with original implementation
2. **Performance**: Minimum 2 times speedup for Q8_0 models
3. **Stability**: No crashes or memory leaks after 24 hour testing
4. **Compatibility**: Works with all common quantization formats

## Open Questions

1. How does llama.cpp's new backend system affect interception?
2. Can we handle dynamic quantization (different scales per layer)?
3. Should we upstream this to llama.cpp eventually?

## References

- [Intel AVX-512 VNNI Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=vnni)
- [AMD Zen 5 Optimization Guide](https://developer.amd.com/resources/developer-guides-manuals/)
- [llama.cpp Quantization Formats](https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c)

## Status Tracking

- [ ] Proof of concept
- [ ] Q8_0 support
- [ ] Q4_0 support  
- [ ] Performance validation
- [ ] Production ready

---

**Note**: This is experimental research. Results may vary significantly from projections. The complexity of implementation may reveal unforeseen challenges.

*Last Updated: 2025-09-23*