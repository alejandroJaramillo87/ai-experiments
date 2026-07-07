# AOCL and llama.cpp: Accelerating Prefill on the Ryzen 9950X

Experiment evaluating AMD Optimizing CPU Libraries (AOCL) as an acceleration
path for llama.cpp CPU inference on the Ryzen 9950X (Zen 5), with focus on
the prefill phase. Coding harnesses (agents, editors, RAG pipelines) send
long prompts and read short completions, so prefill throughput dominates
end-to-end latency for these workloads.

## Table of Contents

- [Summary](#summary)
- [Why Prefill Is the Target](#why-prefill-is-the-target)
- [AOCL Library Survey](#aocl-library-survey)
- [The Classic BLAS Route Is a Trap for Quantized Models](#the-classic-blas-route-is-a-trap-for-quantized-models)
- [The Viable Path: ZenDNN Backend Backed by AOCL-DLP](#the-viable-path-zendnn-backend-backed-by-aocl-dlp)
- [AOCL-LibMem: A Second Preload Wrapper](#aocl-libmem-a-second-preload-wrapper)
- [Interaction with Existing Optimizations](#interaction-with-existing-optimizations)
- [Implementation in This Repository](#implementation-in-this-repository)
- [Benchmark Plan](#benchmark-plan)
- [Expected Outcomes and Risks](#expected-outcomes-and-risks)
- [References](#references)

## Summary

- AOCL is a suite of 13+ libraries. Only three matter for llama.cpp
  inference: AOCL-BLAS (BLIS), AOCL-DLP (low-precision GEMM), and
  AOCL-LibMem (memcpy/memset replacement).
- Linking llama.cpp against AOCL-BLIS through the generic ggml BLAS
  backend (`GGML_BLAS=ON -DGGML_BLAS_VENDOR=FLAME`) is counterproductive
  for quantized models: ggml dequantizes weights to F32 before every
  `cblas_sgemm` call, discarding the AVX-512 VNNI advantage the native
  kernels already exploit. Community measurements show order-of-magnitude
  prompt-processing regressions from this path.
- The productive route is the ZenDNN backend merged into llama.cpp master
  in December 2025 (`-DGGML_ZENDNN=ON`). It routes `MUL_MAT` and
  `MUL_MAT_ID` through ZenDNN's LowOHA matmul operator, which dispatches
  to AOCL-DLP low-precision GEMM kernels (AVX512_VNNI, AVX512_BF16).
  Supported types: F32, BF16, and Q8_0 via dynamic quantization. AMD
  reports 1.5-2.95x prompt-processing gains on Zen 4/5 EPYC; Q8_0
  benefits prefill more than generation.
- AOCL-LibMem is an LD_PRELOAD drop-in for memcpy/memset/memmove,
  the same interposition mechanism as our zen5_optimizer hugepage
  wrapper. The two preloads compose (different symbols: string functions
  vs mmap).
- This experiment adds `docker/llama-cpu/Dockerfile.cpu-aocl`, an
  experimental image combining the ZenDNN backend, the zen5_optimizer
  hugepage preload, and an optional AOCL-LibMem preload, exposed as
  compose service `llama-cpu-aocl` on port 8006.

## Why Prefill Is the Target

The two inference phases stress different hardware limits.

| Phase | Operation shape | Bound by | 9950X ceiling |
|-------|----------------|----------|---------------|
| Prefill (prompt processing) | GEMM: weights x hundreds of tokens | Compute (FLOPs) | ~5 TFLOPS F32; 2x with BF16; 4x ops with INT8 VNNI |
| Decode (token generation) | GEMV: weights x 1 token | Memory bandwidth | ~89.6 GB/s theoretical (DDR5-6000, dual channel) |

Zen 5 executes two full-width 512-bit FMA operations per cycle per core.
At ~5 GHz across 16 cores that is roughly 5 TFLOPS F32 peak. The
`vdpbf16ps` (AVX512_BF16) and `vpdpbusd` (AVX512_VNNI) instructions raise
per-cycle arithmetic throughput 2x and 4x respectively over F32 FMA.
Any prefill acceleration strategy on this CPU therefore reduces to one
question: how close does the GEMM implementation get to the VNNI/BF16
instruction ceiling?

Decode cannot benefit from better GEMM kernels. Streaming an N-GB model
through 89.6 GB/s of bandwidth bounds decode at roughly `89.6 / N` tokens
per second regardless of compute. This is why BLAS-style libraries state
they do not affect generation performance, and why our existing
optimizations (huge pages, FCLK, cache residency) target the memory
subsystem instead.

Coding harnesses invert the usual chat ratio: a typical agent turn sends
5k-50k prompt tokens (system prompt, file contents, tool results) and
generates a few hundred. Time-to-first-token is prefill time. Prefill
optimization is therefore the highest-leverage remaining work on the CPU
service.

## AOCL Library Survey

AOCL 5.1 (current on amd.com/en/developer/aocl.html) with library
releases now at 5.3 (May 2026). Relevance to llama.cpp inference:

| Library | Function | Relevance |
|---------|----------|-----------|
| AOCL-BLAS (BLIS) | Dense BLAS, Zen-tuned sgemm/dgemm; `aocl_gemm` LPGEMM addon | Indirect. Classic cblas path harmful for quantized models (see below). LPGEMM addon superseded by AOCL-DLP. |
| AOCL-DLP | Low-precision GEMM: FP32/FP16/BF16/INT8/INT4, post-ops (bias, GELU, SiLU), batch GEMM. AVX512_VNNI + AVX512_BF16 kernels | High. This is the engine under the ZenDNN llama.cpp backend. |
| AOCL-LibMem | Zen-optimized memcpy/mempcpy/memmove/memset/memcmp + string functions, LD_PRELOAD drop-in | Moderate. Prefill copies large activation/KV buffers; same preload mechanism as zen5_optimizer. |
| AOCL-LibM | Scalar/vector libm (expf, tanhf, powf) | Low. ggml ships its own vectorized AVX-512 exp/tanh approximations in softmax/activation hot paths; libm calls are rare. |
| AOCL-LAPACK (libFLAME) | Dense factorizations (LU, QR, SVD) | None. Inference performs no factorizations. |
| AOCL-ScaLAPACK | Distributed-memory LAPACK (MPI) | None. Single workstation. |
| AOCL-FFTW | Zen-tuned FFT | None. Transformers contain no FFTs. |
| AOCL-Sparse | Sparse BLAS (CSR/ELL SpMV, SpGEMM) | None today. GGUF weights are dense; revisit only if pruning/sparsity experiments land in llama-zen-turbo. |
| AOCL-RNG / SecureRNG | Random number generation | None. Sampling cost is negligible. |
| AOCL-Cryptography | AES/SHA/RSA | None. |
| AOCL-Compression | LZ4/Snappy/zlib etc. | None for inference. |
| AOCL-DA | Data analytics primitives | None. |
| AOCL-Utils | CPU feature detection helper | Incidental (dependency of others). |

The user-facing conclusion from the survey: the matrix-multiplication
libraries are indeed the ones that dominate prefill, but the *precision*
of the GEMM matters more than the vendor tuning. AMD's own stack
acknowledges this by splitting low-precision GEMM out of BLIS
(`aocl_gemm` addon, AOCL-BLAS 4.0+) into a dedicated library (AOCL-DLP)
and wiring it into llama.cpp through ZenDNN rather than through the BLAS
interface.

## The Classic BLAS Route Is a Trap for Quantized Models

ggml's BLAS backend (`ggml/src/ggml-blas/ggml-blas.cpp`) supports BLIS
directly (`GGML_BLAS_VENDOR=FLAME`). Its dispatch logic explains why it
fails for our workload:

1. It claims `MUL_MAT` only when all matrix dimensions are >= 32
   (`min_batch = 32`), i.e. exactly the prefill phase. Decode
   (batch of 1) never reaches BLAS. This matches the recollection that
   these libraries are matrix-times-matrix engines for prefill.
2. BLAS implementations expose only floating-point GEMM. For any
   quantized weight tensor, ggml first dequantizes the entire tensor to
   F32 in a scratch buffer (`to_float`), then calls `cblas_sgemm`, every
   single time the op executes.

The cost is twofold. The F32 conversion inflates a Q8_0 weight matrix 4x
(Q4_K_M ~8x) in the working set, evicting everything useful from L3.
And the GEMM itself then runs at F32 FMA rates, capped at ~5 TFLOPS,
while llama.cpp's native AVX-512 kernels compute directly on int8 blocks
using `vpdpbusd` at up to 4x that arithmetic throughput with no
conversion pass. BLIS being an excellent sgemm does not overcome
starting the race 4-8x behind.

Field evidence agrees. A packaging change that silently enabled OpenBLAS
produced a ~12x prompt-processing regression on a Q4_K_M model
(llama.cpp issue tracker, 2026), and the llamafile/tinyBLAS work that
became ggml's native `sgemm.cpp` was demonstrated beating MKL-class
libraries for exactly these quantized shapes. The same applies to
AOCL-BLIS through this interface: wrong abstraction, not wrong library.

Verdict: do not ship `GGML_BLAS=ON` with BLIS for the quantized CPU
service. It is only defensible for pure F32/F16 GGUF models, which we do
not run on CPU for capacity reasons.

## The Viable Path: ZenDNN Backend Backed by AOCL-DLP

AMD contributed a ZenDNN backend to llama.cpp (RFC discussion #17684,
PR #17690, merged to master 2025-12-06, extended since). Architecture:

```
llama.cpp ggml graph
  -> ggml-zendnn backend (claims MUL_MAT, MUL_MAT_ID; all else falls back to CPU backend)
    -> ZenDNN LowOHA ("low overhead") matmul operator, weight caching
      -> AOCL-DLP LPGEMM kernels (AVX512_VNNI, AVX512_BF16) | LibXSMM | oneDNN
```

Key properties:

- Build: `-DGGML_ZENDNN=ON`. The ggml CMake auto-clones a pinned ZenDNN
  release and its dependencies (AOCL-DLP, oneDNN, LibXSMM, aocl-utils)
  at configure time; no system install required.
- Data types: F32, BF16 (fastest on Zen 4/5), and Q8_0 through dynamic
  quantization. AMD's documentation states Q8_0 benefits prefill more
  than token generation - consistent with the compute-bound/
  bandwidth-bound split above.
- K-quants (Q4_K_M etc.) are not claimed by the backend; those matmuls
  fall back to the native CPU kernels unchanged.
- Runtime tuning: `ZENDNNL_MATMUL_ALGO=1` selects the blocked AOCL-DLP
  algorithm (AMD-recommended).
- Reported results: 1.5-2.95x on EPYC 9004 (Zen 4, 96 threads) across
  Llama/Qwen dense models; a Zen 5 Ryzen AI Max community report showed
  large-batch prefill gains. Dense models benefit more than MoE.

Implications for model selection on the CPU service: to exercise this
path we need Q8_0 or BF16 GGUF variants of the CPU model. Q8_0 of an 8B
model is ~8.5 GB - comfortable in the 96 GB container budget, and Q8_0
quality is near-lossless. BF16 doubles that but engages the fastest
kernels; both belong in the benchmark matrix.

Honest expectation-setting: AMD's published gains come from high-core-
count EPYC parts where llama.cpp's native threading scales poorly and
ZenDNN's blocking shines. A 16-core Ryzen with the native repack path
enabled is a stronger baseline. Gains here may land well under 1.5x, or
even negative versus repack. That is precisely what the benchmark must
establish - the same methodology that validated huge pages at 26%.

## AOCL-LibMem: A Second Preload Wrapper

AOCL-LibMem replaces 17 glibc data-movement and string functions
(memcpy, mempcpy, memmove, memset, memcmp, strcpy, strcmp, strlen, ...)
with Zen-tuned implementations, loaded exactly like our hugepage
wrapper:

```bash
LD_PRELOAD=/path/to/libaocl-libmem.so ./llama-server ...
```

Inference calls memcpy on real data: KV-cache writes, `ggml_cpy` /
`ggml_dup` graph ops, dequantization scratch, server-side prompt
buffers. glibc's memcpy is already well tuned for Zen (`__memmove_avx_unaligned_erms`),
so treat this as a marginal, free-to-test experiment: single-digit
percent at best, zero code changes, composable with zen5_optimizer since
the two preloads interpose disjoint symbols:

```bash
LD_PRELOAD="/app/libzen5_optimizer.so:/app/libaocl-libmem.so"
```

Build note: LibMem must be built with dynamic dispatch
(`ALMEM_DYN_DISPATCH=Y`) or pinned to an AVX-512 target; an AVX-512
build crashes on non-AVX-512 machines, which is irrelevant for this
pinned-hardware repo but matters if the image is ever rebuilt elsewhere.

## Interaction with Existing Optimizations

The pieces of the current stack interact with an AOCL path in specific,
testable ways:

**zen5_optimizer hugepage wrapper.** The wrapper intercepts `mmap()` for
allocations >= 1 GB, which covers the GGUF weight mapping. ZenDNN's
weight caching reorders weight blocks into its own heap buffers on first
use; those go through malloc/aligned_alloc, not mmap, so they receive no
explicit huge pages from our wrapper (only kernel THP, if eligible).
Two consequences: peak RSS rises by roughly the size of the reordered
weights, and a future zen5_optimizer extension intercepting large
aligned allocations (the LibMem-style interposition applied to
`posix_memalign`) becomes worth prototyping if ZenDNN wins the
benchmark.

**`--no-repack`.** The production entrypoint passes `--no-repack`,
disabling llama.cpp's native runtime weight repacking (block-interleaved
layouts feeding AVX-512 GEMM microkernels). Repack is the native
engine's own prefill optimization and the most direct competitor to
ZenDNN in this experiment; it also copies weights out of the mmap
region, which bypasses the hugepage wrapper and increases RSS - likely
why it is off in production. The benchmark matrix must include
native-with-repack as a first-class arm, otherwise we risk crediting
ZenDNN for gains the native engine already had available.

**mlock and the 96 GB limit.** ZenDNN weight caching and BLAS-style
scratch buffers add resident memory on top of the mlocked model. Q8_0 8B
(~8.5 GB weights, ~8.5 GB cached reorder worst case) fits trivially;
watch this before trying 30B-class models.

**Core pinning and threading.** ZenDNN parallelizes with OpenMP and
inherits `OMP_NUM_THREADS` from the entrypoint. Keep cpuset 0-15 and
`OMP_PROC_BIND=spread` / `OMP_PLACES=cores` identical across arms so the
comparison isolates the GEMM engine.

## Implementation in This Repository

Added by this experiment:

```
docker/llama-cpu/Dockerfile.cpu-aocl       # llama.cpp + GGML_ZENDNN + zen5_optimizer + AOCL-LibMem
docker/llama-cpu/entrypoint.cpu-aocl.sh    # ZENDNNL_MATMUL_ALGO=1, preload chaining, repack toggle
docker-compose.yaml                        # llama-cpu-aocl service, port 8006, profile "experimental"
```

The service is profile-gated so `make up` is unaffected:

```bash
# Build and start the experimental service
docker compose --profile experimental build llama-cpu-aocl
docker compose --profile experimental up -d llama-cpu-aocl

# Point it at a Q8_0 model to exercise the ZenDNN path
LLAMA_AOCL_MODEL=/app/models/<model>-Q8_0.gguf \
  docker compose --profile experimental up -d llama-cpu-aocl
```

Entrypoint toggles (environment variables):

| Variable | Default | Effect |
|----------|---------|--------|
| `ENABLE_LIBMEM` | `false` | Adds libaocl-libmem.so to LD_PRELOAD |
| `NO_REPACK` | `true` | Passes `--no-repack` (set `false` to test native repack) |

## Benchmark Plan

Prefill-weighted comparison using the existing harness. All arms: same
model file per format, cpuset 0-15, 16 threads, ctx 32768, ubatch 512.

| Arm | Image | Model | Flags |
|-----|-------|-------|-------|
| A baseline | Dockerfile.cpu | Q8_0 | `--no-repack` (production config) |
| B native repack | Dockerfile.cpu | Q8_0 | repack enabled |
| C zendnn | Dockerfile.cpu-aocl | Q8_0 | `--no-repack` |
| D zendnn bf16 | Dockerfile.cpu-aocl | BF16 | `--no-repack` |
| E zendnn + libmem | Dockerfile.cpu-aocl | winner of C/D | `ENABLE_LIBMEM=true` |

Commands:

```bash
# Server-level benchmark (existing harness; prefill-heavy prompt source)
./scripts/benchmark/benchmark_cpu.sh -c llama-cpu-aocl -n aocl-zendnn-q8 -l detailed

# Kernel-level prefill measurement inside the container
docker exec llama-cpu-aocl /app/llama-bench \
  -m /app/models/<model>-Q8_0.gguf -t 16 -p 512,2048,8192 -n 32
```

Report `pp512`, `pp2048`, `pp8192` (tokens/s) and `tg32` per arm, plus
time-to-first-token at a realistic 8k-token coding-harness prompt.
Decision rule: adopt the ZenDNN image for the CPU service only if the
best AOCL arm beats the best native arm (A or B) by >= 10% on pp2048
without regressing tg by more than 5%.

## Expected Outcomes and Risks

- Most likely: arm B (native repack) and arm C (ZenDNN Q8_0) land close;
  the interesting result is either one clearly winning at the 8k prompt
  length that matters for coding harnesses.
- BF16 (arm D) is the wildcard: highest kernel throughput, but 2x weight
  traffic; on a 16-core part the compute/bandwidth crossover may sit
  right at this operating point.
- ZenDNN's pinned-release auto-download makes the image build heavier
  (four extra dependency builds) and pins us to AMD's release cadence.
- The backend is EPYC-first; Ryzen is officially supported by the
  underlying AOCL-DLP kernels (AVX512_VNNI/BF16 are identical silicon
  features on Zen 5 client) but less benchmarked upstream.
- If AOCL loses across the board, the negative result is still blog
  material and cleanly retires the "should we link AOCL?" question with
  data; the experimental service is profile-gated and removable without
  touching production.

## References

- AMD AOCL portal: https://www.amd.com/en/developer/aocl.html
- What's new in AOCL 5.1: https://www.amd.com/en/developer/resources/technical-articles/2025/what-s-new-in-aocl-5-1--faster--and-more-scalable-math-libraries.html
- AOCL-BLIS: https://github.com/amd/blis
- AOCL-DLP (low-precision GEMM): https://github.com/amd/aocl-dlp and https://amd.github.io/aocl-dlp/
- AOCL-LibMem: https://github.com/amd/aocl-libmem
- LPGEMM addon design (BLIS Retreat, AMD): https://www.cs.utexas.edu/~flame/BLISRetreat2024/slides/Bhaskar_BLIS_Retreat_2024_AMD_LPGEMM_0.pdf
- ZenDNN backend RFC: https://github.com/ggml-org/llama.cpp/discussions/17684
- ZenDNN backend PR (merged 2025-12-06): https://github.com/ggml-org/llama.cpp/pull/17690
- ZenDNN backend docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/ZenDNN.md
- ggml BLAS backend dispatch: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-blas/ggml-blas.cpp
- llamafile GEMM background: https://justine.lol/matmul/
