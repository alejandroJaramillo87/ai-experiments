# Inference parameters

The runtime parameters each engine is served with, and what each was measured to do on
this workstation. The compose file is the source of truth for the values in use; these
pages explain them.

<!-- inventory:pages:start -->

| Document | Covers |
|----------|--------|
| [llama-cpp-parameters.md](llama-cpp-parameters.md) | Detailed analysis of runtime parameters for llama.cpp server (CPU and GPU variants) optimized for low-latency single-request inference on the AMD Ryzen 9950X + RTX 5090 workstation. |
| [vllm-parameters.md](vllm-parameters.md) | Detailed analysis of runtime parameters for vLLM server optimized for high-throughput concurrent inference on the RTX 5090 with focus on maximizing request handling capacity. |

<!-- inventory:pages:end -->
