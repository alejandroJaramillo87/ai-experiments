# AI Experiments Documentation Hub

Central index for all project documentation: hardware setup, software
configuration, the inference stack, the training and interpretability tiers,
optimization strategies, and performance benchmarking for a local AI
workstation (RTX 5090, AMD Ryzen 9950X).

## Quick Navigation

- **[Back to Main README](../README.md)** - Project overview and quick start
- **[Hardware Setup](hardware/README.md)** - Component selection and assembly
- **[Getting Started](os/README.md)** - Ubuntu installation and initial setup

## Documentation by Category

### Infrastructure Setup

| Document | Description |
|----------|-------------|
| [hardware/README.md](hardware/README.md) | Hardware selection, compatibility, and assembly guide |
| [bios/README.md](bios/README.md) | BIOS configuration for Gigabyte X870E Aorus Elite WiFi |
| [os/README.md](os/README.md) | Ubuntu 24.04 LTS installation and configuration |
| [os/storage-configuration.md](os/storage-configuration.md) | Dual-SSD architecture and AI data storage setup |
| [os/backup_and_recovery.md](os/backup_and_recovery.md) | Data protection and recovery strategies |
| [sandbox/README.md](sandbox/README.md) | Docker sandboxed environment setup |
| [sandbox/docker_compose_overview.md](sandbox/docker_compose_overview.md) | Multi-service orchestration configuration |

### GPU Software Stack

| Document | Description | Version |
|----------|-------------|---------|
| [os/gpu-stack/README.md](os/gpu-stack/README.md) | Three-layer GPU stack overview | - |
| [os/gpu-stack/nvidia-drivers.md](os/gpu-stack/nvidia-drivers.md) | NVIDIA driver installation and configuration | 580.65.06 |
| [os/gpu-stack/cuda.md](os/gpu-stack/cuda.md) | CUDA parallel computing platform | 13.0.88 |
| [os/gpu-stack/cudnn.md](os/gpu-stack/cudnn.md) | cuDNN deep learning acceleration library | 9.13.0.50-1 |

### Python Environment

| Document | Description |
|----------|-------------|
| [os/python/uv-dependencies.md](os/python/uv-dependencies.md) | uv dependencies reference for all AI/ML packages |
| [os/python/setup_python.md](os/python/setup_python.md) | Python 3.12 installation via uv |
| [os/python/pytorch_cuda_update_guide.md](os/python/pytorch_cuda_update_guide.md) | PyTorch installation with CUDA 13.0 compatibility |

### Inference Optimization

| Document | Description |
|----------|-------------|
| [inference/README.md](inference/README.md) | Optimization philosophy and overview |
| [inference/engines/inference-engines.md](inference/engines/inference-engines.md) | Inference engines versus training frameworks |
| [inference/engines/llama-cpp.md](inference/engines/llama-cpp.md) | llama.cpp architecture and memory management |
| [inference/engines/vllm.md](inference/engines/vllm.md) | vLLM PagedAttention and continuous batching |
| [inference/parameters/llama-cpp-parameters.md](inference/parameters/llama-cpp-parameters.md) | llama.cpp parameter tuning guide |
| [inference/parameters/vllm-parameters.md](inference/parameters/vllm-parameters.md) | vLLM configuration for throughput (vLLM 0.25.0) |

### Rust Harness

Design and reference documentation for the `rust-harness` workspace.

| Document | Description |
|----------|-------------|
| [rust-harness/README.md](rust-harness/README.md) | Directory index |
| [rust-harness/runtime-design.md](rust-harness/runtime-design.md) | Harness and runtime architecture, and the implementation roadmap |
| [rust-harness/observability.md](rust-harness/observability.md) | Observability and debuggability as first-class code |
| [rust-harness/gemma-4-ingestion.md](rust-harness/gemma-4-ingestion.md) | What the runtime requires to ingest a Gemma 4 checkpoint, and what it refuses |

### Model References

Start with the document for the tier being worked on; each is scoped to what that tier's code
consumes. The family reference below is broader and is the citation target of the conformance
audit.

| Document | Description |
|----------|-------------|
| [rust-harness/gemma-4-ingestion.md](rust-harness/gemma-4-ingestion.md) | Gemma 4 for the inference runtime: GGUF contract, translation, refusals (E2B, E4B) |
| [training/gemma-4-training.md](training/gemma-4-training.md) | Gemma 4 for the training tier: parameter accounting, memory, tensor names, tokenizer (E2B, E4B) |
| [interpretability/gemma-4-interp.md](interpretability/gemma-4-interp.md) | Gemma 4 for the interp tier: residual width, layer counts, hook hazards (E2B, E4B) |
| [models/gemma-4.md](models/gemma-4.md) | Gemma 4 dense family (E2B, E4B, 12B, 31B): general architecture reference, and the citation target of `verification/gemma-4-conformance.md` |

### System Optimizations

| Document | Description | Impact |
|----------|-------------|--------|
| [optimizations/README.md](optimizations/README.md) | Optimization overview and verification | +26% CPU performance |
| [optimizations/bios/bios-optimizations.md](optimizations/bios/bios-optimizations.md) | BIOS settings for AMD Ryzen 9950X | FCLK 2100MHz |
| [optimizations/os/os-optimizations.md](optimizations/os/os-optimizations.md) | Operating system tuning | Performance governor |

### Container Services

| Document | Service |
|----------|---------|
| [sandbox/docker_llama_cpu_overview.md](sandbox/docker_llama_cpu_overview.md) | llama-cpu service (Port 8001) |
| [sandbox/docker_llama_gpu_overview.md](sandbox/docker_llama_gpu_overview.md) | llama-gpu service (Port 8002) |
| [sandbox/docker_vllm_gpu_overview.md](sandbox/docker_vllm_gpu_overview.md) | vllm-gpu service (Port 8003) |
| [sandbox/docker_training_overview.md](sandbox/docker_training_overview.md) | training container (Phase 0 D1, no port) |
| [sandbox/docker_compose_overview.md](sandbox/docker_compose_overview.md) | Service orchestration |

### Training Tier

Living documentation for the training stack (`python -m training`); see
[Design history](#design-history-frozen) for the per-deliverable
investigations.

| Document | Description |
|----------|-------------|
| [training/data_pipeline.md](training/data_pipeline.md) | Full-scale data pipeline (P2-D1): filter, dedup, decontaminate, shard |
| [training/tokenizer_training.md](training/tokenizer_training.md) | Byte-level BPE tokenizer training (D3) |
| [training/memory_fit.md](training/memory_fit.md) | VRAM fit calculator and calibration (D4) |
| [training/tracking_checkpointing.md](training/tracking_checkpointing.md) | Run tracking, checkpoint layout, resume (D6) |
| [training/train_serve_roundtrip.md](training/train_serve_roundtrip.md) | Export, serve, smoke-evaluate a checkpoint (D7) |
| [training/heterogeneous_compute.md](training/heterogeneous_compute.md) | Prefetch loader, staging, affinity (D9) |
| [training/operational_watchdogs.md](training/operational_watchdogs.md) | Thermal, RAM, and disk watchdogs (D10) |
| [training/sft_entry_point.md](training/sft_entry_point.md) | Supervised fine-tuning entry point (F1/F3) |
| [training/teacher_synthetic_data.md](training/teacher_synthetic_data.md) | Teacher generation harness (F4) |
| [training/distillation.md](training/distillation.md) | Response and logit distillation (F5) |
| [training/eval_wrapper.md](training/eval_wrapper.md) | Baseline-versus-tuned evaluation (F7) |
| [training/scorecards/README.md](training/scorecards/README.md) | Committed reference scorecards (F8) |
| [training/gemma-4-training.md](training/gemma-4-training.md) | Gemma 4 for the training tier: parameter accounting, memory, tensor names, tokenizer |

### Interpretability Tier

Living documentation for the interp stack (`python -m interp`).

| Document | Description |
|----------|-------------|
| [interpretability/platform.md](interpretability/platform.md) | Workbench, Neuronpedia harness, platform services |
| [interpretability/writeback.md](interpretability/writeback.md) | Steering, probes, log proxy, probe monitor (W1-W4) |
| [interpretability/experiments/README.md](interpretability/experiments/README.md) | The experiments log convention and entries |
| [interpretability/gemma-4-interp.md](interpretability/gemma-4-interp.md) | Gemma 4 for the interp tier: residual width, layer counts, hook hazards |

### Agents and Testing

| Document | Description |
|----------|-------------|
| [agents/README.md](agents/README.md) | Agent sandbox and tooling |
| [agents/tooling-parity.md](agents/tooling-parity.md) | OpenCode / Claude Code parity conventions |
| [report_schema.md](report_schema.md) | Shared JSON report envelope and validator |

### Hardware Verification

| Document | Description |
|----------|-------------|
| [verification/README.md](verification/README.md) | The manual verification campaign: protocol and phase order |
| [verification/step-zero-environment.md](verification/step-zero-environment.md) | Lockfile, images, smoke tests, quality gate (blocks all phases) |
| [verification/training-phase0.md](verification/training-phase0.md) | Phase 0 deliverables |
| [verification/training-phase1.md](verification/training-phase1.md) | F1 through F8 |
| [verification/training-phase2.md](verification/training-phase2.md) | P2-D1 through P2-D5 |
| [verification/interp.md](verification/interp.md) | The interpretability tier |

## Design history (frozen)

Per-deliverable investigation documents record what was decided and why at
implementation time. They are frozen: drift against the current repository
is expected and correct in them, and their bodies are never edited.

| Location | Contents |
|----------|----------|
| [training/investigations/](training/investigations/) | D3-D10, F1-F8, and P2 deliverable investigations (19 documents) |
| [interpretability/investigations/](interpretability/investigations/) | I1, P4, and PS deliverable investigations (8 documents) |
| [unified_cli_investigation.md](unified_cli_investigation.md) | The unified-CLI design investigation (July 2026 snapshot) |
| [testing/test_coverage_investigation.md](testing/test_coverage_investigation.md) | Test-coverage audit snapshot (2026-07-16); frozen point-in-time analysis |

## Performance Summary

### Benchmarking

Run the benchmark scripts in `scripts/benchmark/` (`benchmark_cpu.sh`,
`benchmark_gpu.sh`) to measure inference performance on this workstation.

### Key Optimizations Applied

- **GPU**: Batch 2048, Ubatch 512, Flash Attention v3
- **CPU**: Batch 2048, 16 threads, huge pages enabled
- **BIOS**: FCLK 2100MHz, C-states disabled, PBO enabled
- **OS**: Performance governor, swap disabled, CPU pinning

## Documentation Standards

This documentation follows Linux/Unix conventions; see
[.styles/DOCUMENTATION_STYLE.md](../.styles/DOCUMENTATION_STYLE.md) for the
full writing standard:

- **Plain text focus** - Clear, technical writing without decorative elements
- **No emojis in code** - Professional documentation uses standard ASCII
- **Command-line oriented** - Examples use shell commands and Unix tools
- **Minimalist approach** - Essential information only, no redundancy

## Contributing to Documentation

1. **Placement**:
   - Hardware guides: `docs/hardware/`
   - Software setup: `docs/os/`
   - Optimization guides: `docs/optimizations/`
   - Container docs: `docs/sandbox/`
   - Inference tuning: `docs/inference/`
   - Training tier: `docs/training/` (investigations are frozen; new design
     history goes to `docs/training/investigations/`)
   - Interpretability tier: `docs/interpretability/` (same split)
2. **Naming Convention**: lowercase with hyphens or underscores, descriptive
   but concise.
3. **Content Standards**: table of contents for long documents, concrete
   commands, actual measurements, cross-references to related documentation.

---

*Last Updated: July 2026 | [Main README](../README.md)*
