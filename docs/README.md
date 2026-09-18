# AI Experiments Documentation Hub

Central index for all project documentation: hardware setup, software
configuration, the inference stack, the training and interpretability tiers,
optimization strategies, and performance benchmarking for a local AI
workstation (RTX 5090, AMD Ryzen 9950X).

## Quick Navigation

- **[Back to Main README](../README.md)** - Project overview and quick start
- **[Hardware Setup](hardware/README.md)** - Component selection and assembly
- **[Getting Started](os/README.md)** - Ubuntu installation and initial setup

## Source directory orientation

Each top-level source directory carries a README covering its purpose, what it consumes and
what consumes it, its entry-point commands, and the mistakes newcomers make. They are
orientation, not explanation: they point here for the detail.

| Directory | Purpose |
|-----------|---------|
| [../common/README.md](../common/README.md) | Shared substrate: output, hashing, provenance, observability. A contract package checked across the Python/Rust boundary |
| [../training/README.md](../training/README.md) | The training tier, and the recipe home for both tiers |
| [../interp/README.md](../interp/README.md) | The interpretability tier: harvesting, SAEs, probes, steering, organisms |
| [../tests/README.md](../tests/README.md) | The pytest suite, its tiers, and the skip-count discipline |
| [../scripts/README.md](../scripts/README.md) | Host-side tooling, and which scripts run in a sandbox |
| [../docker/README.md](../docker/README.md) | Build contexts; the compose files above them are the source of truth |
| [../rust-harness/README.md](../rust-harness/README.md) | The Rust harness workspace, its test tiers, and its quality gate |
| [../hugepages/README.md](../hugepages/README.md) | The huge-page interceptor library |
| [../schemas/README.md](../schemas/README.md) | Cross-language artifact contracts |
| [../site/README.md](../site/README.md) | The user-facing documentation site for the rust harness, distinct from this knowledge base |

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
| [rust-harness/roadmap.md](rust-harness/roadmap.md) | The scheduled work in five phases, the interp feature specifications, and what each estimate turned out to be worth |
| [rust-harness/runtime-design.md](rust-harness/runtime-design.md) | Harness and runtime architecture, and the implementation roadmap |
| [rust-harness/observability.md](rust-harness/observability.md) | Observability and debuggability as first-class code |
| [rust-harness/observation-frames.md](rust-harness/observation-frames.md) | What the agent loop and chat mode contribute to the training and interp programs, and the acceptance bar for new features |
| [rust-harness/live-events.md](rust-harness/live-events.md) | Live run-state events as JSONL, with a serve mode and an HTML view that also replays finished runs |

### Model References

Start with the document for the tier being worked on; each is scoped to what that tier's code
consumes. The family reference below is broader and is the citation target of the conformance
audit.

| Document | Description |
|----------|-------------|
| [training/gemma-4-training.md](training/gemma-4-training.md) | Gemma 4 for the training tier: parameter accounting, memory, tensor names, tokenizer (E2B, E4B) |
| [interp/gemma-4-interp.md](interp/gemma-4-interp.md) | Gemma 4 for the interp tier: residual width, layer counts, hook hazards (E2B, E4B) |
| [models/gemma-4.md](models/gemma-4.md) | Gemma 4 dense family (E2B, E4B, 12B, 31B): general architecture reference, and the citation target of `verification/investigations/gemma-4-conformance_investigation.md` |
| [models/README.md](models/README.md) | Index for the above, plus the committed GGUF and safetensors header transcripts and how to regenerate them |

### System Optimizations

| Document | Description | Impact |
|----------|-------------|--------|
| [optimizations/README.md](optimizations/README.md) | Optimization overview and verification | +26% CPU performance |
| [optimizations/bios/bios-optimizations.md](optimizations/bios/bios-optimizations.md) | BIOS settings for AMD Ryzen 9950X | FCLK 2100MHz |
| [optimizations/os/os-optimizations.md](optimizations/os/os-optimizations.md) | Operating system tuning | Performance governor |
| [optimizations/bios/](optimizations/bios/) | `cpu_benchmark_results_{no,final}_bios.json` — the raw before and after measurements behind the figure above | Captured data |

Unscheduled optimization work is recorded separately; see
[Parked proposals](#parked-proposals).

### Container Services

| Document | Service |
|----------|---------|
| [sandbox/docker_llama_cpu_overview.md](sandbox/docker_llama_cpu_overview.md) | llama-cpu service (Port 8001) |
| [sandbox/docker_llama_gpu_overview.md](sandbox/docker_llama_gpu_overview.md) | llama-gpu service (Port 8002) |
| [sandbox/docker_vllm_gpu_overview.md](sandbox/docker_vllm_gpu_overview.md) | vllm-gpu service (Port 8003) |
| [sandbox/docker_training_overview.md](sandbox/docker_training_overview.md) | training container (Phase 0 D1, no port) |
| [sandbox/docker_compose_overview.md](sandbox/docker_compose_overview.md) | Service orchestration |

### Training Tier

Living documentation for the training stack (`python -m training`).
[training/README.md](training/README.md) indexes this tier and pairs each
living document with the design record it was implemented from; see
[Frozen records](#frozen-records) for those per-deliverable
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
| [training/training-scale-reference.md](training/training-scale-reference.md) | What fits on this workstation: model sizes, batch and sequence budgets, wall-clock expectations |

### Interpretability Tier

Living documentation for the interp stack (`python -m interp`).
[interp/README.md](interp/README.md) indexes this tier.

| Document | Description |
|----------|-------------|
| [interp/writeback.md](interp/writeback.md) | Steering, probes, log proxy, probe monitor (W1-W4) |
| [interp/experiments/README.md](interp/experiments/README.md) | The experiments log convention and entries |
| [interp/gemma-4-interp.md](interp/gemma-4-interp.md) | Gemma 4 for the interp tier: residual width, layer counts, hook hazards |
| [interp/interp-scale-reference.md](interp/interp-scale-reference.md) | What fits on this workstation for interp work: activation volumes, SAE sizes, harvest budgets |
| [interp/rust-harness-interp.md](interp/rust-harness-interp.md) | Running the write-back interpretability path on the Rust runtime |

### Numerics and Precision

| Document | Description |
|----------|-------------|
| [precision/](precision/README.md) | Numeric dtype policy across the three tiers, by concern: compute and weights, activation storage, gradients and optimizer state, KV cache, quantization, and the probe-precision hazard |
| [precision/dtype-decisions.md](precision/dtype-decisions.md) | Why each tier chose its dtype and what would change it: f32 in the Rust reference backend, the four accepted GGUF storage types, bf16 over fp16 for training, and whether the three tiers can share one format |

### Agents and Testing

| Document | Description |
|----------|-------------|
| [agents/README.md](agents/README.md) | Agent sandbox and tooling |
| [agents/tooling-parity.md](agents/tooling-parity.md) | OpenCode / Claude Code parity conventions |
| [testing/README.md](testing/README.md) | The testing tier index: the gate, and the coverage audit |
| [testing/quality-gate.md](testing/quality-gate.md) | What `make quality` runs, and what a sandbox run does not cover |
| [sandbox/agent-sandbox.md](sandbox/agent-sandbox.md) | The contained agent sandbox: image, mounts, and egress allowlist |

### Observability

| Document | Description |
|----------|-------------|
| [observability/README.md](observability/README.md) | Operating the stack: where the files are, the `OBS_` variables, the event and metrics schemas, finding a run, opening a dashboard, and example `jq` queries |
| [observability/design-and-roadmap.md](observability/design-and-roadmap.md) | Why the system is shaped this way, and the OBS item ledger with each item's status |
| [observability/report_schema.md](observability/report_schema.md) | The shared JSON report envelope (`schema_version`, `git_sha`) and its validator. Distinct from `schemas/obs_record.json`, which covers the run event and catalog records |

The Rust half of the same contract is documented at
[rust-harness/observability.md](rust-harness/observability.md), and the cross-language record
shapes both sides write live in [../schemas/README.md](../schemas/README.md).

## Document classes

Not every document here is maintained, and that is deliberate. A document's class decides
whether drift against the current repository is a bug to fix or a record to preserve.
`.styles/DOCUMENTATION_STYLE.md` defines the four classes and the line each one carries; the
sections above list the living reference documentation, and the three tables below list the
rest.

Two of the classes are enforced by path, because both documentation gates read them that way:
`scripts/docs/check_doc_refs.py` and `scripts/docs/check_retired_names.py` exempt anything
under an `investigations/` directory, anything whose name ends `_investigation.md`, and
everything under `docs/verification/`.

### Frozen records

What was decided and why at implementation time. Bodies are never edited; a reference that has
since moved is accurate history rather than drift.

**A frozen record is not a source to check the tree against, and an audit should not read it as
one.** These files describe the code as it stood on their date, so they accumulate names,
formats and commands that no longer exist — `docs/verification/investigations/` alone still
carries hundreds of references to the GGUF model loader deleted in P8b. That is correct and
must stay. Anything sweeping the tree for stale facts should skip these paths, as both
documentation gates already do, rather than rediscover the same expired history each time.

| Location | Contents |
|----------|----------|
| [training/investigations/](training/investigations/) | D3-D10, F1-F8, and P2 deliverable investigations, plus [the unified-CLI investigation](training/investigations/unified_cli_investigation.md) (20 documents, 2026-07-14 to 2026-07-15). [README.md](training/investigations/README.md) carries the directory's conventions |
| [interp/investigations/](interp/investigations/) | I1, P4, and PS deliverable investigations (8 documents, 2026-07-14 to 2026-07-15) |
| [testing/investigations/](testing/investigations/) | [test_coverage_investigation.md](testing/investigations/test_coverage_investigation.md) — audit of the suite at 2026-07-16, commit 8316e5e |
| [verification/investigations/](verification/investigations/) | [gemma-4-conformance_investigation.md](verification/investigations/gemma-4-conformance_investigation.md) — the 2026-08-01 audit of the three consumer tiers against the Gemma 4 reference. Its 50 `GH.n` host checks were never executed; [verification/README.md](verification/README.md) records that backlog |
| [interp/experiments/](interp/experiments/) | Experiment post-mortems, one file per experiment, immutable after the verdict. [README.md](interp/experiments/README.md) is the living convention for writing them |
| [models/](models/) | `gemma4-gguf.txt` and `gemma4-safetensors.txt` — machine transcripts carrying their own `Generated:` timestamps. Cited by line number; regenerating shifts those lines, and `tests/models/test_gemma4_facts_contract.py` fails on every citation the shift broke |

### Parked proposals

Recorded menus of work that is not scheduled. Dated at authoring, and edited only as items
graduate into the living documentation named in each.

| Document | Contents |
|----------|----------|
| [optimizations/ideation/os-optimization-candidates.md](optimizations/ideation/os-optimization-candidates.md) | Host OS and GPU candidates (2026-07-30); graduate into `optimizations/os/os-optimizations.md` |
| [optimizations/ideation/python-pipeline-optimization-candidates.md](optimizations/ideation/python-pipeline-optimization-candidates.md) | Training and interp pipeline candidates (2026-07-30) |
| [optimizations/ideation/rust-harness-optimization-candidates.md](optimizations/ideation/rust-harness-optimization-candidates.md) | x86 CPU backend candidates (2026-07-30); blocked on runbook entry RH.1 |
| [optimizations/ideation/future-inference-optimizations.md](optimizations/ideation/future-inference-optimizations.md) | Parked custom CUDA and C++ findings (2026-07-27); gates not met |

### Operational runbooks

Executed against the repository or the host rather than read for understanding. The audience
is whoever is running the checks, and the human's pasted output is the evidence.

| Document | Contents |
|----------|----------|
| [verification/README.md](verification/README.md) | The manual verification campaign: recording protocol and phase order |
| [verification/step-zero-environment.md](verification/step-zero-environment.md) | Lockfile, images, smoke tests, quality gate (blocks all phases) |
| [verification/training/training-phase0.md](verification/training/training-phase0.md) | Phase 0 deliverables |
| [verification/training/training-phase1.md](verification/training/training-phase1.md) | F1 through F8 |
| [verification/training/training-phase2.md](verification/training/training-phase2.md) | P2-D1 through P2-D5 |
| [verification/interp/interp.md](verification/interp/interp.md) | The interpretability tier |
| [verification/observability/observability.md](verification/observability/observability.md) | OBS-V1 to OBS-V8: the observability stack on the workstation |
| [verification/rust-harness/rust-harness.md](verification/rust-harness/rust-harness.md) | The rust-harness CPU backend on the real E4B GGUF |
| [verification/rust-harness/rust-harness-qa.md](verification/rust-harness/rust-harness-qa.md) | Q0-Q6: the sandbox-executable code-quality campaign |
| [verification/hardware/hardware-stress.md](verification/hardware/hardware-stress.md) | One-time firmware stability soak for the BIOS undervolt profile |
| [verification/bugs/known-bugs.md](verification/bugs/known-bugs.md) | The defect ledger the runbook entries cross-reference |
| [verification/bugs/ledger-archive.md](verification/bugs/ledger-archive.md) | Closed defect history moved out of the ledger, and the candidates the panel refused |

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

1. **Class first**: decide whether the document is living reference, a frozen
   record, an operational runbook, or a parked proposal, and give it the class
   line `.styles/DOCUMENTATION_STYLE.md` specifies. Living reference is the
   default and carries no marker.
2. **Placement**, by topic:
   - Hardware guides: `docs/hardware/`; BIOS: `docs/bios/`
   - Operating system, GPU stack, Python environment: `docs/os/`
   - Optimization guides: `docs/optimizations/` (unscheduled menus go to
     `docs/optimizations/ideation/`)
   - Container and sandbox docs: `docs/sandbox/`
   - Inference tuning: `docs/inference/`
   - Model architecture references and header transcripts: `docs/models/`
   - Rust harness design: `docs/rust-harness/`
   - Observability and the JSON contracts commands emit: `docs/observability/`
   - Agent tooling and parity: `docs/agents/`
   - The quality gate and testing policy: `docs/testing/`
   - Training tier: `docs/training/`
   - Interpretability tier: `docs/interp/`
   - Host and repository verification runbooks: `docs/verification/`
   - Cross-tier numeric precision policy: `docs/precision/`
3. **Then by class, within the topic**: a frozen record goes to
   `<topic>/investigations/` and its filename ends `_investigation.md`. Both
   documentation gates key on those two patterns, so a frozen record placed
   anywhere else is gated as living and will be asked to keep its historical
   paths resolving.
4. **Index it**: add a row to a section above. A document reachable from
   nowhere is a document nobody will read or maintain.
5. **Naming Convention**: lowercase with hyphens or underscores, descriptive
   but concise.
6. **Content Standards**: table of contents for long documents, concrete
   commands, actual measurements, cross-references to related documentation.

---

*Last updated: 2026-08-06 | [Main README](../README.md)*
