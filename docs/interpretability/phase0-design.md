# Interpretability Phase 0 product design: workbench

This document is the implementation-ready product design for the infrastructure deliverable of "Phase 0: Workbench and fundamentals" in the interpretability roadmap (docs/interpretability/roadmap.md). It follows the same contract as the training-tier phase designs: it can be picked up in a fresh coding-agent session and implemented without re-deriving context. It records the decisions already made, the interfaces to build, the repository and ecosystem facts the design depends on, and the tests that define done.

The roadmap defines what Phase 0 delivers; this document defines how, for the buildable part. Where the two disagree, this document wins and the roadmap should be corrected.

## Scope of this document

Phase 0 in the roadmap has four deliverables. Only the first is coding-agent work; the other three are the owner's learning work by design (the roadmap's rule 1: research pulls infrastructure, and its learning pillar depends on the owner implementing techniques by hand). This document therefore specifies:

- **In scope: I1, the workbench image and compose profile** (roadmap deliverable 1, tagged build), plus the smoke test, storage bootstrap, and Make targets that make it verifiable.

Out of scope, deliberately:

- ARENA curriculum chapters, the core-technique checklist notebooks, and the steering exercise (roadmap deliverables 2-4). The workbench's acceptance criteria below prove these *can* run (imports, hooked forward passes, GPU-backed notebooks); actually running them is owner work tracked in the experiments log, not in this design.
- Everything the roadmap assigns to later phases: Neuronpedia and the platform stack (Phase 2), repeng and the steering deployment path (Phase 3), SAE training, SAEBench, and diffing-toolkit runs (Phase 4). SAELens is installed in the image now because ARENA chapter 1.3.2 uses a pretrained SAE, but no SAE is trained in Phase 0.
- The llama.cpp activation read path (exploratory, explicitly deferred by the roadmap).

## Cross-cutting conventions

Facts and decisions the deliverable inherits. Verified against the repository and the ecosystem as of 2026-07; re-verify version pins at implementation time.

**Repository facts.**

- Inference services and their real ports come from docker-compose.yaml: llama-cpu on 8001, llama-gpu on 8002, vllm-gpu on 8003, open-webui on 3000, comfyui-flux on 8188. The compose file is the source of truth; documentation claiming other ports is stale.
- Existing GPU images build from nvidia/cuda:13.0.2-\*-ubuntu24.04 with a non-root appuser, cap_drop: ALL, no-new-privileges, and GPU device reservations in compose. The workbench follows this pattern with one documented deviation (writable filesystem, below).
- No training tier exists in the repository yet. This deliverable depends on nothing from the training designs; it must build and run on the repository as it stands today. Where the training Phase 0 design (D1) later lands a container, the two images stay separate: the interp stack (TransformerLens/nnsight/SAELens) and the training stack (TRL/DeepSpeed/flash-attn) have different and mutually fragile dependency sets. Shared conventions, separate images.
- Heavy artifacts never live in git. The interp storage root is /mnt/ai-data/interp/ per the roadmap's storage layout.
- Style: plain technical prose, no emojis, black and ruff conventions for any Python. Note: the .claude/ style-guide files referenced across the planning documents do not currently exist in this repository; follow the conventions as stated here until they land.

**Policy decisions (settled, do not re-litigate).**

- Interpretability runs at bf16. Quantized models are objects of study, never the substrate.
- Concurrency: interactive workbench sessions require the GPU inference services (llama-gpu, vllm-gpu, comfyui-flux) to be down. llama-cpu may stay up on its dedicated cores. This is enforced by a preflight, not by convention alone.
- The interp compose profile is off by default: make up never starts it.
- PyTorch for sm_120 is pinned first and nothing may downgrade it. ARENA and interp libraries ship requirements files that pin older torch versions; the image install order and a pip constraints file make a torch downgrade a build failure, not a silent regression.

**Ecosystem facts (verified 2026-07, re-check pins when implementing).**

- torch cu130 pip wheels support sm_120 (Blackwell); the repository already standardized on cu130 elsewhere.
- TransformerLens v3 (TransformerBridge) covers roughly 9,000 HF models, including GPT-2 Small and Gemma-2-2B, the Phase 0 reference models.
- nnsight 0.6 plus nnterp cover hooked execution on plain HF models.
- SAELens v6 no longer requires TransformerLens; Gemma Scope SAEs for Gemma-2-2B download from Hugging Face (google/gemma-scope).
- Gemma-2-2B weights are gated on Hugging Face behind license acceptance; an HF token is required to download them.
- ARENA 3.0 (callummcdougall/ARENA_3.0) is the curriculum source; its exercises assume a CUDA GPU and a Jupyter-capable environment.

## I1: Workbench image and compose profile

**Tag:** build. **Depends on:** nothing.

### Purpose

One reproducible environment where all Phase 0 learning work and all later interpretability project work runs: PyTorch cu130 on sm_120 with the hooked-execution stack imported and CUDA-verified at build time, JupyterLab served locally, model weights cached on NVMe, and the concurrency policy enforced mechanically. The workbench is the only interpretability infrastructure deployed on faith; everything after it is pulled by a project.

### Decisions

1. **Base image: nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04.** Runtime, not devel: the interp stack is pure-Python-plus-wheels (TransformerLens, nnsight, SAELens compile nothing), so the devel toolchain buys only image size. If a dependency later needs nvcc, switch to the devel base with a dated comment; do not add a second image.
2. **Torch pinned first, protected by a constraints file.** The Dockerfile installs an exact torch/cu130 pin (record as a build ARG, e.g. TORCH_VERSION) as its first pip step, then installs everything else with `pip install -c /opt/constraints.txt`, where constraints.txt pins torch (and torchvision if installed) to the same version. Any package whose requirements conflict with the pinned torch fails the build loudly. This implements the roadmap's "never let a pinned requirements file downgrade it" rule structurally.
3. **Package set (pin exact versions at implementation):** torch/cu130, transformer-lens (v3.x), nnsight (0.6.x), nnterp, sae-lens (v6.x), transformers, datasets, accelerate, einops, jaxtyping, plotly, pandas, matplotlib, circuitsvis, ipywidgets, jupyterlab, huggingface-hub, safetensors, pynvml. ARENA's per-chapter requirements are installed at image build from a vendored requirements file (docker/interp/arena-requirements.txt) filtered against the constraints file, so chapter notebooks run without in-container pip installs. wandb and sae-bench are not installed; they arrive with the phases that need them.
4. **Dockerfile at docker/interp/Dockerfile.interp, entrypoint at docker/interp/entrypoint.sh.** The entrypoint is thin: create the storage subdirectories if missing (decision 8), echo the environment (torch version, CUDA availability, GPU name), then exec the given command (default: JupyterLab).
5. **Compose service `interp` under `profiles: ["interp"]`, restart "no".** GPU device reservation copied from llama-gpu. Non-root appuser, cap_drop: ALL, no-new-privileges, matching the repository hardening pattern. Deliberate deviation, recorded here: no read_only rootfs. Jupyter, pip caches, and HF downloads need a writable filesystem; the hardening compensations are the non-root user, dropped capabilities, and localhost-only port binding. shm_size: 8g for dataloader workers.
6. **Mounts:** the repository bind-mounted read-write at /workspace (notebooks and code live in git); /mnt/ai-data/interp bind-mounted read-write (heavy artifacts); /mnt/ai-data/models read-only (parity with the inference services, for later phases that read served GGUFs' provenance). ARENA is cloned to /mnt/ai-data/interp/arena by a make target (decision 9), not baked into the image and not committed to this repository.
7. **JupyterLab on 127.0.0.1:8888**, token-authenticated (token generated per launch and printed by the make target; never disabled, never bound beyond localhost). 8888 collides with nothing in the compose file. Notebooks default to /workspace/interp/notebooks/, a new directory committed with a .gitkeep and a README stating the owner-work convention (checklist notebooks land here).
8. **Storage bootstrap.** The entrypoint (and a host-side make interp-dirs target for use outside the container) creates /mnt/ai-data/interp/{activations,saes,probes,vectors,neuronpedia,hf-cache} per the roadmap layout. HF_HOME=/mnt/ai-data/interp/hf-cache inside the container, so GPT-2, Gemma-2-2B, and Gemma Scope artifacts cache on NVMe once and never land in the image or the repository.
9. **Secrets: .env.interp**, untracked (add to .gitignore), passed via compose env_file, carrying HF_TOKEN for the gated Gemma-2-2B download. Nothing is baked into the image. This mirrors the .env.training convention the training designs define; whichever lands first establishes the pattern.
10. **Concurrency preflight in the Make targets, not in the container.** make interp-up and make interp-shell first check `docker compose ps` for running llama-gpu, vllm-gpu, or comfyui-flux and refuse with a one-line message naming the offender (make down or stopping the specific services clears it). llama-cpu running is permitted per the roadmap. The check lives in a small script (scripts/interp/preflight.sh) so both targets share it.

### Interfaces

- make interp-build — build the image.
- make interp-up — preflight, then start JupyterLab on 127.0.0.1:8888 and print the tokened URL.
- make interp-shell — preflight, then an interactive shell in the container with GPU attached.
- make interp-smoke — run the smoke test (below) in the container.
- make interp-down — stop the profile's services.
- make interp-dirs — create the /mnt/ai-data/interp tree on the host.
- All targets under a new `##@ Interpretability` Makefile section.

### Smoke test

scripts/interp/smoke_test.py, run via make interp-smoke and as the final image-build step (build-time run skips CUDA and network steps gracefully; the workstation run enforces them):

1. Import torch; assert torch.cuda.is_available() and device capability (12, 0).
2. Import transformer_lens, nnsight, nnterp, sae_lens, transformers, datasets, einops, circuitsvis, pynvml; assert torch.__version__ matches the pinned version (the anti-downgrade check, again, at runtime).
3. TransformerLens: load gpt2-small, run a forward pass with a hook attached to a residual-stream point, assert the hook fired and logits have the expected shape.
4. nnsight: trace the same prompt on the HF gpt2 model and read an intermediate activation.
5. bf16: run the gpt2-small forward in bf16 and assert finite logits (the substrate policy exercised).
6. Gated-model step, skipped with a warning when HF_TOKEN is absent: load Gemma-2-2B in bf16, one forward pass; load one Gemma Scope SAE via SAELens and encode the captured residual activations. This proves the full ARENA 1.3.2 dependency chain on this GPU.
7. Write a versions manifest (JSON: package versions, torch/CUDA/driver, GPU name) to stdout and, on the workstation, to /mnt/ai-data/interp/versions.json.

### Risks

- Dependency skew among transformer-lens, nnsight, and sae-lens (all move fast and pin transformers ranges): resolve at build time against the constraints file; if two of them cannot share one transformers version, pin to the newest mutually compatible set and record the conflict as a dated comment in the Dockerfile.
- ARENA requirements drift: the vendored arena-requirements.txt is a snapshot; chapters may add packages. The convention is to update the vendored file and rebuild rather than pip-installing inside a running container, so the image stays the lockfile.
- Gemma-2-2B gating: without an accepted license and HF_TOKEN, smoke step 6 cannot run. The step is skippable so the image is buildable anywhere, but Phase 0 acceptance on the workstation requires it.
- Jupyter is a shell on the machine: mitigated by localhost-only binding and token auth; never expose the port beyond 127.0.0.1 (remote access goes through an SSH tunnel).

### Acceptance

- make interp-build completes; the build fails if any non-CUDA smoke step fails.
- make interp-smoke on the workstation passes all seven steps, including the gated Gemma-2-2B and Gemma Scope step, and writes versions.json under /mnt/ai-data/interp/.
- make interp-up serves JupyterLab on 127.0.0.1:8888; a notebook kernel sees the GPU (torch.cuda.get_device_name returns the RTX 5090) and an ARENA chapter 1.2 notebook executes its setup cell without pip installs.
- The preflight demonstrably refuses to start the workbench while llama-gpu is up and permits it with only llama-cpu up.
- make up does not start the interp service; docker compose config confirms the profile gating.
- git status after a full smoke and one notebook session shows no untracked heavy artifacts: weights, caches, and outputs all landed under /mnt/ai-data/interp/.
- No secrets in the repository: .env.interp is gitignored and `git grep -iE 'hf_[A-Za-z0-9]{8}'` finds nothing.

## Phase 0 exit checklist

Restating the roadmap's Phase 0 acceptance criteria against this design and the owner work it enables:

1. Workbench built, smoked, and serving notebooks per the I1 acceptance above (this document).
2. ARENA exercises (1.1, 1.2 first three sections, 1.4.1, 1.3.2 skimmed) run end to end on the local GPU with no cloud fallback — owner work, enabled by I1; blockers found while running them are I1 bugs and come back here.
3. Core-technique checklist notebooks (activation patching, linear probes, pretrained-SAE usage, max-activating examples; logit lens and steering as nice-to-have) exist under interp/notebooks/, hand-written by the owner.
4. The steering exercise measurably shifts LLM-judge scores through a local service — owner work; the judge runs against a local OpenAI-compatible endpoint (llama-gpu or vllm-gpu) in the serving-dependent mode, which by the concurrency policy means the workbench GPU work is idle while judging runs.
5. One month has elapsed, whichever comes first. Unfinished items become backlog, not blockers.

Items 2-4 are tracked in the experiments log convention (docs/interpretability/experiments/), which is a Phase 1 deliverable exercised early; a simple checklist file under docs/interpretability/ suffices until it exists.

*Last updated: 2026-07-10. Written against docs/interpretability/roadmap.md as recorded in the planning document; implement I1 before starting the curriculum.*
