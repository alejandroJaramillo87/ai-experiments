# Image and video generation

Local image and video generation on the RTX 5090 using ComfyUI as the single serving
frontend. One container (`comfyui`, port 8188) serves both modalities; models are
selected per workflow.

## Table of contents

- [Stack](#stack)
- [Models](#models)
- [Setup](#setup)
- [GPU exclusivity](#gpu-exclusivity)
- [VRAM strategy](#vram-strategy)
- [Acceptance checks](#acceptance-checks)

## Stack

| Component | Version | Notes |
|-----------|---------|-------|
| ComfyUI | v0.27.1 | Native FLUX.2, Wan 2.2, and LTX video nodes |
| PyTorch | 2.9.1+cu130 | Matches host driver 580.65.06 / CUDA 13.0.88 |
| Base image | nvidia/cuda:13.0.0-ubuntu24.04 | Built for Blackwell sm_120 |

The service definition lives in `docker-compose.inference.yaml` (`comfyui`); the image build in
`/docker/comfyui/Dockerfile.comfyui`. The web interface is bound to
`127.0.0.1:8188`.

## Models

Downloaded by `scripts/models/download_art_models.sh` into `/mnt/ai-data/models`,
which mounts read-only at `/app/models` in the container.

### Image

| Model | Size | VRAM | Role |
|-------|------|------|------|
| FLUX.2 klein 4B | 4B | ~13GB | Fast iteration, 4-step generation, Apache 2.0 |
| FLUX.2 klein base 9B FP8 | 9B | ~20GB | Higher quality, non-commercial license |

### Video

| Model | Size | VRAM | Role |
|-------|------|------|------|
| LTX-2.3 FP8 | — | 32GB (streams above) | Video with synchronized audio |
| Wan 2.2 T2V/I2V 14B FP8 | 27B MoE, 14B active | ~28GB | Photorealistic video |

The FLUX.2 repositories require accepting the license on huggingface.co and an
authenticated CLI (`hf auth login`) before download.

## Setup

```bash
$ make comfy-build          # build the ComfyUI image
$ make comfy-models-image   # download image models
$ make comfy-models-video   # download video models
$ make art-mode             # stop GPU LLM services, start ComfyUI
```

Open http://localhost:8188 and load a starter template as described in
`/workflows/README.md`. Generated files land in `/outputs/comfyui`.

## GPU exclusivity

The repository rule of one GPU consumer at a time applies. `make art-mode` stops
`llama-gpu` and `vllm-gpu` before starting ComfyUI; `make llm-mode` reverses the
switch. `llama-cpu` (port 8001) stays available in both modes for prompt-expansion
work on the CPU.

## VRAM strategy

- The default `COMFY_VRAM_MODE=highvram` keeps model weights resident, which is
  correct for images and short clips.
- LTX-2.3 fits a 720p 24fps 4-second clip inside 32GB; longer clips engage weight
  streaming into system RAM, which is why the container memory limit is 64G.
- For long video clips that OOM, set `COMFY_VRAM_MODE=normalvram` in `.env` and
  restart the service.
- Quantized GGUF variants of each model (loadable through the ComfyUI-GGUF node)
  trade quality for headroom when composing multi-model workflows.

## Acceptance checks

Run on the workstation after building; the container requires the GPU.

```bash
$ make comfy-build && make art-mode
$ curl -sf http://localhost:8188/system_stats   # healthcheck endpoint
```

Then generate one image (FLUX.2 klein template, default settings) and one 4-second
720p video (LTX-2.3 template) from the web interface, and confirm both files appear
under `/outputs/comfyui`.
