# Neuronpedia platform runbook

Operating guide for the self-hosted Neuronpedia platform (interp Phase 2). The platform
runs from a pinned upstream checkout plus this repository's compose override; this
repository holds no upstream logic. Design and rationale:
`docs/interpretability/ps1_neuronpedia_harness_investigation.md`.

## Table of contents

- [Port map](#port-map)
- [Layout](#layout)
- [Bring-up and teardown](#bring-up-and-teardown)
- [Concurrency matrix](#concurrency-matrix)
- [Autointerp through local serving](#autointerp-through-local-serving)
- [Attribution graphs](#attribution-graphs)
- [Shared network behavior](#shared-network-behavior)
- [Pin-bump procedure](#pin-bump-procedure)
- [Trust model](#trust-model)

## Port map

All host ports bind to 127.0.0.1 only. `scripts/interp/np_smoke.sh` fails if any platform
port is reachable beyond loopback.

| Service | Host port | Stack |
|---------|-----------|-------|
| llama-cpu | 8001 | inference (`docker-compose.yaml`) |
| llama-gpu | 8002 | inference |
| vllm-gpu | 8003 | inference |
| open-webui | 3000 | inference |
| comfyui | 8188 | inference |
| JupyterLab workbench | 8888 | interp workbench (`docker-compose.interp.yaml`) |
| Neuronpedia webapp | 3001 | platform (upstream compose + override) |
| Neuronpedia postgres | 5432 | platform |
| Neuronpedia inference | 5002 | platform |
| Neuronpedia autointerp | 5003 | platform |
| Neuronpedia graph | 5004 | platform |

The upstream graph-crm service (5005) is not deployed. The webapp is remapped from
upstream's 3000 to 3001 because open-webui holds 3000.

## Layout

| Path | Contents |
|------|----------|
| `/mnt/ai-data/interp/neuronpedia/src` | Pinned upstream checkout (multi-gigabyte after builds; local edits are discarded by `make np-clone`) |
| `/mnt/ai-data/interp/neuronpedia/postgres` | Database bind mount (replaces the upstream named volume) |
| `/mnt/ai-data/interp/hf-cache` | Shared HF cache; platform containers mount `hf-cache/hub` at `/root/.cache/huggingface/hub` |
| `docker/interp/neuronpedia-pin.txt` | Exact upstream SHA |
| `docker/interp/neuronpedia.override.yaml` | Port, volume, and network deltas |
| `docker/interp/neuronpedia.env-deltas` | Non-secret `.env.localhost` deltas (merged by `make np-env`) |
| `docker/interp/patches/` | Vendored patches applied to the checkout by `make np-clone` (see the directory README) |
| `.env.interp` (untracked) | `HF_TOKEN`, optional `OPENAI_API_KEY` |

## Bring-up and teardown

From a clean machine state:

```bash
$ make np-clone      # clone/pin the checkout (idempotent, discards local edits)
$ make np-build      # build images; runs np-env first (NEXT_PUBLIC_* is build-time)
$ make np-up         # webapp + postgres + db-init
$ make np-smoke      # loopback bindings, webapp, postgres, sm_120 when GPU instance up
```

GPU services, each behind the concurrency preflight:

```bash
$ make np-inference-up                 # MODEL_SOURCESET=gemma-2-2b-it.gemmascope-res-16k default
$ make np-graph-up                     # attribution graph server
$ make np-autointerp-up                # CPU by default; NP_GPU=1 attaches the GPU
$ make np-down                         # stop everything; database and caches persist
```

Database backup: `make np-down`, then snapshot `/mnt/ai-data/interp/neuronpedia/postgres`.

## Concurrency matrix

One GPU consumer at a time (AGENTS.md). llama-cpu may stay up throughout. The preflight
(`scripts/interp/preflight.sh`) blocks GPU platform targets while llama-gpu, vllm-gpu, or
comfyui is running; the failure mode is a policy message, not a CUDA OOM.

| Mode | Inference stack | Platform |
|------|-----------------|----------|
| Interactive platform work (PS3, PS5) | llama-gpu/vllm-gpu/comfyui down; llama-cpu allowed | webapp + postgres + one GPU service (np-inference-up or np-graph-up) |
| Serving-dependent autointerp (PS4) | vllm-gpu up | webapp + postgres + autointerp on CPU (`make np-autointerp-up`, no `NP_GPU`) |
| Browsing only | anything | webapp + postgres (`make np-up` is CPU-only, never preflighted) |

## Autointerp through local serving

PS4 mode: Delphi-based feature explanation batched against the vllm-gpu
OpenAI-compatible endpoint, no external API spend. `make np-autointerp-up` (default, CPU)
runs `scripts/interp/np_autointerp_preflight.sh`, which requires vllm-gpu healthy on
127.0.0.1:8003 and every other GPU consumer down — the inverse of the interactive
preflight. `NP_GPU=1` instead attaches the GPU for Delphi's CUDA-only embedding scorer
and runs the standard preflight (vllm-gpu down); the two modes are mutually exclusive.

Start order:

```bash
$ docker compose up -d vllm-gpu    # or make gpu-up; creates ai-network
$ make np-up                       # webapp + postgres
$ make np-autointerp-up            # serving preflight, then autointerp on 127.0.0.1:5003
```

Operational facts:

- The explainer model is whatever vllm-gpu serves; the request's `model` field must match
  the served model id. Source of truth:
  `curl -s http://127.0.0.1:8003/v1/models | jq -r '.data[].id'`.
- The autointerp client is OpenRouter-shaped with a per-request `openrouter_key`; against
  vLLM the key is a dummy string (vLLM ignores it). The base URL is redirected to
  `http://vllm-gpu:8003/v1` via the `OPENROUTER_BASE_URL` delta, honored through upstream
  support or the vendored patch in `docker/interp/patches/`.
- Acceptance evidence that no external endpoint was touched: vllm-gpu request logs
  (`docker compose logs vllm-gpu`) covering the batch window.
- The Phase 2 acceptance batch is small by design — at least 20 features of one imported
  Gemma Scope source — proving the path, not explaining the dictionary. Full-dictionary
  runs are project-pulled, sized with vllm-gpu throughput numbers, and run overnight like
  benchmarks.
- Generated explanations land in Postgres and render on the corresponding feature pages
  on `localhost:3001`.

## Attribution graphs

PS5 mode: circuit-tracer attribution graphs generated locally on Gemma-2-2B with the
8x/k64 transcoder configuration (verified inside the 32GB envelope; 32x configs need
40GB+ and are out of scope), rendered in the webapp's graph viewer. Interactive
concurrency mode: the graph server owns the GPU, so llama-gpu/vllm-gpu/comfyui and the
PS3 inference instance must be down (enforced by the preflight). Design and open
verification items: `docs/interpretability/ps5_attribution_graphs_investigation.md`.

Start order:

```bash
$ make np-up          # webapp + postgres
$ make np-graph-up    # preflight, then graph server on 127.0.0.1:5004
$ make np-smoke       # includes the sm_120 torch check inside the graph container
```

Operational facts:

- Transcoder weights download into the shared HF cache on first start (tens of GB;
  gated Gemma weights need `HF_TOKEN` from `.env.interp`).
- First graph generation includes weight load and compile and can look like a hang;
  warm generations are much faster.
- Graphs are generated and viewed through the webapp on `localhost:3001`; the hosted
  S3-upload path is not configured and graphs stay in the local stack.
- Mini-project post-mortems land in `docs/interpretability/experiments/` per that
  directory's README.

Measured on first host run (record here, per the PS5 acceptance):

| Measurement | Value |
|-------------|-------|
| Transcoder cache size on disk | not yet measured |
| First-run graph generation time | not yet measured |
| Warm graph generation time | not yet measured |
| Peak VRAM during generation | not yet measured |
| Graph persistence location (survives `np-down`?) | not yet verified |

## Shared network behavior

The autointerp container also joins the inference stack's `ai-network` so
`http://vllm-gpu:8003/v1` resolves directly (PS4). Docker compose behavior to know:

- The network is created by the inference stack (`make up`, or any `docker compose up`
  service) and removed by `docker compose down`. A stopped-but-not-downed stack keeps it.
- `make np-autointerp-up` fails if the network does not exist. Start order for PS4:
  `make gpu-up` (or at least vllm-gpu) first, then the platform targets.
- The external network name defaults to `ai-experiments_ai-network` (compose prefixes the
  network key with the project directory name). If `docker network ls` reports a
  different name, pass `AI_NETWORK_NAME=<name>` to the np-* targets.

## Pin-bump procedure

1. Update the SHA and dated comment in `docker/interp/neuronpedia-pin.txt`.
2. Re-verify the override against upstream's compose files at the new SHA: service names,
   compose file layout under `docker/`, env file conventions. Re-verify every patch in
   `docker/interp/patches/` still applies (`make np-clone` fails loudly if not) and
   delete any patch upstream has absorbed.
3. `make np-clone && make np-build && make np-up && make np-inference-up`.
4. `make np-smoke` must pass, including the sm_120 torch check inside the inference
   container. If upstream's torch pin predates sm_120 wheel support, override the torch
   version via build arg or a one-line patch and record it beside the pin.
5. Merge only with the smoke check green; record the reported output in the PR.

## Trust model

Localhost mode disables sign-in (`NEXT_PUBLIC_ENABLE_SIGNIN=false`) and authenticates
inter-service calls with static shared secrets. This is acceptable only because every
port binds to loopback. Do not rotate or "fix" the dummy secrets; do keep the loopback
smoke check green. Remote access, if ever needed, goes through an SSH tunnel, matching
the workbench convention.
