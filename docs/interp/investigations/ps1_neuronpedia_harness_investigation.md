# Interpretability PS1 Neuronpedia deployment harness: implementation investigation

**Frozen record — 2026-07-14. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing interp Phase 2 deliverable PS1 (Neuronpedia deployment
harness) entails in this repository. The planning document's design is taken as intent;
where it conflicts with the repository, the repository is the source of truth and the
deviations are recorded here. PS1 is pure orchestration: a pinned upstream checkout, a
compose override, wrapper make targets, and a smoke check. It writes no interpretability
code and vendors no upstream logic.

## Table of contents

- [What PS1 delivers](#what-ps1-delivers)
- [Design-versus-repository deviations](#design-versus-repository-deviations)
- [Pinned checkout and pin file](#pinned-checkout-and-pin-file)
- [Compose override](#compose-override)
- [Environment files](#environment-files)
- [Make targets and preflight](#make-targets-and-preflight)
- [Smoke check script](#smoke-check-script)
- [Runbook page](#runbook-page)
- [Upstream facts to re-verify at implementation time](#upstream-facts-to-re-verify-at-implementation-time)
- [File inventory](#file-inventory)
- [Acceptance checks and host-side split](#acceptance-checks-and-host-side-split)
- [Risks](#risks)
- [Open questions](#open-questions)

## What PS1 delivers

One command brings the self-hosted Neuronpedia platform (webapp, postgres, db-init,
inference, autointerp, graph) up in a known state from a pinned upstream revision, with
this repository's port, storage, and network conventions applied as a thin override; one
command tears it down. Upstream orchestration is adopted, not ported: this repository
holds only a pin file, an override file, an env-delta file, wrapper make targets, a smoke
script, and a runbook. Everything else runs from upstream's own compose files and make
targets inside the pinned checkout.

PS1 depends on nothing else in Phase 2 and everything else (PS2 import, PS3 steering,
PS4 autointerp, PS5 attribution graphs) sits on it.

## Design-versus-repository deviations

The planning document largely matches the repository, but several references are stale.
Repository reality wins in each case.

| Design says | Repository reality | Resolution |
|-------------|--------------------|------------|
| Python style per `.claude/PYTHON_STYLE.md` | Style guides live in `.styles/` (`BASH_OUTPUT_STYLE.md`, `PYTHON_OUTPUT_STYLE.md`, `DOCUMENTATION_STYLE.md`) | Follow `.styles/`; the smoke script follows `BASH_OUTPUT_STYLE.md` |
| Concurrency preflight covers llama-gpu and vllm-gpu | `scripts/interp/preflight.sh` already exists and also blocks on `comfyui` (GPU consumer, port 8188) | Reuse the existing preflight for GPU-attached platform services; comfyui is a blocker too |
| `USE_LOCAL_HF_CACHE=1` pointed at "the existing HF cache location" | The interp tier's HF cache is `/mnt/ai-data/interp/hf-cache` (`HF_HOME` in `docker/interp/Dockerfile.interp:78`); upstream's hf-cache overlay defaults to `${HOME}/.cache/huggingface/hub` | The override remaps the HF cache mount to `/mnt/ai-data/interp/hf-cache` so Gemma weights are shared with the workbench and download once |
| Storage tree "adds neuronpedia/ under /mnt/ai-data/interp" | `make interp-dirs` and `docker/interp/entrypoint.sh` already create `/mnt/ai-data/interp/neuronpedia` | No new top-level directory; PS1 adds `src/` and `postgres/` beneath it (extend `interp-dirs`) |
| Workbench "exists or lands alongside PS1" | Interp Phase 0 I1 is merged: `docker-compose.interp.yaml`, `docker/interp/`, `make interp-*` all exist | Dependency satisfied; PS1 builds beside it, not into it |
| Roadmap wording: platform "under the interp compose profile" | Repository precedent is a separate compose file per tier, invoked only through make targets; the doc itself already amends this | The Neuronpedia stack runs from upstream compose plus our override, never from this repository's compose files; `make up` cannot start it |
| Documentation may claim inference ports 8004/8005 | `docker-compose.yaml` binds llama-cpu 8001, llama-gpu 8002, vllm-gpu 8003, open-webui 3000, comfyui 8188 — all `127.0.0.1` | Compose file is the source of truth; the doc's port block (webapp 3001, postgres 5432, inference 5002, autointerp 5003, graph 5004) avoids all of them |

The inference-stack network the override must join is named `ai-network`
(`docker-compose.yaml:242`, bridge driver, project-scoped name at runtime). The doc calls
it "this repository's compose network" without naming it; the override declares it as an
external network using the actual runtime name (`ai-experiments_ai-network` or whatever
`docker network ls` reports — verify on the host, since compose prefixes by project
directory name).

## Pinned checkout and pin file

- `docker/interp/neuronpedia-pin.txt`: one line, the exact upstream git SHA, with a dated
  comment — same convention as the llama.cpp pin
  (`docker/llama-gpu/Dockerfile:61`, `ARG LLAMA_CPP_COMMIT=...  # tag b9976 (2026-07-11)`).
- Checkout target: `/mnt/ai-data/interp/neuronpedia/src`. Not a submodule: it is
  multi-gigabyte after builds (images, `node_modules`) and lives on NVMe per the storage
  convention.
- `make np-clone` is idempotent: clone if absent, otherwise `git fetch` plus
  `git checkout <pinned SHA>`. It should refuse (or warn loudly) if the working tree is
  dirty, since builds write into the checkout.
- Pin selection is an implementation-time task: pick the current upstream default-branch
  SHA, re-verify the upstream facts below against it, and record it. Bumping the pin later
  requires a green `np-smoke` before merge.

## Compose override

`docker/interp/neuronpedia.override.yaml`, checked in, passed with `-f` after upstream's
compose files. It carries only the deltas:

1. Host port remaps, all `127.0.0.1`-bound: webapp `127.0.0.1:3001:3000` (3000 collides
   with open-webui), postgres `127.0.0.1:5432:5432`, inference `127.0.0.1:5002:5002`,
   autointerp `127.0.0.1:5003:5003`, graph `127.0.0.1:5004:5004`. The upstream graph-crm
   service (5005) is not deployed.
2. Postgres data as a bind mount to `/mnt/ai-data/interp/neuronpedia/postgres`, replacing
   the upstream named volume (backup by stopping postgres and snapshotting the directory).
3. HF cache mount remapped to `/mnt/ai-data/interp/hf-cache` (see deviations table).
4. External network attachment joining the autointerp container to `ai-network` so
   `http://vllm-gpu:8003/v1` resolves directly (consumed by PS4; wired now because it is a
   one-line delta and the override is PS1's deliverable).

Everything else — service definitions, images, healthchecks, inter-service wiring — stays
upstream's. The override references upstream service names, which is the main pin-bump
fragility (see risks).

## Environment files

- `docker/interp/neuronpedia.env-deltas`, checked in: non-secret deltas seeded into upstream's
  `.env.localhost` — `NEXT_PUBLIC_URL=http://localhost:3001`,
  `NEXTAUTH_URL=http://localhost:3001`, `USE_LOCALHOST_AUTOINTERP=true`.
- Secrets come from the existing untracked `.env.interp` (already gitignored, line 17 of
  `.gitignore`; template `env.interp.example` carries `HF_TOKEN`). PS1 extends the
  template with an optional `OPENAI_API_KEY` (explanation-search embeddings, a PS2
  concern) and the wrapper targets merge `.env.interp` at invocation. Secrets are never
  written into the checkout in committed form.
- The localhost trust model (sign-in disabled, static `x-secret-key` shared secrets) is
  acceptable only because every port binds to loopback. The dummy secrets are not to be
  "fixed"; the smoke check instead fails loudly if any port mapping loses its `127.0.0.1`
  prefix.

## Make targets and preflight

New `##@ Interpretability Platform` section in the Makefile, following the existing
color/echo conventions and `.PHONY` declarations:

| Target | Action |
|--------|--------|
| `np-clone` | Idempotent clone/fetch/checkout of the pinned SHA into `/mnt/ai-data/interp/neuronpedia/src` |
| `np-build` | Upstream image builds via upstream make/compose, with the override applied |
| `np-up` | webapp + postgres + db-init (CPU-only trio) |
| `np-inference-up` | GPU inference instance, `MODEL_SOURCESET=gemma-2-2b-it.gemmascope-res-16k` default; preflight-gated |
| `np-autointerp-up` | Autointerp service; preflight semantics depend on mode (PS4 runs it beside vllm-gpu) |
| `np-graph-up` | Graph server; preflight-gated |
| `np-down` | Tear down the platform stack; always safe |
| `np-status` | `docker compose ps` for the platform project plus port-binding summary |

Concurrency enforcement: GPU-attached platform targets run `scripts/interp/preflight.sh`
first and refuse with a clear message and nonzero exit if llama-gpu, vllm-gpu, or comfyui
is running — the policy failure mode is a refusal message, not a CUDA OOM. `np-up`
(CPU-only services) does not need the GPU preflight but the doc's acceptance wants the
refusal observable, so the gate sits on every target that requests a GPU service.
llama-cpu may stay up throughout.

Targets shell into the pinned checkout and drive upstream's own make/compose with our
`-f` override and env merge; they do not duplicate upstream logic. Implementation detail
to settle at build time: whether wrapping upstream's make targets or calling
`docker compose -f <upstream files> -f <override>` directly is more stable across pin
bumps — whichever requires fewer assumptions about upstream's Makefile internals wins.

## Smoke check script

`scripts/interp/np_smoke.sh`, following `.styles/BASH_OUTPUT_STYLE.md` (`[OK]`/`[ERROR]`
lines, `NO_COLOR` support, exit code 0/1):

1. Every mapped platform port (3001, 5432, 5002, 5003, 5004) that is listening is bound
   to `127.0.0.1` only (`ss -tlnp` inspection); any `0.0.0.0` or `[::]` binding is a hard
   failure.
2. Webapp answers on `http://127.0.0.1:3001`.
3. Postgres accepts connections (e.g. `pg_isready` inside the container).
4. When the GPU inference instance is up: the container sees the RTX 5090 and
   `torch.cuda.get_device_capability()` reports sm_120 support. Upstream GPU images build
   on CUDA 12.1 runtime bases, which is fine under the CUDA 13 host driver, but the torch
   wheel inside must support sm_120; if upstream's pin predates sm_120, override the torch
   version via build arg or a one-line patch and record it with the pin.

A companion `make np-smoke` target runs it. The script is host-side only (needs the
running stack and the GPU).

## Runbook page

`docs/interpretability/platform.md`, per `.styles/DOCUMENTATION_STYLE.md`:

- Port map (platform 3001/5432/5002/5003/5004 beside inference 8001/8002/8003, open-webui
  3000, comfyui 8188, Jupyter 8888).
- Start/stop matrix per concurrency mode: interactive platform work (GPU inference
  quiesced), serving-dependent PS4 (vllm-gpu up, interp GPU idle), llama-cpu always
  permitted.
- Docker network behavior for the shared `ai-network`: the external attachment requires
  the inference stack's network to exist even when its services are stopped; `docker
  compose down` removes it. The start matrix covers the required order.
- Pin-bump procedure: update `neuronpedia-pin.txt`, `np-clone`, `np-build`, `np-smoke`
  green, then merge.
- The PS2 import runbook slots in here later.

## Upstream facts to re-verify at implementation time

From the planning doc (verified 2026-07) and a fresh look at the upstream repository
(github.com/hijohnnylin/neuronpedia, MIT). Upstream moves quickly; re-verify all of these
against the chosen pin SHA before writing the override:

- Compose layout: `docker/compose.yaml` plus per-service overlays
  (`compose.inference.gpu.yaml`, `compose.autointerp.gpu.yaml`, `compose.graph.gpu.yaml`,
  `compose.hf-cache.yaml`), driven by upstream targets (`webapp-localhost-*`,
  `inference-localhost-dev-gpu MODEL_SOURCESET=...`, `autointerp-localhost-dev-gpu`,
  `graph-localhost-dev-gpu`).
- Services and container ports: webapp (Next.js, node:22-alpine) 3000; db-init (Prisma
  migrations, run-once); postgres (pgvector/pgvector:pg15) 5432; inference (Python 3.10,
  FastAPI) 5002; autointerp (FastAPI wrapping EleutherAI Delphi) 5003; graph (FastAPI
  wrapping safety-research/circuit-tracer) 5004; graph-crm 5005 (not deployed).
- `.env.localhost` knobs: `PORT`, `NEXT_PUBLIC_URL`, `NEXTAUTH_URL`,
  `USE_LOCALHOST_INFERENCE=true`, `USE_LOCALHOST_AUTOINTERP`, three `*_SERVER_SECRET`s,
  `OPENAI_API_KEY`.
- Per-instance inference env files `.env.inference.<model>.<sourceset>`;
  `gemma-2-2b-it.gemmascope-res-16k` ships as a preset; localhost mode runs exactly one
  inference instance at a time.
- Practical minima: ≥16GB RAM (trivial on 128GB), linux/amd64 images, `HF_TOKEN` for
  gated Gemma weights.

## File inventory

New files:

| File | Purpose |
|------|---------|
| `docker/interp/neuronpedia-pin.txt` | Exact upstream SHA with dated comment |
| `docker/interp/neuronpedia.override.yaml` | Port/volume/network deltas over upstream compose |
| `docker/interp/neuronpedia.env-deltas` | Non-secret `.env.localhost` deltas |
| `scripts/interp/np_smoke.sh` | Loopback-binding, webapp, postgres, sm_120 checks |
| `docs/interpretability/platform.md` | Runbook: port map, concurrency matrix, pin-bump procedure |

Modified files:

| File | Change |
|------|--------|
| `Makefile` | New `##@ Interpretability Platform` section: `np-clone`, `np-build`, `np-up`, `np-inference-up`, `np-autointerp-up`, `np-graph-up`, `np-down`, `np-status`, `np-smoke`; extend `interp-dirs` with `neuronpedia/postgres` |
| `env.interp.example` | Add optional `OPENAI_API_KEY` with comment |

Host-side state (not in git): `/mnt/ai-data/interp/neuronpedia/{src,postgres}`.

## Acceptance checks and host-side split

All three acceptance checks are host-side (need Docker, the GPU, and `/mnt/ai-data`);
the agent outputs the commands and records reported output verbatim:

1. From a machine with neither stack running: `make np-clone && make np-build && make
   np-up`, then `make np-smoke` passes; `make np-down` returns to clean state.
2. `ss -tlnp` (via `np_smoke.sh`) verifies all five host ports loopback-only.
3. With llama-gpu running, `make np-inference-up` (a GPU platform service) fails with the
   concurrency policy message, not a CUDA OOM.

Sandbox-executable beforehand: shellcheck on `np_smoke.sh`, `docker compose config`
syntax validation of the override in isolation (against a stub, since the upstream files
live on the host), Makefile target inspection.

## Risks

- **Upstream compose/service-name churn.** The override names upstream services; a pin
  bump can rename them. Mitigated by the pin-bump procedure requiring green `np-smoke`.
- **Two compose projects, one network.** The `ai-network` external attachment fails if
  the inference stack's network does not exist; `make down` removes it. The runbook start
  matrix documents the order; the actual runtime network name must be verified on the
  host before writing the override.
- **torch/sm_120 inside upstream GPU images.** Caught at smoke time; remedy is a build
  arg or one-line patch recorded with the pin.
- **Upstream Dockerfiles that do not build at the pin.** Not listed at design time, and
  met on the first `make np-build` (2026-09-17, KB-68): the graph image installs
  python3.12 from apt on an Ubuntu 22.04 base, and the second build found the inference
  image installing 3.10 for a project that requires 3.11 (KB-76), and the third found
  the same Dockerfile still installing with poetry for a project that had moved to uv
  (KB-77). Re-verifying compose layout at a pin does not exercise the Dockerfiles; only
  a build does, and one failing stage hides the next in the same stage. Same remedy as
  the row above, and the pin-bump procedure now names it.
- **First `np-build` is heavy** (multi-gigabyte images, `node_modules` in the checkout).
  All of it lands on NVMe under `/mnt/ai-data`, not in this repository; the existing
  disk-free watchdog floor applies.

## Resolved at implementation (2026-07-14)

1. **Pin SHA:** `e5488b2a37976333e6b31f7f1cd6a56b4b5cf7ab` (upstream main, 2026-07-14).
   Upstream's `docker/compose.yaml`, GPU/dev overlays, and localhost make targets were
   re-verified against this SHA; service names in the override match.
2. **Direct compose, not upstream make.** Upstream's localhost targets accept no extra
   `-f`, so the wrapper targets mirror upstream's exact compose invocations (same file
   stacks, same `--env-file` order, `RELOAD=0`) with the override appended last. A pin
   bump re-checks the mirrored invocations against upstream's Makefile.
3. **Network name** defaults to `ai-experiments_ai-network` via the `AI_NETWORK_NAME`
   variable (overridable per invocation) rather than a hardcoded value.
4. **Delta file named `neuronpedia.env-deltas`, not `neuronpedia.env`.** The agent
   sandbox guard (`scripts/agents/guard.sh`) denies access to any `*.env` path; the file
   holds no secrets, so it takes a non-matching name instead of a guard exception.
5. **Env merge step (`make np-env`, `scripts/interp/np_env.sh`).** `NEXT_PUBLIC_*` values
   are baked into the webapp image at build time, so deltas cannot ride an env file at
   `up`; they are merged into the checkout's `.env.localhost` before every build/up, and
   `np-clone` uses `checkout -f` so a re-clone plus re-merge is always clean.
6. **Autointerp defaults to CPU** (`NP_GPU=1` opts into the GPU overlay plus preflight),
   so the PS4 mode — autointerp beside a running vllm-gpu — does not trip the
   one-GPU-consumer gate.
