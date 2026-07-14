# Interpretability I1 workbench and M3 commons: implementation investigation

Investigation of what implementing the interpretability Phase 0 deliverables — I1 (workbench
image and compose integration) and M3 (mini-project harness commons) — entails in this
repository. The planning document's design is taken as intent; where it conflicts with the
repository, the repository is the source of truth and the deviations are recorded here.

## Table of contents

- [Design-versus-repository deviations](#design-versus-repository-deviations)
- [I1: workbench image](#i1-workbench-image)
- [I1: compose integration](#i1-compose-integration)
- [I1: Makefile targets and preflight](#i1-makefile-targets-and-preflight)
- [I1: smoke test](#i1-smoke-test)
- [M3: mini-project harness commons](#m3-mini-project-harness-commons)
- [Dependency findings (verified 2026-07-14)](#dependency-findings-verified-2026-07-14)
- [File inventory](#file-inventory)
- [Acceptance checks and host-side split](#acceptance-checks-and-host-side-split)
- [Open questions](#open-questions)

## Design-versus-repository deviations

The planning document was written before the training tier landed. The repository has since
established concrete patterns that supersede several of its decisions.

| Design says | Repository reality | Resolution |
|-------------|--------------------|------------|
| pip with a hand-written `/opt/constraints.txt`, `TORCH_VERSION` build ARG | Dependencies are managed by uv; `docker/training/Dockerfile.training` installs with `uv sync --frozen --group <group>` from `uv.lock`; torch is pinned to 2.13.0+cu130 via the `pytorch-cu130` index in `pyproject.toml` | Install with `uv sync --frozen --no-install-project --no-default-groups --group interp`. The lockfile is the constraints file: any interp package that requires a different torch fails `uv lock`, which is the anti-downgrade rule enforced structurally. |
| nnsight 0.6.x | `pyproject.toml` already carries an `interp` group with `transformer-lens==3.5.1` and `nnsight==0.7.0`, both resolved in `uv.lock` against transformers 5.13.1 | Keep the repository pins; extend the group with the missing packages (see dependency findings). |
| `interp` profile inside `docker-compose.yaml` | The repository precedent is a separate compose file per non-inference tier (`docker-compose.training.yaml`, `docker-compose.agents.yaml`) invoked only through make targets; `make up` uses the default file and cannot start them | A separate `docker-compose.interp.yaml` is the consistent choice. The acceptance criterion "docker compose config confirms the profile gating" becomes "`make up` uses `docker-compose.yaml` only, verified by inspection of the Makefile and `docker compose -f docker-compose.yaml config --services`". |
| Preflight checks `comfyui-flux` | The compose service is named `comfyui` (container `comfyui`, port 8188) | Preflight checks `llama-gpu`, `vllm-gpu`, `comfyui`. |
| "The .claude/ style-guide files do not currently exist" | Style guides exist under `.styles/` and are loaded via `CLAUDE.md` | Follow `.styles/PYTHON_OUTPUT_STYLE.md` (the `Printer` class), `BASH_OUTPUT_STYLE.md`, `COMMENT_STYLE.md`, `DOCUMENTATION_STYLE.md`. |
| "No training tier exists yet" | Training D1 (container), D2 (data pipeline), and D5 (recipe system) are merged | The workbench copies the training container's hardening and layout patterns rather than inventing them. Images stay separate as designed. |
| Scripts under `scripts/interp/` | Training settled on `scripts/training/` (a `scripts/train/` rename was already repaired once) | `scripts/interp/` is consistent; no change needed. |
| `.env.interp` gitignored | `.gitignore` covers `.env` and `.env.*.local` but not `.env.interp` | Add an explicit `.env.interp` entry. No `.env.training` exists yet, so this establishes the per-tier env-file pattern. |
| No `read_only` rootfs (documented deviation for Jupyter) | The training service runs `read_only: true` with tmpfs mounts for `/tmp` and the HF cache, and `user: ${UID}:${GID}` | Attempt the stricter pattern first: `read_only: true` plus tmpfs `/tmp` (exec flag for Triton) and a writable home via tmpfs, with `HF_HOME` on the `/mnt/ai-data/interp/hf-cache` bind mount. Jupyter needs a writable runtime dir (`JUPYTER_RUNTIME_DIR`, `JUPYTER_DATA_DIR`) which tmpfs covers. Fall back to the design's writable-rootfs deviation only if this demonstrably breaks, with a dated comment. |

## I1: workbench image

`docker/interp/Dockerfile.interp`, modeled line-for-line on `Dockerfile.training` with these
differences:

- **Base**: `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04` (runtime, not devel). The interp
  stack is pure wheels; nothing compiles. The training image needed devel only for DeepSpeed
  CPUAdam. Keep the pinned uv stage (`ghcr.io/astral-sh/uv:0.11.28`) and the
  `uv venv /opt/venv --python 3.12` pattern.
- **Install**: `COPY pyproject.toml uv.lock` then
  `uv sync --frozen --no-install-project --no-default-groups --group interp`. Note the interp
  group must be self-contained the way the training group is: packages already in core
  dependencies (torch, transformers, datasets, accelerate, einops, jupyterlab, matplotlib,
  pandas, huggingface-hub) are pulled in because `uv sync` for a group still resolves against
  the project. Verify with the same flags training uses; if `--no-default-groups` excludes
  core dependencies, mirror exactly what the training Dockerfile does today (it works).
- **Environment**: `TORCH_CUDA_ARCH_LIST="12.0"`, `HF_HOME=/mnt/ai-data/interp/hf-cache`
  (bind mount, not tmpfs — Gemma-2-2B weights must persist across runs, unlike the training
  container's disposable cache), `HOME=/app`.
- **Non-root**: the repo-standard `appuser` line; compose overrides with `${UID}:${GID}` for
  bind-mount write parity, as training does.
- **Build-time gate**: `RUN python scripts/interp/smoke_test.py --build` as the final layer,
  exactly like training. `--build` degrades CUDA and network steps to `[SKIP]`.
- **ARENA requirements**: the design wants a vendored `docker/interp/arena-requirements.txt`
  installed at build. Under uv this cannot be a raw `pip install -r` (it would bypass the
  lock). Two options, decide at implementation: (a) fold ARENA's needed packages into the
  `interp` group so the lockfile owns them — cleanest, and most are already listed
  (einops, jaxtyping, plotly, circuitsvis, ipywidgets); (b) vendor the file and install with
  `uv pip install -r arena-requirements.txt --constraint <lock-derived pins>` as a separate
  layer. Option (a) is recommended; the vendored file then documents provenance only.

`docker/interp/entrypoint.sh`: thin, copied from the training entrypoint's structure
(color helpers per `BASH_OUTPUT_STYLE.md`, environment echo, `exec "$@"`), plus:

- Create `/mnt/ai-data/interp/{activations,saes,probes,vectors,neuronpedia,hf-cache}` if
  missing (idempotent `mkdir -p`).
- Default command starts JupyterLab:
  `jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --IdentityProvider.token=${JUPYTER_TOKEN}`
  with the token generated by the make target and passed through the environment. Binding
  inside the container is 0.0.0.0; localhost-only exposure is enforced by the compose port
  mapping `127.0.0.1:8888:8888`.

## I1: compose integration

`docker-compose.interp.yaml`, mirroring `docker-compose.training.yaml`:

- Service `interp`, `restart: "no"`, `init: true`, `user: "${UID:-1000}:${GID:-1000}"`,
  `security_opt: no-new-privileges`, `cap_drop: ALL`, GPU device reservation copied from
  llama-gpu, `shm_size: "8g"`.
- Unlike training this is a long-running service (`up -d`, healthcheck optional), with
  `ports: ["127.0.0.1:8888:8888"]`.
- Mounts: repository at `/workspace` read-write (notebooks live in git);
  `/mnt/ai-data/interp` read-write; `/mnt/ai-data/models` read-only.
- `env_file: .env.interp` (documented as optional-but-required-for-Gemma; compose fails on a
  missing env_file unless marked `required: false`, so ship the file as required and document
  `cp env.interp.example .env.interp` in setup, or mark it optional — decide at
  implementation; `required: false` is friendlier for first build).
- The design's "profile gating" acceptance is satisfied structurally: `make up` never touches
  this file.

## I1: Makefile targets and preflight

New `##@ Interpretability` section following the `##@ Training` section's shape, with
`INTERP_COMPOSE := docker-compose.interp.yaml`:

- `interp-build` — `docker compose -f $(INTERP_COMPOSE) build`.
- `interp-up` — run `scripts/interp/preflight.sh`, generate a token
  (`openssl rand -hex 24`), `JUPYTER_TOKEN=... docker compose -f $(INTERP_COMPOSE) up -d interp`,
  print `http://127.0.0.1:8888/?token=...`.
- `interp-shell` — preflight, then `run --rm interp /bin/bash`.
- `interp-smoke` — `run --rm interp python scripts/interp/smoke_test.py`.
- `interp-down` — `docker compose -f $(INTERP_COMPOSE) down`.
- `interp-dirs` — host-side `mkdir -p` of the storage tree.

`scripts/interp/preflight.sh`: checks
`docker compose ps --status running --format '{{.Service}}'` for `llama-gpu`, `vllm-gpu`, or
`comfyui`; refuses with one `[ERROR]` line naming the offender and the clearing command
(`make down` or `docker compose stop <service>`). `llama-cpu` running is permitted. Output per
`BASH_OUTPUT_STYLE.md`. Note the Makefile's existing `comfyui-mode`/`llm-mode` GPU-switching
targets: the preflight is the same policy extended to a third GPU consumer.

## I1: smoke test

`scripts/interp/smoke_test.py`, structured like `scripts/training/smoke_test.py` (the
`Printer` class, `--build` mode, exit codes 0/1/2, versions manifest). Steps per the design:

1. torch import; CUDA available; capability (12, 0) — enforced on workstation, `[SKIP]` at
   build.
2. Imports: `transformer_lens`, `nnsight`, `nnterp`, `sae_lens`, `transformers`, `datasets`,
   `einops`, `circuitsvis`, `pynvml`; assert `torch.__version__` startswith the pinned
   version (read the pin from `importlib.metadata`, not a hardcoded string, so the lockfile
   remains the single source).
3. TransformerLens: `gpt2-small` forward with a residual-stream hook; assert hook fired and
   logit shape. Network-dependent (downloads GPT-2 on first run): `[SKIP]` at build, required
   on workstation.
4. nnsight: trace HF `gpt2`, read an intermediate activation.
5. bf16 forward on gpt2-small, assert finite logits.
6. Gated step: Gemma-2-2B bf16 forward plus one Gemma Scope SAE encode via SAELens;
   `[WARN]`+skip when `HF_TOKEN` absent; required for workstation acceptance.
7. Versions manifest JSON to stdout and `/mnt/ai-data/interp/versions.json` on the
   workstation.

## M3: mini-project harness commons

Deliberately thin importable package at `interp/miniprojects/common/` (total budget ~600
lines), host-runnable tests at `tests/interp/`, no CLI, no config system.

- `models.py` — `load_subject(model_id, revision)` returning a TransformerLens
  `HookedTransformer` where supported, else an nnsight `LanguageModel`, bf16 on cuda;
  `format_chat(model_id, messages)` always through `tokenizer.apply_chat_template`; a small
  registry of known model quirks; unknown ids raise.
- `artifacts.py` — `fetch(spec)` for HF artifacts pinned to explicit revisions in a
  checked-in `artifacts.yaml`, downloading to `/mnt/ai-data/models/` or
  `/mnt/ai-data/interp/`, idempotent, verifying size/hash where available. Uses
  `huggingface_hub.snapshot_download` with `revision=`, which is already idempotent via the
  cache; verification is the added value.
- `pairs.py` — contrastive pair sets from JSONL; difference-in-means over a hook point;
  train/test split with normalized-string dedup across the boundary. Pure torch + stdlib;
  unit-testable on CPU.
- `judge.py` — `score(items, rubric, endpoint)` against a local OpenAI-compatible endpoint,
  default `http://localhost:8003/v1` (vllm-gpu), fallback `http://localhost:8002/v1`
  (llama-gpu). Healthcheck preflight before scoring (vllm exposes `/health`, llama.cpp
  `/v1/health` — both verified in the Makefile's `status` target); schema-constrained JSON
  where supported, parse-with-retry otherwise; warn if a hooked model is resident on the GPU
  (`torch.cuda.memory_allocated() > 0`). Endpoint mocked in unit tests.
- `record.py` — plot/JSON drop helpers with the naming convention the experiments log links
  to.
- Tests: `tests/interp/conftest.py` mirrors `tests/training/conftest.py` (repo root on
  `sys.path`); `pairs` and `judge` tests run under `make test` (`uv run pytest -v`) with no
  GPU and no network. Note `make test` runs on the host: importing `transformer_lens` at
  module top level in `models.py` would require the interp group host-side. Either instruct
  `uv sync --group interp` for development (the group is host-installable) or import lazily
  inside `load_subject`; lazy import keeps `make test` working from a plain `uv sync` and is
  recommended.
- Makefile: `interp-fetch PROJECT=<slug>` added to the `##@ Interpretability` section. The
  M3 GPU check (Gemma-2-2B hooked forward, difference-in-means on two pairs) extends
  `interp-smoke` rather than adding a target.

## Dependency findings (verified 2026-07-14)

Current lockfile state and PyPI checks:

| Package | Status |
|---------|--------|
| torch 2.13.0+cu130, torchvision 0.28.0 | Core dependency, cu130 index, in lock |
| transformers 5.13.1, datasets 5.0.0, accelerate 1.14.0, einops 0.8.2, jupyterlab 4.6.1, matplotlib 3.11.0, pandas 2.3.3, huggingface-hub 1.23.0, safetensors (transitive) | Already core dependencies |
| transformer-lens 3.5.1 | Already in the `interp` group, resolved in lock |
| nnsight 0.7.0 | Already in the `interp` group (newer than the design's 0.6.x; keep) |
| sae-lens | Not yet in group. Latest 6.46.0; requires `transformer-lens>=2.16.1` (satisfied), `datasets>=3.1.0` (satisfied), plotly, tenacity — no torch or transformers ceiling observed |
| nnterp | Not yet in group. Latest 1.3.0; requires `nnsight>=0.6` (satisfied) |
| circuitsvis | Not yet in group. Latest 1.43.3; requires `numpy>=1.26`, `torch>=2.1.1` (both satisfied) |
| jaxtyping | Not yet in group. Latest 0.3.11, dependency-light |
| plotly, ipywidgets, pynvml | Not in core; pynvml 12.0.0 is in the training group — repeat it in the interp group (groups may overlap) |

Additions to the `interp` dependency group therefore: `sae-lens`, `nnterp`, `circuitsvis`,
`jaxtyping`, `plotly`, `ipywidgets`, `pynvml`. Then `uv lock` must be regenerated; the
existing pyproject note says lock regeneration is confirmed on the workstation, so the
resolved pins are provisional until `uv lock` runs there.

Risk noted in the design and still live: transformer-lens, nnsight, and sae-lens all move
fast. The lockfile turns any incompatibility into a loud `uv lock` failure rather than a
silent downgrade, which is the design's intent delivered by existing repository machinery.

## File inventory

New files:

- `docker/interp/Dockerfile.interp`
- `docker/interp/entrypoint.sh`
- `docker/interp/arena-requirements.txt` (provenance snapshot; see ARENA decision above)
- `docker-compose.interp.yaml`
- `scripts/interp/preflight.sh`
- `scripts/interp/smoke_test.py`
- `interp/notebooks/README.md` + `.gitkeep` (owner-work convention)
- `interp/miniprojects/common/{__init__.py,models.py,artifacts.py,pairs.py,judge.py,record.py}` (M3)
- `interp/miniprojects/artifacts.yaml` (M3, or per-project files — decide at implementation)
- `tests/interp/{conftest.py,test_pairs.py,test_judge.py}` (M3)
- `env.interp.example` (documented template, mirrors `env.example`)

Modified files:

- `pyproject.toml` (extend `interp` group), `uv.lock` (regenerate)
- `Makefile` (`##@ Interpretability` section)
- `.gitignore` (`.env.interp`)

## Acceptance checks and host-side split

Per AGENTS.md the sandbox cannot run Docker, the GPU, or services. The split:

Sandbox-executable: all file authoring, `uv lock` (network permitting), host-side unit tests
(`uv run pytest tests/interp`), shellcheck-style review of scripts, `git grep` secret scan.

Host-side (human runs, reports output back):

```bash
$ make interp-build
$ make interp-smoke          # all seven steps, HF_TOKEN set
$ make interp-up             # then open the printed URL, run a notebook cell
$ docker compose up -d llama-gpu && make interp-up   # preflight must refuse
$ docker compose stop llama-gpu
$ make test                  # host-side unit tests
$ git status                 # no untracked heavy artifacts after a session
```

## Open questions

1. `read_only: true` with tmpfs home versus the design's writable-rootfs deviation: attempt
   strict first (recommended), or take the design's deviation as settled?
2. ARENA requirements: fold into the `interp` group (recommended) or keep a separately
   installed vendored file?
3. Should I1 and M3 land as one change or two? M3 depends on the workbench for its GPU check
   but its host-side modules and tests are independent; two commits on one branch is the
   natural split.
4. `env_file: .env.interp` marked `required: false` (buildable without secrets) versus
   required (fails loudly when missing)?
