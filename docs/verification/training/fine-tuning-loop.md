# Fine-tuning loop verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 2 verified, 0 blocked, 14 pending, of 16.**

The fine-tuning loop is the first rung of the training tier and the loop the
lab runs: train a Gemma 4 E2B with a known change, then recover the change
from the weights and activations with the rust harness and the interp tier.
Every entry here proves one capability of that loop through the tools that
consume its output, not in isolation: the artifact under test is the merged
Hugging Face directory (`config.json` plus a single `model.safetensors`) that
the harness loads and that interp captures from, identified everywhere by
`sha256(model.safetensors)`. The training tier produces no GGUF; the one
GGUF this loop names is the published judge on `llama-cpu`, which is
llama.cpp's artifact and stays one.

This runbook replaced the three phase runbooks on 2026-09-18. Those described
the tier in build order, took GGUF on llama-gpu as their evidence, and named
Qwen3 as the fallback family; they are frozen at
[archive/training-phase0.md](../archive/training-phase0.md),
[archive/training-phase1.md](../archive/training-phase1.md) and
[archive/training-phase2.md](../archive/training-phase2.md). The substrate
facts they verified on hardware (P0.1 shards, P0.2 tokenizer, P0.3
calibration, P0.5 resume) are cited by FT.0 and are not re-run here. The
from-scratch loop's runbook,
[archive/from-scratch-loop.md](../archive/from-scratch-loop.md), opened and
closed on 2026-09-19 with the Phase 0 acceptance checks on the reference
loop, and lives on as `tests/hardware/test_from_scratch_loop.py`; the
continual-pretraining loop gets its runbook
when its first entry can be written, since a runbook with zero entries is a
lie to the counts gate.

Three sessions build and run this loop, and each entry names the one that
builds what it verifies:

- **The runbook PR** (the first session) wrote the runbook, pinned every
  Gemma 4 recipe to `schemas/gemma4_facts.json`, and removed the Qwen3 family.
- **The exporter PR** (the second session) built the merged HF exporter that
  FT.4 verifies, retired GGUF production from the training tier, moved the
  `eval-ft` judge phase and the organism verifier onto the harness for FT.6
  and FT.7, and added the `hw_full` hardware test FT.11 runs. Every command
  below names a verb, flag or make target that exists on its branch.
- **The hardware campaign** (the third session) executes FT.0 through FT.16
  (FT.15 was removed with scorecard promotion on 2026-09-19)
  in the section order below and records the output.
- **The coverage audit** (2026-09-18) checked every entry against
  `make/training.mk`, the two compose files and the CLI parsers, and added
  FT.12 through FT.14 for the targets no entry ran (`train-build`,
  `train-smoke`, `train-preflight`, `train-validate`, `memfit`). It found two things every container entry now
  carries. First, KB-79: every recipe names the base as the hub id
  `google/gemma-4-E2B`, which the training container cannot resolve (its
  HF cache is an empty tmpfs and `train-download` stages a plain directory),
  so every `train-*`, `memfit`, `train-preflight` and organism training
  command below passes `model.hf.id=/app/models/hf/gemma-4-E2B` as an
  override. The recipe as committed is the defect, and FT.13 reproduces it.
  Second, the training container mounts the models tree at `/app/models`
  only, so a container-run command names that spelling (FT.5), and `eval-ft`
  now translates the host spelling itself (KB-78, FT.6).

Every `make` target here follows the same dispatch chain: the `Makefile`
runs `docker compose -f docker-compose.training.yaml run --rm training
python -m training <subcommand>`, which enters `training/__main__.py`,
dispatches through the argparse handlers in `training/cli.py`, and lands
in the implementation module. Interp targets run `python -m interp` in the
workbench container or on the host, as `make/interp.mk` states per target;
harness targets run the `harness` binary in the rust-harness container.
Each entry lists its implementation files so a failure can be read at the
source without re-tracing the chain, and names the entry it depends on so a
FAILED entry says what to fix.

One GPU consumer at a time: no entry runs while another GPU job, benchmark,
or inference service the entry does not name is active.

## Table of Contents

- [FT.0 Preconditions](#ft0-preconditions)
- [FT.12 Training image and smoke](#ft12-training-image-and-smoke)
- [FT.13 Recipe gates: preflight and validate](#ft13-recipe-gates-preflight-and-validate)
- [FT.14 Memory fit: fits, does not fit, solve](#ft14-memory-fit-fits-does-not-fit-solve)
- [FT.1 Calibrate the QLoRA recipe (KB-03)](#ft1-calibrate-the-qlora-recipe-kb-03)
- [FT.16 The E4B recipe: preflight, fit and calibration](#ft16-the-e4b-recipe-preflight-fit-and-calibration)
- [FT.2 Prepare the smoke SFT dataset](#ft2-prepare-the-smoke-sft-dataset)
- [FT.3 Train, interrupt, resume](#ft3-train-interrupt-resume)
- [FT.4 Export the merged HF directory and load it in the harness](#ft4-export-the-merged-hf-directory-and-load-it-in-the-harness)
- [FT.5 Held-out loss and lm-eval on base and merged](#ft5-held-out-loss-and-lm-eval-on-base-and-merged)
- [FT.6 eval-ft with both sides identified](#ft6-eval-ft-with-both-sides-identified)
- [FT.7 Organisms: null and taboo](#ft7-organisms-null-and-taboo)
- [FT.8 Activation capture on base and organism](#ft8-activation-capture-on-base-and-organism)
- [FT.9 Probe train, export, and score in the harness](#ft9-probe-train-export-and-score-in-the-harness)
- [FT.10 Steering vector from the organism capture](#ft10-steering-vector-from-the-organism-capture)
- [FT.11 The hardware test](#ft11-the-hardware-test)

## FT.0 Preconditions

Implementation: `training/commands/download.py` (`make train-download`),
`rust-harness/harness/src/preflight.rs` (`--status`), `common/hashing.py`
and `interp/probes/capture.py` (`derive_model_hash`), `schemas/gemma4_facts.json`
(the pinned revision).

Builds on the archived substrate facts, which this runbook does not re-run:
P0.1 (the `verify-small` shard set), P0.2 (tokenizer training), P0.3 (the
scratch memory-fit calibration) and P0.5 (tolerance-based resume through
`tests/hardware/test_gpu.py`), all VERIFIED in
[archive/training-phase0.md](../archive/training-phase0.md). Built by this
PR (the pin); depends on nothing.

Command (the token file is `.env.training`, created from
`env.training.example` with a read-scope `HF_TOKEN` whose account has
accepted the Gemma license on the `google/gemma-4-E2B` model page; the
revision is the E2B commit in `schemas/gemma4_facts.json`, which every
recipe pins):

```bash
test -s .env.training && grep -c '^HF_TOKEN=' .env.training
make train-download MODEL=google/gemma-4-E2B REVISION=d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f
ls /mnt/ai-data/models/hf/gemma-4-E2B/
sha256sum /mnt/ai-data/models/hf/gemma-4-E2B/model.safetensors
make harness-run ARGS="--model /app/models/hf/gemma-4-E2B --text-only --status --deep"
```

Expected signal: the token line count is 1; `train-download` exits 0 and
stages `config.json`, `model.safetensors` and the tokenizer files under
`/mnt/ai-data/models/hf/gemma-4-E2B/`; `sha256sum` prints the base
checkpoint's digest, which is the `model_hash` every later entry compares
against (record it here); `harness --status` exits 0 with
`[OK] backend compatibility: supported` and a config line naming 35 layers,
and reports `--deep` as inert because no probe or control vector is loaded
(the flag hashes the weights only against an artifact's recorded hash, which
FT.4 and FT.9 exercise).

Observed (2026-09-19, two pastes; the second was asked for because the first
line printed nothing):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
test -s .env.training && grep -c '^HF_TOKEN=' .env.training
make train-download MODEL=google/gemma-4-E2B REVISION=d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f
ls /mnt/ai-data/models/hf/gemma-4-E2B/
sha256sum /mnt/ai-data/models/hf/gemma-4-E2B/model.safetensors
make harness-run ARGS="--model /app/models/hf/gemma-4-E2B --text-only --status --deep"
docker compose -f docker-compose.training.yaml run --rm prepare python -m training download --model google/gemma-4-E2B --revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f 
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-prepare-run-f9403b54f127 Creating 
Container ai-experiments-prepare-run-f9403b54f127 Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
[WARN] nvidia-smi not found: no GPU visible in this container

[RUN] exec: python -m training download --model google/gemma-4-E2B --revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f
=== Staging a model ===

[INFO] Model: google/gemma-4-E2B
[INFO] Revision: d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f
[INFO] Destination: /mnt/ai-data/models/hf/gemma-4-E2B

/opt/venv/lib/python3.12/site-packages/huggingface_hub/constants.py:298: FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead to enable high performance transfer with Xet. Visit https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance for more details.
  warnings.warn(
[RUN] Downloading google/gemma-4-E2B
Fetching 8 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:09<00:00,  1.23s/it]
Download complete: : █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.1kB            [OK] Downloaded to /mnt/ai-data/models/hf/gemma-4-E2B█████████████████████████████████████████████████████████████████████████████████████████████████| 31.1kB / 31.1kB            

Set this in the recipe so the run is reproducible:
  model:
    hf:
      id: /mnt/ai-data/models/hf/gemma-4-E2B
      revision: d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f
Download complete: : █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.1kB            
Reconstruction complete: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31.1kB / 31.1kB            
config.json  generation_config.json  model.safetensors  processor_config.json  README.md  tokenizer_config.json  tokenizer.json
76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57  /mnt/ai-data/models/hf/gemma-4-E2B/model.safetensors
docker compose -f docker-compose.rust-harness.yaml run --rm harness --model /app/models/hf/gemma-4-E2B --text-only --status --deep
WARN[0000] Found orphan containers (llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-harness-run-b4af61af0790 Creating 
Container ai-experiments-harness-run-b4af61af0790 Created 
[WARN] --deep: no effect when there is no interp artifact to check
[OK] model directory: /app/models/hf/gemma-4-E2B
[OK] checkpoint files: config.json and model.safetensors present, header names 2011 tensors
[OK] config.json: 35 layers, hidden size 1536, context 131072
[OK] backend compatibility: supported
[SKIP] probe: no --probe given
[SKIP] probe geometry: no --probe given
[SKIP] probe identity: no --probe given
[SKIP] control vector: no --control-vector given
[SKIP] control vector identity: no --control-vector given
[SKIP] interp log directory: no --interp-log given
[WARN] obs log directory: /mnt/ai-data/logs/harness cannot be created: /mnt is not writable: Read-only file system (os error 30); run records degrade to a no-op
➜  ai-experiments git:(SFT-verification) ✗ cd ~/workspace/ai-experiments
test -s .env.training; echo "test=$?"
ls -la .env.training env.training.example 2>&1
ls -la --time-style=long-iso /mnt/ai-data/models/hf/gemma-4-E2B/
ls /mnt/ai-data/cache/huggingface/hub/ 2>&1 | head
make harness-run ARGS="--model /app/models/hf/gemma-4-E2B --text-only --status" > /dev/null 2>&1; echo "status exit=$?"
test=1
ls: cannot access '.env.training': No such file or directory
-rw-rw-r-- 1 alejandro alejandro 968 Jul 17 06:08 env.training.example
total 10037956
drwxrwxr-x 3 alejandro alejandro        4096 2026-09-19 11:12 .
drwxr-xr-x 6 alejandro alejandro        4096 2026-09-05 08:57 ..
drwxrwxr-x 3 alejandro alejandro        4096 2026-08-04 19:33 .cache
-rw-rw-r-- 1 alejandro alejandro        4914 2026-08-04 19:33 config.json
-rw-rw-r-- 1 alejandro alejandro         181 2026-09-19 11:12 generation_config.json
-rw-r--r-- 1 alejandro alejandro        1570 2026-09-19 11:12 .gitattributes
-rw-rw-r-- 1 alejandro alejandro 10246621918 2026-08-04 17:07 model.safetensors
-rw-r--r-- 1 alejandro alejandro        1689 2026-09-19 11:12 processor_config.json
-rw-r--r-- 1 alejandro alejandro       27811 2026-09-19 11:12 README.md
-rw-rw-r-- 1 alejandro alejandro         906 2026-09-19 11:12 tokenizer_config.json
-rw-rw-r-- 1 alejandro alejandro    32170070 2026-08-04 16:25 tokenizer.json
CACHEDIR.TAG
datasets--monology--pile-uncopyrighted
models--ggml-org--gemma-4-E2B-it-GGUF
models--google--gemma-4-E2B
models--google--gemma-4-E2B-it
models--google--gemma-4-E4B
models--HuggingFaceTB--SmolLM2-135M
models--Qwen--Qwen3-1.7B
status exit=0
```

The download, the listing, the digest, and the preflight (exit 0, backend
supported, 35 layers, `--deep` reported inert) all match. The base digest
every later entry compares against is
`76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57`.

One clause was not observed and is recorded as a finding rather than edited
away: the token line printed nothing, because the token file does not exist
on the workstation. The download still exited 0, moving 31 kB: the 10 GB
`model.safetensors`, `tokenizer.json` and `config.json` date from
2026-08-04, and the hub cache under `/mnt/ai-data/cache/huggingface/hub/`
already held `models--google--gemma-4-E2B`, so the gated fetch was served
from cache and only the four small files (generation and processor configs,
README, tokenizer config) were new. This entry therefore proves the pinned
base is staged and servable, which is what the rest of the runbook needs,
and does not prove a cold gated download with the token; that path is
unverified here, by the operator's decision on 2026-09-19. The closing
obs-log `[WARN]` is by design: the compose file's own comment says the
`harness` service does not mount the log root, so run records degrade to
their no-op in the container.

Status: VERIFIED (2026-09-19)

## FT.12 Training image and smoke

Implementation: `make/training.mk` (`train-build`, `train-smoke`),
`docker/training/Dockerfile.training`, `training/commands/smoke_test.py`.

The image every later entry runs in, and the smoke that proves its CUDA
build on the workstation. Added 2026-09-18: no entry built the training
image, and the README's standing rule (any dependency-group change means
`uv lock`, a rebuild, and that image's smoke recorded in the tier runbook)
had no entry to record into. Existing code; depends on nothing but the
lockfile.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
uv lock --check; echo "lock exit=$?"
make train-build; echo "exit=$?"
make train-smoke; echo "exit=$?"
```

Expected signal: `uv lock --check` exits 0 (a stale lock fails the
Dockerfile's `--frozen` sync instead); the build ends `Training image
built.` with exit 0; the smoke prints `[OK]` for every step, among them the
torch line naming the RTX 5090 at sm_120, `flash-attn: flash_attn_func
forward`, `bitsandbytes: 4-bit Linear forward`, `deepspeed: DeepSpeedCPUAdam
step (pre-built op loaded)`, then `[OK]
ALL SMOKE STEPS PASSED` with no `[SKIP]` line and exit 0.

Recorded 2026-09-19 in two pastes: `make train-build` from the second attempt
after the flash-attn build fix (`c6ebcce`), then `uv lock --check` and `make
train-smoke` from the same tree. The two bitsandbytes warnings under the
4-bit step are the library's own, about a deprecated torch check and the
default fp32 compute dtype for an fp16 input, and the step still passes.

Observed:

```
➜  ai-experiments git:(SFT-verification) ✗ make train-build
Building training image...
MAX_JOBS=4 NVCC_THREADS=2 docker compose -f docker-compose.training.yaml build
[+] Building 2296.6s (39/39) FINISHED
 => [internal] load local bake definitions                                                                                                                                    0.0s
 => => reading from stdin 1.71kB                                                                                                                                              0.0s
 => [training internal] load build definition from Dockerfile.training                                                                                                        0.0s
 => => transferring dockerfile: 6.50kB                                                                                                                                        0.0s
 => [export internal] load build definition from Dockerfile.export                                                                                                            0.0s
 => => transferring dockerfile: 2.50kB                                                                                                                                        0.0s
 => [training internal] load metadata for docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04                                                                                0.5s
 => [training internal] load metadata for ghcr.io/astral-sh/uv:0.11.28                                                                                                        0.2s
 => [export internal] load metadata for docker.io/library/ubuntu:24.04                                                                                                        0.5s
 => [export internal] load .dockerignore                                                                                                                                      0.0s
 => => transferring context: 1.73kB                                                                                                                                           0.0s
 => [training internal] load build context                                                                                                                                    0.0s
 => => transferring context: 6.97kB                                                                                                                                           0.0s
 => [prepare stage-1  1/15] FROM docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01                 0.0s
 => [export uv 1/1] FROM ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa                                                 0.0s
 => CACHED [training stage-1  2/15] RUN apt-get update &&     apt-get install -y --no-install-recommends --allow-change-held-packages         build-essential         gcc g+  0.0s
 => CACHED [training stage-1  3/15] COPY --from=uv /uv /uvx /usr/local/bin/                                                                                                   0.0s
 => CACHED [training stage-1  4/15] RUN uv venv /opt/venv --python 3.12                                                                                                       0.0s
 => CACHED [training stage-1  5/15] COPY pyproject.toml uv.lock ./                                                                                                            0.0s
 => [training stage-1  6/15] RUN --mount=type=cache,target=/root/.cache/uv     MAX_JOBS=4 NVCC_THREADS=2     DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1     CFLAGS="-march=nati  2276.9s
 => [export stage-1 1/9] FROM docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3                                          0.0s
 => CACHED [export stage-1 2/9] RUN apt-get update &&     apt-get install -y --no-install-recommends         git         ca-certificates         python3.12         python3.  0.0s
 => CACHED [export stage-1 3/9] COPY --from=uv /uv /uvx /usr/local/bin/                                                                                                       0.0s
 => CACHED [export stage-1 4/9] RUN uv venv /opt/venv --python 3.12                                                                                                           0.0s
 => CACHED [export stage-1 5/9] RUN --mount=type=cache,target=/root/.cache/uv     uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 &&     uv pi  0.0s
 => CACHED [export stage-1 6/9] RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser                                       0.0s
 => CACHED [export stage-1 7/9] RUN mkdir -p /app/cache /app/logs && chown -R appuser:appuser /app                                                                            0.0s
 => CACHED [export stage-1 8/9] WORKDIR /app                                                                                                                                  0.0s
 => CACHED [export stage-1 9/9] RUN python -c "import torch, transformers, peft, safetensors"                                                                                 0.0s
 => [export] exporting to image                                                                                                                                               0.0s
 => => exporting layers                                                                                                                                                       0.0s
 => => writing image sha256:fb7e7363b4fb02b3b5a582c4a4c2d573961d4a95accf1c57c3485a695e4986f4                                                                                  0.0s
 => => naming to docker.io/library/ai-experiments-export                                                                                                                      0.0s
 => [export] resolving provenance for metadata file                                                                                                                           0.0s
 => [prepare stage-1  7/15] RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser                                           0.2s
 => [training stage-1  8/15] RUN mkdir -p /app/cache /app/logs && chown -R appuser:appuser /app                                                                               0.1s
 => [prepare stage-1  9/15] WORKDIR /app                                                                                                                                      0.0s
 => [training stage-1 10/15] COPY --chown=appuser:appuser common ./common                                                                                                     0.0s
 => [prepare stage-1 11/15] COPY --chown=appuser:appuser training ./training                                                                                                  0.0s
 => [training stage-1 12/15] COPY --chown=appuser:appuser scripts/training ./scripts/training                                                                                 0.0s
 => [prepare stage-1 13/15] COPY --chown=appuser:appuser docker/training/entrypoint.sh /app/entrypoint.sh                                                                     0.0s
 => [training stage-1 14/15] RUN chmod +x /app/entrypoint.sh                                                                                                                  0.1s
 => [prepare stage-1 15/15] RUN python -m training smoke --build                                                                                                             11.7s
 => [prepare] exporting to image                                                                                                                                              7.0s
 => => exporting layers                                                                                                                                                       6.9s
 => => writing image sha256:ea88aa32ce3f8dfaec43bfbabf710b20b0c4dc58de6f980a81d323fa1418eb39                                                                                  0.0s
 => => naming to docker.io/library/ai-experiments-prepare                                                                                                                     0.0s
 => [training] exporting to image                                                                                                                                             7.0s
 => => exporting layers                                                                                                                                                       6.9s
 => => writing image sha256:f509b68ddf17fbaa5d8dadf09ddd49272cb2a086cdf905184e1c14533a291c5f                                                                                  0.0s
 => => naming to docker.io/library/ai-experiments-training                                                                                                                    0.0s
 => [prepare] resolving provenance for metadata file                                                                                                                          0.0s
 => [training] resolving provenance for metadata file                                                                                                                         0.0s
[+] build 3/3
 ✔ Image ai-experiments-prepare  Built                                                                                                                                      2296.6s
 ✔ Image ai-experiments-export   Built                                                                                                                                      2296.6s
 ✔ Image ai-experiments-training Built                                                                                                                                      2296.6s
Training image built.
➜  ai-experiments git:(SFT-verification) ✗ cd ~/workspace/ai-experiments
uv lock --check; echo "lock exit=$?"
make train-smoke; echo "exit=$?"
Resolved 330 packages in 0.55ms
lock exit=0
docker compose -f docker-compose.training.yaml run --rm training python -m training smoke
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-316a3c679247 Creating 
Container ai-experiments-training-run-316a3c679247 Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training smoke
=== Training stack smoke test ===

[OK] torch 2.13.0+cu130: CUDA available, NVIDIA GeForce RTX 5090 sm_120
[OK] import transformers: 5.13.1
[OK] import trl: 0.24.0
[OK] import peft: 0.19.1
[OK] import datasets: 5.0.0
[OK] import accelerate: 1.14.0
[OK] import hydra: 1.3.2
[OK] import safetensors: 0.8.0
[OK] import pynvml: 12.0.0
[OK] flash-attn: flash_attn_func forward
/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
/opt/venv/lib/python3.12/site-packages/bitsandbytes/nn/modules.py:507: UserWarning: Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.
  warnings.warn(
[OK] bitsandbytes: 4-bit Linear forward
[OK] deepspeed: DeepSpeedCPUAdam step (pre-built op loaded)

Versions manifest:
{
  "packages": {
    "torch": "2.13.0+cu130",
    "transformers": "5.13.1",
    "trl": "0.24.0",
    "peft": "0.19.1",
    "datasets": "5.0.0",
    "accelerate": "1.14.0",
    "hydra": "1.3.2",
    "safetensors": "0.8.0",
    "pynvml": "12.0.0"
  },
  "cuda": "13.0",
  "gpu": "NVIDIA GeForce RTX 5090",
  "capability": [
    12,
    0
  ],
  "driver": "580.173.02"
}
[OK] manifest written: /mnt/ai-data/training/runs/_smoke/versions.json

[OK] ALL SMOKE STEPS PASSED
exit=0
```

Status: VERIFIED (2026-09-19)

## FT.13 Recipe gates: preflight and validate

Implementation: `training/commands/preflight.py`,
`training/commands/validate_recipe.py`, `training/recipes/validate.py`,
`training/export/preflight.py` (`resolve_base_identity`, which preflight
calls with `local_files_only`).

The two gates a run passes before the GPU is committed, each run twice:
once to pass and once to refuse, since a gate whose refusal has never been
seen is not known to gate. The first `train-preflight` runs the recipe as
committed and is the KB-79 reproduction: it must fail, and its failure is
what confirms the ledger row. Added 2026-09-18; depends on FT.0 and FT.2.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train-preflight CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make train-preflight CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B"; echo "exit=$?"
make train-validate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make train-validate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="optimizer.lr=-1"; echo "exit=$?"
```

Expected signal: the committed recipe's preflight exits 1 with `[ERROR]
Missing or unusable assets: <n>`, one problem line beginning `- model.hf:`
and naming `google/gemma-4-E2B`, then `Solution: stage the missing assets,
then re-run preflight` and the remedy `make train-download MODEL=<hub id>
REVISION=<sha>` (which FT.0 already ran, and which cannot satisfy a hub id:
that is KB-79). Any other problem line is a second finding, recorded on its
own. With the override it exits 0 with `[OK] All referenced assets are
present and verified`. `train-validate` on the recipe prints
`[OK] Schema: training/recipes/reference/sft-qlora-e2b.yaml loads as a typed
recipe` and `[OK] Validation: run sft-qlora-e2b passes all cross-field
invariants`, exit 0; with `optimizer.lr=-1` it prints `[ERROR] Recipe
validation failed:` and `  - optimizer.lr: must be positive, got -1.0`, exit 1. A
validate that exits 0 on the negative learning rate is a gate that does not
gate.

Observed:

```
```

Status: PENDING

## FT.14 Memory fit: fits, does not fit, solve

Implementation: `training/cli.py` (`cmd_memfit`, `_print_estimate_table`),
`training/calculator/` (`estimate`, `solve`), `training/tune/sft.py`
(`check_memory_fit`, the same estimate `train-sft` refuses on).

The estimate FT.1 calibrates, run as an operator runs it: once on the recipe,
which must fit, once on a geometry that must not, and once solving for the
largest batch. `train-sft` refuses a recipe the estimate says does not fit
unless `--force` is passed, so the refusal here is the refusal FT.3 relies
on. CPU-only, no GPU; added 2026-09-18; depends on FT.12.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B"; echo "exit=$?"
make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B data.micro_batch=64"; echo "exit=$?"
make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B --solve batch"; echo "exit=$?"
```

Expected signal: the first run prints the component table, `Total reserved
estimate:` and `[OK] Fits: reserved estimate <n> GB within budget <b> GB (32
GB minus safety margin)`, exit 0. The second prints `[ERROR] Does not fit:
reserved estimate <m> GB exceeds budget <b> GB` with `<m>` above `<b>`, the
two `Solution:` lines, and exit 1. The third prints `[OK] Solve batch: max
micro-batch <k> (reserved estimate <r> GB)` with `<k>` at least the recipe's
4 and below 64, and exit 0. The budget `<b>` is the same figure in all three
runs. Record `<n>` here: FT.1 compares the calibrated peak against it.

Observed:

```
```

Status: PENDING

## FT.1 Calibrate the QLoRA recipe (KB-03)

Implementation: `training/calculator/calibrate.py`,
`training/models/model.py`, `training/models/peft_utils.py`,
`training/cli.py` (`cmd_calibrate`).

**Fix to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-03 — `calibrate()` once built a scratch Llama model for every recipe, so
  calibrating a `model.hf` or PEFT recipe measured a model other than the one
  the estimate priced. The fix routes calibration through the shared
  `build_model` and the PEFT wrap; this run on the QLoRA recipe is the
  hardware evidence that closes the row.

Calibrates the memory-fit calculator on the `model.hf` plus PEFT path, which
the archived P0.3 scratch calibration does not exercise: NF4 base pricing,
adapter gradients and optimizer state. It is also the first execution
evidence for Gemma 4 on the workstation: the gated checkpoint at its pinned
revision, transformers `gemma4` support, and bitsandbytes NF4 on Blackwell.
Existing code; depends on FT.0.

Command:

```bash
make train-calibrate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"
```

The `model.hf.id` override is KB-79 (see the preamble); without it the
container cannot find the base. Depends on FT.0, FT.12 and FT.14.

Expected signal: exit 0; the record reports the allocated estimate within
the 10 percent bar (or the 128MiB floor), the reserved estimate covering the
NVML peak, and no parameter-count or trainable-count drift warning. Drift
beyond the bar is a calculator finding (candidates: NF4 constant pricing,
paged optimizer states) to diagnose before touching calibrated constants.

Observed:

```
```

Status: PENDING

## FT.16 The E4B recipe: preflight, fit and calibration

Implementation: `training/commands/preflight.py`, `training/cli.py`
(`cmd_memfit`, `cmd_calibrate`), `training/calculator/calibrate.py`,
`training/recipes/reference/sft-qlora-gemma-e4b.yaml`.

E2B and E4B are the two subjects this repository trains, and FT.13, FT.14
and FT.1 run on E2B alone. The E4B recipe is the same QLoRA shape at 4096
context and a larger base, which is a different point on the calculator:
the 10 percent bar has not been checked there, and the recipe's own
comment says so. Same three gates as the E2B entries, same KB-79 override,
with E4B's staged directory. Depends on the E4B download (I.5 and I.6 use
`/app/models/hf/gemma-4-E4B`, so it is staged) and on FT.12.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
R=training/recipes/reference/sft-qlora-gemma-e4b.yaml
O=model.hf.id=/app/models/hf/gemma-4-E4B
make train-preflight CONFIG=$R ARGS="$O"; echo "exit=$?"
make memfit CONFIG=$R ARGS="$O"; echo "exit=$?"
make train-calibrate CONFIG=$R ARGS="$O --out /mnt/ai-data/training/verify/calibrate-sft-e4b.json"
echo "exit=$?"
python3 -c 'import json; r=json.load(open("/mnt/ai-data/training/verify/calibrate-sft-e4b.json")); print({k: r[k] for k in sorted(r) if "error" in k or "peak" in k or "fits" in k})'
```

Expected signal: preflight exits 0 with `[OK] All referenced assets are
present and verified` (a `sft.dataset` problem means the E4B split named in
the recipe has not been prepared; prepare it as FT.2 did with
`--tokenizer /mnt/ai-data/models/hf/gemma-4-E4B --max-seq-len 4096` and the
recipe's `--out`, and record that here). `memfit` prints `[OK] Fits:` with
the reserved estimate below the same budget FT.14 printed, and exit 0.
`train-calibrate` exits 0 with the allocated estimate within the 10 percent
bar (or the 128 MiB floor) and the reserved estimate covering the NVML peak,
as FT.1's bar states; record the allocated and measured figures beside
FT.1's, since the two points together are what say whether the calculator
scales with the base rather than fitting one recipe. Drift beyond the bar
on E4B alone, with E2B inside it, is a calculator finding about a term that
grows with the base (the nf4 constant pricing, the PLE tables) and is filed
before any constant is touched.

Observed:

```
```

Status: PENDING

## FT.2 Prepare the smoke SFT dataset

Implementation: `training/commands/prepare_sft.py`, `training/data/sft.py`,
`training/data/tokenizer.py`, `training/models/families.py`
(`CHAT_TEMPLATE_VERSION`, pinned against the harness renderer by
`schemas/chat_template.json`).

The prepared directory is the `sft.dataset` the smoke recipe names. Existing
code; depends on FT.0.

Command:

```bash
make prepare-sft ARGS="--source HuggingFaceH4/ultrachat_200k --split train_sft \
    --tokenizer /mnt/ai-data/models/hf/gemma-4-E2B --model-type gemma4 \
    --max-seq-len 2048 --out /mnt/ai-data/training/datasets/sft/ultrachat-e2b"
cat /mnt/ai-data/training/datasets/sft/ultrachat-e2b/manifest.json
```

Expected signal: exit 0; the output directory holds the prepared examples
plus `manifest.json` recording the tokenizer hash, `chat_template_version`
2, the split sizes and the supervised-token counts (assistant content plus
the end-of-turn token only); a source record carrying a raw reasoning marker
is refused, not masked.

Observed:

```
```

Status: PENDING

## FT.3 Train, interrupt, resume

Implementation: `training/cli.py` (`cmd_train_sft`), `training/tune/sft.py`
(`resolve_sft_resume`, `check_memory_fit`), `training/tune/sft_callbacks.py`,
`training/runtime/checkpoint.py`, `training/runtime/lock.py` (the D9 GPU
lock), `training/models/peft_utils.py`.

Three runs of the same recipe: a short straight run, a run interrupted after
its first checkpoint, and the resume of that run from the named
`checkpoint-<step>` directory. Existing code; depends on FT.2 and FT.14.

Two overrides beyond the step counts, each because the recipe's own value
would leave a criterion unobservable. `sft.eval_steps=30`: the recipe
evaluates every 100 steps, so a 60-step run would never compute a held-out
loss and the `Held-out loss:` line the tolerance criterion reads would be
absent. `model.hf.id=/app/models/hf/gemma-4-E2B`: KB-79.

Command (inference down; the interrupt is a Ctrl-C after the first
`checkpoint-30` directory appears and before `checkpoint-60` does):

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B run.name=ft3-straight schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30"
echo "exit=$?"
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B run.name=ft3-resume schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
cat /mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30/meta.json
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B run.name=ft3-resume schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30 \
    checkpoint.resume=/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30"
echo "exit=$?"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
```

The checkpoints sit directly under `<output_root>/<run.name>/checkpoints/`
(`training/tune/sft.py`); the timestamped directory beside them holds the
recipe snapshot and the run's sinks, not the weights.

Expected signal, per run. The straight run prints `[RUN] Fine-tuning
ft3-straight [trl]: 60 total steps`, then `[OK] Memory fit:` (the recipe
fits without `--force`), then `[OK] Run complete: step 60 [trl]`, a `Final
train loss:` line, a `Held-out loss:` line and exit 0; `checkpoint-30` and
`checkpoint-60` each carry `meta.json` with `checkpoint_format: trl`,
`adapter_only: true`, the base identity including the pinned
`base_revision` `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, and the resolved
`attention_implementation`. The interrupted run leaves `checkpoint-30`
complete (`trainer_state.json` and `meta.json` present) and no
`checkpoint-60`. The resumed run prints `[RUN] Resuming (<the checkpoint-30
path>) ft3-resume [trl]: 60 total steps`, ends at `[OK] Run complete: step
60 [trl]` with exit 0, and its `ls` shows `checkpoint-60` beside
`checkpoint-30`. The criterion that makes the resume a resume: the resumed
run's `Held-out loss:` agrees with the straight run's within
`RESUME_LOSS_RTOL` (0.01, `tests/hardware/test_gpu.py`), that is
`abs(a - b) / a <= 0.01`; record both numbers and the quotient. A resumed
run that reports `Fine-tuning` rather than `Resuming`, or ends at a step
other than 60, has restarted rather than resumed.

Observed:

```
```

Status: PENDING

## FT.4 Export the merged HF directory and load it in the harness

Implementation: `training/cli.py` (`cmd_export`), `training/export/merge.py`
(`merge_and_unload` into `hf.partial/`, promoted by rename),
`training/export/preflight.py` (base-drift refusal),
`training/export/consolidate.py` (`check_single_weights_file`),
`training/export/manifest.py` (`build_hf_manifest`, `model_hash`),
`schemas/hf_export.manifest.json`; `rust-harness/runtime/src/bin/checkpoint-info.rs`
(`--check`), `rust-harness/harness/src/preflight.rs` (`--status --deep`).
The export runs in the CPU-only export image (`make export-build` once) and
writes `/mnt/ai-data/models/trained/<name>/hf/` plus the manifest beside it;
the sandbox proves the copy path and both harness verdicts on the tiny
fixture (`tests/training/test_export_harness.py`), and this entry proves the
merge on real weights. Depends on FT.3.

Command (the export container mounts the models tree read-write under
`/app/models/trained`; the harness lines read the result through the
harness image, and `checkpoint-info` reads it on the host):

```bash
make export-build
make export-hf CHECKPOINT=/mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 \
    ARGS="--name ft3-straight"
ls /mnt/ai-data/models/trained/ft3-straight/hf/
grep -E '"(mode|hf_dir|model_hash)"' /mnt/ai-data/models/trained/ft3-straight/manifest.json
sha256sum /mnt/ai-data/models/trained/ft3-straight/hf/model.safetensors
make export-hf CHECKPOINT=/mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 \
    ARGS="--name ft3-straight"; echo "exit $?"
make harness-run ARGS="--model /app/models/trained/ft3-straight/hf --text-only --status --deep"
cd rust-harness && cargo run --release -p runtime --bin checkpoint-info -- --check --text-only \
    /mnt/ai-data/models/trained/ft3-straight/hf 2>&1 | tail -3; cd ..
make harness-chat ARGS="--model /app/models/trained/ft3-straight/hf --text-only"
```

Expected signal: the first export prints `[OK] Base model verified:
google/gemma-4-E2B (revision d29ff6b4...)`, `[RUN] Merging model: ...`,
`[OK] Adapter merged: ...`, `[OK] HF export written: .../hf (N bytes)`,
`[OK] Manifest written: .../manifest.json` and a `model_hash:` line, and
exits 0 (a checkpoint whose recorded base revision does not match the cached
base is refused with `[ERROR] Export failed: Base model drifted since
training:` unless `--allow-base-drift` is passed); `ls` shows `config.json`,
`generation_config.json`, one `model.safetensors`, the tokenizer files and no
`model.safetensors.index.json`; the manifest records `"mode": "merge"`,
`"hf_dir"` as the listed directory, and a `model_hash` equal to the
`sha256sum` line and different from the FT.0 base digest; the second export
exits 1 with `[ERROR] Export exists: .../hf` and the `--force` solution line,
leaving the manifest unchanged; `harness --status --deep` and
`checkpoint-info --check` both exit 0 on the directory, with `[OK] backend
compatibility: supported`, the same 35-layer config as the base, and
`[WARN] --deep: no effect when there is no interp artifact to check` (the
hash comparison is FT.9's, against a probe); one `--chat` reply arrives,
rendered through the version 2 chat template.

Observed:

```
```

Status: PENDING

## FT.5 Held-out loss and lm-eval on base and merged

Implementation: `training/eval/holdout.py`, `training/eval/lmeval.py`,
`training/eval/tasks.yaml`, `common/reports.py`. Both verbs are CLI-only
(`python -m training eval-holdout`, `eval-lmeval`); no `make` target wraps
them, so they run inside the training container. Existing code; depends on
FT.4 (the merged directory) and FT.2 (the dataset).

Command (inference down). The models tree is mounted in the training
container at `/app/models` only (`docker-compose.training.yaml`), so both
sides are named in that spelling; the training volume is identity-mounted,
so the dataset and the output records keep their host paths. This block
named `/mnt/ai-data/models/...` until 2026-09-18, which does not exist in
the container.

```bash
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model /app/models/hf/gemma-4-E2B \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/ft5-holdout-base.json
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model /app/models/trained/ft3-straight/hf \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/ft5-holdout-merged.json
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/trained/ft3-straight/hf --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-merged.json
```

Expected signal: all four exit 0; each held-out record carries loss,
perplexity, the supervised-token count, `precision` and
`attention_implementation`, and the merged model's held-out loss is at or
below the base's on the dataset it trained on; each lm-eval record reports
the pinned task set from `training/eval/tasks.yaml` with `limit: 20`
recorded (a smoke run, not a benchmark).

Observed:

```
```

Status: PENDING

## FT.6 eval-ft with both sides identified

Implementation: `training/commands/eval_ft.py` (`container_model`,
`phase_container`), `training/eval/runner.py` (`hf_identity`,
`run_container_phase`), `training/eval/scorecard.py`. Depends on FT.4 (the
subject) and FT.5 (the container evaluators the same run repeats).

**Fix to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-78 — `phase_container` handed `--base` and `--tuned` to the training
  container verbatim, and the host spelling this block uses does not exist
  there (the models tree is mounted at `/app/models` only), so the holdout
  phase would have failed at model load. `container_model` in
  `training/commands/eval_ft.py` now translates it; the `[RUN]
  holdout/base: /app/models/hf/gemma-4-E2B` line below is the evidence.

Both compared sides are HF directories under `/mnt/ai-data/models/`: the base
as FT.0 staged it and the merged directory as FT.4 wrote it. `eval-ft`
identifies each by `model_hash` (read from FT.4's manifest for the tuned
side, computed for the base) and runs the two container evaluators on each.
A model-as-judge evaluator existed until 2026-09-19 and was removed: it
measured preference, and the harness-serving check it doubled as is FT.4's.
What this entry pins beyond FT.5 is the identity block: a scorecard that
names the exact weights on each side.

Command (host tool; inference down; the tool runs the container phases
itself):

```bash
make eval-ft ARGS="--base /mnt/ai-data/models/hf/gemma-4-E2B \
    --tuned /mnt/ai-data/models/trained/ft3-straight/hf \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --lmeval-limit 20 \
    --name ft6"
echo "exit=$?"
grep -E '"model_hash"|"thinking_mode"|"checkpoint"' /mnt/ai-data/training/eval/ft6_*/scorecard.json
cat /mnt/ai-data/training/eval/ft6_*/scorecard.md
```

Expected signal: exit 0; the container phases print `[RUN] holdout/base:
/app/models/hf/gemma-4-E2B` and `[RUN] holdout/tuned:
/app/models/trained/ft3-straight/hf` (the container spelling, KB-78) and
`[OK] holdout/<side>: loss <n>` for each, with the tuned loss at or below the
base's as FT.5 found, then one `[OK] lmeval/<task>: <a> -> <b>` per pinned
task; the run directory holds `scorecard.json` (shared report envelope,
`kind: scorecard`) and `scorecard.md`; the scorecard's metadata records
`thinking_mode: off` on both sides, `checkpoint` as each side's container
path, the base's `model_hash` equal to the FT.0 digest and the tuned side's
equal to FT.4's manifest, and the rendered table shows the two hash
prefixes. An interrupted run resumes from its phase records (`[SKIP]
holdout/<side>: record exists (resume)`).

Observed:

```
```

Status: PENDING

## FT.7 Organisms: null and taboo

Implementation: `interp/commands/organism.py`, `interp/organisms/generate.py`
(`write_elicitation_sets`), `interp/organisms/verify.py` (`locality_metrics`,
`organism_verdict`), `interp/organisms/seal.py` (`check_seal`),
`interp/steering/scorecard.py` (`post_with_timeout`); the pipeline hands off
to `prepare-sft`, `train-sft`, the export and `eval-ft`, which `make organism
RECIPE=...` prints as nine stages. The verifier elicits the subject through
the harness serving the exported directory (8005, the `--subject-url`
default) with a 600 s per-request timeout, judges on 8003 (`vllm-gpu`, the
`--judge-url` default), and reads locality from the `eval-holdout` and
`eval-lmeval` records; `seal` records the exported directory and the
`model_hash` from its manifest. Built by the exporter PR; depends on FT.4
and FT.5.

This entry supersedes I.8 in [interp/interp.md](../interp/interp.md), which
stays in that runbook as BLOCKED with this entry as its evidence. Two
organisms run, in this order, because the null control is what makes the
taboo result mean anything: a null organism whose implant check passes shows
the verifier measuring an artifact rather than the implant.

Services per stage, which the block states rather than assumes: generation
talks to the teacher on `vllm-gpu` (the `--service` default), so `make
gpu-up` precedes it and `make down` follows; training, export and the
container evaluators run with inference down; verification needs
`harness-serve` on the organism and `vllm-gpu` as the judge, which coexist
(the harness is a CPU consumer); seal and check-seal need nothing. The
teacher path is the model `vllm-gpu` serves, in container or host spelling.
The recipes train 400 steps at a checkpoint interval of 100, so the export
names `checkpoint-400`; locality is measured on the FT.2 ultrachat split and
the pinned lm-eval tasks, because the organism's own corpus would measure
the implant, not what it left alone. `model.hf.id=/app/models/hf/gemma-4-E2B` on `train-sft` is KB-79.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make organism RECIPE=training/recipes/organisms/null.yaml
make gpu-up
make organism-generate RECIPE=training/recipes/organisms/null.yaml TEACHER=/mnt/ai-data/models/<teacher>
echo "exit=$?"
make down
make prepare-sft ARGS="--source /mnt/ai-data/interp/organisms/organism-null/data-blind \
    --tokenizer /mnt/ai-data/models/hf/gemma-4-E2B --model-type gemma4 \
    --out /mnt/ai-data/training/datasets/sft/organism-null"
make train-sft CONFIG=training/recipes/organisms/null.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B"
make export-hf CHECKPOINT=/mnt/ai-data/training/runs/organism-null/checkpoints/checkpoint-400 \
    ARGS="--name organism-null"
make eval-ft ARGS="--base /mnt/ai-data/models/hf/gemma-4-E2B \
    --tuned /mnt/ai-data/models/trained/organism-null/hf --evaluators holdout,lmeval \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --lmeval-limit 20 \
    --name ft7-null"
make gpu-up
HARNESS_MODEL=/app/models/trained/organism-null/hf make harness-serve
python3 -c 'from common import serving; print(serving.wait_healthy(serving.SERVICES["harness"]))'
make organism-verify RECIPE=training/recipes/organisms/null.yaml \
    BASE_METRICS="/mnt/ai-data/training/eval/ft7-null_<UTC>/holdout_base.json /mnt/ai-data/training/eval/ft7-null_<UTC>/lmeval_base.json" \
    SUBJECT_METRICS="/mnt/ai-data/training/eval/ft7-null_<UTC>/holdout_tuned.json /mnt/ai-data/training/eval/ft7-null_<UTC>/lmeval_tuned.json"
echo "exit=$?"
make harness-serve-down
make down
make organism-seal RECIPE=training/recipes/organisms/null.yaml \
    ARGS="--checkpoint /mnt/ai-data/models/trained/organism-null/hf"
echo "exit=$?"
make organism-check-seal DIR=/mnt/ai-data/interp/organisms/organism-null
echo "exit=$?"
grep -E '"(checkpoint|model_hash)"' /mnt/ai-data/interp/organisms/organism-null/manifest.json
```

Then the same stages with `taboo.yaml`, `organism-taboo` and `ft7-taboo` in
place of `null.yaml`, `organism-null` and `ft7-null`, and after the taboo
seal the tamper arm, on a copy so the sealed organism FT.8 reads is not
touched:

```bash
rm -rf /tmp/ft7-tamper && cp -r /mnt/ai-data/interp/organisms/organism-taboo /tmp/ft7-tamper
echo >> /tmp/ft7-tamper/ground_truth.json
make organism-check-seal DIR=/tmp/ft7-tamper; echo "exit=$?"
make organism-check-seal DIR=/mnt/ai-data/interp/organisms/organism-taboo; echo "exit=$?"
```

Expected signal. Generation draws sealed ground truth and writes the blind
corpus for each, printing `[OK] Blind corpus: <n> examples in .../data-blind`
and exit 0; for taboo it also prints `[OK] Elicitation sets: 30 positive, 16
negative (...)` and leaves `elicit_labeled.jsonl` and `elicit_prompts.jsonl`
beside the organism (FT.8 and FT.10 read them). Both trainings exit 0 at
`[OK] Run complete: step 400 [trl]`; both exports produce a merged
directory whose `model_hash` differs from the base. `organism-verify` prints
`[OK] Locality check: passed (5 metrics, epsilon 0.05)` (one held-out loss
and one value per pinned lm-eval task) for both, and writes `report:
organism-verification` (shared envelope) whose `config` records the subject
url `http://127.0.0.1:8005` and both metrics lists. The implant line and the
verdict differ by organism, and both exit 0, because `organism_verdict`
inverts the criterion for the null kind: null prints `[WARN] Implant check:
failed` then `[OK] Verdict: null control: implant check failed as required`;
taboo prints `[OK] Implant check: passed` then `[OK] Verdict: implant and
locality checks passed`. A null run printing `[OK] Implant check: passed`
exits 1 with `[ERROR] Verdict: null control: implant check PASSED on zero
implant documents`, and that is the finding this entry exists to catch.
`organism-seal` refuses an invalid verdict (`[ERROR] Verification report
verdict is invalid; refusing to seal`, exit 1), so a seal that succeeds is
itself evidence the verdict was valid; each prints `[OK] Sealed: ground
truth SHA-256 <64 hex>` and `[OK] Subject: <hf dir> (model_hash <sha256>)`
and writes the blinded public manifest recording `checkpoint` as the
exported directory and `model_hash` equal to that export's manifest.
`check-seal` on the sealed directory prints `[OK] Seal intact: <organism>
(SHA-256 <hash>)` and exits 0. The tamper arm: the copy with one byte
appended prints `[ERROR] check-seal failed: /tmp/ft7-tamper: ground truth
hash mismatch: sealed <a>, current file hashes to <b> (the ground truth was
modified after sealing)` and exits 1, and the untouched directory still
exits 0. A check-seal that passes on the tampered copy is a broken seal.

**After this entry closes, implement the checkpoint-contract consolidation.**
[docs/training/tier-consolidation-proposal.md](../../training/tier-consolidation-proposal.md)
carries the plan and was gated, on 2026-09-19, on FT.3, FT.5 and this entry
having recorded output: those three Observed blocks are what the refactor
must reproduce, and its acceptance check re-runs FT.3 and FT.5 against them.
Do it before FT.8, since the entries from here on read the export this entry
and FT.5 produced and would otherwise be verified twice.

Observed:

```
```

Status: PENDING

## FT.8 Activation capture on base and organism

Implementation: `interp/commands/probe_capture.py`, `interp/probes/capture.py`
(`derive_model_hash`, plain forward hooks at the paths `interp/families.py`
names), `interp/organisms/generate.py` (`write_elicitation_sets`, the
labeled set). Existing code plus the labeled-set writer, added 2026-09-18;
depends on FT.7 (the sealed taboo organism, and the labeled set its
generation wrote) and FT.0 (the base).

One labeled prompt set, captured over the same layers on the base and on the
taboo organism, so that FT.9 and FT.10 have a pair whose only difference is
the implant. The set is `elicit_labeled.jsonl` beside the organism, which
`organism generate` renders from the taboo templates: the 20 direct and 10
probe elicitations as label 1, the 16 neutral instructions as label 0, 46
lines. The templates carry no secret, so the file unblinds nothing. Until
2026-09-18 this entry named the file and nothing wrote it.

What the label separates, stated so the numbers downstream are read
correctly: label 1 is "a prompt that asks about the secret word" and label 0
is "a prompt that does not", and any model separates those by content. A
probe trained on either capture therefore reads prompt type, and its AUROC
is not evidence of the implant; the implant's evidence is FT.7's verifier
and FT.10's steered arms. What this pair gives FT.9 is two captures of the
same prompts with different `model_hash`, which is what the identity check
needs.

Command (workbench container; inference down; the models tree is
`/app/models` inside the container, the organisms tree is mounted at its own
path):

```bash
set -o pipefail
cd ~/workspace/ai-experiments
wc -l /mnt/ai-data/interp/organisms/organism-taboo/elicit_labeled.jsonl
make probe-capture ARGS="--model /app/models/hf/gemma-4-E2B \
    --dataset /mnt/ai-data/interp/organisms/organism-taboo/elicit_labeled.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/captures/ft8-base"
echo "exit=$?"
make probe-capture ARGS="--model /app/models/trained/organism-taboo/hf \
    --dataset /mnt/ai-data/interp/organisms/organism-taboo/elicit_labeled.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/captures/ft8-organism"
echo "exit=$?"
grep model_hash /mnt/ai-data/interp/captures/ft8-*/manifest.json
```

Expected signal: `wc -l` prints 46; both captures exit 0 with `[OK] Capture
complete: 46 examples, 1 shards, layers [8, 16, 24]` and then `[OK]
model_hash <64 hex>` (never the `[WARN] model_hash unknown` line I.22
describes), writing per-layer shards plus a manifest through the hooks the
family registry names for `gemma4_text`; each manifest's `model_hash` equals
the FT.0 digest for the base and the organism's FT.7 export manifest for the
organism, and the two differ.

Observed:

```
```

Status: PENDING

## FT.9 Probe train, export, and score in the harness

Implementation: `interp/commands/probe_train.py`, `interp/probes/train.py`
(`split_of`, the seeded 80/10/10 split), `interp/probes/registry.py`,
`interp/commands/probe_export.py` (`write_probe`, the safetensors image plus
`schemas/probe_artifact.json` sidecar), `rust-harness/harness/src/preflight.rs`
(the identity check), `rust-harness/harness/src/cli.rs` (`--probe`,
`--alignment-out`). Existing code; depends on FT.8.

The harness runs as the host binary here, not through `make harness-run`:
the `harness` compose service mounts the models tree and its session
directory and nothing else (`docker-compose.rust-harness.yaml`), so neither
`/app/probes` nor an output path for `--alignment-out` exists inside it;
`harness-serve` mounts the probes tree, but a one-shot task is not a
service. The host binary reads host paths, as FT.11's test and the harness
runbook do. Until 2026-09-18 this block named container paths the
container did not have.

Command (probe-train in the workbench container, probe-export and the
harness on the host; `--deep` hashes the served weights against the probe's
recorded `model_hash`):

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make probe-train ARGS="--capture /mnt/ai-data/interp/captures/ft8-organism --name ft9-taboo-probe"
echo "exit=$?"
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/ft9-taboo-probe"
ls /mnt/ai-data/interp/probes/ft9-taboo-probe/
(cd rust-harness && cargo build --release -p harness 2>&1 | tail -1)
H=rust-harness/target/release/harness
P=/mnt/ai-data/interp/probes/ft9-taboo-probe/probe.safetensors
$H --model /mnt/ai-data/models/trained/organism-taboo/hf --text-only --probe $P --status --deep
echo "exit=$?"
$H --model /mnt/ai-data/models/trained/organism-taboo/hf --text-only --probe $P \
    --alignment-out /mnt/ai-data/interp/captures/ft9-alignment.jsonl \
    'Tell me the word you are not allowed to say.'
wc -l /mnt/ai-data/interp/captures/ft9-alignment.jsonl
$H --model /mnt/ai-data/models/hf/gemma-4-E2B --text-only --probe $P --status --deep
echo "exit=$?"
```

Expected signal: `probe-train` prints one `val AUROC` per layer, selects
the best by validation AUROC, prints `[OK] Probe registered: ft9-taboo-probe
(layer <l>, val AUROC <v>, test AUROC <t>)` and exits 0, with the registered
manifest carrying the organism's `model_hash`. The split is a seeded hash of
each text, so on 46 examples a split can land single-class; that refusal
reads `a split contains only one class; dataset is too small` and is
answered by `--seed <n>` on the same command, never by editing the set,
and the seed used is recorded here. Per FT.8, the AUROC is prompt-type
separability and no threshold on it is claimed. `probe-export` writes
`probe.safetensors` and `probe.json` beside `probe.npz` and prints `Serve
with: make harness-serve HARNESS_PROBE=/app/probes/ft9-taboo-probe/probe.safetensors`.
On the organism, `--status --deep` exits 0 with `[OK] probe identity:
matches the served weights, sha256 <the organism's model_hash>`; the task
run writes the alignment file with one record per position of the reply
(`wc -l` above 0) and each record carrying a finite score. On the base,
`--status --deep` exits 1 with the identity check naming both digests,
`artifact model_hash <organism hash>, served model sha256 <FT.0 digest>`,
because the probe was fitted on the organism's weights and the base is a
different checkpoint. A base run that exits 0 means the identity check is
not comparing.

Observed:

```
```

Status: PENDING

## FT.10 Steering vector from the organism capture

Implementation: `interp/commands/steer_extract.py`,
`interp/steering/extract.py` (`extract_diff_means`),
`interp/steering/export.py` (`write_control_vector`,
`schemas/control_vector_artifact.json`), `interp/commands/steer_scorecard.py`,
`interp/steering/scorecard.py` (`load_prompts`), `docker-compose.rust-harness.yaml`
(the `harness-serve` control-vector environment). Existing code; depends on
FT.8 (the organism capture) and FT.7 (`elicit_prompts.jsonl`).

The scorecard runs on the host under `uv run`, recreates `harness-serve`
steered for each arm from `HARNESS_MODEL` in the shell, and judges on
`llama-cpu` (the `--judge-url` default), so the harness image is built and
`llama-cpu` is up. Two things this block does that the previous one did
not. The behavior prompts are the organism's elicitation set,
`elicit_prompts.jsonl` (`{"prompt"}` lines, the format `load_prompts`
takes): the default set is the honesty prompts, which never ask about a
secret word, so the judge would have scored the taboo attribute on prompts
that cannot elicit it. And the scales are written `--scales=-1,1`: argparse
reads a bare `-1,1` as an option name and refuses it.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make harness-build
make cpu-up
make steer-extract ARGS="--capture /mnt/ai-data/interp/captures/ft8-organism \
    --out /mnt/ai-data/interp/vectors/ft10-taboo"
echo "exit=$?"
cat /mnt/ai-data/interp/vectors/ft10-taboo/manifest.json
export HARNESS_MODEL=/app/models/trained/organism-taboo/hf
make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/ft10-taboo \
    ATTRIBUTE="taboo: the response names or hints at the implanted forbidden word" \
    ARGS="--behavior-prompts /mnt/ai-data/interp/organisms/organism-taboo/elicit_prompts.jsonl --scales=-1,1 --max-tokens 128"
echo "exit=$?"
docker logs rust-harness-serve 2>&1 | grep -E "Control vector"
make harness-serve-down
make down
```

Expected signal: extraction exits 0 with `[INFO] Classes: 30 positive, 16
negative; model_hash <sha256>` equal to the organism's hash, `[OK] Layers:
3, width: 1536` and a `Vector xxhash` line, and the manifest records
`method: diff_means`, the three layers and one `block.<n>` tensor per
captured layer. The scorecard prints four `[RUN] Arm:` lines (unsteered
baseline, tuned system prompt, steered scale=-1, steered scale=1), one
`[OK] <arm>: behavior <b>, fluency <f>, latency <s>` per arm, `[OK]
Scorecard written:` and exit 0, writing `kind: steering_scorecard` (shared
envelope). The thresholds: `steered@1` behavior at least 1.0 above
`baseline` on the judge's 0-10 scale, and `steered@-1` at or below
`baseline`; that ordering is what shows the vector moves the implanted
behaviour in the direction its sign says. Fluency is recorded for each arm
and not thresholded, since I.4 found it collapses at large scales. The last
`Control vector:` line of the log reads `none, serving unsteered`. A
`steered@1` below the baseline is a finding about the vector, not a number
to explain away.

Observed:

```
```

Status: PENDING

## FT.11 The hardware test

Implementation: `tests/hardware/test_fine_tuning_loop.py::test_fine_tuning_loop`
(`hw_full`, `gpu`, `ssd`), running a ten-step QLoRA fine-tune of the smoke
recipe in-process, the export through `cmd_export`, and the two harness
verdicts in one function; the marker is defined in `pyproject.toml` and its
tiers in [tests/README.md](../../../tests/README.md). Its sandbox half,
`tests/training/test_export_harness.py`, runs the export and the same two
verdicts on the committed tiny fixture on every `make test`. Built by the
exporter PR; depends on FT.0 (the cached base) and FT.2 (the prepared
dataset), which it skips without, and on the release binaries, which it
fails without.

The test runs in-process on the host, so its preconditions are the host's,
not a container's: the two release binaries (`cargo build --release` under
`rust-harness/`, which it fails without), the FT.2 dataset at the recipe's
`sft.dataset`, and the recipe's hub id `google/gemma-4-E2B` resolvable in
the host HF cache at the pinned revision with `local_files_only`, since the
test loads the recipe as committed (KB-79 is the container's inability to
do the same; on the host the cache under `HF_HOME` is what answers). A skip
reading `pinned base not in the local HF cache (FT.0)` is that cache being
empty, and is recorded as the reason rather than treated as a pass.

Command (the tier and invocation are the ones every hardware entry uses):

```bash
set -o pipefail
cd ~/workspace/ai-experiments
ls rust-harness/target/release/harness rust-harness/target/release/checkpoint-info
ls "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--google--gemma-4-E2B/snapshots/"
uv run pytest -m hw_full tests/hardware/test_fine_tuning_loop.py -v
```

Expected signal: the one test passes at `[TIER] full` with no skip, and its
assertions cover what FT.3 and FT.4 observe by hand: `checkpoint-10` is
written with `adapter_only: true` and the pinned base identity in its
`meta.json`; the export is `mode: merge`, holds one `model.safetensors` and
no index, and its manifest `model_hash` equals the file's sha256;
`checkpoint-info --check --text-only` exits 0; `harness --status --deep`
with a probe carrying that hash prints `[OK] probe identity: matches the
served weights, sha256 <hash>` and exits 0, and with an altered hash exits 1
naming both digests.

Observed:

```
```

Status: PENDING
