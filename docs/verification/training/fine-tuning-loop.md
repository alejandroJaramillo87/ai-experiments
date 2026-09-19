# Fine-tuning loop verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 4 verified, 0 blocked, 12 pending, of 16.**

Retired: FT.15 (2026-09-19, scorecard promotion).

An entry id is a name, not a position: it is assigned once, kept when the
title changes, and never reused. The Table of Contents is the execution
order, and `scripts/docs/check_runbook_ids.py` holds the sections to it. The
entries carried numbers until 2026-09-19, when every live one took a slug; the old numbers are in the retired-names table of
`scripts/docs/check_retired_names.py`, and the frozen records keep them.

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
calibration, P0.5 resume) are cited by FT.preconditions and are not re-run here. The
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
  FT.export-merged verifies, retired GGUF production from the training tier, moved the
  `eval-ft` judge phase and the organism verifier onto the harness for FT.eval-ft
  and FT.organisms, and added the `hw_full` hardware test FT.hardware-test runs. Every command
  below names a verb, flag or make target that exists on its branch.
- **The hardware campaign** (the third session) executes every entry
  in the section order below and records the output.
- **The coverage audit** (2026-09-18) checked every entry against
  `make/training.mk`, the two compose files and the CLI parsers, and added
  FT.image-smoke, FT.recipe-gates and FT.memfit for the targets no entry ran (`train-build`,
  `train-smoke`, `train-preflight`, `train-validate`, `memfit`). It found two things every container entry now
  carries. First, KB-79: every recipe names the base as the hub id
  `google/gemma-4-E2B`, which the training container cannot resolve (its
  HF cache is an empty tmpfs and `train-download` stages a plain directory),
  so until the fix of 2026-09-19 (below) every `train-*`, `memfit`,
  `train-preflight` and organism training command passed
  `model.hf.id=/app/models/hf/gemma-4-E2B` as an override. The recipe as
  committed was the defect, and FT.recipe-gates reproduced it.
  Second, the training container mounts the models tree at `/app/models`
  only, so a container-run command names that spelling (FT.heldout-eval), and `eval-ft`
  now translates the host spelling itself (KB-78, FT.eval-ft).
- **The re-check** (2026-09-19, before FT.memfit ran) read every entry again
  against the tree after the from-scratch loop closed, and found KB-79's
  sibling, KB-80: every SFT recipe's `tokenizer.path` named
  `/mnt/ai-data/training/tokenizers/gemma-4-e2b`, which nothing stages (a
  fine-tune reuses the base tokenizer, and the staged HF directory holds
  it). It was handled for one pass by a second override, then fixed in the
  tree the same day: `tokenizer.path` is optional for an HF base and every
  reader resolves the tokenizer from `model.hf.id`, so the six HF recipes
  carry no tokenizer block and the KB-79 override was, until that day's
  later fix, the only one a container command passed. FT.recipe-gates's
  committed-recipe preflight then reproduced KB-79 alone, and its pass
  command is what confirmed KB-80
  fixed. It also found that preflight
  verified an SFT recipe's `data.source` as token shards and crashed on a
  prepared dataset (KB-81, fixed in the tree, confirmed by FT.recipe-gates), and
  reordered the sections so no entry precedes what it depends on: FT.smoke-dataset
  before FT.recipe-gates, since the preflight gate reads the dataset FT.smoke-dataset prepares.
- **The calculator decision** (2026-09-19, after FT.memfit first passed) read
  the memory-fit calculator against the Gemma 4 notes and against what the
  loop needs, and reduced it rather than rip it out or specify it fully
  ([memory_fit.md](../../training/memory_fit.md)): the static terms and the
  parameter count stay (the count fixed for Gemma 4's two head widths, its
  norms, the per-layer embedding projection and the towers, KB-82), the
  analytic activation formula and `solve` left, activations are a rate
  `train-calibrate` measures per profile and the operator records in
  `training/calculator/measured.py`, and `train-sft` warns instead of
  refusing. FT.memfit's first pass ran under the analytic model and is kept in
  the entry as a superseded observation; the entry is redefined and now
  follows FT.calibrate-qlora, which supplies the rate it reads.
- **FT.calibrate-qlora's first attempt** (2026-09-19) got past the model load and died
  in the PEFT wrap: the Gemma 4 towers reuse the decoder's projection names
  behind wrappers PEFT cannot adapt (KB-83, fixed in the tree the same day
  by confining the adapter to `model.language_model.`). The paste is kept in
  FT.calibrate-qlora as a superseded observation; the entry is unchanged otherwise.
- **FT.recipe-gates ran** (2026-09-19, after FT.smoke-dataset) and found
  KB-79's fourth reader: with the prepared dataset on disk, validate's
  shard-manifest hash check loads the tokenizer from `model.hf.id`, the
  hub id, so the recipe as committed fails validation before preflight's
  asset gate, from `train-validate` as well as `train-preflight`. The
  override preflight passed and closed KB-80 and KB-81; the entry is
  blocked on KB-79 for its two committed-recipe commands. Fixed the same
  day, in the tree: the recipe keeps its hub id and revision, and every
  reader (`build_model`, the trainer, `resolve_base_identity`, the
  calculator, `tokenizer_source`) resolves the id to the directory
  `train-download` staged through `training/models/staged.py`, whichever
  spelling the environment mounts; validate's hash checks read a directory
  only and leave an unstaged base to preflight. Every override left the
  commands below, `tests/hardware/test_recipe_gates.py` holds the gates as
  an `hw_smoke` test, and FT.recipe-gates's re-run as committed passed the
  same day, which closed KB-79.

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

1. [FT.preconditions Preconditions](#ftpreconditions-preconditions)
1. [FT.image-smoke Training image and smoke](#ftimage-smoke-training-image-and-smoke)
1. [FT.smoke-dataset Prepare the smoke SFT dataset](#ftsmoke-dataset-prepare-the-smoke-sft-dataset)
1. [FT.recipe-gates Recipe gates: preflight and validate](#ftrecipe-gates-recipe-gates-preflight-and-validate)
1. [FT.calibrate-qlora Calibrate the QLoRA recipe (KB-03)](#ftcalibrate-qlora-calibrate-the-qlora-recipe-kb-03)
1. [FT.memfit Memory fit: the measured rate reproduces the calibration](#ftmemfit-memory-fit-the-measured-rate-reproduces-the-calibration)
1. [FT.e4b-recipe The E4B recipe: preflight, fit and calibration](#fte4b-recipe-the-e4b-recipe-preflight-fit-and-calibration)
1. [FT.train-resume Train, interrupt, resume](#fttrain-resume-train-interrupt-resume)
1. [FT.export-merged Export the merged HF directory and load it in the harness](#ftexport-merged-export-the-merged-hf-directory-and-load-it-in-the-harness)
1. [FT.heldout-eval Held-out loss and lm-eval on base and merged](#ftheldout-eval-held-out-loss-and-lm-eval-on-base-and-merged)
1. [FT.eval-ft eval-ft with both sides identified](#fteval-ft-eval-ft-with-both-sides-identified)
1. [FT.organisms Organisms: null and taboo](#ftorganisms-organisms-null-and-taboo)
1. [FT.capture Activation capture on base and organism](#ftcapture-activation-capture-on-base-and-organism)
1. [FT.probe Probe train, export, and score in the harness](#ftprobe-probe-train-export-and-score-in-the-harness)
1. [FT.steering Steering vector from the organism capture](#ftsteering-steering-vector-from-the-organism-capture)
1. [FT.hardware-test The hardware test](#fthardware-test-the-hardware-test)

## FT.preconditions Preconditions

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
FT.export-merged and FT.probe exercise).

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

## FT.image-smoke Training image and smoke

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

## FT.smoke-dataset Prepare the smoke SFT dataset

Implementation: `training/commands/prepare_sft.py`, `training/data/sft.py`,
`training/data/tokenizer.py`, `training/models/families.py`
(`CHAT_TEMPLATE_VERSION`, pinned against the harness renderer by
`schemas/chat_template.json`).

The prepared directory is the `sft.dataset` the smoke recipe names. Existing
code; depends on FT.preconditions.

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

Observed (2026-09-19; the paste is a second invocation, run to capture the
output after the first had already written the directory, so the
preparing run's own console lines were not recorded and the manifest it
wrote is the evidence):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
make prepare-sft ARGS="--source HuggingFaceH4/ultrachat_200k --split train_sft \
    --tokenizer /mnt/ai-data/models/hf/gemma-4-E2B --model-type gemma4 \
    --max-seq-len 2048 --out /mnt/ai-data/training/datasets/sft/ultrachat-e2b"
echo "exit=$?"
cat /mnt/ai-data/training/datasets/sft/ultrachat-e2b/manifest.json
docker compose -f docker-compose.training.yaml run --rm prepare python -m training prepare-sft --source HuggingFaceH4/ultrachat_200k --split train_sft     --tokenizer /mnt/ai-data/models/hf/gemma-4-E2B --model-type gemma4     --max-seq-len 2048 --out /mnt/ai-data/training/datasets/sft/ultrachat-e2b
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-prepare-run-e6643c617715 Creating
Container ai-experiments-prepare-run-e6643c617715 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
[WARN] nvidia-smi not found: no GPU visible in this container

[RUN] exec: python -m training prepare-sft --source HuggingFaceH4/ultrachat_200k --split train_sft --tokenizer /mnt/ai-data/models/hf/gemma-4-E2B --model-type gemma4 --max-seq-len 2048 --out /mnt/ai-data/training/datasets/sft/ultrachat-e2b
=== SFT data preparation (Phase 1 F1) ===

[INFO] Source: HuggingFaceH4/ultrachat_200k
[INFO] Tokenizer: /mnt/ai-data/models/hf/gemma-4-E2B
[INFO] Model family: gemma4
[INFO] Output: /mnt/ai-data/training/datasets/sft/ultrachat-e2b
[RUN] Rendering, tokenizing, and masking conversations
[ERROR] /mnt/ai-data/training/datasets/sft/ultrachat-e2b already contains manifest.json; refusing to overwrite
  Solution: choose a new output directory or delete the existing one

make: *** [make/training.mk:38: prepare-sft] Error 1
exit=2
{
  "chat_template_version": 2,
  "command": "/app/training/__main__.py prepare-sft --source HuggingFaceH4/ultrachat_200k --split train_sft --tokenizer /mnt/ai-data/models/hf/gemma-4-E2B --model-type gemma4 --max-seq-len 2048 --out /mnt/ai-data/training/datasets/sft/ultrachat-e2b",
  "created_at": "2026-09-19T21:43:47.597784+00:00",
  "dataset": "HuggingFaceH4/ultrachat_200k",
  "dropped_examples": 155,
  "files": [
    {
      "examples": 203644,
      "file": "train.jsonl",
      "xxhash64": "b7bf7e4d3a2fe832"
    },
    {
      "examples": 4066,
      "file": "holdout.jsonl",
      "xxhash64": "1835216bffa05489"
    }
  ],
  "git_sha": "unknown",
  "holdout_examples": 4066,
  "holdout_fraction": 0.02,
  "holdout_seed": 42,
  "kind": "sft",
  "license": "unknown",
  "manifest_version": 1,
  "max_seq_len": 2048,
  "model_type": "gemma4",
  "revision": null,
  "split": "train_sft",
  "supervised_tokens": 184870319,
  "tokenizer": "/mnt/ai-data/models/hf/gemma-4-E2B",
  "tokenizer_hash": "d1e84d8e83e7f712",
  "total_tokens": 241238839,
  "train_examples": 203644
}
```

What the evidence shows: the refusal to overwrite is a gate seen refusing,
and the manifest carries the fields the entry names, `chat_template_version`
2, `tokenizer_hash` `d1e84d8e83e7f712` for the staged base, the two split
files with their sizes and content hashes, and the supervised-token count
against the total. FT.recipe-gates's override preflight then loaded this
directory through the F1 loader and reported it verified, which is the
hash check the tokenizer line stands for. Not observed: the reasoning-marker
refusal, which `tests/training/test_sft.py` covers in the sandbox and no
ultrachat record triggers. Noted, not filed: `git_sha` is `unknown` in the
manifest because the prepare container carries no `.git`, and `license`
is `unknown` because the hub card is not read; both are provenance fields
that fail soft, and neither reader checks them.

Status: VERIFIED 2026-09-19 on the manifest and the refusal; the preparing
run's console output was not captured.

## FT.recipe-gates Recipe gates: preflight and validate

Implementation: `training/commands/preflight.py`,
`training/commands/validate_recipe.py`, `training/recipes/validate.py`,
`training/export/preflight.py` (`resolve_base_identity`, which preflight
calls with `local_files_only`), `training/models/staged.py` (the hub id
to staged directory resolution every reader shares, KB-79), and
`tests/hardware/test_recipe_gates.py` (`hw_smoke`, `cpu`), which holds the
same gates for every HF recipe on the host.

The two gates a run passes before the GPU is committed, each run to pass
and to refuse, since a gate whose refusal has never been seen is not known
to gate. Every command runs the recipe as committed: the recipe names its
base by hub id and revision, and each reader resolves the id to the
directory `train-download` staged. Added 2026-09-18; depends on
FT.preconditions, FT.image-smoke and FT.smoke-dataset (the dataset the
preflight must find).

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train-preflight CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make train-validate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make train-validate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="optimizer.lr=-1"; echo "exit=$?"
uv run --no-sync pytest -m hw_smoke tests/hardware/test_recipe_gates.py -v; echo "exit=$?"
```

Expected signal: `train-preflight` prints `=== Preflight: sft-qlora-e2b
===` and `[OK] All referenced assets are present and verified`, exit 0,
with no `model.hf`, `tokenizer` or `shard` line; a `- model.hf:` line with
the `make train-download` remedy is the base not staged (FT.preconditions),
and a `tokenizer.path: cannot load` line is KB-79 back. `train-validate`
prints `[OK] Schema: training/recipes/reference/sft-qlora-e2b.yaml loads
as a typed recipe` and `[OK] Validation: run sft-qlora-e2b passes all
cross-field invariants`, exit 0; with `optimizer.lr=-1` it prints `[ERROR]
Recipe validation failed:` and one problem line, `  - optimizer.lr: must be
positive, got -1.0`, the command exits 1 and `make` reports `Error 1` and
`exit=2`. A validate that exits 0 on the negative learning rate is a gate
that does not gate, and a second problem line is a new finding. The test
passes its three cases for each E2B recipe (the resolution lands on the
staged directory, validate passes and refuses, preflight names the dataset
or nothing) and skips the E4B recipe unless E4B is staged. That paste closes
KB-79.

Superseded expected signal, as first written (before FT.smoke-dataset had run): the
committed recipe's preflight prints `[ERROR] Missing or unusable assets:
1` and one problem line, `- model.hf:` naming `google/gemma-4-E2B`
(KB-79); then `Solution: stage the missing assets, then re-run preflight`
and one remedy, `make train-download MODEL=<hub id> REVISION=<sha>` (which
FT.preconditions already ran, and which cannot satisfy a hub id). The
command exits 1, which `make` reports as `make: *** [make/training.mk:44:
train-preflight] Error 1` and `exit=2`. A second problem line is a new
finding, recorded on its own; a `- tokenizer.path:` line is KB-80 back,
and a line naming `data.source` or a traceback out of `verify_shards` is
KB-81 back. With the model override, and FT.smoke-dataset's dataset on
disk, it exits 0 with `[OK] All referenced assets are present and
verified` and no `shard` or `tokenizer` in its output; that run is what
confirms KB-80 fixed. `train-validate` on the recipe prints `[OK] Schema:
training/recipes/reference/sft-qlora-e2b.yaml loads as a typed recipe` and
`[OK] Validation: run sft-qlora-e2b passes all cross-field invariants`,
exit 0; with `optimizer.lr=-1` it prints `[ERROR] Recipe validation
failed:` and `  - optimizer.lr: must be positive, got -1.0`, the command
exits 1 and `make` reports `Error 1` and `exit=2`. A validate that exits 0
on the negative learning rate is a gate that does not gate.

What the tree did once the dataset was on disk (2026-09-19, before the fix): the `- model.hf:`
line is unreachable. Every SFT recipe sets `data.source` to the prepared
dataset directory, and `_check_tokenizer_matches_shards`
(`training/recipes/validate.py`) hash-checks that directory's manifest
against the tokenizer `tokenizer_source()` names, which with no tokenizer
block is `model.hf.id`, the hub id; `load_tokenizer` fetches a hub id
through `AutoTokenizer` and the container is offline. While the directory
was absent the check returned early and preflight reached its asset gate;
with FT.smoke-dataset done, validation fails first, from both
`train-preflight` and `train-validate`, with `- tokenizer.path: cannot
load google/gemma-4-E2B: We couldn't connect to 'https://huggingface.co'`.
That is KB-79 through its fourth reader (the ledger row lists them), not
KB-80 back: KB-80's line was `tokenizer.path: not found` on a path nothing
staged, and its fix is what routes this check to `model.hf.id`. The
override run is unaffected, and the negative learning rate is still
refused, with the KB-79 line beside it. So of the four commands the two
that run the recipe as committed are blocked on KB-79, and the two that
prove the gates gate pass.

Observed (2026-09-19, superseded: the run that found KB-79's fourth reader,
under the override the entry then carried):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
make train-preflight CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make train-preflight CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B"; echo "exit=$?"
make train-validate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make train-validate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="optimizer.lr=-1"; echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training python -m training preflight --config training/recipes/reference/sft-qlora-e2b.yaml
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-6233c042088a Creating
Container ai-experiments-training-run-6233c042088a Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training preflight --config training/recipes/reference/sft-qlora-e2b.yaml
[ERROR] Recipe validation failed:
  - tokenizer.path: cannot load google/gemma-4-E2B: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

make: *** [make/training.mk:44: train-preflight] Error 1
exit=2
docker compose -f docker-compose.training.yaml run --rm training python -m training preflight --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-3724bcb10192 Creating
Container ai-experiments-training-run-3724bcb10192 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training preflight --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B
=== Preflight: sft-qlora-e2b ===

[OK] All referenced assets are present and verified
exit=0
docker compose -f docker-compose.training.yaml run --rm training python -m training validate training/recipes/reference/sft-qlora-e2b.yaml
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-dfc4e7614539 Creating
Container ai-experiments-training-run-dfc4e7614539 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training validate training/recipes/reference/sft-qlora-e2b.yaml
[OK] Schema: training/recipes/reference/sft-qlora-e2b.yaml loads as a typed recipe
[ERROR] Recipe validation failed:
  - tokenizer.path: cannot load google/gemma-4-E2B: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

make: *** [make/training.mk:47: train-validate] Error 1
exit=2
docker compose -f docker-compose.training.yaml run --rm training python -m training validate training/recipes/reference/sft-qlora-e2b.yaml optimizer.lr=-1
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-abcd205db52b Creating
Container ai-experiments-training-run-abcd205db52b Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training validate training/recipes/reference/sft-qlora-e2b.yaml optimizer.lr=-1
[OK] Schema: training/recipes/reference/sft-qlora-e2b.yaml loads as a typed recipe
[INFO] Overrides applied: optimizer.lr=-1
[ERROR] Recipe validation failed:
  - optimizer.lr: must be positive, got -1.0
  - tokenizer.path: cannot load google/gemma-4-E2B: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

make: *** [make/training.mk:47: train-validate] Error 1
exit=2
```

The override preflight's `[OK] All referenced assets are present and
verified` is the confirmation KB-80 and KB-81 were waiting on: the
dataset FT.smoke-dataset prepared was found, hash-verified through the F1
loader against the staged base's tokenizer, and reported without a
`shard`, `tokenizer` or traceback line. Both rows move to resolved. The
committed-recipe preflight no longer prints the `Missing or unusable
assets` block, so the KB-79 reproduction is now the validation line above
rather than the asset gate; the ledger row records both.

Observed (2026-09-19, the re-run as committed, after the KB-79 fix):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
make train-preflight CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make train-validate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make train-validate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="optimizer.lr=-1"; echo "exit=$?"
uv run --no-sync pytest -m hw_smoke tests/hardware/test_recipe_gates.py -v; echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training python -m training preflight --config training/recipes/reference/sft-qlora-e2b.yaml
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-b3b8537133ba Creating
Container ai-experiments-training-run-b3b8537133ba Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training preflight --config training/recipes/reference/sft-qlora-e2b.yaml
=== Preflight: sft-qlora-e2b ===

[OK] All referenced assets are present and verified
exit=0
docker compose -f docker-compose.training.yaml run --rm training python -m training validate training/recipes/reference/sft-qlora-e2b.yaml
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-1975874c3121 Creating
Container ai-experiments-training-run-1975874c3121 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training validate training/recipes/reference/sft-qlora-e2b.yaml
[OK] Schema: training/recipes/reference/sft-qlora-e2b.yaml loads as a typed recipe
[OK] Validation: run sft-qlora-e2b passes all cross-field invariants
exit=0
docker compose -f docker-compose.training.yaml run --rm training python -m training validate training/recipes/reference/sft-qlora-e2b.yaml optimizer.lr=-1
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-34155d487f76 Creating
Container ai-experiments-training-run-34155d487f76 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training validate training/recipes/reference/sft-qlora-e2b.yaml optimizer.lr=-1
[OK] Schema: training/recipes/reference/sft-qlora-e2b.yaml loads as a typed recipe
[INFO] Overrides applied: optimizer.lr=-1
[ERROR] Recipe validation failed:
  - optimizer.lr: must be positive, got -1.0

make: *** [make/training.mk:47: train-validate] Error 1
exit=2
=============================================================================== test session starts ===============================================================================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/alejandro/workspace/ai-experiments/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/alejandro/workspace/ai-experiments
configfile: pyproject.toml
plugins: asyncio-1.4.0, mock-3.15.1, jaxtyping-0.3.11, cov-7.1.0, hydra-core-1.3.2, anyio-4.14.2, xdist-3.8.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 18 items

tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[false-fact.yaml] PASSED                                                               [  5%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[null.yaml] PASSED                                                                     [ 11%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[taboo.yaml] PASSED                                                                    [ 16%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[continual-gemma4-e2b.yaml] PASSED                                                     [ 22%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[sft-qlora-e2b.yaml] PASSED                                                            [ 27%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[sft-qlora-gemma-e4b.yaml] PASSED                                                      [ 33%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[false-fact.yaml] PASSED                                                    [ 38%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[null.yaml] PASSED                                                          [ 44%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[taboo.yaml] PASSED                                                         [ 50%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[continual-gemma4-e2b.yaml] PASSED                                          [ 55%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[sft-qlora-e2b.yaml] PASSED                                                 [ 61%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[sft-qlora-gemma-e4b.yaml] PASSED                                          [ 66%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[false-fact.yaml] PASSED                                                              [ 72%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[null.yaml] PASSED                                                                    [ 77%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[taboo.yaml] PASSED                                                                   [ 83%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[continual-gemma4-e2b.yaml] PASSED                                                    [ 88%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[sft-qlora-e2b.yaml] PASSED                                                           [ 94%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[sft-qlora-gemma-e4b.yaml] PASSED                                                     [100%]

[TIER] full: torch present, the whole suite was in scope.
=============================================================================== 18 passed in 5.64s ================================================================================
exit=0
```

Every line matched the expected signal: preflight verified all assets with
no `model.hf`, `tokenizer` or `shard` line, validate passed both `[OK]`
lines, the negative learning rate was refused on one problem line and
nothing else, and the test passed all 18 cases, E4B included, since its
base was staged. That is the confirmation KB-79 was waiting on: every
recipe as committed resolves in the offline container and on the host.

Status: VERIFIED 2026-09-19 on the re-run. Was BLOCKED the same day on
KB-79 for the two committed-recipe commands after the superseded run above;
KB-79 was fixed in the tree and the entry re-ran as one, with no override.

## FT.calibrate-qlora Calibrate the QLoRA recipe (KB-03)

Implementation: `training/calculator/calibrate.py`,
`training/calculator/measured.py`, `training/models/model.py`,
`training/models/peft_utils.py`, `training/cli.py` (`cmd_calibrate`).

**Fixes to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-03 — `calibrate()` once built a scratch Llama model for every recipe, so
  calibrating a `model.hf` or PEFT recipe measured a model other than the one
  the estimate priced. The fix routes calibration through the shared
  `build_model` and the PEFT wrap; this run on the QLoRA recipe is the
  hardware evidence that closes the row.
- KB-82 — the calculator's Gemma 4 parameter count missed the seven
  512-wide full-attention layers, the q and k norms, two of the four block
  norms, the per-layer embedding projection, and the towers `from_pretrained`
  loads. The `Parameter count` line this run prints is the confirmation.
- KB-83 — PEFT matched the recipe's target-module names against the whole
  composite model, and Gemma 4's vision and audio towers reuse `q_proj`,
  `gate_proj` and the rest behind `Gemma4ClippableLinear` wrappers PEFT
  cannot adapt, so the wrap died on the first tower module (this entry's
  first attempt, below). `build_peft_model` now confines the adapter to
  `model.language_model.`; the trainable-parameter report and the
  `Activation rate:` line are the confirmation.

Measures the QLoRA recipe on the `model.hf` plus PEFT path, which the
archived P0.3 scratch calibration does not exercise: NF4 base pricing,
adapter gradients and optimizer state, and the activation rate of the
profile `gemma4_text/35x1536/qlora/ckpt`, which no entry has measured and
which FT.memfit and FT.train-resume read. It is also the first execution evidence for
Gemma 4 on the workstation: the gated checkpoint at its pinned revision,
transformers `gemma4` support, and bitsandbytes NF4 on Blackwell. Existing
code; depends on FT.preconditions.

Command:

```bash
make train-calibrate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"
```

The base is the recipe's hub id resolved to the staged directory (KB-79,
fixed 2026-09-19; the superseded attempt below passed it as an override).
Depends on FT.preconditions and FT.image-smoke.

Expected signal: exit 0. `[OK] Parameter count: <p> (analytic matches
model)`, or a `[WARN] Parameter count drift` whose two figures differ by
exactly 15,733,760, the shared-KV residue the modeling code does not
construct (KB-82; any other drift is a finding); no trainable-count drift
warning. Then `Resident after build:`, `Activation rate: <r> bytes/token`,
the line `Record: "gemma4_text/35x1536/qlora/ckpt": <r>,` and
`[INFO] Allocated error: not on file`, since this is the profile's first
measurement and there is no rate to hold to the bar. Record `<r>` and the
measured allocated, reserved and NVML figures here, then, before FT.memfit:

1. Paste the `Record:` line into `RATES` in `training/calculator/measured.py`
   with a comment naming this entry, the date, the measured allocated and
   the resident figure it was derived from.
2. Commit. No image rebuild: the training container bind-mounts
   `training/`, so the next `make memfit` reads the row.

A second calibrate of the same profile later (after a transformers or TRL
upgrade) is the drift check: it holds the estimate to the 10 percent bar and
the reserved estimate to the NVML peak, and a miss there is re-recorded, not
argued with.

First attempt (2026-09-19, superseded, kept verbatim: the model loaded with
the KB-79 override, and the PEFT wrap failed on a vision-tower module, which
is KB-83):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
make train-calibrate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-6d4cb765ba00 Creating 
Container ai-experiments-training-run-6d4cb765ba00 Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
[RUN] Calibrating sft-qlora-e2b: 5 steps on cuda:0
Loading weights:   0%|                                                                                                                                   | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 2871.98it/s]
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/training/__main__.py", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/app/training/cli.py", line 683, in main
    return dispatch("training", build_parser, argv)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/common/obs/dispatch.py", line 70, in dispatch
    result = args.func(args)
             ^^^^^^^^^^^^^^^
  File "/app/training/cli.py", line 139, in cmd_calibrate
    record = calibrate_mod.calibrate(recipe, steps=args.steps)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/training/calculator/calibrate.py", line 156, in calibrate
    model = _build_calibration_model(recipe)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/training/calculator/calibrate.py", line 134, in _build_calibration_model
    model = build_peft_model(model, recipe.peft)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/training/models/peft_utils.py", line 90, in build_peft_model
    return get_peft_model(model, lora_config)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/peft/mapping_func.py", line 122, in get_peft_model
    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/peft/peft_model.py", line 1955, in __init__
    super().__init__(model, peft_config, adapter_name, **kwargs)
  File "/opt/venv/lib/python3.12/site-packages/peft/peft_model.py", line 129, in __init__
    self.base_model = cls(model, {adapter_name: peft_config}, adapter_name)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/peft/tuners/tuners_utils.py", line 315, in __init__
    self.inject_adapter(self.model, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)
  File "/opt/venv/lib/python3.12/site-packages/peft/tuners/tuners_utils.py", line 913, in inject_adapter
    self._create_and_replace(
  File "/opt/venv/lib/python3.12/site-packages/peft/tuners/lora/model.py", line 269, in _create_and_replace
    new_module = self._create_new_module(lora_config, adapter_name, target, device_map=device_map, **kwargs)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/peft/tuners/lora/model.py", line 424, in _create_new_module
    raise ValueError(
ValueError: Target module Gemma4ClippableLinear(
  (linear): Linear4bit(in_features=768, out_features=768, bias=False)
) is not supported. Currently, only the following modules are supported: `torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv1d`, `torch.nn.Conv2d`, `torch.nn.Conv3d`, `transformers.pytorch_utils.Conv1D`, `torch.nn.MultiheadAttention.`.

make: *** [make/training.mk:53: train-calibrate] Error 1
exit=2
```

Observed:

```
```

Status: PENDING

## FT.memfit Memory fit: the measured rate reproduces the calibration

Implementation: `training/cli.py` (`cmd_memfit`, `_print_estimate_table`),
`training/calculator/estimate.py`, `training/calculator/measured.py` (the
rate FT.calibrate-qlora records), `training/tune/sft.py` (`check_memory_fit`, the same
estimate `train-sft` prints as advice).

The estimate as an operator reads it, once FT.calibrate-qlora has measured the recipe's
activation rate and the rate is on file: once on the recipe, which must fit,
and once on a geometry that must not. The calculator was reduced on
2026-09-19 ([memory_fit.md](../../training/memory_fit.md)): activations are
the measured rate times tokens, `solve` is gone, and `train-sft` warns rather
than refuses, so the verdict here is the line FT.train-resume prints before it trains.
CPU-only, no GPU; added 2026-09-18, redefined 2026-09-19; depends on FT.calibrate-qlora.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml; echo "exit=$?"
make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="data.micro_batch=64"; echo "exit=$?"
```

Expected signal: the first run prints `Profile: gemma4_text/35x1536/qlora/ckpt`,
the component table with an `activations` row equal to FT.calibrate-qlora's rate times
8,192 tokens (no `unmeasured`), `Total reserved estimate:` within 10 percent
of FT.calibrate-qlora's measured allocated plus the 2.50 GB overhead, and `[OK] Fits:
reserved estimate <n> GB within budget 31.00 GB (32 GB minus safety margin)`,
exit 0. A `[WARN] Activations unmeasured` line here means the `Record:` line
FT.calibrate-qlora printed was not recorded in `training/calculator/measured.py`. The
second prints the same rate times 131,072 tokens, `[ERROR] Does not fit:
reserved estimate <m> GB exceeds budget 31.00 GB`, the two `Solution:` lines,
and the command exits 1, which `make` reports as
`make: *** [make/training.mk:50: memfit] Error 1` and `exit=2`. Record `<n>`
here. The hardware mirror is `tests/hardware/test_memfit_sft.py` in the
`hw_smoke` tier: the same two verdicts through the calculator the CLI and
`train-sft`'s advisory share, on the staged base's `config.json` and
checkpoint header, skipped until the rate is on file.

Superseded observation (2026-09-19, under the analytic activation model the
reduction replaced, kept verbatim as the record of what that model said:
21.46 GB reserved at micro-batch 4, a batch solve of 6, no measurement
behind either figure):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B"; echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training python -m training memfit --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-761a83458a02 Creating
Container ai-experiments-training-run-761a83458a02 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training memfit --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B
=== Memory-fit estimate ===

Model parameters: 4,583,959,040
Micro-batch 4, sequence length 2048

Component            Bytes             Size
  weights               6,527,114,560     6.08 GB
  gradients                37,847,040     0.04 GB
  optimizer state          38,438,400     0.04 GB
  activations          14,277,672,960    13.30 GB
  framework overhead    2,684,354,560     2.50 GB

Total allocated estimate: 18.96 GB
  Peak phase: head_backward — gradient checkpointing makes the
  total a temporal max over backward phases, below the component sum
Total reserved estimate:  21.46 GB
[OK] Fits: reserved estimate 21.46 GB within budget 31.00 GB (32 GB minus safety margin)
exit=0
➜  ai-experiments git:(SFT-verification) ✗ make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B data.micro_batch=64"; echo "exit=$?"

docker compose -f docker-compose.training.yaml run --rm training python -m training memfit --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B data.micro_batch=64
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-eeed1399888c Creating
Container ai-experiments-training-run-eeed1399888c Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training memfit --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B data.micro_batch=64
=== Memory-fit estimate ===

Model parameters: 4,583,959,040
Micro-batch 64, sequence length 2048

Component            Bytes             Size
  weights               6,527,114,560     6.08 GB
  gradients                37,847,040     0.04 GB
  optimizer state          38,438,400     0.04 GB
  activations         228,442,767,360   212.75 GB
  framework overhead    2,684,354,560     2.50 GB

Total allocated estimate: 211.61 GB
  Peak phase: head_backward — gradient checkpointing makes the
  total a temporal max over backward phases, below the component sum
Total reserved estimate:  214.11 GB
[ERROR] Does not fit: reserved estimate 214.11 GB exceeds budget 31.00 GB
  Solution: reduce data.micro_batch or data.seq_len, enable
  memory.gradient_checkpointing, or switch optimizer to adamw8bit

make: *** [make/training.mk:50: memfit] Error 1
exit=2
➜  ai-experiments git:(SFT-verification) ✗ make memfit CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="model.hf.id=/app/models/hf/gemma-4-E2B --solve batch"; echo "exit=$?"

docker compose -f docker-compose.training.yaml run --rm training python -m training memfit --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B --solve batch
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-96ea18571c23 Creating
Container ai-experiments-training-run-96ea18571c23 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training memfit --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B --solve batch
=== Memory-fit estimate ===

Model parameters: 4,583,959,040
Micro-batch 4, sequence length 2048

Component            Bytes             Size
  weights               6,527,114,560     6.08 GB
  gradients                37,847,040     0.04 GB
  optimizer state          38,438,400     0.04 GB
  activations          14,277,672,960    13.30 GB
  framework overhead    2,684,354,560     2.50 GB

Total allocated estimate: 18.96 GB
  Peak phase: head_backward — gradient checkpointing makes the
  total a temporal max over backward phases, below the component sum
Total reserved estimate:  21.46 GB
[OK] Fits: reserved estimate 21.46 GB within budget 31.00 GB (32 GB minus safety margin)

[OK] Solve batch: max micro-batch 6 (reserved estimate 27.88 GB)
exit=0
```

Observed:

```
```

Status: PENDING

## FT.e4b-recipe The E4B recipe: preflight, fit and calibration

Implementation: `training/commands/preflight.py`, `training/cli.py`
(`cmd_memfit`, `cmd_calibrate`), `training/calculator/calibrate.py`,
`training/recipes/reference/sft-qlora-gemma-e4b.yaml`.

E2B and E4B are the two subjects this repository trains, and FT.recipe-gates, FT.calibrate-qlora
and FT.memfit run on E2B alone. The E4B recipe is the same QLoRA shape at 4096
context and a larger base, which is a different profile
(`gemma4_text/42x2560/qlora/ckpt`) with no rate on file: calibrate measures
it first, the operator records it as FT.calibrate-qlora did, and memfit then prices the
recipe. Same three gates as the E2B entries, the hub id resolved to E4B's
staged directory the same way (KB-79), and the same composite base, so its PEFT wrap is the
second confirmation of KB-83 (the adapter confined to
`model.language_model.`).
The recipe's `sft.dataset` is an E4B split nobody has prepared, so the
block's first line prepares it as FT.smoke-dataset did, at the recipe's 4096 context,
when the directory is absent. Depends on the E4B download (I.probe-loop and I.served-request-log use
`/app/models/hf/gemma-4-E4B`, so it is staged), on FT.image-smoke, and on FT.smoke-dataset for
the shape of the prepare step.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
R=training/recipes/reference/sft-qlora-gemma-e4b.yaml
test -d /mnt/ai-data/training/datasets/sft/ultrachat-gemma-e4b || \
    make prepare-sft ARGS="--source HuggingFaceH4/ultrachat_200k --split train_sft \
    --tokenizer /mnt/ai-data/models/hf/gemma-4-E4B --model-type gemma4 \
    --max-seq-len 4096 --out /mnt/ai-data/training/datasets/sft/ultrachat-gemma-e4b"
make train-preflight CONFIG=$R; echo "exit=$?"
make train-calibrate CONFIG=$R ARGS="--out /mnt/ai-data/training/verify/calibrate-sft-e4b.json"
echo "exit=$?"
python3 -c 'import json; r=json.load(open("/mnt/ai-data/training/verify/calibrate-sft-e4b.json")); print({k: r[k] for k in sorted(r) if "error" in k or "peak" in k or "rate" in k or "static" in k})'
```

Then record the `Record:` line in `training/calculator/measured.py` as FT.calibrate-qlora
did, commit, and:

```bash
make memfit CONFIG=$R; echo "exit=$?"
```

Expected signal: the prepare, when it runs, exits 0 and writes the
manifest as FT.smoke-dataset's did (record it here); preflight exits 0 with `[OK] All
referenced assets are present and verified`. `train-calibrate` exits 0 with
the E4B parameter-count line matching the built model (or drifting by the
E4B shared-KV residue alone, 18 layers of `k_proj`, `v_proj` and `k_norm`;
KB-82), `Activation rate:` and the `Record:` line for
`gemma4_text/42x2560/qlora/ckpt`, and `[INFO] Allocated error: not on
file`. `memfit`, once the rate is on file, prints `[OK] Fits:` with the
reserved estimate within 10 percent of the measured allocated plus 2.50 GB,
and exit 0. Record both rates beside each other: E2B's and E4B's per-token
figures are the two points that say whether activation cost grows with the
base the way the geometry says it should.

Observed:

```
```

Status: PENDING

## FT.train-resume Train, interrupt, resume

Implementation: `training/cli.py` (`cmd_train_sft`), `training/tune/sft.py`
(`resolve_sft_resume`, `check_memory_fit`), `training/tune/sft_callbacks.py`,
`training/runtime/checkpoint.py`, `training/runtime/lock.py` (the D9 GPU
lock), `training/models/peft_utils.py`.

Three runs of the same recipe: a short straight run, a run interrupted after
its first checkpoint, and the resume of that run from the named
`checkpoint-<step>` directory. Existing code; depends on FT.smoke-dataset and FT.memfit.

One override beyond the step counts, because the recipe's own value would
leave a criterion unobservable. `sft.eval_steps=30`: the recipe evaluates
every 100 steps, so a 60-step run would never compute a held-out loss and
the `Held-out loss:` line the tolerance criterion reads would be absent.
The base is the recipe's hub id resolved to the staged directory (KB-79,
fixed), and the dataset hash check reads the tokenizer from that same
directory (KB-80, fixed), so a `tokenizer hash ... does not match` error
here is one of them back. The PEFT wrap is the one FT.calibrate-qlora confirms (KB-83): a
`Gemma4ClippableLinear ... is not supported` error here is KB-83 back.

Command (inference down; the interrupt is a Ctrl-C after the first
`checkpoint-30` directory appears and before `checkpoint-60` does):

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-straight schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30"
echo "exit=$?"
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-resume schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
cat /mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30/meta.json
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-resume schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30 \
    checkpoint.resume=/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30"
echo "exit=$?"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
```

The checkpoints sit directly under `<output_root>/<run.name>/checkpoints/`
(`training/tune/sft.py`); the timestamped directory beside them holds the
recipe snapshot and the run's sinks, not the weights.

Expected signal, per run. The straight run prints `[RUN] Fine-tuning
ft3-straight [trl]: 60 total steps`, then `[OK] Memory fit:` (the advisory
line FT.memfit verified; a `[WARN]` here is a finding, not a stop, since
`train-sft` no longer refuses on it), then `[OK] Run complete: step 60 [trl]`, a `Final
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

## FT.export-merged Export the merged HF directory and load it in the harness

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
merge on real weights. Depends on FT.train-resume.

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
`sha256sum` line and different from the FT.preconditions base digest; the second export
exits 1 with `[ERROR] Export exists: .../hf` and the `--force` solution line,
leaving the manifest unchanged; `harness --status --deep` and
`checkpoint-info --check` both exit 0 on the directory, with `[OK] backend
compatibility: supported`, the same 35-layer config as the base, and
`[WARN] --deep: no effect when there is no interp artifact to check` (the
hash comparison is FT.probe's, against a probe); one `--chat` reply arrives,
rendered through the version 2 chat template.

Observed:

```
```

Status: PENDING

## FT.heldout-eval Held-out loss and lm-eval on base and merged

Implementation: `training/eval/holdout.py`, `training/eval/lmeval.py`,
`training/eval/tasks.yaml`, `common/reports.py`. Both verbs are CLI-only
(`python -m training eval-holdout`, `eval-lmeval`); no `make` target wraps
them, so they run inside the training container. Existing code; depends on
FT.export-merged (the merged directory) and FT.smoke-dataset (the dataset).

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

## FT.eval-ft eval-ft with both sides identified

Implementation: `training/commands/eval_ft.py` (`container_model`,
`phase_container`), `training/eval/runner.py` (`hf_identity`,
`run_container_phase`), `training/eval/scorecard.py`. Depends on FT.export-merged (the
subject) and FT.heldout-eval (the container evaluators the same run repeats).

**Fix to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-78 — `phase_container` handed `--base` and `--tuned` to the training
  container verbatim, and the host spelling this block uses does not exist
  there (the models tree is mounted at `/app/models` only), so the holdout
  phase would have failed at model load. `container_model` in
  `training/commands/eval_ft.py` now translates it; the `[RUN]
  holdout/base: /app/models/hf/gemma-4-E2B` line below is the evidence.

Both compared sides are HF directories under `/mnt/ai-data/models/`: the base
as FT.preconditions staged it and the merged directory as FT.export-merged wrote it. `eval-ft`
identifies each by `model_hash` (read from FT.export-merged's manifest for the tuned
side, computed for the base) and runs the two container evaluators on each.
A model-as-judge evaluator existed until 2026-09-19 and was removed: it
measured preference, and the harness-serving check it doubled as is FT.export-merged's.
What this entry pins beyond FT.heldout-eval is the identity block: a scorecard that
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
base's as FT.heldout-eval found, then one `[OK] lmeval/<task>: <a> -> <b>` per pinned
task; the run directory holds `scorecard.json` (shared report envelope,
`kind: scorecard`) and `scorecard.md`; the scorecard's metadata records
`thinking_mode: off` on both sides, `checkpoint` as each side's container
path, the base's `model_hash` equal to the FT.preconditions digest and the tuned side's
equal to FT.export-merged's manifest, and the rendered table shows the two hash
prefixes. An interrupted run resumes from its phase records (`[SKIP]
holdout/<side>: record exists (resume)`).

Observed:

```
```

Status: PENDING

## FT.organisms Organisms: null and taboo

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
`model_hash` from its manifest. Built by the exporter PR; depends on FT.export-merged
and FT.heldout-eval.

This entry supersedes I.organisms in [interp/interp.md](../interp/interp.md), which
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
names `checkpoint-400`; locality is measured on the FT.smoke-dataset ultrachat split and
the pinned lm-eval tasks, because the organism's own corpus would measure
the implant, not what it left alone. `wait_healthy` is given 600 s, the compose file's `start_period`,
rather than the registry's 60 s default, which a cold 10 GB load can miss.

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
make train-sft CONFIG=training/recipes/organisms/null.yaml
make export-hf CHECKPOINT=/mnt/ai-data/training/runs/organism-null/checkpoints/checkpoint-400 \
    ARGS="--name organism-null"
make eval-ft ARGS="--base /mnt/ai-data/models/hf/gemma-4-E2B \
    --tuned /mnt/ai-data/models/trained/organism-null/hf --evaluators holdout,lmeval \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --lmeval-limit 20 \
    --name ft7-null"
make gpu-up
HARNESS_MODEL=/app/models/trained/organism-null/hf make harness-serve
python3 -c 'from common import serving; print(serving.wait_healthy(serving.SERVICES["harness"], 600))'
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
seal the tamper arm, on a copy so the sealed organism FT.capture reads is not
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
beside the organism (FT.capture and FT.steering read them). Both trainings exit 0 at
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
exits 1 with a line beginning `[ERROR] Verdict: null control: implant check
PASSED on zero implant documents` (the note goes on to say the harness is
measuring an artifact), and that is the finding this entry exists to catch.
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
carries the plan and was gated, on 2026-09-19, on FT.train-resume, FT.export-merged and this entry
having recorded output: those three Observed blocks are what the refactor
must reproduce, and its acceptance check re-runs FT.train-resume and FT.export-merged against them.
Do it before FT.capture, since the entries from here on read the export this entry
and FT.heldout-eval produced and would otherwise be verified twice.

Observed:

```
```

Status: PENDING

## FT.capture Activation capture on base and organism

Implementation: `interp/commands/probe_capture.py`, `interp/probes/capture.py`
(`derive_model_hash`, plain forward hooks at the paths `interp/families.py`
names), `interp/organisms/generate.py` (`write_elicitation_sets`, the
labeled set). Existing code plus the labeled-set writer, added 2026-09-18;
depends on FT.organisms (the sealed taboo organism, and the labeled set its
generation wrote) and FT.preconditions (the base).

One labeled prompt set, captured over the same layers on the base and on the
taboo organism, so that FT.probe and FT.steering have a pair whose only difference is
the implant. The set is `elicit_labeled.jsonl` beside the organism, which
`organism generate` renders from the taboo templates: the 20 direct and 10
probe elicitations as label 1, the 16 neutral instructions as label 0, 46
lines. The templates carry no secret, so the file unblinds nothing. Until
2026-09-18 this entry named the file and nothing wrote it.

What the label separates, stated so the numbers downstream are read
correctly: label 1 is "a prompt that asks about the secret word" and label 0
is "a prompt that does not", and any model separates those by content. A
probe trained on either capture therefore reads prompt type, and its AUROC
is not evidence of the implant; the implant's evidence is FT.organisms's verifier
and FT.steering's steered arms. What this pair gives FT.probe is two captures of the
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
model_hash <64 hex>`, not the `[WARN] model_hash unknown` line I.probe-unchecked-load
describes: `derive_model_hash` hashes the single `model.safetensors`, which
FT.preconditions's listing shows the base has and FT.export-merged's export contract guarantees for
the organism, so the warning here would mean a sharded checkpoint, writing per-layer shards plus a manifest through the hooks the
family registry names for `gemma4_text`; each manifest's `model_hash` equals
the FT.preconditions digest for the base and the organism's FT.organisms export manifest for the
organism, and the two differ.

Observed:

```
```

Status: PENDING

## FT.probe Probe train, export, and score in the harness

Implementation: `interp/commands/probe_train.py`, `interp/probes/train.py`
(`split_of`, the seeded 80/10/10 split; it writes the registry manifest
itself), `interp/commands/probe_export.py` (`write_probe`, the safetensors
image plus `schemas/probe_artifact.json` sidecar, reading the manifest
through `interp/probes/registry.py`), `rust-harness/harness/src/preflight.rs`
(the identity check), `rust-harness/harness/src/cli.rs` (`--probe`,
`--alignment-out`). Existing code; depends on FT.capture.

The harness runs as the host binary here, not through `make harness-run`:
the `harness` compose service mounts the models tree and its session
directory and nothing else (`docker-compose.rust-harness.yaml`), so neither
`/app/probes` nor an output path for `--alignment-out` exists inside it;
`harness-serve` mounts the probes tree, but a one-shot task is not a
service. The host binary reads host paths, as FT.hardware-test's test and the harness
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

Expected signal: `probe-train` prints one `layer <n>: val <v>` line per
layer, selects the best by validation AUROC, prints `[OK] Probe registered: ft9-taboo-probe
(layer <l>, val AUROC <v>, test AUROC <t>)` and exits 0, with the registered
manifest carrying the organism's `model_hash`. The split is a seeded hash of
each text, so on 46 examples a split can land single-class; that refusal
reads `a split contains only one class; dataset is too small` and is
answered by `--seed <n>` on the same command, never by editing the set,
and the seed used is recorded here. Per FT.capture, the AUROC is prompt-type
separability and no threshold on it is claimed. `probe-export` writes
`probe.safetensors` and `probe.json` beside `probe.npz` and prints `Serve
with: make harness-serve HARNESS_PROBE=/app/probes/ft9-taboo-probe/probe.safetensors`.
On the organism, `--status --deep` exits 0 with `[OK] probe identity:
matches the served weights, sha256 <the organism's model_hash>`; the task
run writes the alignment file with one header line and then one record
per position of the reply (`wc -l` at least 2) and each record carrying a
finite score. On the base,
`--status --deep` exits 1 with the identity check naming both digests,
`artifact model_hash <organism hash>, served model sha256 <FT.preconditions digest>`,
because the probe was fitted on the organism's weights and the base is a
different checkpoint. A base run that exits 0 means the identity check is
not comparing.

Observed:

```
```

Status: PENDING

## FT.steering Steering vector from the organism capture

Implementation: `interp/commands/steer_extract.py`,
`interp/steering/extract.py` (`extract_diff_means`),
`interp/steering/export.py` (`write_control_vector`,
`schemas/control_vector_artifact.json`), `interp/commands/steer_scorecard.py`,
`interp/steering/scorecard.py` (`load_prompts`), `docker-compose.rust-harness.yaml`
(the `harness-serve` control-vector environment). Existing code; depends on
FT.capture (the organism capture) and FT.organisms (`elicit_prompts.jsonl`).

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
`[OK] <arm>: behavior <b>, fluency <f>, latency <t>s` per arm, `[OK]
Scorecard written:` and exit 0, writing `kind: steering_scorecard` (shared
envelope). The `system_prompt` arm runs the default honesty file (no taboo
system prompt exists, and a tuned baseline needs three recorded
iterations), so its number is recorded and not thresholded. The thresholds: `steered@1` behavior at least 1.0 above
`baseline` on the judge's 0-10 scale, and `steered@-1` at or below
`baseline`; that ordering is what shows the vector moves the implanted
behaviour in the direction its sign says. Fluency is recorded for each arm
and not thresholded, since I.steered-scorecard found it collapses at large scales. The last
`Control vector:` line of the log reads `none, serving unsteered`. A
`steered@1` below the baseline is a finding about the vector, not a number
to explain away.

Observed:

```
```

Status: PENDING

## FT.hardware-test The hardware test

Implementation: `tests/hardware/test_fine_tuning_loop.py::test_fine_tuning_loop`
(`hw_full`, `gpu`, `ssd`), running a ten-step QLoRA fine-tune of the smoke
recipe in-process, the export through `cmd_export`, and the two harness
verdicts in one function; the marker is defined in `pyproject.toml` and its
tiers in [tests/README.md](../../../tests/README.md). Its sandbox half,
`tests/training/test_export_harness.py`, runs the export and the same two
verdicts on the committed tiny fixture on every `make test`. Built by the
exporter PR; depends on FT.preconditions (the cached base) and FT.smoke-dataset (the prepared
dataset), which it skips without, and on the release binaries, which it
fails without.

The test runs in-process on the host, so its preconditions are the host's,
not a container's: the two release binaries (`cargo build --release` under
`rust-harness/`, which it fails without), the FT.smoke-dataset dataset at the recipe's
`sft.dataset`, and the recipe's hub id `google/gemma-4-E2B` resolved to
the staged directory at its host spelling (`training/models/staged.py`,
KB-79), since the test loads the recipe as committed. A skip reading
`pinned base not staged (FT.preconditions)` is that directory being absent,
and is recorded as the reason rather than treated as a pass. The
dataset hash check reads `tokenizer.json` from the base the recipe names
(KB-80, fixed), the same file FT.smoke-dataset tokenized with, so the test passes no
tokenizer override. Before training, the test reads the dataset's
`manifest.json` and fails, naming FT.smoke-dataset, when its
`chat_template_version` is not the one `training/models/families.py`
renders or its `tokenizer_hash` is not the pinned base's: a prepared
directory left over from an earlier template or base is a stale artifact,
not a pass, and the loader's own refusal would otherwise surface as an
exception from inside `train_sft`.

Command (the tier and invocation are the ones every hardware entry uses;
`--no-sync` because a plain `uv run` resyncs the default groups and, on a
host without a `flash-attn` wheel, starts a source build on every core):

```bash
set -o pipefail
cd ~/workspace/ai-experiments
ls rust-harness/target/release/harness rust-harness/target/release/checkpoint-info
ls "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--google--gemma-4-E2B/snapshots/"
uv run --no-sync pytest -m hw_full tests/hardware/test_fine_tuning_loop.py -v
```

Expected signal: the one test passes at `[TIER] full` with no skip, and its
assertions cover what FT.train-resume and FT.export-merged observe by hand: `checkpoint-10` is
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
