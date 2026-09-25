# Fine-tuning loop verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 5 verified, 0 blocked, 9 pending, of 14.**

Host verification of the fine-tuning loop, the first rung of the training tier: train
a Gemma 4 E2B with a known change, then recover it through the rust harness and the
interp tier.

An entry id is a name, not a position: it is assigned once, kept when the
title changes, and never reused. The Table of Contents is the execution
order, and `scripts/docs/check_runbook_ids.py` holds the sections to it. The
entries carried numbers until 2026-09-19, when every live one took a slug; the old numbers are in the retired-names table of
`scripts/docs/check_retired_names.py`, and the frozen records keep them.

Retired: FT.15 (2026-09-19, scorecard promotion), FT.memfit (2026-09-20, the memory-fit calculator left the tree), FT.e4b-recipe (2026-09-20, the E4B recipe card left the tree: no organism and no make target ran it, and the Phase 1 audit found nothing it verified that FT.calibrate-qlora does not).

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
  FT.image-smoke, FT.recipe-gates and FT.memfit (since retired) for the targets no entry ran (`train-build`,
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
  loop needs, and reduced it rather than rip it out or specify it fully:
  the static terms and the parameter count stayed (the count fixed for
  Gemma 4's two head widths, its norms, the per-layer embedding projection
  and the towers, KB-82), the analytic activation formula and `solve` left,
  activations became a rate `train-calibrate` measured per profile, and
  `train-sft` warned instead of refusing. FT.memfit was redefined to follow
  FT.calibrate-qlora and read the rate it recorded.
- **The calculator retired** (2026-09-20, after FT.calibrate-qlora's fifth
  attempt). Three days on it had filed seven ledger rows, two of them
  calculator defects and five training defects it was the first code to
  execute; its remaining prediction, a per-token rate needing one workstation
  run per family, width, mode and checkpointing flag, answered "does this
  fit" no earlier than the one-minute step it was calibrated against, on a
  one-GPU machine. The estimate, the rate table, `memfit`, the trainer's
  advisory line and FT.memfit left (its first pass under the analytic model
  went with the entry). What stayed moved to where its consumers are: the
  parameter count to `training/models/architecture.py`, and the calibration
  loop to `python -m training step` (`make train-step`,
  [train_step.md](../../training/base/train_step.md)), which builds the real
  model, steps it through the trainer's loss, checks the parameter count
  against the built model and prints the resident and peak figures.
  FT.calibrate-qlora keeps its slug and is retitled; FT.e4b-recipe (since
  retired) dropped its fit step; FT.train-resume no longer expects a `Memory fit:` line.
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
- **FT.calibrate-qlora's second attempt** (2026-09-19) got through the PEFT
  wrap, confirming KB-83, and died on the first forward pass:
  `FlashAttention forward only supports head dimension at most 256`.
  Gemma 4's full-attention layers are 512 wide, and the tree selected
  flash-attn whenever the image had it. KB-84, fixed the same day:
  `attention_implementation` reads the config and takes sdpa above the
  cap, and the provenance stamps read the loaded model. The paste is kept
  in FT.calibrate-qlora as a superseded observation; the entry is unchanged
  otherwise and re-runs as committed.
- **FT.calibrate-qlora's third attempt** (2026-09-19, as committed) selected
  sdpa, confirming KB-84, and ran the GPU out of memory in the first step:
  30.63 GiB allocated against a 21.46 GB estimate. Two defects and the
  recipe's geometry, all fixed the same day: PEFT's k-bit preparation upcast
  the 2.75 B frozen embedding parameters of E2B to fp32, about 5.5 GB the
  calculator never priced, and enabled checkpointing regardless of the
  recipe (KB-85; the wrap now owns that step and upcasts the norms only);
  calibrate reported the OOM as "run on the workstation" (KB-86; it now
  prints the resident and peak figures and names `data.micro_batch`); and
  8192 tokens per micro-batch is more than the 262144-wide head fits beside
  the base, so the two reference SFT recipes halve their micro-batch and
  double `grad_accum`. The paste is kept in FT.calibrate-qlora as a
  superseded observation; the entry re-runs as committed.
- **FT.calibrate-qlora's fourth attempt** (2026-09-20, as committed)
  confirmed KB-85 (6.37 GB resident after build, against the 6.08 GB
  `weights` row) and KB-86 (the new OOM line printed), and ran out of memory
  again at 4096 tokens per micro-batch: 30.57 GB peak, 24 GB above the
  resident model. That is the head: the tree's loss path held six copies of
  the `tokens x 262144` logits (bf16 through Gemma 4's softcap, fp32 through
  transformers' cross-entropy, their gradients back), about 6 bytes per
  vocabulary entry per token, outside every checkpointed block, and TRL's
  trainer read them twice more for metrics (KB-87, fixed the same day: the
  decoder runs once and the head, softcap and cross-entropy run per chunk of
  `memory.head_chunk_tokens` positions under torch checkpointing, in both
  calibrate and the SFT trainer). The same reading found the holdout
  evaluation accumulating full logits across the holdout (KB-88, fixed:
  `prediction_loss_only`). The paste is kept in FT.calibrate-qlora as a
  superseded observation; the entry re-runs as committed.
- **FT.calibrate-qlora's fifth attempt** (2026-09-20) printed the fourth
  attempt's figures to the hundredth of a GB after `git pull` reported
  `Already up to date`, and was read as a checkout without the KB-87 fix.
  The paste is kept as a superseded observation.
- **FT.calibrate-qlora's sixth attempt** (2026-09-20, the first through
  `make train-step`, so the chunked loss ran) printed the same figures
  again, which settled what an identical peak means: a peak read at an OOM
  is the card, and none of the three pastes had measured the head. The
  term the head hid is KB-89: the step loop never put the model in training
  mode, and transformers' checkpointing layers checkpoint only while
  `self.training`, so every decoder layer kept its activations at 4096
  tokens; the Trainer calls `train()` itself, so the loop was measuring a
  model `train-sft` never runs. Fixed the same day (`_build_step_model`
  returns the model in training mode); the paste is kept as a superseded
  observation and the entry re-runs as committed.
- **FT.calibrate-qlora's seventh attempt** (2026-09-20, as committed after
  KB-89) stepped: 6.37 GB resident, 10.67 GB peak allocated at 2 x 2048,
  confirming KB-87 and KB-89 and closing KB-03, KB-83 and KB-84 with them.
  Its two drift lines were findings: the adapter analytic priced one layer
  and multiplied, 5.2 M short of the 24,158,208 the built adapter carries
  (KB-90, fixed the same day, per-layer pricing pinned for E2B), and
  the documented shared-KV residue had been priced at the sliding width for
  all 20 shared layers, 15,733,760 against the 18,880,512 four full-attention
  layers make it; 928 of the measured 18,881,440 stays unexplained. The
  entry gains `tests/hardware/test_step_qlora.py` (`hw_full`) as its mirror
  and re-ran through it the same day: `1 passed`, no drift line, which
  confirms KB-90 and verifies the entry.
- **FT.train-resume's first attempt** (2026-09-25) stopped at the recipe
  validator before the model loaded: the entry's `schedule.total_steps=60`
  override left the recipe's `warmup_steps` at 100, which the validator
  refuses. A runbook defect, not a tree defect; the entry gains a
  `schedule.warmup_steps=3` override, the paste is kept as a superseded
  observation, and the entry re-runs as committed.

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

<!-- toc:start -->

1. [FT.preconditions Preconditions](#ftpreconditions-preconditions)
2. [FT.image-smoke Training image and smoke](#ftimage-smoke-training-image-and-smoke)
3. [FT.smoke-dataset Prepare the smoke SFT dataset](#ftsmoke-dataset-prepare-the-smoke-sft-dataset)
4. [FT.recipe-gates Recipe gates: preflight and validate](#ftrecipe-gates-recipe-gates-preflight-and-validate)
5. [FT.calibrate-qlora Step the QLoRA recipe on the GPU (KB-03)](#ftcalibrate-qlora-step-the-qlora-recipe-on-the-gpu-kb-03)
6. [FT.train-resume Train, interrupt, resume](#fttrain-resume-train-interrupt-resume)
7. [FT.export-merged Export the merged HF directory and load it in the harness](#ftexport-merged-export-the-merged-hf-directory-and-load-it-in-the-harness)
8. [FT.heldout-eval Held-out loss and lm-eval on base and merged](#ftheldout-eval-held-out-loss-and-lm-eval-on-base-and-merged)
9. [FT.eval-ft eval-ft with both sides identified](#fteval-ft-eval-ft-with-both-sides-identified)
10. [FT.organisms Organisms: null and taboo](#ftorganisms-organisms-null-and-taboo)
11. [FT.capture Activation capture on base and organism](#ftcapture-activation-capture-on-base-and-organism)
12. [FT.probe Probe train, export, and score in the harness](#ftprobe-probe-train-export-and-score-in-the-harness)
13. [FT.steering Steering vector from the organism capture](#ftsteering-steering-vector-from-the-organism-capture)
14. [FT.hardware-test The hardware test](#fthardware-test-the-hardware-test)

<!-- toc:end -->

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
or nothing): 15 cases since FT.e4b-recipe retired the E4B card on 2026-09-20,
18 while it was in the tree. That paste closes KB-79.

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

Re-ran the test half on 2026-09-20 after FT.e4b-recipe retired the E4B card,
together with FT.calibrate-qlora's mirror, to see the gates hold without the
E4B cases (the pull reported `Already up to date` at `634d982`, the head the
hand-over named):

Observed:

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
rm -rf training/calculator
git pull origin SFT-verification; git log --oneline -1
uv run --no-sync pytest -m "hw_smoke or hw_full" tests/hardware/test_recipe_gates.py tests/hardware/test_step_qlora.py -v; echo "exit=$?"
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch            SFT-verification -> FETCH_HEAD
Already up to date.
=============================================================================== test session starts ===============================================================================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/alejandro/workspace/ai-experiments/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/alejandro/workspace/ai-experiments
configfile: pyproject.toml
plugins: asyncio-1.4.0, mock-3.15.1, jaxtyping-0.3.11, cov-7.1.0, hydra-core-1.3.2, anyio-4.14.2, xdist-3.8.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 16 items

tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[false-fact.yaml] PASSED                                                               [  6%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[null.yaml] PASSED                                                                     [ 12%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[taboo.yaml] PASSED                                                                    [ 18%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[continual-gemma4-e2b.yaml] PASSED                                                     [ 25%]
tests/hardware/test_recipe_gates.py::test_the_hub_id_resolves_to_the_staged_directory[sft-qlora-e2b.yaml] PASSED                                                            [ 31%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[false-fact.yaml] PASSED                                                    [ 37%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[null.yaml] PASSED                                                          [ 43%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[taboo.yaml] PASSED                                                         [ 50%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[continual-gemma4-e2b.yaml] PASSED                                          [ 56%]
tests/hardware/test_recipe_gates.py::test_validate_passes_as_committed_and_refuses_a_negative_lr[sft-qlora-e2b.yaml] PASSED                                                 [ 62%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[false-fact.yaml] PASSED                                                              [ 68%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[null.yaml] PASSED                                                                    [ 75%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[taboo.yaml] PASSED                                                                   [ 81%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[continual-gemma4-e2b.yaml] PASSED                                                    [ 87%]
tests/hardware/test_recipe_gates.py::test_preflight_reports_the_dataset_never_the_base[sft-qlora-e2b.yaml] PASSED                                                           [ 93%]
tests/hardware/test_step_qlora.py::test_step_holds_the_counts_and_the_peak PASSED                                                                                           [100%]

================================================================================ warnings summary =================================================================================
tests/hardware/test_step_qlora.py: 14 warnings
  /home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

tests/hardware/test_step_qlora.py::test_step_holds_the_counts_and_the_peak
  /home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
    torch._check_is_size(blocksize)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[TIER] full: torch present, the whole suite was in scope.
======================================================================== 16 passed, 15 warnings in 10.16s =========================================================================
exit=0
```

15 cases, the five recipes by three gates, and the mirror, exit 0 at `[TIER]
full`: nothing the E4B card verified was lost with it, which is what its
retirement note claimed. The 15 warnings are the same torch and bitsandbytes
deprecations the mirror's first run printed.

## FT.calibrate-qlora Step the QLoRA recipe on the GPU (KB-03)

Implementation: `training/commands/train_step.py`,
`training/models/architecture.py`, `training/models/model.py`,
`training/models/peft_utils.py`, `training/models/head_loss.py`.

**Fixes to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-03 — the calibrate loop once built a scratch Llama model for every
  recipe, so calibrating a `model.hf` or PEFT recipe measured a model other
  than the recipe's. The fix routes the build through the shared
  `build_model` and the PEFT wrap, which `python -m training step` inherits;
  this run on the QLoRA recipe is the hardware evidence that closes the row.
- KB-82 — the Gemma 4 parameter count missed the seven
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
- KB-84 — flash-attn was selected for a base whose full-attention heads are
  512 wide, above the kernel's 256 cap (this entry's second attempt). The
  `[model] ... attention runs on sdpa` note the third attempt printed is the
  confirmation; the row closes with this run.
- KB-87 — the loss materialized the full `micro_batch x seq_len x 262144`
  logits and held six copies of them through Gemma 4's softcap,
  transformers' fp32 cross-entropy and the backward, 24 GB at 4096 tokens
  per micro-batch beside a 6.37 GB resident model (this entry's fourth
  attempt). `training/models/head_loss.py` now runs the head, softcap and
  cross-entropy per chunk of `memory.head_chunk_tokens` positions under
  torch checkpointing, in the step loop and in the SFT trainer alike. The
  `Peak allocated:` figure is the confirmation: on the order of 12 to 15 GB
  at the committed geometry, not 30.
- KB-88 — the SFT trainer's holdout evaluation left `prediction_loss_only`
  at the Trainer default and would have accumulated every eval batch's
  262144-wide logits across the holdout. Fixed with the KB-87 loss (there
  are no logits to accumulate); confirmed by FT.train-resume's first
  evaluation, not here.

Steps the QLoRA recipe on the `model.hf` plus PEFT path, which the archived
P0.3 scratch calibration does not exercise: the NF4 base resident on the
card, adapter gradients and optimizer state, and the peak of five real steps
at the recipe's geometry, which FT.train-resume launches on. It is also the
first execution evidence for Gemma 4 on the workstation: the gated checkpoint
at its pinned revision, transformers `gemma4` support, and bitsandbytes NF4
on Blackwell. Existing code; depends on FT.preconditions. Until 2026-09-20
the entry calibrated the memory-fit calculator's activation rate (the five
superseded attempts below ran `make train-calibrate`); the calculator
retired that day and the entry became the step it always was.

Command:

```bash
set -o pipefail
cd ~/workspace/ai-experiments
git log --oneline -1
make train-step CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/step-sft-e2b.json"; echo "exit=$?"
uv run --no-sync pytest -m hw_full tests/hardware/test_step_qlora.py -v; echo "exit=$?"
```

The `git log` line is part of the record since the fifth attempt, which ran
a checkout that had not received the fix it was meant to confirm.

The base is the recipe's hub id resolved to the staged directory (KB-79,
fixed 2026-09-19; the superseded attempts below passed it as an override).
Attention runs on sdpa: FlashAttention 2 refuses a head dimension above
256 and Gemma 4's full-attention layers are 512 wide, so
`attention_implementation(config)` selects sdpa for every Gemma 4 base
even though the image ships flash-attn (KB-84, found by the second attempt
below). A `FlashAttention forward only supports head dimension at most
256` error here is KB-84 back. The recipe steps at its committed
geometry, 2 x 2048 tokens per micro-batch since the third attempt, through
the chunked head loss the SFT trainer uses (KB-87, the fourth attempt): the
head, softcap and cross-entropy run `memory.head_chunk_tokens` positions at
a time, so the 262144-wide vocabulary is resident one chunk at a time
whatever the micro-batch. Depends on FT.preconditions and FT.image-smoke.

Expected signal: exit 0. `[OK] Parameter count: <p> (analytic matches
model)`, or a `[WARN] Parameter count drift` whose two figures differ by
18,881,440: the shared-KV residue the modeling code does not construct,
18,880,512 (`k_proj`, `v_proj` and `k_norm` on the 20 shared layers at each
layer's width), plus 928 the seventh attempt measured and nothing yet
explains (KB-82; any other drift is a finding); no trainable-count drift
warning (KB-90). Then `Resident after build:` near 6.4 GB (KB-85), `Peak
allocated:` well under the card (12 to 15 GB is the expectation at this
geometry with a 1024-token head chunk; 30 GB is KB-87 or KB-89 back), `Peak
reserved:`, `NVML peak:`, `Tokens per micro-batch: 4,096` and `[OK] Stepped
5 steps`. Record the four figures here; FT.train-resume launches on this
geometry and reads nothing else from this entry. Then the hardware mirror
passes its one test, holding the same four figures.

A later step of the same recipe (after a transformers or TRL upgrade) is the
drift check: a peak that moved is a finding to read, not a number to argue
with.

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

Second attempt (2026-09-19, superseded, kept verbatim: run before the
KB-79 fix was pulled, so still with the override; the wrap passed, which
is KB-83's confirmation, and the first forward pass died in the
FlashAttention kernel, which is KB-84):

```
➜  ai-experiments git:(SFT-verification) ✗ make train-calibrate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"
docker compose -f docker-compose.training.yaml run --rm training python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml model.hf.id=/app/models/hf/gemma-4-E2B --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-dadfcf03375d Creating
Container ai-experiments-training-run-dadfcf03375d Created
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
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 2965.80it/s]
[transformers] Casting fp32 inputs back to torch.bfloat16 for flash-attn compatibility.
[ERROR] Calibration failed: FlashAttention forward only supports head dimension at most 256
  Solution: run on the workstation via 'make train-calibrate CONFIG=...'

make: *** [make/training.mk:53: train-calibrate] Error 1
```

The wrap no longer names a vision-tower module, so KB-83 is confirmed
fixed by this attempt. The forward pass failed in flash-attn's kernel: the
model loaded with `attn_implementation=flash_attention_2`, the host
resolution, and Gemma 4's full-attention layers carry `global_head_dim`
512, above the kernel's cap of 256 (`schemas/gemma4_facts.json` records
both widths). Every Gemma 4 load in the tree, the trainer, the calculator,
the evaluators and the hardware tests, would have hit the same wall on the
first step; the from-scratch loop did not, because its Llama-shaped
scratch model has 128-wide heads. Fixed in the tree the same day:
`attention_implementation` takes the config and returns sdpa when its
widest head exceeds the cap, with one note on stderr, and the checkpoint
and evaluation records stamp the implementation the loaded model reports
rather than the host resolution.

Third attempt (2026-09-19, superseded, kept verbatim: as committed, the
first run with no override; attention on sdpa, which confirms KB-84; the
first calibration step ran the GPU out of memory, which is KB-85 and KB-86):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
make train-calibrate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-f81262b0824e Creating
Container ai-experiments-training-run-f81262b0824e Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
[RUN] Calibrating sft-qlora-e2b: 5 steps on cuda:0
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
Loading weights:   0%|                                                                                                                                   | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 2956.51it/s]
[W919 22:23:36.107218551 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 50331648 bytes (free: 37552128, total: 33668726784).
[W919 22:23:36.114189739 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 67108864 bytes (free: 45940736, total: 33668726784).
[W919 22:23:36.117163322 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 50331648 bytes (free: 29163520, total: 33668726784).
[W919 22:23:36.117209749 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 50331648 bytes (free: 29163520, total: 33668726784).
[ERROR] Calibration failed: CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 27.81 MiB is free. Including non-PyTorch memory, this process has 31.32 GiB memory in use. Of the allocated memory 30.63 GiB is allocated by PyTorch, and 116.78 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
  Solution: run on the workstation via 'make train-calibrate CONFIG=...'

make: *** [make/training.mk:53: train-calibrate] Error 1
exit=2
```

The model loaded and wrapped, and the first step allocated 30.63 GiB of
the 31.36 GiB card before the failed 48 MiB request, against a memfit
estimate of 21.46 GB reserved. Three things account for the gap. PEFT's
`prepare_model_for_kbit_training`, which `build_peft_model` called, upcasts
every non-4-bit bf16 parameter to fp32; bitsandbytes quantizes `nn.Linear`
only, so E2B's token embedding (262144 x 1536) and per-layer embedding
table (262144 x 35 x 256), about 2.75 B frozen parameters, went from bf16
to fp32, roughly 5.5 GB the calculator's `weights` row prices at bf16
(KB-85, fixed the same day: the wrap prepares the base itself, upcasting
the one-dimensional norms only, and leaves checkpointing to the recipe).
The head materializes `micro_batch x seq_len x 262144` logits, 4.3 GB in
bf16 at 8192 tokens, and transformers' causal-LM loss holds them in fp32
through backward with the softmax and the logit gradient at the same size,
a term outside every checkpointed block; the 7 full-attention layers add
the sdpa math kernel's `[4, 8, 2048, 2048]` scores, since PyTorch's fused
kernels cap the head width as flash-attn does. That is the recipe's
geometry, not a defect: `sft-qlora-e2b.yaml` now runs 2 x 2048 with
`grad_accum` 16 (the same 32 sequences per step), `sft-qlora-gemma-e4b.yaml`
1 x 4096 with 32, and `tests/training/test_recipe_pins.py` holds every
Gemma 4 recipe to 4096 tokens per micro-batch. Last, the failure landed in
calibrate's generic arm with the no-GPU remedy (KB-86, fixed the same day:
`CalibrationOutOfMemory` carries the resident and peak figures and the CLI
names `data.micro_batch`). The geometry reading was incomplete: the fourth
attempt measured the head's term at 4096 tokens and it is KB-87, below.

Fourth attempt (2026-09-20, superseded, kept verbatim: as committed after
KB-85 and KB-86; both confirmed, and the head's logits term ran the first
step out of memory, which is KB-87):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
make train-calibrate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-86c7aafbe27f Creating 
Container ai-experiments-training-run-86c7aafbe27f Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
[RUN] Calibrating sft-qlora-e2b: 5 steps on cuda:0
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
Loading weights:   0%|                                                                                                                                   | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 2953.83it/s]
[W920 00:15:29.007096531 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 201326592 bytes (free: 27066368, total: 33668726784).
[W920 00:15:29.007153478 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 201326592 bytes (free: 27066368, total: 33668726784).
[ERROR] Calibration ran out of GPU memory at step 1: resident after build 6.37 GB, peak allocated 30.57 GB of 31.36 GB, at 4,096 tokens per micro-batch
  Solution: lower data.micro_batch (the rate is per token; memfit solves the batch once the rate is on file)

make: *** [make/training.mk:53: train-calibrate] Error 1
exit=2
```

`Resident after build 6.37 GB` is the KB-85 confirmation: the calculator's
`weights` row is 6.08 GB, the adapter and the CUDA workspaces are the rest,
and the 5.5 GB upcast is gone. The line itself is the KB-86 confirmation.
The peak, 30.57 GB, is 24.2 GB above the resident model at 4096 tokens per
micro-batch, and the head accounts for that much by itself (the sixth
attempt found a second term underneath it, KB-89). One `4096 x 262144` copy
is 2.15 GB in bf16 and 4.3 GB in fp32; the tree's loss path
(`Gemma4ForConditionalGeneration.forward`, transformers' `ForCausalLMLoss`)
kept the `lm_head` output, the softcap's `tanh` output and product, the
`.float()` copy and the saved log-softmax alive for the backward, and the
backward added the fp32 logit gradient and its bf16 chain: about 6 bytes
per vocabulary entry per token, 24 GB, outside every checkpointed decoder
block; in `train-sft` TRL's `compute_loss` read the same logits twice more
for its entropy and accuracy metrics. Halving the micro-batch again would
have fit E2B and not E4B (one 4096-token sequence is the same 4096 tokens).
Fixed the same day: `training/models/head_loss.py` runs the decoder once
and then the head, softcap and cross-entropy over `memory.head_chunk_tokens`
positions (1024) at a time under `torch.utils.checkpoint`, so one chunk's
logits are alive in the forward and again in the backward; the calibration
loop and the SFT trainer (`compute_chunked_loss`, replacing TRL's
`compute_loss` and its logits-reading metrics) both go through it, and the
trainer sets `prediction_loss_only` so evaluation stops accumulating logits
it never read (KB-88). The geometry stays at 2 x 2048. Re-runs as committed.

Fifth attempt (2026-09-20, superseded, kept verbatim: `git pull` reported
`Already up to date` and the figures are the fourth attempt's to the
hundredth of a GB. Read at the time as a checkout that had not received the
KB-87 fix; the sixth attempt, on a checkout that provably carried it, printed
the same figures, so the identical peak was the card, not the checkout, and
this paste is as much evidence of KB-89 as the sixth is):

```
➜  ai-experiments git:(SFT-verification) ✗ set -o pipefail
cd ~/workspace/ai-experiments
git pull origin SFT-verification
make train-calibrate CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"; echo "exit=$?"
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch            SFT-verification -> FETCH_HEAD
Already up to date.
docker compose -f docker-compose.training.yaml run --rm training python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-89daa469c0c7 Creating 
Container ai-experiments-training-run-89daa469c0c7 Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training calibrate --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/calibrate-sft-e2b.json
[RUN] Calibrating sft-qlora-e2b: 5 steps on cuda:0
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
Loading weights:   0%|                                                                                                                                   | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 2931.24it/s]
[W920 00:30:59.679843114 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 201326592 bytes (free: 35454976, total: 33668726784).
[W920 00:30:59.679894561 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 201326592 bytes (free: 35454976, total: 33668726784).
[ERROR] Calibration ran out of GPU memory at step 1: resident after build 6.37 GB, peak allocated 30.57 GB of 31.36 GB, at 4,096 tokens per micro-batch
  Solution: lower data.micro_batch (the rate is per token; memfit solves the batch once the rate is on file)

make: *** [make/training.mk:53: train-calibrate] Error 1
exit=2
```

The same day the memory-fit calculator retired and this entry became the
step it always was (`make train-step`, above).

Sixth attempt (2026-09-20, superseded, kept verbatim: the first run through
`make train-step`, which exists only in a checkout carrying the retirement
commit, so the chunked loss ran this time and the peak did not move, which is
KB-89; the operator pasted from the compose output onward, without the
`git log` line):

```
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-80663815f1ef Creating
Container ai-experiments-training-run-80663815f1ef Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training step --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/step-sft-e2b.json
[RUN] Stepping sft-qlora-e2b: 5 steps on cuda:0
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
Loading weights:   0%|                                                                                                                                   | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 2859.50it/s]
[W920 01:03:14.394287229 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 201326592 bytes (free: 35454976, total: 33668726784).
[W920 01:03:14.394343706 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 201326592 bytes (free: 35454976, total: 33668726784).
[ERROR] Step ran out of GPU memory at step 1: resident after build 6.37 GB, peak allocated 30.57 GB of 31.36 GB, at 4,096 tokens per micro-batch
  Solution: lower data.micro_batch or data.seq_len

make: *** [make/training.mk:50: train-step] Error 1
exit=2
```

The figures are the fourth and fifth attempts' to the hundredth of a GB, and
that is the finding: a peak read at an OOM is the card less the request that
failed (31.36 GB less 192 MiB of allocator slack is 30.57 GB whatever was
asked for), so none of the three pastes measured a term, and the fourth
attempt's "24 GB is the head" was the head plus a second term the head had
hidden. `from_pretrained` returns the model in eval mode, PEFT's wrap keeps
it there, and nothing in the step loop called `train()`; transformers'
`GradientCheckpointingLayer` checkpoints only while `self.training`, so
every one of E2B's 35 decoder layers kept its activations for the backward
at 4096 tokens, the 7 full-attention layers with the sdpa math kernel's
fp32 scores among them. The Trainer calls `train()` itself before its first
step, which is why `train-sft` would not have shown this and why the step
loop was measuring a model the trainer never runs (KB-89, fixed the same
day: `_build_step_model` returns the model in training mode, and
`tests/training/test_cli_handlers_step.py` holds it). Re-runs as committed.

Seventh attempt (2026-09-20, as committed after KB-89; the operator pasted
from the pull onward, so the `git log` line is absent, and the fast-forward
to `a199bd1` shows the checkout):

```
make train-step CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="--out /mnt/ai-data/training/verify/step-sft-e2b.json"; echo "exit=$?"
remote: Enumerating objects: 35, done.
remote: Counting objects: 100% (35/35), done.
remote: Compressing objects: 100% (1/1), done.
remote: Total 18 (delta 15), reused 18 (delta 15), pack-reused 0 (from 0)
Unpacking objects: 100% (18/18), 5.41 KiB | 1.80 MiB/s, done.
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch              SFT-verification -> FETCH_HEAD
   9ae4b3aa..a199bd18  SFT-verification -> origin/SFT-verification
Updating 9ae4b3a..a199bd1
Fast-forward
 docs/training/base/train_step.md                    | 12 +++++++++---
 docs/verification/bugs/known-bugs.md           |  7 ++++---
 docs/verification/training/fine-tuning-loop.md | 85 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--------------
 tests/training/test_cli_handlers_step.py       | 27 ++++++++++++++++++++++-----
 training/commands/train_step.py                |  9 ++++++++-
 training/models/head_loss.py                   |  4 +++-
 6 files changed, 117 insertions(+), 27 deletions(-)
?? neuronpedia_replacement_investigation.md
?? rust-harness/.gdb_history
?? rust-harness/fixtures/reference/e2b/
?? rust-harness/flamegraph.svg
?? rust-harness/runtime/tests/data/
?? rust-harness/runtime/tests/rh5_replay.rs
?? rust-harness/session.kv
docker compose -f docker-compose.training.yaml run --rm training python -m training step --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/step-sft-e2b.json
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-1f5c8f2e3598 Creating
Container ai-experiments-training-run-1f5c8f2e3598 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training step --config training/recipes/reference/sft-qlora-e2b.yaml --out /mnt/ai-data/training/verify/step-sft-e2b.json
[RUN] Stepping sft-qlora-e2b: 5 steps on cuda:0
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
Loading weights:   0%|                                                                                                                                   | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 2965.15it/s]
[OK] Step record written: /mnt/ai-data/training/verify/step-sft-e2b.json
[WARN] Parameter count drift: model 5,128,455,712 vs analytic 5,142,102,464
  Solution: align the param formulas in training/models/architecture.py with the built model
[WARN] Trainable count drift: model 24,158,208 vs analytic 18,923,520
  Solution: align adapter_param_count in training/models/architecture.py with the built adapter

Resident after build:   6.37 GB
Peak allocated:         10.67 GB
Peak reserved:          12.60 GB
NVML peak:              13.68 GB
Tokens per micro-batch: 4,096
[OK] Stepped 5 steps
exit=0
```

The step ran: `Resident after build: 6.37 GB` again, and `Peak allocated:
10.67 GB` where the sixth read 30.57, which confirms KB-87 and KB-89
together (the reserved and NVML figures, 12.60 and 13.68 GB, are the
allocator's and the driver's slack over it). KB-03, KB-83 and KB-84 close
with it: the model stepped is the recipe's, wrapped, on sdpa. The two drift
lines are findings against the expected signal, both read from the figures
without a second run:

- The trainable drift is exact: 24,158,208 is rank 16 over the seven
  modules priced per layer on what the modeling code builds (the seven
  full-attention layers at 512-wide heads, the double-wide MLP on the
  20-layer tail, and no `k_proj` or `v_proj` on that tail, which shares
  another layer's K and V), and 18,923,520 is the same rank over every
  layer at the sliding width with a flat MLP. `adapter_param_count` priced
  one layer and multiplied (KB-90, fixed the same day: it prices each layer,
  and `tests/training/test_architecture.py` pins both variants).
- The parameter drift is 18,881,440, not the 15,733,760 the expected signal
  named: that figure priced the residue at the sliding width for all 20
  shared layers, and 4 of them are full-attention layers with 512-wide
  `k_proj`, `v_proj` and `k_norm` (16 x 786,688 + 4 x 1,573,376 =
  18,880,512). The documented figure is corrected; the remaining 928 has no
  explanation yet and is pinned as measured.

The entry gains a hardware mirror, `tests/hardware/test_step_qlora.py`
(`hw_full`, `gpu`), which steps the recipe as committed and holds the four
figures: the adapter count equal, the parameter drift exactly 18,881,440,
resident under 7 GB and peak allocated under 14 GB. Re-ran as committed
through it the same day (the operator pasted from the pull's last `create
mode` line onward, so the `git log` line is absent; the pull is the
fast-forward to `c6a7181`, the commit that added the mirror):

Observed:

```
 create mode 100644 tests/hardware/test_step_qlora.py
=============================================================================== test session starts ===============================================================================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/alejandro/workspace/ai-experiments/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/alejandro/workspace/ai-experiments
configfile: pyproject.toml
plugins: asyncio-1.4.0, mock-3.15.1, jaxtyping-0.3.11, cov-7.1.0, hydra-core-1.3.2, anyio-4.14.2, xdist-3.8.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1 item                                                                                                                                                                  

tests/hardware/test_step_qlora.py::test_step_holds_the_counts_and_the_peak PASSED                                                                                           [100%]

================================================================================ warnings summary =================================================================================
tests/hardware/test_step_qlora.py: 14 warnings
  /home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

tests/hardware/test_step_qlora.py::test_step_holds_the_counts_and_the_peak
  /home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
    torch._check_is_size(blocksize)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[TIER] full: torch present, the whole suite was in scope.
========================================================================= 1 passed, 15 warnings in 7.30s ==========================================================================
exit=0
```

`1 passed`, exit 0, with no drift line and no assertion message: the built
adapter equals the analytic 24,158,208 (KB-90 confirmed), the parameter
drift is exactly 18,881,440 (the corrected residue plus the 928 KB-82 keeps),
resident under 7 GB and peak allocated under 14 GB. The 15 warnings are torch
and bitsandbytes deprecations, not the tree's. This is the entry's re-run and
its standing mirror: `make quality-full HW=full` repeats it.
The mirror ran again on 2026-09-20 at `634d982` beside FT.recipe-gates' re-run
(the paste is recorded there): `16 passed`, exit 0, the mirror unchanged.

Status: VERIFIED 2026-09-20 on the seventh attempt through the mirror. Was
PENDING through six attempts the same two days: the first two on KB-83 and
KB-84 (fixed in the tree), the third on KB-85 and KB-86, the fourth, fifth and
sixth out of memory at the card on KB-87 and then KB-89, each fixed in the
tree before the next; the seventh stepped and its two drift lines were read
as KB-90 and the residue correction without another run.

## FT.train-resume Train, interrupt, resume

Implementation: `training/cli.py` (`cmd_train_sft`), `training/tune/sft.py`
(`resolve_sft_resume`), `training/tune/sft_callbacks.py`,
`training/runtime/checkpoint.py`, `training/runtime/lock.py` (the D9 GPU
lock), `training/models/peft_utils.py`.

**Fixes to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-80 — every SFT recipe once named a tokenizer directory nothing staged,
  and `build_dataset` hash-checked the dataset against it, so `train-sft`
  failed at dataset load. FT.recipe-gates confirmed the preflight half; the
  `train-sft` half closes here, when the straight run loads the dataset
  without a `tokenizer hash ... does not match` error and reaches its first
  step.
- KB-88 — the holdout evaluation left `prediction_loss_only` at the Trainer
  default and would have gathered every eval batch's 262144-wide logits
  across the holdout. Fixed with the KB-87 loss; the straight run's
  `Held-out loss:` line, the first `eval_loss` the tree has computed, is
  the confirmation.

Three runs of the same recipe: a short straight run, a run interrupted after
its first checkpoint, and the resume of that run from the named
`checkpoint-<step>` directory. Existing code; depends on FT.smoke-dataset and FT.calibrate-qlora.

Two overrides beyond the step counts, because the recipe's own values do
not fit a 60-step run. `schedule.warmup_steps=3`: the recipe warms up for
100 steps, and the validator refuses a warmup longer than the run
(`training/recipes/validate.py`), which is what the first attempt below
hit; 3 keeps the recipe's 100-in-2000 ratio, and the straight and resumed
runs share it, as the tolerance criterion requires. `sft.eval_steps=30`:
the recipe evaluates every 100 steps, so a 60-step run would never compute
a held-out loss and the `Held-out loss:` line the tolerance criterion reads
would be absent.
The evaluation runs at steps 30 and 60 and prints nothing while it runs;
the one `Held-out loss:` line is the last `eval_loss` in the trainer's log
history, the step-60 value (`training/tune/sft.py`, after `trainer.train`).
Each evaluation reads the smoke dataset's whole holdout split (4066
examples, FT.smoke-dataset) at the recipe's micro-batch of 2, about 2033
eval batches twice per run, and is the entry's dominant wall-clock cost: a
long silence after step 30 or step 60 is the evaluation, not a hang.
The base is the recipe's hub id resolved to the staged directory (KB-79,
fixed), and the dataset hash check reads the tokenizer from that same
directory (KB-80, fixed), so a `tokenizer hash ... does not match` error
here is one of them back. The PEFT wrap is the one FT.calibrate-qlora confirms (KB-83): a
`Gemma4ClippableLinear ... is not supported` error here is KB-83 back.

The interrupt is one Ctrl-C in the terminal running the second command,
sent after `meta.json` exists inside `checkpoint-30` and before `checkpoint-60`
appears. `meta.json` is the last file the checkpoint callback writes
(`training/tune/sft_callbacks.py`, after transformers' `trainer_state.json`),
so its presence is what makes `checkpoint-30` complete; the directory
itself appears earlier, mid-save. Watch for it from a second terminal:

```bash
watch -n 5 ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30/
```

One Ctrl-C only. The compose service runs under tini (`init: true`), so a
single SIGINT reaches the trainer as a `KeyboardInterrupt`, nothing in
`training/cli.py` catches it, and the process ends with a traceback while
the lock context manager (`training/runtime/lock.py`) removes the training
lock on the way out. A second Ctrl-C kills the container hard and leaves
the lock behind, which the next run reports as `Stale training lock` with
the `rm` to run; that is operator error, not a finding.

Command (inference down):

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-straight schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30"
echo "exit=$?"
ls /mnt/ai-data/training/runs/ft3-straight/checkpoints/
cat /mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60/meta.json
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-resume schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30"
echo "exit=$?"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
cat /mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30/meta.json
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-resume schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30 \
    checkpoint.resume=/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30"
echo "exit=$?"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
```

The checkpoints sit directly under `<output_root>/<run.name>/checkpoints/`
(`training/tune/sft.py`); the timestamped directory beside them holds the
recipe snapshot and the run's sinks, not the weights.

Expected signal, per run. The straight run prints `[RUN] Fine-tuning
ft3-straight [trl]: 60 total steps` (no `Memory fit:` line: the advisory
left with the calculator on 2026-09-20, and FT.calibrate-qlora is the fit
evidence), then `[OK] Run complete: step 60 [trl]`, a `Final
train loss:` line, a `Held-out loss:` line and exit 0; `checkpoint-30` and
`checkpoint-60` each carry `meta.json` with `checkpoint_format: trl`,
`adapter_only: true`, the base identity including the pinned
`base_revision` `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`, and the resolved
`attention_implementation`. The interrupted run prints the same `[RUN]`
line for `ft3-resume`, ends in a `KeyboardInterrupt` traceback with a
non-zero exit, and leaves `checkpoint-30` complete (`trainer_state.json`
and `meta.json` present) and no `checkpoint-60`. A `checkpoint-60`
directory without `meta.json` means the Ctrl-C landed inside the step-60
save: remove the `ft3-resume` run directory and repeat the interrupted run,
because a resume from that state proves nothing. The resumed run prints
`[RUN] Resuming (/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30)
ft3-resume [trl]: 60 total steps` (the verb carries the override as
passed), ends at `[OK] Run complete: step 60 [trl]` with exit 0, and its
`ls` shows `checkpoint-60` beside `checkpoint-30`. The criterion that makes
the resume a resume: the resumed run's `Held-out loss:` agrees with the
straight run's within `RESUME_LOSS_RTOL` (0.01, `tests/hardware/test_gpu.py`),
that is `abs(a - b) / a <= 0.01`; record both numbers and the quotient.
The constant is borrowed from `test_resume_loss_curve_within_tolerance`,
which holds the from-scratch loop's per-step losses to it on the GPU; no
hardware test mirrors a TRL resume, so this entry is the only evidence for
one. A resumed run that reports `Fine-tuning` rather than `Resuming`, or
ends at a step other than 60, has restarted rather than resumed.

Observed:

First attempt (2026-09-25, superseded, kept verbatim: stage 1 as the entry
then stood, without the warmup override; the validator refused the recipe
before anything loaded, and no GPU work ran):

```
➜  ai-experiments git:(SFT-verification-2) cd ~/workspace/ai-experiments
git pull origin SFT-verification-2
git log -1 --oneline
set -o pipefail
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-straight schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30"
echo "exit=$?"
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch            SFT-verification-2 -> FETCH_HEAD
Already up to date.
docker compose -f docker-compose.training.yaml run --rm training python -m training train-sft --config training/recipes/reference/sft-qlora-e2b.yaml run.name=ft3-straight schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30
WARN[0001] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-25153eb6e87f Creating 
Container ai-experiments-training-run-25153eb6e87f Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training train-sft --config training/recipes/reference/sft-qlora-e2b.yaml run.name=ft3-straight schedule.total_steps=60 checkpoint.interval_steps=30 sft.eval_steps=30
[ERROR] Recipe validation failed:
  - schedule: warmup_steps (100) exceeds total_steps (60)

make: *** [make/training.mk:56: train-sft] Error 1
exit=2
```

Straight run (2026-09-25, as committed after the warmup fix; the `ls` and
`cat` of its checkpoints were added to the command block afterwards and are
recorded with the interrupted run):

```
➜  ai-experiments git:(SFT-verification-2) make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-straight schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training python -m training train-sft --config training/recipes/reference/sft-qlora-e2b.yaml run.name=ft3-straight schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-596881c37182 Creating 
Container ai-experiments-training-run-596881c37182 Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training train-sft --config training/recipes/reference/sft-qlora-e2b.yaml run.name=ft3-straight schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30
[RUN] Fine-tuning ft3-straight [trl]: 60 total steps
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
Loading weights:   0%|                                                                                                                                 | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:03<00:00, 505.89it/s]
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 1, 'bos_token_id': 2, 'pad_token_id': 0}.
{'loss': '1.469', 'grad_norm': '0.5555', 'learning_rate': '0.0001946', 'epoch': '0.002076'}                                                                                     
 18%|█████████████████████████▍                                                                                                                 | 11/60 [02:58<13:35, 16.64s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
{'loss': '1.231', 'grad_norm': '0.3388', 'learning_rate': '0.0001636', 'epoch': '0.004152'}                                                                                     
 45%|██████████████████████████████████████████████████████████████▌                                                                            | 27/60 [07:14<08:47, 15.99s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
 48%|███████████████████████████████████████████████████████████████████▏                                                                       | 29/60 [07:44<07:56, 15.37s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
{'loss': '1.102', 'grad_norm': '0.359', 'learning_rate': '0.0001137', 'epoch': '0.006227'}                                                                                      
 50%|█████████████████████████████████████████████████████████████████████▌                                                                     | 30/60 [08:00<07:52, 15.77s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 155/2033 [00:26<04:13,  7.40it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 493/2033 [01:26<04:51,  5.29it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5████▉                                                                                    | 776/2033 [02:16<03:47,  5.52it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5██████████████████████████████████▌                                                     | 1229/2033 [03:36<02:14,  5.98it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5██████████████████████████████████████████▏                                             | 1343/2033 [03:56<02:27,  4.69it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5████████████████████████████████████████████████████████████▋                           | 1622/2033 [04:46<01:10,  5.79it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5███████████████████████████████████████████████████████████████████████▉                | 1791/2033 [05:17<00:46,  5.22it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5███████████████████████████████████████████████████████████████████████████████████     | 1959/2033 [05:46<00:14,  5.27it/s]
{'eval_loss': '1.038', 'eval_runtime': '361.7', 'eval_samples_per_second': '11.24', 'eval_steps_per_second': '5.621', 'epoch': '0.006227'}                                      
 52%|███████████████████████████████████████████████████████████████████████▎                                                                  | 31/60 [14:16<59:49, 123.78s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                                                            
 65%|██████████████████████████████████████████████████████████████████████████████████████████▎                                                | 39/60 [16:26<07:41, 21.98s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
{'loss': '1.099', 'grad_norm': '0.2819', 'learning_rate': '5.983e-05', 'epoch': '0.008303'}                                                                                     
{'loss': '1.085', 'grad_norm': '0.6077', 'learning_rate': '1.782e-05', 'epoch': '0.01038'}                                                                                      
 87%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▍                  | 52/60 [19:59<02:13, 16.68s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
 88%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                | 53/60 [20:14<01:52, 16.12s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
 95%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████       | 57/60 [21:20<00:49, 16.57s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
{'loss': '1.078', 'grad_norm': '0.3607', 'learning_rate': '1.518e-07', 'epoch': '0.01245'}                                                                                     
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [22:06<00:00, 15.74s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                          | 36/2033 [00:06<06:08,  5.41it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 149/2033 [00:26<05:11,  6.05it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 663/2033 [01:56<03:37,  6.30it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5████████▍                                                                                | 828/2033 [02:25<03:31,  5.70it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5███████████████▉                                                                         | 940/2033 [02:46<03:48,  4.78it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5██████████████████████████████▎                                                         | 1165/2033 [03:26<02:27,  5.87it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5█████████████████████████████████████████████████████▏                                  | 1508/2033 [04:26<01:36,  5.46it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5███████████████████████████████████████████████████████████████████▉                    | 1731/2033 [05:06<01:03,  4.76it/s]
{'eval_loss': '1.03', 'eval_runtime': '361.5', 'eval_samples_per_second': '11.25', 'eval_steps_per_second': '5.624', 'epoch': '0.01245'}                                        
{'train_runtime': '1688', 'train_samples_per_second': '1.137', 'train_steps_per_second': '0.036', 'train_loss': '1.177', 'epoch': '0.01245'}                                    
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [28:08<00:00, 28.14s/it]
[OK] Run complete: step 60 [trl]
  [INFO] trainable parameters: 24,158,208 of 5,128,455,712 (0.47 percent)
Final train loss: 1.077697
Held-out loss:    1.029761
Run directory: /mnt/ai-data/training/runs/ft3-straight/2026-09-25_13-32-04
Checkpoints:   /mnt/ai-data/training/runs/ft3-straight/checkpoints
exit=0
```

Every stage-1 criterion observed: `[RUN] Fine-tuning ft3-straight [trl]: 60
total steps`, the sdpa note (KB-84), the dataset loaded from the staged
base's tokenizer without a hash error (KB-80's `train-sft` half, confirmed),
`eval_loss` 1.038 at step 30 and 1.03 at step 60, each over the 2033 holdout
batches in 361 s with no out-of-memory (KB-88 confirmed), `[OK] Run complete:
step 60 [trl]`, `Final train loss: 1.077697`, `Held-out loss:    1.029761`,
exit 0, and the adapter's 24,158,208 trainable parameters, the count
FT.calibrate-qlora measured. The step-60 held-out loss, 1.029761, is the
straight-run side of the tolerance criterion. Observation, not a mismatch:
`[watchdog] ALERT gpu_throttle: value 1, threshold 0.5` printed on most
samples, in both the step loop and the evaluation. The sampler
(`training/runtime/watchdog.py`) reports 1 for any active clock-throttle
reason other than idle, and the signal is alert-only
([operational_watchdogs.md](../../training/base/operational_watchdogs.md)),
so a standing reason such as a locked clock or a power cap trips it on every
sample and the line carries no information on this workstation.

Interrupted run (2026-09-25; the straight run's checkpoint `ls` and `cat`
were not pasted with it and are still owed):

```
➜  ai-experiments git:(SFT-verification-2) cd ~/workspace/ai-experiments
set -o pipefail
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-resume schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30"
echo "exit=$?"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
cat /mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30/meta.json
docker compose -f docker-compose.training.yaml run --rm training python -m training train-sft --config training/recipes/reference/sft-qlora-e2b.yaml run.name=ft3-resume schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-cb3ccfc7b6fe Creating 
Container ai-experiments-training-run-cb3ccfc7b6fe Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training train-sft --config training/recipes/reference/sft-qlora-e2b.yaml run.name=ft3-resume schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30
[RUN] Fine-tuning ft3-resume [trl]: 60 total steps
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
Loading weights:   0%|                                                                                                                                 | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3196.33it/s]
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 1, 'bos_token_id': 2, 'pad_token_id': 0}.
{'loss': '1.464', 'grad_norm': '0.608', 'learning_rate': '0.0001946', 'epoch': '0.002076'}                                                                                      
 17%|███████████████████████▏                                                                                                                   | 10/60 [02:39<13:19, 15.99s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
 27%|█████████████████████████████████████                                                                                                      | 16/60 [04:16<11:42, 15.97s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
 28%|███████████████████████████████████████▍                                                                                                   | 17/60 [04:34<11:46, 16.43s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
{'loss': '1.225', 'grad_norm': '0.4617', 'learning_rate': '0.0001636', 'epoch': '0.004152'}                                                                                     
 37%|██████████████████████████████████████████████████▉                                                                                        | 22/60 [05:53<10:00, 15.81s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
 48%|███████████████████████████████████████████████████████████████████▏                                                                       | 29/60 [07:42<07:55, 15.35s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
{'loss': '1.104', 'grad_norm': '1.176', 'learning_rate': '0.0001137', 'epoch': '0.006227'}                                                                                      
 50%|█████████████████████████████████████████████████████████████████████▌                                                                     | 30/60 [07:59<07:52, 15.74s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 157/2033 [00:27<05:02,  6.21it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 387/2033 [01:07<04:40,  5.86it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 669/2033 [01:57<03:53,  5.84it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5████████████▋                                                                            | 892/2033 [02:37<03:48,  5.00it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5██████████████████████████████▊                                                         | 1171/2033 [03:27<03:02,  4.73it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5████████████████████████████████████████████████████████████████████████████            | 1853/2033 [05:27<00:33,  5.33it/s]
{'eval_loss': '1.038', 'eval_runtime': '361.5', 'eval_samples_per_second': '11.25', 'eval_steps_per_second': '5.624', 'epoch': '0.006227'}                                      
 55%|████████████████████████████████████████████████████████████████████████████▍                                                              | 33/60 [14:47<30:55, 68.73s/it]^CTraceback (most recent call last):                                                                                                                                             
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/training/__main__.py", line 7, in <module>
    sys.exit(main())
             ^^^^^^
  File "/app/training/cli.py", line 484, in main
    return dispatch("training", build_parser, argv)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/app/common/obs/dispatch.py", line 70, in dispatch
    result = args.func(args)
             ^^^^^^^^^^^^^^^
  File "/app/training/cli.py", line 115, in cmd_train_sft
    result = train_sft(recipe)
             ^^^^^^^^^^^^^^^^^
  File "/app/training/tune/sft.py", line 389, in train_sft
    return _train_locked(recipe)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/app/training/tune/sft.py", line 479, in _train_locked
    trainer.train(resume_from_checkpoint=resume_path)
  File "/opt/venv/lib/python3.12/site-packages/transformers/trainer.py", line 1433, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/transformers/trainer.py", line 1515, in _inner_training_loop
    self._run_epoch(
  File "/opt/venv/lib/python3.12/site-packages/transformers/trainer.py", line 1743, in _run_epoch
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/trl/trainer/sft_trainer.py", line 1185, in training_step
    return super().training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/transformers/trainer.py", line 1943, in training_step
    self.accelerator.backward(loss, **kwargs)
  File "/opt/venv/lib/python3.12/site-packages/accelerate/accelerator.py", line 2850, in backward
    loss.backward(**kwargs)
  File "/opt/venv/lib/python3.12/site-packages/torch/_tensor.py", line 623, in backward
    torch.autograd.backward(
  File "/opt/venv/lib/python3.12/site-packages/torch/autograd/__init__.py", line 395, in backward
    _engine_run_backward(
  File "/opt/venv/lib/python3.12/site-packages/torch/autograd/graph.py", line 979, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
 55%|████████████████████████████████████████████████████████████████████████████▍                                                              | 33/60 [15:07<12:22, 27.51s/it]

make: *** [make/training.mk:56: train-sft] Error 130
exit=2
checkpoint-30  latest  README.md
{
  "step": 30,
  "checkpoint_format": "trl",
  "adapter_only": true,
  "base_model": "google/gemma-4-E2B",
  "base_revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
  "base_config_sha256": "e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8",
  "backend": "trl",
  "recipe_hash": "beda71298666ffce4ba8d9547915371a7deb8994fd648057f8b769e6b80380ec",
  "git_sha": "unknown",
  "library_versions": {
    "torch": "2.13.0+cu130",
    "transformers": "5.13.1",
    "trl": "0.24.0",
    "peft": "0.19.1"
  },
  "wall_clock": "2026-09-25T15:02:53+0000",
  "attention_implementation": "sdpa"
}
```

Every stage-2 criterion observed: `[RUN] Fine-tuning ft3-resume [trl]: 60
total steps`, the same `eval_loss` 1.038 at step 30 as the straight run (the
two runs are seeded alike, so agreement here is expected), one Ctrl-C at step
33 ending in the `KeyboardInterrupt` traceback with `make` reporting error
130 and the shell `exit=2` (`make` exits 2 when a recipe fails, whatever the
recipe's own code), `checkpoint-30` present and `checkpoint-60` absent, and
the `meta.json` inside `checkpoint-30` carrying `checkpoint_format: trl`, `adapter_only:
true`, `base_revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` and
`attention_implementation: sdpa`. The other two listing entries, `latest` and
`README.md`, are the checkpoint callback's own outputs (the symlink
`update_latest` maintains and the run readme, `training/tune/sft_callbacks.py`),
not stray checkpoints. `git_sha: unknown` is the same soft-fail
FT.smoke-dataset noted for the prepare container: the training image carries
no `.git` either, and no reader checks the field.

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
`run_container_phase`, `check_holdout_pair`, `check_lmeval_pair`). Depends on
FT.export-merged (the subject) and FT.heldout-eval (the container evaluators
the same run repeats).

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
A scorecard assembled over the four records existed until 2026-09-20 and was
removed: nothing read it, where `organism-verify` reads the records. What this
entry pins beyond FT.heldout-eval is the driver: both sides reach the container
in its spelling, each side's `model_hash` is printed, and the four records the
organism pipeline consumes are written under one run directory.

Command (host tool; inference down; the tool runs the container phases
itself):

```bash
make eval-ft ARGS="--base /mnt/ai-data/models/hf/gemma-4-E2B \
    --tuned /mnt/ai-data/models/trained/ft3-straight/hf \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --lmeval-limit 20 \
    --name ft6"
echo "exit=$?"
ls /mnt/ai-data/training/eval/ft6_*/
```

Expected signal: exit 0; the container phases print `[RUN] holdout/base:
/app/models/hf/gemma-4-E2B` and `[RUN] holdout/tuned:
/app/models/trained/ft3-straight/hf` (the container spelling, KB-78) and
`[OK] holdout/<side>: loss <n>` for each, with the tuned loss at or below the
base's as FT.heldout-eval found, then one `[OK] lmeval/<task>: <a> -> <b>` per pinned
task; before the phases, one `[INFO] <side>: <container path> model_hash
<sha256> thinking_mode off` line per side, the base's hash equal to the
FT.preconditions digest and the tuned side's equal to FT.export-merged's manifest;
the run directory lists `holdout_base.json`, `holdout_tuned.json`,
`lmeval_base.json` and `lmeval_tuned.json`, the four records FT.organisms
hands to `organism-verify`. An interrupted run resumes from its phase records
(`[SKIP] holdout/<side>: record exists (resume)`).

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
