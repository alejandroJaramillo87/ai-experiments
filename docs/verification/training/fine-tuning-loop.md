# Fine-tuning loop verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 10 verified, 0 blocked, 4 pending, of 14.**

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
- **FT.export-merged's first attempt** (2026-09-25) died before the export
  began: the export image did not carry `xxhash` or `omegaconf`, which
  `python -m training` reaches while registering every subcommand, so the
  CLI never dispatched. A tree defect, KB-91, the same shape as KB-59: both
  packages are pinned in the image, a sandbox test holds the image's pins to
  the CLI's imports, the paste is kept as a superseded observation, and the
  entry re-runs as committed.
- **FT.export-merged verified** (2026-09-25) on its second run, with one
  runbook finding: the entry asked for a single-line chat reply, the
  behaviour of the instruction-tuned model the rust-harness chats ran, and
  the 60-step adapter over the base ran to the sampling cap. The harness
  promises no line count, so the clause was corrected with the observation
  kept beside it; the reply's shape is the adapter's, judged at
  FT.heldout-eval. Two host-state fixes on the way, neither a tree defect:
  a root-owned `trained` directory older than the setup script's chown, and
  a root-owned `sessions/rust-harness` left by a run before the harness
  targets pre-created it.
- **Two hardware mirrors and a subsystem axis** (2026-09-25, after
  FT.export-merged verified): `tests/hardware/test_train_resume.py` runs
  FT.train-resume's three runs at 20 steps through a `stop_after` seam on
  `train_sft`, and `test_fine_tuning_loop.py` gained the base-verified line,
  the refused re-export and a chat turn from FT.export-merged's hand run.
  Every hardware test now carries one of `harness`, `training`, `interp`,
  `platform`, and `make hardware-tests HW=full SUITE=training` runs the
  training slice alone, about twenty minutes of GPU, without the E4B chats.
- **FT.hardware-test verified** (2026-09-25) on the two mirrors' first run:
  `test_train_interrupt_resume` and `test_fine_tuning_loop` passed at
  `[TIER] full` in 25 minutes, no skip. It closed ahead of the five entries
  before it in the order because its own dependencies were all met. One
  observation: 16.6 GB of CUDA tensors survived the resume test into the loop
  test, which the `cuda` fixture exists to report and the loop test ran
  beside; no ledger row.
- **FT.heldout-eval read against the tree** (2026-09-25, before it ran) and
  found KB-92: `eval-lmeval` could not run in the training container, whose
  offline cache held none of the four task datasets and whose `lm-eval` pin
  lacked the ifeval extra. Fixed the same day (a dataset form on
  `train-download`, a read-only hub-cache mount, the extra); the entry
  gains a precondition block, the printed lines and a KB-92 confirmation,
  and runs as committed once the workstation locks the extra.
- **FT.heldout-eval's first evaluation attempt** (2026-09-25): both holdout
  passes met every criterion; both lm-eval passes died at task load, offline,
  because KB-92's first fix mounted the half of the HF cache that `datasets`
  never reads offline (KB-92 reopened, second fix: the datasets half,
  read-write), and the run showed two more things the task set reaches that
  the container lacked, nltk's punkt models and tinyMMLU's git-only scoring
  package (KB-93: the models baked into the image by digest, a repo-owned
  `tinyMMLU_acc` with plain accuracy). The holdout half stands on that paste;
  the lm-eval half re-runs.
- **FT.heldout-eval's second evaluation attempt** (2026-09-25): the two
  probes and the base's task loading passed, so KB-92's second fix and
  KB-93 hold; the base `eval-lmeval` ran to scoring and was refused by the
  tree's own extractor, because the pinned metric keys lacked the
  `,<filter>` suffix lm-eval puts on every result, `none` for an
  unfiltered task (KB-94, fixed in `tasks.yaml`). The lm-eval half re-runs
  on a `git pull` alone: the yaml is compose-mounted.
- **FT.heldout-eval verified** (2026-09-25) on its third evaluation
  attempt, the lm-eval half after a `git pull` to the KB-94 fix: both
  sides exit 0 with the four `[OK]` lines spelled as the harness keys
  them, the records carrying the pin and the limit, about four minutes a
  side; KB-92, KB-93 and KB-94 confirmed, and KB-84's third loader had
  its first run on the first attempt's holdout passes. The harness's
  batch-size search, some forty allocator warnings a phase, is read from
  the wheel and recorded as the noise it is.
- **FT.heldout-eval's hardware mirror** (2026-09-25):
  `tests/hardware/test_heldout_eval.py` trains 20 steps, exports, and runs
  `eval-holdout` and `eval-lmeval` on both sides as host processes with the
  HF cache at the prepare service's root, holding the printed lines, both
  record shapes, the driver's pair checks and the loss order. Its first
  run the same day failed at the base's lm-eval task load on a host venv
  synced before `f52c158` locked the ifeval extra, after the train, export
  and holdout phases had passed; recorded in the entry, host state, no
  ledger row. The mirror now checks the extra before any GPU work, and
  passed the same day after `make install` (`1 passed` in 21:34), recorded
  in the entry.
- **FT.eval-ft's readiness review** (2026-09-25) read the entry against
  the tree before handover and found two latent defects on its path, both
  of KB-78's host-and-container-spelling shape: `make eval-ft` ran the
  system `python3`, which has none of the training group that `python -m
  training` imports at import (KB-95, with the other host-run python
  targets), and `load_hf_manifest` refused the manifest the export wrote
  because it records the container spelling and the host driver asks with
  the host's, so the tuned side's hash was silently recomputed from the
  weights (KB-96). Both fixed in the tree; the entry now names the venv,
  the two rows, the printed lines in order and the two digests it expects.
  The entry passed the same day on its first run (about sixteen minutes,
  exit 0), closing KB-78, KB-95 and KB-96.
- **FT.eval-ft's hardware mirror** (2026-09-25):
  `tests/hardware/test_eval_ft.py` runs the driver as one host process on
  the staged base and the campaign's export, the first hardware test to
  drive compose, holding the printed lines in order, both digests, the
  four records, the driver's pair checks, the loss order and a resume on
  the same run directory. Its first run passed on 2026-09-26, 1 passed in
  fifteen minutes, exit 0, recorded in the entry.
- **FT.organisms's first attempt** (2026-09-26): `make verify` died before
  the script ran, at `set -o pipefail` under the dash that make runs recipes
  with (KB-97). Fixed in the target the same day with a test under the same
  shell; the entry re-runs as committed.
- **FT.organisms's readiness review** (2026-09-26): the entry and the
  tree's two hints passed `prepare-sft` the blind corpus directory where
  the verb opens a file (the JSONL inside it); fixed in both. The entry
  now runs as one logged, resumable script through `make verify
  ENTRY=ft-organisms`, states its cost per stage (about three hours for the
  null, two and a half for the taboo) and the two design facts it runs
  under: 400 steps over 16 or 32 examples, and locality at lm-eval limit
  20, where one item is 0.05 and epsilon is 0.05.

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
straight run's within `RESUME_LOSS_RTOL` (0.01, `tests/hardware/conftest.py`),
that is `abs(a - b) / a <= 0.01`; record both numbers and the quotient.
The constant is shared with `test_resume_loss_curve_within_tolerance`,
which holds the from-scratch loop's per-step losses to it on the GPU. Since
2026-09-25 this entry has a hardware mirror,
`tests/hardware/test_train_resume.py` (`hw_full`, `gpu`, `ssd`, `training`),
written after the hand run verified: the same three runs at 20 steps with
the checkpoint at 10, in-process through `train_sft`'s `stop_after` in
place of the Ctrl-C, no held-out passes, and the tolerance applied to the
trainer's ten-step train-loss log, whose entry at step 20 is the mean over
steps 11 to 20 in both the straight and the resumed run. That is the
quantity the pastes below print every ten steps, and the hand run's own
pairs at steps 40, 50 and 60 sit inside the tolerance too; the held-out
comparison stays this entry's, at the recipe's scale. A resumed run that
reports `Fine-tuning` rather than `Resuming`, or ends at a step other than
60, has restarted rather than resumed.

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

Resumed run (2026-09-25), with the straight run's checkpoint listing and
its step-60 `meta.json` owed from stage 1:

```
➜  ai-experiments git:(SFT-verification-2) cd ~/workspace/ai-experiments
set -o pipefail
ls /mnt/ai-data/training/runs/ft3-straight/checkpoints/
cat /mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60/meta.json
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-resume schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30 \
    checkpoint.resume=/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30"
echo "exit=$?"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
checkpoint-30  checkpoint-60  latest  README.md
{
  "step": 60,
  "checkpoint_format": "trl",
  "adapter_only": true,
  "base_model": "google/gemma-4-E2B",
  "base_revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
  "base_config_sha256": "e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8",
  "backend": "trl",
  "recipe_hash": "98a8eab285605b237344e8151270ded92cb4f704b37b0207a6628ec517ae14b4",
  "git_sha": "unknown",
  "library_versions": {
    "torch": "2.13.0+cu130",
    "transformers": "5.13.1",
    "trl": "0.24.0",
    "peft": "0.19.1"
  },
  "wall_clock": "2026-09-25T14:00:37+0000",
  "attention_implementation": "sdpa"
}docker compose -f docker-compose.training.yaml run --rm training python -m training train-sft --config training/recipes/reference/sft-qlora-e2b.yaml run.name=ft3-resume schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30     checkpoint.resume=/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container ai-experiments-training-run-a67e2e96b1cc Creating 
Container ai-experiments-training-run-a67e2e96b1cc Created 
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training train-sft --config training/recipes/reference/sft-qlora-e2b.yaml run.name=ft3-resume schedule.total_steps=60 schedule.warmup_steps=3 checkpoint.interval_steps=30 sft.eval_steps=30 checkpoint.resume=/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30
[RUN] Resuming (/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30) ft3-resume [trl]: 60 total steps
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
Loading weights:   0%|                                                                                                                                 | 0/1951 [00:00<?, ?it/s]/opt/venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
  torch._check_is_size(blocksize)
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3202.55it/s]
[transformers] The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 1, 'bos_token_id': 2, 'pad_token_id': 0}.
{'loss': '1.106', 'grad_norm': '0.542', 'learning_rate': '5.983e-05', 'epoch': '0.008303'}                                                                                      
 67%|████████████████████████████████████████████████████████████████████████████████████████████▋                                              | 40/60 [02:40<03:51, 11.59s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
 72%|███████████████████████████████████████████████████████████████████████████████████████████████████▌                                       | 43/60 [03:30<04:08, 14.62s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
{'loss': '1.093', 'grad_norm': '0.894', 'learning_rate': '1.782e-05', 'epoch': '0.01038'}                                                                                       
 83%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                       | 50/60 [05:21<02:38, 15.84s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5
{'loss': '1.083', 'grad_norm': '0.2802', 'learning_rate': '1.518e-07', 'epoch': '0.01245'}                                                                                      
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [08:04<00:00, 16.08s/it][watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                          | 30/2033 [00:04<04:55,  6.79it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 143/2033 [00:24<05:54,  5.33it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5                                                                                         | 657/2033 [01:55<03:03,  7.48it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5█████████████████████████████▉                                                          | 1159/2033 [03:24<02:54,  5.01it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5█████████████████████████████████████▌                                                  | 1274/2033 [03:44<02:31,  5.00it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5████████████████████████████████████████████████████████████                            | 1613/2033 [04:44<01:32,  4.54it/s]
                                                                                                                                                                                [watchdog] ALERT gpu_throttle: value 1, threshold 0.5███████████████████████████████████████████████████████████████████████▎                | 1782/2033 [05:14<00:51,  4.92it/s]
{'eval_loss': '1.03', 'eval_runtime': '361.4', 'eval_samples_per_second': '11.25', 'eval_steps_per_second': '5.626', 'epoch': '0.01245'}                                        
{'train_runtime': '846.4', 'train_samples_per_second': '2.268', 'train_steps_per_second': '0.071', 'train_loss': '0.547', 'epoch': '0.01245'}                                   
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 60/60 [14:06<00:00, 14.11s/it]
[OK] Run complete: step 60 [trl]
  [INFO] trainable parameters: 24,158,208 of 5,128,455,712 (0.47 percent)
Final train loss: 1.083399
Held-out loss:    1.030488
Run directory: /mnt/ai-data/training/runs/ft3-resume/2026-09-25_15-14-51
Checkpoints:   /mnt/ai-data/training/runs/ft3-resume/checkpoints
exit=0
checkpoint-30  checkpoint-60  latest  README.md
```

Every remaining criterion observed. Straight run's checkpoints:
`checkpoint-30` and `checkpoint-60` listed, and the step-60 `meta.json`
carries `checkpoint_format: trl`, `adapter_only: true`, `base_revision
d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` and `attention_implementation:
sdpa`. Resumed run: `[RUN] Resuming
(/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30) ft3-resume
[trl]: 60 total steps`, the step counter picking up past 30 with the
learning rate already on the cosine tail (5.983e-05 at the first log line),
`[OK] Run complete: step 60 [trl]`, `Held-out loss:    1.030488`, exit 0, and
`checkpoint-60` beside `checkpoint-30` in the final listing. Tolerance:
abs(1.029761 - 1.030488) / 1.029761 = 0.000706, inside `RESUME_LOSS_RTOL`
(0.01) by a factor of 14. Two observations, neither a mismatch: the
trainer's own summary line reports `train_loss` 0.547 because transformers
divides the loss accumulated over the 30 resumed steps by all 60, whereas
the `Final train loss: 1.083399` the command prints is the last logged step
loss; and the two runs' `recipe_hash` values differ because `run.name` is
part of the hashed recipe, which is why the resumed run carries the
interrupted run's hash (both `ft3-resume`) and the straight run its own.

The hardware mirror's first run (2026-09-25, after this entry verified):
`tests/hardware/test_train_resume.py::test_train_interrupt_resume PASSED`
at `[TIER] full`, the three runs at 20 steps with both window losses inside
`RESUME_LOSS_RTOL`; the paste is recorded once, verbatim, under
FT.hardware-test.

Status: VERIFIED (2026-09-25)

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
merge on real weights. Its hardware mirror is
`tests/hardware/test_fine_tuning_loop.py` (`hw_full`, `training`), the
test FT.hardware-test runs: since 2026-09-25 it also asserts the
`[OK] Base model verified:` line, the refused second export with its
`--force` remedy, and one chat turn on a pipe ending in `[INFO] Chat ended:
1 model turns`, the three checks this entry's hand run added to what the
test held before. The export image itself is not what the test runs, since
every hardware test is in-process; the image's dependency set is
`tests/training/test_export_image_dependencies.py`'s (KB-91). Depends on
FT.train-resume.

Command (the export container mounts the models tree read-write under
`/app/models/trained`; the harness lines read the result through the
harness image, and `checkpoint-info` reads it on the host). `make
export-build` rebuilds the image's pip layer after KB-91's two pins, minutes
rather than the cache hit FT.preconditions observed. The merge loads
the 10.2 GB bf16 base on the CPU and writes the same again into the
`hf.partial` staging directory before the rename; with the base in the page
cache from the training runs it took seconds on the first run (2026-09-25),
and a cold cache costs one read of the base on top. An `hf.partial`
directory left behind marks a killed run, and the next export removes it.
The last line is interactive: at the `> ` marker type one short prompt
(`Reply with one sentence: what is a huge page?`), wait for the reply (CPU
inference on a 5 B model, so minutes rather than seconds), then type
`/quit`. Every `echo` records the exit of the line above it; `make` reports
a failing recipe as its own exit 2, whatever the command inside returned.

```bash
set -o pipefail
make export-build
make export-hf CHECKPOINT=/mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 \
    ARGS="--name ft3-straight"
echo "exit=$?"
ls /mnt/ai-data/models/trained/ft3-straight/hf/
grep -E '"(mode|hf_dir|model_hash)"' /mnt/ai-data/models/trained/ft3-straight/manifest.json
sha256sum /mnt/ai-data/models/trained/ft3-straight/hf/model.safetensors
make export-hf CHECKPOINT=/mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 \
    ARGS="--name ft3-straight"
echo "exit=$?"
make harness-run ARGS="--model /app/models/trained/ft3-straight/hf --text-only --status --deep"
echo "exit=$?"
cd rust-harness && cargo run --release -p runtime --bin checkpoint-info -- --check --text-only \
    /mnt/ai-data/models/trained/ft3-straight/hf 2>&1 | tail -3; echo "exit=$?"; cd ..
make harness-chat ARGS="--model /app/models/trained/ft3-straight/hf --text-only"
echo "exit=$?"
```

Expected signal: the first export prints `[OK] Base model verified:
/app/models/hf/gemma-4-E2B (revision
d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f)` (the container spelling of the
staged base, because the line prints the path the trainer loaded and PEFT
recorded, not the recipe's hub id; `training/export/merge.py`'s
`check_meta_consistency` names this one-base-two-spellings case), `[RUN]
Merging model: ...`, `[OK] Adapter merged: ...`, `[OK] HF export written:
.../hf (N bytes)`, `[OK] Manifest written: .../manifest.json` and a
`model_hash:` line, and exits 0 (a checkpoint whose recorded base revision
does not match the cached base is refused with `[ERROR] Export failed: Base
model drifted since training:` unless `--allow-base-drift` is passed); `ls`
shows `config.json`, `generation_config.json`, one `model.safetensors`, the
tokenizer files and no `model.safetensors.index.json`; the manifest records
`"mode": "merge"`, `"hf_dir"` as the container spelling of the listed
directory (`/app/models/trained/ft3-straight/hf`), and a `model_hash` equal
to the `sha256sum` line and different from the FT.preconditions base digest;
the second export prints `[ERROR] Export exists: .../hf` and the `--force`
solution line, returns 1 so `make` reports `Error 1` and the echo shows 2,
and leaves the manifest unchanged; `harness --status --deep` and
`checkpoint-info --check` both exit 0 on the directory, with `[OK] backend
compatibility: supported` (`checkpoint-info` spells it `CPU backend
compatibility`), the same 35-layer config as the base, and `[WARN] --deep:
no effect when there is no interp artifact to check` (the hash comparison is
FT.probe's, against a probe); one `--chat` reply arrives after the `> `
marker, ends at the model's end-of-turn or at the 256-token sampling cap,
and `/quit` ends with `[INFO] Chat ended: 1 model turns`, exit 0. The
reply's length and whether it closes its own turn are the adapter's, judged
at FT.heldout-eval, not the harness's: until 2026-09-25 this clause asked
for "a single line", the behaviour of the instruction-tuned model the
rust-harness chats ran, and the 60-step adapter over the base ran to the
cap, recorded below. The prompt was rendered through the version 2 chat
template, which is `CHAT_TEMPLATE_VERSION` in
`rust-harness/harness/src/template.rs` pinned against
`training/models/families.py`; the reply arriving cleanly framed is its
observable, not a printed version.

Observed:

First attempt (2026-09-25, superseded, kept verbatim: stage A stopped before
the export began. The image build was the cache hit FT.preconditions
observed, and `python -m training` died at import in the container: the
export image carried neither `xxhash` nor `omegaconf`, which the CLI reaches
while registering every subcommand. KB-91, fixed in the tree the same day;
no criterion was reached, so none is judged):

```
➜  ai-experiments git:(SFT-verification-2) cd ~/workspace/ai-experiments
git pull origin SFT-verification-2
git log -1 --oneline
set -o pipefail
make export-build
remote: Enumerating objects: 18, done.
remote: Counting objects: 100% (18/18), done.
remote: Compressing objects: 100% (12/12), done.
remote: Total 18 (delta 9), reused 15 (delta 6), pack-reused 0 (from 0)
Unpacking objects: 100% (18/18), 11.34 KiB | 1.89 MiB/s, done.
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch              SFT-verification-2 -> FETCH_HEAD
   9509f1b3..e9703d73  SFT-verification-2 -> origin/SFT-verification-2
Updating 9509f1b..e9703d7
Fast-forward
 docs/verification/training/fine-tuning-loop.md | 292 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++---------
 1 file changed, 271 insertions(+), 21 deletions(-)
Building export image...
docker compose -f docker-compose.training.yaml build export
[+] Building 0.6s (17/17) FINISHED
 => [internal] load local bake definitions                                                                                                                                  0.0s
 => => reading from stdin 569B                                                                                                                                              0.0s
 => [internal] load build definition from Dockerfile.export                                                                                                                 0.0s
 => => transferring dockerfile: 2.53kB                                                                                                                                      0.0s
 => [internal] load metadata for docker.io/library/ubuntu:24.04                                                                                                             0.5s
 => [internal] load metadata for ghcr.io/astral-sh/uv:0.11.28                                                                                                               0.3s
 => [internal] load .dockerignore                                                                                                                                           0.0s
 => => transferring context: 1.73kB                                                                                                                                         0.0s
 => [stage-1 1/9] FROM docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3                                               0.0s
 => [uv 1/1] FROM ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa                                                      0.0s
 => CACHED [stage-1 2/9] RUN apt-get update &&     apt-get install -y --no-install-recommends         git         ca-certificates         python3.12         python3.12-de  0.0s
 => CACHED [stage-1 3/9] COPY --from=uv /uv /uvx /usr/local/bin/                                                                                                            0.0s
 => CACHED [stage-1 4/9] RUN uv venv /opt/venv --python 3.12                                                                                                                0.0s
 => CACHED [stage-1 5/9] RUN --mount=type=cache,target=/root/.cache/uv     uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 &&     uv pip ins  0.0s
 => CACHED [stage-1 6/9] RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser                                            0.0s
 => CACHED [stage-1 7/9] RUN mkdir -p /app/cache /app/logs && chown -R appuser:appuser /app                                                                                 0.0s
 => CACHED [stage-1 8/9] WORKDIR /app                                                                                                                                       0.0s
 => CACHED [stage-1 9/9] RUN python -c "import torch, transformers, peft, safetensors"                                                                                      0.0s
 => exporting to image                                                                                                                                                      0.0s
 => => exporting layers                                                                                                                                                     0.0s
 => => writing image sha256:fb7e7363b4fb02b3b5a582c4a4c2d573961d4a95accf1c57c3485a695e4986f4                                                                                0.0s
 => => naming to docker.io/library/ai-experiments-export                                                                                                                    0.0s
 => resolving provenance for metadata file                                                                                                                                  0.0s
[+] build 1/1
 ✔ Image ai-experiments-export Built                                                                                                                                         0.6s
Export image built.
➜  ai-experiments git:(SFT-verification-2) make export-hf CHECKPOINT=/mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 \
    ARGS="--name ft3-straight"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm export python -m training export --checkpoint /mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 --name ft3-straight
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-export-run-0aab72d6b2ca Creating
Container ai-experiments-export-run-0aab72d6b2ca Created
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/app/training/__main__.py", line 5, in <module>
    from training.cli import main
  File "/app/training/cli.py", line 24, in <module>
    from training.commands import (
  File "/app/training/commands/build_contamination.py", line 20, in <module>
    from training.data import contamination
  File "/app/training/data/contamination.py", line 29, in <module>
    import xxhash
ModuleNotFoundError: No module named 'xxhash'

make: *** [make/training.mk:64: export-hf] Error 1
exit=2
```

Stage A (2026-09-25, as committed after KB-91; the export, the listing, the
manifest, the digest, and the refused second export):

```
➜  ai-experiments git:(SFT-verification-2) cd ~/workspace/ai-experiments
git pull origin SFT-verification-2
git log -1 --oneline
set -o pipefail
make export-build
remote: Enumerating objects: 17, done.
remote: Counting objects: 100% (17/17), done.
remote: Compressing objects: 100% (4/4), done.
remote: Total 17 (delta 13), reused 17 (delta 13), pack-reused 0 (from 0)
Unpacking objects: 100% (17/17), 8.42 KiB | 2.10 MiB/s, done.
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch              SFT-verification-2 -> FETCH_HEAD
   e9703d73..293d77aa  SFT-verification-2 -> origin/SFT-verification-2
Updating e9703d7..293d77a
Fast-forward
 docker/training/Dockerfile.export                |  12 +++++--
 docs/sandbox/docker_training_overview.md         |  16 +++++----
 docs/verification/bugs/known-bugs.md             |   7 ++--
 docs/verification/training/fine-tuning-loop.md   |  86 ++++++++++++++++++++++++++++++++++++++++++++++-
 tests/training/README.md                         |   1 +
 tests/training/test_export_image_dependencies.py | 219 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 6 files changed, 329 insertions(+), 12 deletions(-)
 create mode 100644 tests/training/test_export_image_dependencies.py
Building export image...
docker compose -f docker-compose.training.yaml build export
[+] Building 24.9s (17/17) FINISHED
 => [internal] load local bake definitions                                                                                                                                  0.0s
 => => reading from stdin 569B                                                                                                                                              0.0s
 => [internal] load build definition from Dockerfile.export                                                                                                                 0.0s
 => => transferring dockerfile: 3.01kB                                                                                                                                      0.0s
 => [internal] load metadata for docker.io/library/ubuntu:24.04                                                                                                             0.8s
 => [internal] load metadata for ghcr.io/astral-sh/uv:0.11.28                                                                                                               0.8s
 => [internal] load .dockerignore                                                                                                                                           0.0s
 => => transferring context: 1.73kB                                                                                                                                         0.0s
 => [stage-1 1/9] FROM docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3                                               0.0s
 => [uv 1/1] FROM ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa                                                      0.0s
 => CACHED [stage-1 2/9] RUN apt-get update &&     apt-get install -y --no-install-recommends         git         ca-certificates         python3.12         python3.12-de  0.0s
 => CACHED [stage-1 3/9] COPY --from=uv /uv /uvx /usr/local/bin/                                                                                                            0.0s
 => CACHED [stage-1 4/9] RUN uv venv /opt/venv --python 3.12                                                                                                                0.0s
 => [stage-1 5/9] RUN --mount=type=cache,target=/root/.cache/uv     uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 &&     uv pip install    17.8s
 => [stage-1 6/9] RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser                                                   0.1s
 => [stage-1 7/9] RUN mkdir -p /app/cache /app/logs && chown -R appuser:appuser /app                                                                                        0.2s
 => [stage-1 8/9] WORKDIR /app                                                                                                                                              0.0s
 => [stage-1 9/9] RUN python -c "import torch, transformers, peft, safetensors, omegaconf, xxhash"                                                                          4.7s
 => exporting to image                                                                                                                                                      1.2s
 => => exporting layers                                                                                                                                                     1.2s
 => => writing image sha256:84c03bf6d68e4b6c1794365545ddcab077851ac24fe7de5478e15569d1b1900a                                                                                0.0s
 => => naming to docker.io/library/ai-experiments-export                                                                                                                    0.0s
 => resolving provenance for metadata file                                                                                                                                  0.0s
[+] build 1/1
 ✔ Image ai-experiments-export Built                                                                                                                                        24.9s
Export image built.
➜  ai-experiments git:(SFT-verification-2) make export-hf CHECKPOINT=/mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 \
    ARGS="--name ft3-straight"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm export python -m training export --checkpoint /mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 --name ft3-straight
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-export-run-47c8bdd51646 Creating
Container ai-experiments-export-run-47c8bdd51646 Created
[OK] Base model verified: /app/models/hf/gemma-4-E2B (revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f)
[ERROR] Export failed: [Errno 13] Permission denied: '/app/models/trained/ft3-straight'

make: *** [make/training.mk:64: export-hf] Error 1
exit=2
➜  ai-experiments git:(SFT-verification-2) id -u; id -g; echo "UID=${UID} GID=${GID}"
ls -ld /mnt/ai-data /mnt/ai-data/models /mnt/ai-data/models/trained /mnt/ai-data/training/runs
ls -la /mnt/ai-data/models/trained/
docker compose -f docker-compose.training.yaml config export 2>/dev/null | grep -E "^\s+user:"
1000
1000
UID=1000 GID=1000
drwxr-xr-x  9 alejandro alejandro 4096 Aug  6 15:41 /mnt/ai-data
drwxr-xr-x  9 alejandro alejandro 4096 Aug  1 05:44 /mnt/ai-data/models
drwxr-xr-x  2 root      root      4096 Jul 23 11:10 /mnt/ai-data/models/trained
drwxr-xr-x 12 alejandro alejandro 4096 Sep 25 07:42 /mnt/ai-data/training/runs
total 8
drwxr-xr-x 2 root      root      4096 Jul 23 11:10 .
drwxr-xr-x 9 alejandro alejandro 4096 Aug  1 05:44 ..
    user: 1000:1000
  user: 1000:1000
➜  ai-experiments git:(SFT-verification-2) sudo chown -R "$(whoami):$(whoami)" /mnt/ai-data/models/trained
➜  ai-experiments git:(SFT-verification-2) ls -ld /mnt/ai-data/models/trained
drwxr-xr-x  2 alejandro alejandro 4096 Jul 23 11:10 /mnt/ai-data/models/trained
➜  ai-experiments git:(SFT-verification-2) make export-hf CHECKPOINT=/mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 \
    ARGS="--name ft3-straight"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm export python -m training export --checkpoint /mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 --name ft3-straight
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-export-run-fb0cd4975a17 Creating
Container ai-experiments-export-run-fb0cd4975a17 Created
[OK] Base model verified: /app/models/hf/gemma-4-E2B (revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f)
[RUN] Merging model: /mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 61658.17it/s]
Writing model shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.55s/it]
[OK] Adapter merged: /mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 into /app/models/hf/gemma-4-E2B
[OK] HF export written: /app/models/trained/ft3-straight/hf (10,208,852,910 bytes)
[OK] Manifest written: /app/models/trained/ft3-straight/manifest.json
model_hash: 684dfd319d75faff9d56e9c1cc911a78427540e12ca53a6ae9a3f7818d478b52
exit=0
➜  ai-experiments git:(SFT-verification-2) ls /mnt/ai-data/models/trained/ft3-straight/hf/
grep -E '"(mode|hf_dir|model_hash)"' /mnt/ai-data/models/trained/ft3-straight/manifest.json
sha256sum /mnt/ai-data/models/trained/ft3-straight/hf/model.safetensors
config.json  generation_config.json  model.safetensors  tokenizer_config.json  tokenizer.json
  "mode": "merge",
  "hf_dir": "/app/models/trained/ft3-straight/hf",
  "model_hash": "684dfd319d75faff9d56e9c1cc911a78427540e12ca53a6ae9a3f7818d478b52",
684dfd319d75faff9d56e9c1cc911a78427540e12ca53a6ae9a3f7818d478b52  /mnt/ai-data/models/trained/ft3-straight/hf/model.safetensors
➜  ai-experiments git:(SFT-verification-2) make export-hf CHECKPOINT=/mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 \
    ARGS="--name ft3-straight"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm export python -m training export --checkpoint /mnt/ai-data/training/runs/ft3-straight/checkpoints/checkpoint-60 --name ft3-straight
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-export-run-fcdcb0f69297 Creating
Container ai-experiments-export-run-fcdcb0f69297 Created
[OK] Base model verified: /app/models/hf/gemma-4-E2B (revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f)
[ERROR] Export exists: /app/models/trained/ft3-straight/hf
  Solution: pass --force to replace it, or --name for a new export (a re-export changes model_hash under anything fitted to the old weights)

make: *** [make/training.mk:64: export-hf] Error 1
exit=2
```

Stage A judged: the base-verified line printed the staged container path
and the full revision as the expected signal states, which is KB-91's close
condition. `[RUN] Merging model:`, `[OK] Adapter merged:`, `[OK] HF export
written:` (10,208,852,910 bytes), `[OK] Manifest written:` and the
`model_hash:` line all printed, `exit=0`. The listing holds `config.json`,
`generation_config.json`, one `model.safetensors`, `tokenizer.json` and
`tokenizer_config.json`, and no index file. The manifest records
`"mode": "merge"`, `"hf_dir"` in the container spelling, and a `model_hash`
that equals the `sha256sum` line and differs from the FT.preconditions base
digest `76dc84a5...`. The second export was refused with `[ERROR] Export
exists:` and the `--force` solution line before staging, `make` reported
`Error 1`, the echo showed 2, and the listing and manifest above are the
ones the refusal left in place. Pass on every stage A criterion.

Two observations, neither a mismatch. First, the export's first run died at
`[Errno 13] Permission denied` creating the export directory: the host
`trained` directory was owned by root, dated 2026-07-23, older than the rest
of the tree, so it predates the `chown -R` in
`scripts/setup/setup_data_ssd.sh` that gives every other directory to the
operator, and the container ran as uid 1000 exactly as the compose anchor
resolves it. One `chown` on the host fixed it, kept in the paste; a host-state
defect outside the tree, so no ledger row. Second, the merge took seconds,
not the minutes the command paragraph predicted: the 1,951 tensors loaded at
61,658 per second and the single shard wrote in 1.55 s, because the staged
base was in the page cache from the training runs and safetensors maps it
rather than reading it. The command paragraph's timing note was operator
guidance, not an expected signal, and now states what was observed.

Stage B (2026-09-25): the two harness readers.

```
➜  ai-experiments git:(SFT-verification-2) cd ~/workspace/ai-experiments
git pull origin SFT-verification-2
git log -1 --oneline
set -o pipefail
make harness-run ARGS="--model /app/models/trained/ft3-straight/hf --text-only --status --deep"
echo "exit=$?"
remote: Enumerating objects: 8, done.
remote: Counting objects: 100% (8/8), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 8 (delta 5), reused 8 (delta 5), pack-reused 0 (from 0)
Unpacking objects: 100% (8/8), 3.86 KiB | 1.29 MiB/s, done.
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch              SFT-verification-2 -> FETCH_HEAD
   293d77aa..d52ace9f  SFT-verification-2 -> origin/SFT-verification-2
Updating 293d77a..d52ace9
Fast-forward
 docs/verification/bugs/known-bugs.md           |   2 +-
 docs/verification/training/fine-tuning-loop.md | 159 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--
 2 files changed, 158 insertions(+), 3 deletions(-)
docker compose -f docker-compose.rust-harness.yaml run --rm harness --model /app/models/trained/ft3-straight/hf --text-only --status --deep
WARN[0000] Found orphan containers (llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-harness-run-2e58861a6c18 Creating
Container ai-experiments-harness-run-2e58861a6c18 Created
[WARN] --deep: no effect when there is no interp artifact to check
[OK] model directory: /app/models/trained/ft3-straight/hf
[OK] checkpoint files: config.json and model.safetensors present, header names 1951 tensors
[OK] config.json: 35 layers, hidden size 1536, context 131072
[OK] backend compatibility: supported
[SKIP] probe: no --probe given
[SKIP] probe geometry: no --probe given
[SKIP] probe identity: no --probe given
[SKIP] control vector: no --control-vector given
[SKIP] control vector identity: no --control-vector given
[SKIP] interp log directory: no --interp-log given
[WARN] obs log directory: /mnt/ai-data/logs/harness cannot be created: /mnt is not writable: Read-only file system (os error 30); run records degrade to a no-op
exit=0
➜  ai-experiments git:(SFT-verification-2) cd rust-harness && cargo run --release -p runtime --bin checkpoint-info -- --check --text-only \
    /mnt/ai-data/models/trained/ft3-straight/hf 2>&1 | tail -3; echo "exit=$?"; cd ..
  layer 34: Global, head 512, ffn 12288, rope base 1000000

[OK] CPU backend compatibility: supported
exit=0
```

Stage B, the container chat, first run (2026-09-25; superseded, host
state: the bind-mounted `sessions/rust-harness` was root-owned, and the
inspection that found it is kept):

```
➜  ai-experiments git:(SFT-verification-2) make harness-chat ARGS="--model /app/models/trained/ft3-straight/hf --text-only"
echo "exit=$?"
docker compose -f docker-compose.rust-harness.yaml run --rm harness --model /app/models/trained/ft3-straight/hf --text-only --chat
WARN[0000] Found orphan containers (llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-harness-run-c00f8d115bcf Creating
Container ai-experiments-harness-run-c00f8d115bcf Created
[ERROR] Task failed: session.jsonl: open /work/session.jsonl for append: Permission denied (os error 13)

make: *** [make/harness.mk:18: harness-chat] Error 1
exit=2
➜  ai-experiments git:(SFT-verification-2) cd ~/workspace/ai-experiments
ls -la sessions/ sessions/rust-harness/
sessions/:
total 12
drwxrwxr-x  3 alejandro alejandro 4096 Sep 20 09:29 .
drwxrwxr-x 24 alejandro alejandro 4096 Sep 25 07:24 ..
drwxr-xr-x  2 root      root      4096 Sep 20 09:29 rust-harness

sessions/rust-harness/:
total 8
drwxr-xr-x 2 root      root      4096 Sep 20 09:29 .
drwxrwxr-x 3 alejandro alejandro 4096 Sep 20 09:29 ..
```

Stage B, the container chat (2026-09-25, after the `sessions/rust-harness`
reset that RH.38's own command block opens with):

```
➜  ai-experiments git:(SFT-verification-2) cd ~/workspace/ai-experiments
rm -rf sessions/rust-harness && mkdir -p sessions/rust-harness
ls -ld sessions/rust-harness
make harness-chat ARGS="--model /app/models/trained/ft3-straight/hf --text-only"
echo "exit=$?"
ls -la sessions/rust-harness
drwxrwxr-x 2 alejandro alejandro 4096 Sep 25 09:23 sessions/rust-harness
docker compose -f docker-compose.rust-harness.yaml run --rm harness --model /app/models/trained/ft3-straight/hf --text-only --chat
WARN[0000] Found orphan containers (llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-harness-run-7fdbb34fb71b Creating
Container ai-experiments-harness-run-7fdbb34fb71b Created
[INFO] Model: /app/models/trained/ft3-straight/hf hash=027443861f18a8f6 layers=35
[INFO] Weights: 8.62 GiB resident at bf16
[INFO] Peak RSS after load: 9.51 GiB
[INFO] KV cache: 131072 tokens, 1.52 GiB f32 resident (sliding layers capped at the window plus a batch of 64)
[INFO] Rewind horizon: 208 tokens behind the tip once the window fills (compaction past it re-prefills)
[INFO] Fold: dot=widen4 batch=64 threads=8
[INFO] Sampling: seed=0 temperature=0.8 max_tokens=256
[INFO] Build: 0.1.0 git unknown
[INFO] Model load: 3.75 s
Reply with one sentence: what is a huge page?
[INFO] Prefill: 20 tokens in 0.72 s (28 tok/s)
A huge page is a large virtual memory page that can be used for efficient memory management and speedier access to data.

<h2>What is a Huge Page?</h2>

A huge page is a large virtual memory page that can be used for efficient memory management and speedier access to data. It is a memory page that is larger than the normal page size, which is typically 4KB.

Virtual memory is an essential part of computer system because it allows programs to access data that is not present in physical memory. However, virtual memory can be slow and inefficient if the page size is not large enough. This is where huge pages come into play.

A huge page can be used for memory management because it can store more data in a single page. This can result in fewer page faults, which are the number of times the operating system has to retrieve data from disk instead of from memory. This can lead to faster processing times.

Additionally, huge pages can improve the performance of applications that require large amounts of memory, such as databases and virtual machines. By allocating a large page to an application, it can access more data in memory, resulting in faster processing times.

Overall, huge pages are a valuable tool for efficient memory management and faster data access
/quit
[INFO] Chat ended: 1 model turns
[INFO] Session snapshot: 276 cached tokens at /work/session.kv
exit=0
total 9952
drwxrwxr-x 2 alejandro alejandro     4096 Sep 25 09:26 .
drwxrwxr-x 3 alejandro alejandro     4096 Sep 25 09:23 ..
-rw-r--r-- 1 alejandro alejandro     1389 Sep 25 09:25 session.jsonl
-rw-r--r-- 1 alejandro alejandro 10175889 Sep 25 09:26 session.kv
```

Stage B judged: `harness --status --deep` exited 0 with `[OK] backend
compatibility: supported`, the base's 35-layer config, the header naming
1,951 tensors, and the `--deep` WARN the signal names. `checkpoint-info
--check` ended on `[OK] CPU backend compatibility: supported`, exit 0. The
chat loaded the export (8.62 GiB resident at bf16, 3.75 s), rendered the
prompt through the pinned template so that the reply came back framed and
on topic, and `/quit` ended with `[INFO] Chat ended: 1 model turns`, exit 0,
leaving a user-owned `session.jsonl` and `session.kv`. The `[WARN] obs log
directory` line is the one FT.preconditions observed on the same service,
which mounts no log root by design. The chat's first run failed on host
state, not the tree: the bind-mounted `sessions/rust-harness` was root-owned
and dated 2026-09-20 09:29, created by docker as the bind source on a run
whose tree predated the `mkdir -p` the harness targets gained on 2026-09-19,
and `mkdir -p` leaves an existing directory alone; the same shape as stage
A's `trained` directory, fixed the same way, no ledger row.

One mismatch, and it is a runbook finding, not a code defect. The signal as
first written asked for a reply "as a single line". The reply answered the
question in one sentence and then continued into a headed essay until the
256-token sampling cap, without emitting its end-of-turn. Nothing in the
harness promises a line count: the generator stops at the tokenizer EOS or
the turn-close token (`rust-harness/harness/src/generator.rs`,
`rust-harness/harness/src/chat.rs`), and the REPL prints what the model
produced. The clause was carried over from the rust-harness chats, which ran
the instruction-tuned `gemma-4-E2B-it`; this artifact is 60 SFT steps over
the pretrained base, about 1,920 examples, and whether it has learned to
close its turn is the adapter's quality, which FT.heldout-eval and
FT.eval-ft measure. What this entry tests, the export loading and serving
through both readers and one generation round trip, was observed in full,
so the clause is corrected above with this record beside it, and the entry
closes. Pass on every stage B criterion as corrected.

The hardware mirror's first run with the three checks this entry added
(2026-09-25, after this entry verified):
`tests/hardware/test_fine_tuning_loop.py::test_fine_tuning_loop PASSED` at
`[TIER] full`, the base-verified line, the refused re-export and the chat
turn on a pipe included; the paste is recorded once, verbatim, under
FT.hardware-test.

Status: VERIFIED (2026-09-25)

## FT.heldout-eval Held-out loss and lm-eval on base and merged

Implementation: `training/eval/holdout.py` (`evaluate_holdout`,
`load_model`), `training/eval/lmeval.py` (`run_lmeval`),
`training/eval/tasks.yaml`, `training/cli.py` (`cmd_eval_holdout`,
`cmd_eval_lmeval`, the printed lines). Both verbs are CLI-only (`python -m
training eval-holdout`, `eval-lmeval`); no `make` target wraps one alone
(`make eval-ft`, FT.eval-ft, drives all four), so they run inside the
training container. The records are plain JSON, not the `common/reports.py`
envelope, which only the run tracker writes. Existing code; depends on
FT.export-merged (the merged directory) and FT.smoke-dataset (the dataset).
Since 2026-09-25 this entry has a hardware mirror,
`tests/hardware/test_heldout_eval.py` (`hw_full`, `gpu`, `ssd`, `training`),
written after the hand run verified: 20 QLoRA steps on the smoke recipe, the
export, then the four verbs as processes of their own on the workstation
venv, `HF_HOME` at the prepare service's cache so the harness reads the
staged datasets offline, held to the printed lines, the record shapes, the
`eval-ft` driver's pair checks, the organism verifier's projection, and the
one ordering below (merged at or below base), and failing with `make
install` named when the venv lacks the ifeval extra the lock carries. `make
hardware-tests HW=full SUITE=training` runs it with the other training
mirrors.

**Fixes to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-92 — `eval-lmeval` could not run in the training container: the four
  pinned tasks load hub datasets the offline container had no copy of, and
  the ifeval task imports two packages the `lm-eval` pin did not carry.
  Fixed 2026-09-25 by a dataset form on `train-download`, a cache mount into
  the training service, and the `ifeval` extra on the pin. The first mount
  was the hub half, which `datasets` offline never reads; the first
  evaluation attempt (below) found that, and the second fix mounts the
  datasets half read-write. Closes when the base's `eval-lmeval` prints one
  `[OK] <task>:` line per pinned task. Confirmed 2026-09-25 by the third
  evaluation attempt.
- KB-93 — two more run-time needs of the task set the offline container did
  not have: the ifeval task loads nltk's punkt models (its import tried to
  download them), and the harness's tinyMMLU scores through a git-only
  package with all 100 items. Fixed 2026-09-25: the training image fetches
  `punkt_tab` at build by digest and sets `NLTK_DATA`; the pinned MMLU task
  is the repo-owned `tinyMMLU_acc`, the same items with plain accuracy.
  Closes with KB-92, on the same `[OK]` lines, after the sentence-count
  probe below prints `2`. Confirmed 2026-09-25: the probe on the second
  evaluation attempt, the lines on the third.
- KB-94 — the pinned task set named its result keys bare, and lm-eval
  keys every result `<metric>,<filter>`, `none` when the task declares no
  filter; the first `eval-lmeval` to reach scoring was refused by the
  tree's own extractor. Fixed 2026-09-25: the three unfiltered keys carry
  `,none`. Closes on the same `[OK]` lines, `ifeval:
  prompt_level_strict_acc,none` among them. Confirmed 2026-09-25 by the
  third evaluation attempt.

`holdout.load_model` is KB-84's third loader (config first, sdpa above the
head-width cap), and this entry is its first run; the fix was confirmed by
FT.calibrate-qlora, so a `FlashAttention forward only supports head
dimension at most 256` here is KB-84 back, not a new row.

Preconditions, in order. The lock first: the `ifeval` extra changed
`pyproject.toml` in the sandbox, and `uv lock` resolves through the cu130
torch index the sandbox cannot reach, so the workstation regenerates and
commits it (until then `make quality` fails its `uv lock --check` step,
which is the gate working). Then the image, whose pip layer the new lock
rebuilds, then the four datasets through the prepare service, the one with
egress: each `download --dataset` prints `[OK] Staged <id> (<config>) at
<sha>: <split>=<n>, ...`, and the sha it prints is the pin, since the
consumer resolves `main` offline and passes no revision. These ran on
2026-09-25 (Observed, below) and the staged data stands.

```bash
set -o pipefail
cd ~/workspace/ai-experiments && git pull origin SFT-verification-2 && git log -1 --oneline
uv lock && git diff --stat uv.lock
git -c user.name="Alejandro Jaramillo" -c user.email="alejandroj1234@proton.me" commit -am "Lock the lm-eval ifeval extra" && git push origin SFT-verification-2
make train-build 2>&1 | tail -5; echo "exit=$?"
make train-download DATASET=google/IFEval; echo "exit=$?"
make train-download DATASET=openai/gsm8k NAME=main; echo "exit=$?"
make train-download DATASET=tinyBenchmarks/tinyMMLU NAME=all; echo "exit=$?"
make train-download DATASET=Rowan/hellaswag; echo "exit=$?"
```

After the first evaluation attempt (KB-92's second fix, KB-93): the image
again, for the nltk layer only (the pip layer is a cache hit, so minutes,
not the 2408 s of the first build), then two probes in the training
container, no GPU, before any evaluation runs. The first loads the four
datasets the way the harness will, offline, and prints a `DatasetDict`
with the staged split counts for each, after `datasets`' warning `Using
the latest cached version of the dataset since <id> couldn't be found on
the Hugging Face Hub (offline mode is enabled).`; a `ConnectionError`
naming a dataset id means the mount is still wrong. The second counts the
sentences of a two-sentence string through the ifeval task's own helper
and prints `2`, with no `[nltk_data]` line and no `Downloaded punkt_tab`
line before it; a `LookupError` naming `punkt` means the image was not
rebuilt or `NLTK_DATA` is not searched. These ran on 2026-09-25 and
passed (Observed, below); they stand. KB-94's fix is a yaml the training
service compose-mounts, so the lm-eval half after it needs the `git
pull` only, no rebuild and no probe.

```bash
set -o pipefail
cd ~/workspace/ai-experiments && git pull origin SFT-verification-2 && git log -1 --oneline
make train-build 2>&1 | tail -5; echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training \
    python -c "from datasets import load_dataset; [print(d, load_dataset(d, n)) for d, n in [('google/IFEval', None), ('openai/gsm8k', 'main'), ('tinyBenchmarks/tinyMMLU', 'all'), ('Rowan/hellaswag', None)]]"
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training \
    python -c "from lm_eval.tasks.ifeval import instructions_util as u; print(u.count_sentences('One sentence. Two sentences.'))"
echo "exit=$?"
```

Command (inference down). The models tree is mounted in the training
container at `/app/models` only (`docker-compose.training.yaml`), so both
sides are named in that spelling; the training volume is identity-mounted,
so the dataset and the output records keep their host paths. This block
named `/mnt/ai-data/models/...` until 2026-09-18, which does not exist in
the container. Each holdout pass loads the model in bf16 on the GPU (10.2
GB for E2B) and runs the 4066-example holdout one example at a time with
full logits per example, so expect minutes per side after the load, with
no progress line until the `[OK]` (the trainer's own evaluation of the same
split took 361 s at batch 2). Each lm-eval pass loads the model again
through the harness's own `hf` backend and runs 20 examples per task; gsm8k
and ifeval generate, so they dominate.

```bash
set -o pipefail
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model /app/models/hf/gemma-4-E2B \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/ft5-holdout-base.json
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model /app/models/trained/ft3-straight/hf \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/ft5-holdout-merged.json
echo "exit=$?"
grep -E '"(loss|perplexity|supervised_tokens|precision|attention_implementation)"' \
    /mnt/ai-data/training/eval/ft5-holdout-base.json /mnt/ai-data/training/eval/ft5-holdout-merged.json
```

The holdout half ran and passed on 2026-09-25 (Observed, below); its code
and the image's Python dependencies are untouched by the fixes that
followed, so it is not re-run. The lm-eval half, after the probes above:

```bash
set -o pipefail
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/trained/ft3-straight/hf --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-merged.json
echo "exit=$?"
grep -E '"(lm_eval_version|limit)"' /mnt/ai-data/training/eval/ft5-lmeval-base.json /mnt/ai-data/training/eval/ft5-lmeval-merged.json
```

Expected signal: all four exit 0. Each holdout run prints `[RUN] Held-out
loss: <model> over <dataset>`, then `[OK] Loss <x>, perplexity <y> (4066
examples, <n> supervised tokens, bf16)` and `[OK] Record written: <out>`;
each record carries `loss`, `perplexity`, `supervised_tokens`, `precision`
`bf16` and `attention_implementation` `sdpa` (the two `grep` lines show
them side by side), and the merged model's held-out loss is at or below the
base's on the dataset it trained on. The merged number is in the
neighbourhood of the `1.030488` FT.train-resume's straight run reported, not
equal to it: that was the nf4 adapter under the trainer's chunked loss at
step 60, this is the merged bf16 weights under plain cross-entropy, and the
entry records the gap without judging it. Each lm-eval run prints `[RUN]
lm-evaluation-harness: <model>`, then one `[OK] <task>: <metric> = <value>
(stderr <s>)` line per pinned task, sorted (`gsm8k`, `hellaswag`, `ifeval`,
`tinyMMLU_acc`), the metric spelled as the harness keys it, filter suffix
included (`exact_match,strict-match`, `acc_norm,none`,
`prompt_level_strict_acc,none`, `acc_norm,none`; KB-94), and `[OK] Record
written:`; each record reports the pinned
task set from `training/eval/tasks.yaml` with `"lm_eval_version": "0.4.12"`
and `"limit": 20` recorded (a smoke run, not a benchmark: 20 examples per
task carry stderr wide enough that base and merged are not ranked here).
Between the `[RUN]` and the first `[OK]`, `datasets` prints its
`Using the latest cached version` warning once per task, and no
`[nltk_data]` or `Downloaded punkt_tab` line appears. Two lines are the
harness's own noise, not failures: a run of `CUDACachingAllocator ...
OOM` warnings while `batch_size=auto` probes descending batch sizes at
the model's full context, ending in `Determined Largest batch size:`,
and `fatal: not a git repository` from its commit-hash lookup in `/app`. An `lm-eval version
mismatch` line means the image was not rebuilt from the new lock; a `Task
missing from harness results` or a `ConnectionError` naming one of the four
dataset ids means the datasets half is not mounted or its staging line did
not run; a `ModuleNotFoundError: tinyBenchmarks` means the harness's own
tinyMMLU was loaded instead of the repo-owned task; a `Metric <m> missing
from <task> results` line after the requests have run means
`training/eval/tasks.yaml` spells a key differently from the harness.

Observed:

Preconditions (2026-09-25). The lock was regenerated and committed on the
workstation (`f52c158`, adding `langdetect` and `immutabledict`), then the
image and the four staging lines, pasted verbatim:

```
➜  ai-experiments git:(SFT-verification-2) make train-build
Building training image...
MAX_JOBS=4 NVCC_THREADS=2 docker compose -f docker-compose.training.yaml build
[+] Building 2428.8s (39/39) FINISHED
 => [internal] load local bake definitions                                                                                                                                  0.0s
 => => reading from stdin 1.71kB                                                                                                                                            0.0s
 => [prepare internal] load build definition from Dockerfile.training                                                                                                       0.0s
 => => transferring dockerfile: 6.44kB                                                                                                                                      0.0s
 => [export internal] load build definition from Dockerfile.export                                                                                                          0.0s
 => => transferring dockerfile: 3.01kB                                                                                                                                      0.0s
 => [export internal] load metadata for ghcr.io/astral-sh/uv:0.11.28                                                                                                        0.3s
 => [prepare internal] load metadata for docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04                                                                               0.5s
 => [export internal] load metadata for docker.io/library/ubuntu:24.04                                                                                                      0.5s
 => [export internal] load .dockerignore                                                                                                                                    0.0s
 => => transferring context: 1.73kB                                                                                                                                         0.0s
 => [export stage-1 1/9] FROM docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3                                        0.0s
 => [training uv 1/1] FROM ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa                                             0.0s
 => CACHED [export stage-1 2/9] RUN apt-get update &&     apt-get install -y --no-install-recommends         git         ca-certificates         python3.12         python  0.0s
 => CACHED [export stage-1 3/9] COPY --from=uv /uv /uvx /usr/local/bin/                                                                                                     0.0s
 => CACHED [export stage-1 4/9] RUN uv venv /opt/venv --python 3.12                                                                                                         0.0s
 => CACHED [export stage-1 5/9] RUN --mount=type=cache,target=/root/.cache/uv     uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 &&     uv   0.0s
 => CACHED [export stage-1 6/9] RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser                                     0.0s
 => CACHED [export stage-1 7/9] RUN mkdir -p /app/cache /app/logs && chown -R appuser:appuser /app                                                                          0.0s
 => CACHED [export stage-1 8/9] WORKDIR /app                                                                                                                                0.0s
 => CACHED [export stage-1 9/9] RUN python -c "import torch, transformers, peft, safetensors, omegaconf, xxhash"                                                            0.0s
 => [prepare stage-1  1/15] FROM docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01               0.0s
 => [prepare internal] load build context                                                                                                                                   0.0s
 => => transferring context: 7.30kB                                                                                                                                         0.0s
 => [export] exporting to image                                                                                                                                             0.0s
 => => exporting layers                                                                                                                                                     0.0s
 => => writing image sha256:84c03bf6d68e4b6c1794365545ddcab077851ac24fe7de5478e15569d1b1900a                                                                                0.0s
 => => naming to docker.io/library/ai-experiments-export                                                                                                                    0.0s
 => CACHED [prepare stage-1  2/15] RUN apt-get update &&     apt-get install -y --no-install-recommends --allow-change-held-packages         build-essential         gcc g  0.0s
 => CACHED [prepare stage-1  3/15] COPY --from=uv /uv /uvx /usr/local/bin/                                                                                                  0.0s
 => CACHED [prepare stage-1  4/15] RUN uv venv /opt/venv --python 3.12                                                                                                      0.0s
 => CACHED [prepare stage-1  5/15] COPY pyproject.toml uv.lock ./                                                                                                           0.0s
 => [training stage-1  6/15] RUN --mount=type=cache,target=/root/.cache/uv     MAX_JOBS=4 NVCC_THREADS=2     DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1     CFLAGS="-march=na  2408.8s
 => [export] resolving provenance for metadata file                                                                                                                         0.0s
 => [training stage-1  7/15] RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser                                        0.1s
 => [prepare stage-1  8/15] RUN mkdir -p /app/cache /app/logs && chown -R appuser:appuser /app                                                                              0.2s
 => [prepare stage-1  9/15] WORKDIR /app                                                                                                                                    0.0s
 => [prepare stage-1 10/15] COPY --chown=appuser:appuser common ./common                                                                                                    0.0s
 => [training stage-1 11/15] COPY --chown=appuser:appuser training ./training                                                                                               0.0s
 => [training stage-1 12/15] COPY --chown=appuser:appuser scripts/training ./scripts/training                                                                               0.0s
 => [training stage-1 13/15] COPY --chown=appuser:appuser docker/training/entrypoint.sh /app/entrypoint.sh                                                                  0.0s
 => [prepare stage-1 14/15] RUN chmod +x /app/entrypoint.sh                                                                                                                 0.1s
 => [prepare stage-1 15/15] RUN python -m training smoke --build                                                                                                           12.1s
 => [prepare] exporting to image                                                                                                                                            6.6s
 => => exporting layers                                                                                                                                                     6.6s
 => => writing image sha256:cb3511decc3f96a46049acb8ffd7d831535b05684320cb1a9e54ee62dabb4c36                                                                                0.0s
 => => naming to docker.io/library/ai-experiments-prepare                                                                                                                   0.0s
 => [training] exporting to image                                                                                                                                           6.6s
 => => exporting layers                                                                                                                                                     6.6s
 => => writing image sha256:a7b5a9d7faf7d1bb23a440ccafda8b371a91d2ba413d08f9b4cdfc1efecb5756                                                                                0.0s
 => => naming to docker.io/library/ai-experiments-training                                                                                                                  0.0s
 => [prepare] resolving provenance for metadata file                                                                                                                        0.0s
 => [training] resolving provenance for metadata file                                                                                                                       0.0s
[+] build 3/3
 ✔ Image ai-experiments-prepare  Built                                                                                                                                    2428.8s
 ✔ Image ai-experiments-export   Built                                                                                                                                    2428.8s
 ✔ Image ai-experiments-training Built                                                                                                                                    2428.8s
Training image built.
➜  ai-experiments git:(SFT-verification-2) make train-download DATASET=google/IFEval; echo "exit=$?"
make train-download DATASET=openai/gsm8k NAME=main; echo "exit=$?"
make train-download DATASET=tinyBenchmarks/tinyMMLU NAME=all; echo "exit=$?"
make train-download DATASET=Rowan/hellaswag; echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm prepare python -m training download --dataset google/IFEval
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-prepare-run-6f923ec81ba3 Creating
Container ai-experiments-prepare-run-6f923ec81ba3 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
[WARN] nvidia-smi not found: no GPU visible in this container

[RUN] exec: python -m training download --dataset google/IFEval
=== Staging a dataset ===

[INFO] Dataset: google/IFEval
[INFO] Config: (default)
[INFO] Cache: /app/cache

/opt/venv/lib/python3.12/site-packages/huggingface_hub/constants.py:298: FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead to enable high performance transfer with Xet. Visit https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance for more details.
  warnings.warn(
[RUN] Downloading google/IFEval
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.52k/5.52k [00:00<00:00, 1.67MB/s]
ifeval_input_data.jsonl: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 207k/207k [00:00<00:00, 35.9MB/s]
Generating train split: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 541/541 [00:00<00:00, 192151.62 examples/s]
[OK] Staged google/IFEval (default) at 966cd89545d6b6acfd7638bc708b98261ca58e84: train=541
The training service reads this snapshot offline; a re-stage moves it.
exit=0
docker compose -f docker-compose.training.yaml run --rm prepare python -m training download --dataset openai/gsm8k --name main
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-prepare-run-ca67280ac86f Creating
Container ai-experiments-prepare-run-ca67280ac86f Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
[WARN] nvidia-smi not found: no GPU visible in this container

[RUN] exec: python -m training download --dataset openai/gsm8k --name main
=== Staging a dataset ===

[INFO] Dataset: openai/gsm8k
[INFO] Config: main
[INFO] Cache: /app/cache

/opt/venv/lib/python3.12/site-packages/huggingface_hub/constants.py:298: FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead to enable high performance transfer with Xet. Visit https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance for more details.
  warnings.warn(
[RUN] Downloading openai/gsm8k
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.93k/7.93k [00:00<00:00, 49.5MB/s]
main/train-00000-of-00001.parquet: downloading bytes: ██████████████████████████████████████████████████████████████████████████████████████████████████████| 2.30MB,  222kB/s
main/train-00000-of-00001.parquet: reconstructing file: 100%|██████████████████████████████████████████████████████████████████████████████████████| 2.31MB / 2.31MB,  223kB/s
main/test-00000-of-00001.parquet: downloading bytes: ███████████████████████████████████████████████████████████████████████████████████████████████████████|  419kB, 41.2kB/s
main/test-00000-of-00001.parquet: reconstructing file: 100%|███████████████████████████████████████████████████████████████████████████████████████|  419kB /  419kB, 41.3kB/s
Generating train split: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 7473/7473 [00:00<00:00, 1333675.17 examples/s]
Generating test split: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [00:00<00:00, 955655.03 examples/s]
[OK] Staged openai/gsm8k (main) at 740312add88f781978c0658806c59bc2815b9866: train=7473, test=1319
The training service reads this snapshot offline; a re-stage moves it.
exit=0
docker compose -f docker-compose.training.yaml run --rm prepare python -m training download --dataset tinyBenchmarks/tinyMMLU --name all
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-prepare-run-399d1651e2e4 Creating
Container ai-experiments-prepare-run-399d1651e2e4 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
[WARN] nvidia-smi not found: no GPU visible in this container

[RUN] exec: python -m training download --dataset tinyBenchmarks/tinyMMLU --name all
=== Staging a dataset ===

[INFO] Dataset: tinyBenchmarks/tinyMMLU
[INFO] Config: all
[INFO] Cache: /app/cache

/opt/venv/lib/python3.12/site-packages/huggingface_hub/constants.py:298: FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead to enable high performance transfer with Xet. Visit https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance for more details.
  warnings.warn(
[RUN] Downloading tinyBenchmarks/tinyMMLU
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.73k/5.73k [00:00<00:00, 33.5MB/s]
all/test-00000-of-00001.parquet: downloading bytes: ████████████████████████████████████████████████████████████████████████████████████████████████████████|  170kB, 16.7kB/s
all/test-00000-of-00001.parquet: reconstructing file: 100%|████████████████████████████████████████████████████████████████████████████████████████|  178kB /  178kB, 17.5kB/s
all/dev-00000-of-00001.parquet: downloading bytes: █████████████████████████████████████████████████████████████████████████████████████████████████████████|  177kB, 17.4kB/s
all/dev-00000-of-00001.parquet: reconstructing file: 100%|█████████████████████████████████████████████████████████████████████████████████████████|  205kB /  205kB, 20.2kB/s
Generating test split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 45724.45 examples/s]
Generating dev split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 285/285 [00:00<00:00, 167022.03 examples/s]
[OK] Staged tinyBenchmarks/tinyMMLU (all) at cd6a8ee9e1e6f3f18f683535f6c096c386a85da3: test=100, dev=285
The training service reads this snapshot offline; a re-stage moves it.
exit=0
docker compose -f docker-compose.training.yaml run --rm prepare python -m training download --dataset Rowan/hellaswag
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-prepare-run-c0c08c182007 Creating
Container ai-experiments-prepare-run-c0c08c182007 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
[WARN] nvidia-smi not found: no GPU visible in this container

[RUN] exec: python -m training download --dataset Rowan/hellaswag
=== Staging a dataset ===

[INFO] Dataset: Rowan/hellaswag
[INFO] Config: (default)
[INFO] Cache: /app/cache

/opt/venv/lib/python3.12/site-packages/huggingface_hub/constants.py:298: FutureWarning: The `HF_HUB_ENABLE_HF_TRANSFER` environment variable is deprecated as 'hf_transfer' is not used anymore. Please use `HF_XET_HIGH_PERFORMANCE` instead to enable high performance transfer with Xet. Visit https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfxethighperformance for more details.
  warnings.warn(
[RUN] Downloading Rowan/hellaswag
README.md: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.02k/7.02k [00:00<00:00, 36.9MB/s]
data/train-00000-of-00001.parquet: downloading bytes: ██████████████████████████████████████████████████████████████████████████████████████████████████████| 23.2MB, 2.16MB/s
data/train-00000-of-00001.parquet: reconstructing file: 100%|██████████████████████████████████████████████████████████████████████████████████████| 24.4MB / 24.4MB, 2.34MB/s
data/test-00000-of-00001.parquet: downloading bytes: ███████████████████████████████████████████████████████████████████████████████████████████████████████| 5.58MB,  544kB/s
data/test-00000-of-00001.parquet: reconstructing file: 100%|███████████████████████████████████████████████████████████████████████████████████████| 6.11MB / 6.11MB,  599kB/s
data/validation-00000-of-00001.parquet: downloading bytes: █████████████████████████████████████████████████████████████████████████████████████████████████| 6.03MB,  583kB/s
data/validation-00000-of-00001.parquet: reconstructing file: 100%|█████████████████████████████████████████████████████████████████████████████████| 6.32MB / 6.32MB,  617kB/s
Generating train split: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 39905/39905 [00:00<00:00, 865723.06 examples/s]
Generating test split: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 10003/10003 [00:00<00:00, 903378.83 examples/s]
Generating validation split: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 10042/10042 [00:00<00:00, 869907.90 examples/s]
[OK] Staged Rowan/hellaswag (default) at 218ec52e09a7e7462a5400043bb9a69a41d06b76: train=39905, test=10003, validation=10042
The training service reads this snapshot offline; a re-stage moves it.
exit=0
```

Preconditions judged: `make train-build` rebuilt the training image's pip
layer from the new lock (2408 s, the flash-attn and DeepSpeed compiles
behind the `uv sync`; the export image was a cache hit) and passed its
build smoke; each of the four staging lines exited 0 with its `[OK] Staged
<id> (<config>) at <sha>: <splits>` line, the pins being
`966cd89545d6b6acfd7638bc708b98261ca58e84` (IFEval, 541 train),
`740312add88f781978c0658806c59bc2815b9866` (gsm8k main, 7473 train and
1319 test), `cd6a8ee9e1e6f3f18f683535f6c096c386a85da3` (tinyMMLU all, 100
test and 285 dev) and `218ec52e09a7e7462a5400043bb9a69a41d06b76`
(hellaswag, 39905 train, 10003 test, 10042 validation). Two lines are
known, neither a finding: `[WARN] nvidia-smi not found` is the prepare
service's, which reserves no GPU, and the `HF_HUB_ENABLE_HF_TRANSFER`
FutureWarning is huggingface_hub deprecating the variable the prepare
service sets (`docker-compose.training.yaml`) in favour of its Xet
transfer; the downloads ran regardless.

First evaluation attempt (2026-09-25; the holdout half verified, the lm-eval
half superseded by KB-92's second fix and KB-93). The four commands as
committed, pasted verbatim, terminal artefacts included:

```
➜  ai-experiments git:(SFT-verification-2) set -o pipefail
cd ~/workspace/ai-experiments && git pull origin SFT-verification-2 && git log -1 --oneline
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model /app/models/hf/gemma-4-E2B \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/ft5-holdout-base.json
echo "exit=$?"
remote: Enumerating objects: 11, done.
remote: Counting objects: 100% (11/11), done.
remote: Compressing objects: 100% (2/2), done.
remote: Total 6 (delta 4), reused 6 (delta 4), pack-reused 0 (from 0)
Unpacking objects: 100% (6/6), 3.34 KiB | 1.11 MiB/s, done.
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch              SFT-verification-2 -> FETCH_HEAD
   f52c158a..afa9bd04  SFT-verification-2 -> origin/SFT-verification-2
Updating f52c158..afa9bd0
Fast-forward
 docs/verification/training/fine-tuning-loop.md | 214 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 214 insertions(+)
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-816becdcc985 Creating
Container ai-experiments-training-run-816becdcc985 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-holdout --model /app/models/hf/gemma-4-E2B --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --out /mnt/ai-data/training/eval/ft5-holdout-base.json
[RUN] Held-out loss: /app/models/hf/gemma-4-E2B over /mnt/ai-data/training/datasets/sft/ultrachat-e2b
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3686.48it/s]
[OK] Loss 1.064038, perplexity 2.8980 (4066 examples, 3631683 supervised tokens, bf16)
[OK] Record written: /mnt/ai-data/training/eval/ft5-holdout-base.json
exit=0
➜  ai-experiments git:(SFT-verification-2) ^[[200~docker compose -f docker-compose.training.yaml run --rm training \
>     python -m training eval-holdout --model /app/models/trained/ft3-straight/hf \
>     --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
>     --out /mnt/ai-data/training/eval/ft5-holdout-merged.json
200~docker compose -f docker-compose.training.yaml run --rm training python200~dockerzsh: bad pattern: ^[[200~docker
➜  ai-experiments git:(SFT-verification-2) echo "exit=$?"
exit=1
➜  ai-experiments git:(SFT-verification-2) docker compose -f docker-compose.training.yaml run --rm training \
>     python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 \
>     --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-e08779292dc3 Creating
Container ai-experiments-training-run-e08779292dc3 Created
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/trained/ft3-straight/hf --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-merged.json
echo "exit=$?"
grep -E '"(loss|perplexity|supervised_tokens|precision|attention_implementation)"' \
    /mnt/ai-data/training/eval/ft5-holdout-base.json /mnt/ai-data/training/eval/ft5-holdout-merged.json
grep -E '"(lm_eval_version|limit)"' /mnt/ai-data/training/eval/ft5-lmeval-base.json /mnt/ai-data/training/eval/ft5-lmeval-merged.json^[[201~=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
[RUN] lm-evaluation-harness: /app/models/hf/gemma-4-E2B
^C^CTraceback (most recent call last):
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
  File "/app/training/cli.py", line 309, in cmd_eval_lmeval
    record = run_lmeval(
             ^^^^^^^^^^^
  File "/app/training/eval/lmeval.py", line 106, in run_lmeval
    raw = lm_eval.simple_evaluate(
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/evaluator.py", line 257, in simple_evaluate
    lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 508, in get_model
    return model_registry.get(model_name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 310, in get
    concrete = self._materialise(fresh)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 270, in _materialise
    return cast("T", _materialise_placeholder(ph))
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/api/registry.py", line 121, in _materialise_placeholder
    return getattr(importlib.import_module(mod), attr)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/models/huggingface.py", line 23, in <module>
    from transformers.models.auto.modeling_auto import (
  File "/opt/venv/lib/python3.12/site-packages/transformers/models/auto/modeling_auto.py", line 21, in <module>
    from .auto_factory import (
  File "/opt/venv/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 42, in <module>
    from ...generation import GenerationMixin
  File "<frozen importlib._bootstrap>", line 1412, in _handle_fromlist
  File "/opt/venv/lib/python3.12/site-packages/transformers/utils/import_utils.py", line 2305, in __getattr__
    module = self._get_module(self._class_to_module[name])
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/transformers/utils/import_utils.py", line 2537, in _get_module
    return importlib.import_module("." + module_name, self.__name__)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/importlib/__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/transformers/generation/utils.py", line 44, in <module>
    from ..masking_utils import create_masks_for_generate
  File "/opt/venv/lib/python3.12/site-packages/transformers/masking_utils.py", line 42, in <module>
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/__init__.py", line 13, in <module>
    from . import (
  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/aot_compile.py", line 17, in <module>
    from torch._dynamo.convert_frame import GraphRuntimeEnv
  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", line 62, in <module>
    from torch._dynamo.symbolic_convert import TensorifyState
  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/symbolic_convert.py", line 62, in <module>
    from . import (
  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/trace_rules.py", line 60, in <module>
    from .variables import (
  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/variables/__init__.py", line 19, in <module>
    from .base import VariableTracker
  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/variables/base.py", line 1943, in <module>
    from . import builder
  File "/opt/venv/lib/python3.12/site-packages/torch/_dynamo/variables/builder.py", line 207, in <module>
    from .builtin import BuiltinVariable
KeyboardInterrupt
^C
➜  ai-experiments git:(SFT-verification-2)
➜  ai-experiments git:(SFT-verification-2) docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model /app/models/trained/ft3-straight/hf \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/ft5-holdout-merged.json
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
echo "exit=$?"
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/trained/ft3-straight/hf --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-merged.json
echo "exit=$?"
grep -E '"(loss|perplexity|supervised_tokens|precision|attention_implementation)"' \
    /mnt/ai-data/training/eval/ft5-holdout-base.json /mnt/ai-data/training/eval/ft5-holdout-merged.json
grep -E '"(lm_eval_version|limit)"' /mnt/ai-data/training/eval/ft5-lmeval-base.json /mnt/ai-data/training/eval/ft5-lmeval-merged.json
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-d31ca4216260 Creating
Container ai-experiments-training-run-d31ca4216260 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-holdout --model /app/models/trained/ft3-straight/hf --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --out /mnt/ai-data/training/eval/ft5-holdout-merged.json
[RUN] Held-out loss: /app/models/trained/ft3-straight/hf over /mnt/ai-data/training/datasets/sft/ultrachat-e2b
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:02<00:00, 721.72it/s]
[OK] Loss 1.019455, perplexity 2.7717 (4066 examples, 3631683 supervised tokens, bf16)
[OK] Record written: /mnt/ai-data/training/eval/ft5-holdout-merged.json
exit=0
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-b5f2148548cf Creating
Container ai-experiments-training-run-b5f2148548cf Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
[RUN] lm-evaluation-harness: /app/models/hf/gemma-4-E2B
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3784.24it/s]
/opt/venv/lib/python3.12/site-packages/nltk/downloader.py:1076: UserWarning: NLTK will not authorize the non-private download directory '/app/nltk_data': it (or an ancestor) is world- or group-writable, so another local user could plant files there. Choose a private location such as ~/nltk_data.
  for msg in self.incr_download(info_or_id, download_dir, force):
[nltk_data] Error loading punkt_tab: <urlopen error pathsec.urlopen:
[nltk_data]     no validated address for host
[nltk_data]     'raw.githubusercontent.com'; refusing to connect by
[nltk_data]     unvalidated hostname>
Downloaded punkt_tab on rank 0
Traceback (most recent call last):
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
  File "/app/training/cli.py", line 309, in cmd_eval_lmeval
    record = run_lmeval(
             ^^^^^^^^^^^
  File "/app/training/eval/lmeval.py", line 106, in run_lmeval
    raw = lm_eval.simple_evaluate(
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/evaluator.py", line 302, in simple_evaluate
    loaded = task_manager.load(tasks)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/manager.py", line 208, in load
    obj = self._load_spec(spec) if not isinstance(spec, (Task, Group)) else spec  # type:ignore[invalid-argument-type]
          ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/manager.py", line 154, in _load_spec
    return self._factory.build(
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/_factory.py", line 61, in build
    return self._build_task(entry, overrides)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/_factory.py", line 81, in _build_task
    obj = ConfigurableTask(config=cfg)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/api/task.py", line 753, in __init__
    self.download(self.config.dataset_kwargs)
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/api/task.py", line 869, in download
    self.dataset = datasets.load_dataset(
                   ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/datasets/load.py", line 1698, in load_dataset
    builder_instance = load_dataset_builder(
                       ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/datasets/load.py", line 1325, in load_dataset_builder
    dataset_module = dataset_module_factory(
                     ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/datasets/load.py", line 1217, in dataset_module_factory
    raise e1 from None
  File "/opt/venv/lib/python3.12/site-packages/datasets/load.py", line 1159, in dataset_module_factory
    raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
ConnectionError: Couldn't reach 'google/IFEval' on the Hub (OfflineModeIsEnabled)

exit=1
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-0750f816f7de Creating
Container ai-experiments-training-run-0750f816f7de Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-lmeval --model /app/models/trained/ft3-straight/hf --limit 20 --out /mnt/ai-data/training/eval/ft5-lmeval-merged.json
[RUN] lm-evaluation-harness: /app/models/trained/ft3-straight/hf
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3676.40it/s]
/opt/venv/lib/python3.12/site-packages/nltk/downloader.py:1076: UserWarning: NLTK will not authorize the non-private download directory '/app/nltk_data': it (or an ancestor) is world- or group-writable, so another local user could plant files there. Choose a private location such as ~/nltk_data.
  for msg in self.incr_download(info_or_id, download_dir, force):
[nltk_data] Error loading punkt_tab: <urlopen error pathsec.urlopen:
[nltk_data]     no validated address for host
[nltk_data]     'raw.githubusercontent.com'; refusing to connect by
[nltk_data]     unvalidated hostname>
Downloaded punkt_tab on rank 0
Traceback (most recent call last):
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
  File "/app/training/cli.py", line 309, in cmd_eval_lmeval
    record = run_lmeval(
             ^^^^^^^^^^^
  File "/app/training/eval/lmeval.py", line 106, in run_lmeval
    raw = lm_eval.simple_evaluate(
          ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/utils.py", line 575, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/evaluator.py", line 302, in simple_evaluate
    loaded = task_manager.load(tasks)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/manager.py", line 208, in load
    obj = self._load_spec(spec) if not isinstance(spec, (Task, Group)) else spec  # type:ignore[invalid-argument-type]
          ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/manager.py", line 154, in _load_spec
    return self._factory.build(
           ^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/_factory.py", line 61, in build
    return self._build_task(entry, overrides)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/tasks/_factory.py", line 81, in _build_task
    obj = ConfigurableTask(config=cfg)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/api/task.py", line 753, in __init__
    self.download(self.config.dataset_kwargs)
  File "/opt/venv/lib/python3.12/site-packages/lm_eval/api/task.py", line 869, in download
    self.dataset = datasets.load_dataset(
                   ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/datasets/load.py", line 1698, in load_dataset
    builder_instance = load_dataset_builder(
                       ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/datasets/load.py", line 1325, in load_dataset_builder
    dataset_module = dataset_module_factory(
                     ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/venv/lib/python3.12/site-packages/datasets/load.py", line 1217, in dataset_module_factory
    raise e1 from None
  File "/opt/venv/lib/python3.12/site-packages/datasets/load.py", line 1159, in dataset_module_factory
    raise ConnectionError(f"Couldn't reach '{path}' on the Hub ({e.__class__.__name__})") from e
ConnectionError: Couldn't reach 'google/IFEval' on the Hub (OfflineModeIsEnabled)

exit=1
/mnt/ai-data/training/eval/ft5-holdout-base.json:  "supervised_tokens": 3631683,
/mnt/ai-data/training/eval/ft5-holdout-base.json:  "loss": 1.064038,
/mnt/ai-data/training/eval/ft5-holdout-base.json:  "perplexity": 2.898049,
/mnt/ai-data/training/eval/ft5-holdout-base.json:  "precision": "bf16",
/mnt/ai-data/training/eval/ft5-holdout-base.json:  "attention_implementation": "sdpa"
/mnt/ai-data/training/eval/ft5-holdout-merged.json:  "supervised_tokens": 3631683,
/mnt/ai-data/training/eval/ft5-holdout-merged.json:  "loss": 1.019455,
/mnt/ai-data/training/eval/ft5-holdout-merged.json:  "perplexity": 2.771684,
/mnt/ai-data/training/eval/ft5-holdout-merged.json:  "precision": "bf16",
/mnt/ai-data/training/eval/ft5-holdout-merged.json:  "attention_implementation": "sdpa"
grep: /mnt/ai-data/training/eval/ft5-lmeval-base.json: No such file or directory
grep: /mnt/ai-data/training/eval/ft5-lmeval-merged.json: No such file or directory
```

First evaluation attempt judged. The holdout half passed every criterion:
both runs printed `[RUN] Held-out loss: <model> over <dataset>`, the sdpa
line KB-84's third loader owes (its first run, and no FlashAttention
error), `[OK] Loss 1.064038, perplexity 2.8980 (4066 examples, 3631683
supervised tokens, bf16)` for the base and `[OK] Loss 1.019455, perplexity
2.7717` for the merged directory over the same 3631683 tokens, `[OK] Record
written:` and `exit=0` each; the `grep` shows `precision` `bf16` and
`attention_implementation` `sdpa` on both records, the merged loss
(1.019455) below the base's (1.064038), and in the neighbourhood of the
trainer's nf4 `1.030488`, 1.1 percent under it, recorded without judgement
as the entry says. The lm-eval half failed at task load on both sides,
`exit=1`, no record written: `ConnectionError: Couldn't reach 'google/IFEval'
on the Hub (OfflineModeIsEnabled)` from `datasets.load` under the
harness task's `download`, after nltk's `[nltk_data] Error loading
punkt_tab` refusal at the ifeval task's import (`no validated address for
host 'raw.githubusercontent.com'`, the internal network doing its job) and
its misleading `Downloaded punkt_tab on rank 0` print. That is the
cache-miss signature the entry named, but not for the reason it named:
the staging lines had run and exited 0, and the mount was the wrong half of
the cache (KB-92, reopened), with two more run-time needs behind it (KB-93).
Three lines are the operator's terminal, not findings: zsh's `bad pattern:
^[[200~docker` (a bracketed paste the shell rejected, so `echo "exit=$?"`
printed 1 for it before the commands were re-issued in order), the Ctrl-C
during the first lm-eval's torch import, and the `Found orphan containers`
WARN of every compose run. One observation, no row: `holdout.load_model`
passes `torch_dtype`, which transformers now deprecates in favour of
`dtype`; it loaded in bf16 regardless, as the records show.

Second evaluation attempt (2026-09-25; the probes verified, the base's
lm-eval superseded by KB-94). The precondition block as committed after
the first attempt, then the base side of the lm-eval block, pasted
verbatim; the merged side was not started:

```
➜  ai-experiments git:(SFT-verification-2) set -o pipefail
cd ~/workspace/ai-experiments && git pull origin SFT-verification-2 && git log -1 --oneline
make train-build
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch            SFT-verification-2 -> FETCH_HEAD
Already up to date.
Building training image...
MAX_JOBS=4 NVCC_THREADS=2 docker compose -f docker-compose.training.yaml build
[+] Building 0.5s (40/40) FINISHED
 => [internal] load local bake definitions                                                                                                                                  0.0s
 => => reading from stdin 1.71kB                                                                                                                                            0.0s
 => [prepare internal] load build definition from Dockerfile.training                                                                                                       0.0s
 => => transferring dockerfile: 7.51kB                                                                                                                                      0.0s
 => [export internal] load build definition from Dockerfile.export                                                                                                          0.0s
 => => transferring dockerfile: 3.01kB                                                                                                                                      0.0s
 => [training internal] load metadata for docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04                                                                              0.3s
 => [export internal] load metadata for ghcr.io/astral-sh/uv:0.11.28                                                                                                        0.3s
 => [export internal] load metadata for docker.io/library/ubuntu:24.04                                                                                                      0.3s
 => [training internal] load .dockerignore                                                                                                                                  0.0s
 => => transferring context: 1.73kB                                                                                                                                         0.0s
 => [export stage-1 1/9] FROM docker.io/library/ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3                                        0.0s
 => [prepare uv 1/1] FROM ghcr.io/astral-sh/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa                                              0.0s
 => CACHED [export stage-1 2/9] RUN apt-get update &&     apt-get install -y --no-install-recommends         git         ca-certificates         python3.12         python  0.0s
 => CACHED [export stage-1 3/9] COPY --from=uv /uv /uvx /usr/local/bin/                                                                                                     0.0s
 => CACHED [export stage-1 4/9] RUN uv venv /opt/venv --python 3.12                                                                                                         0.0s
 => CACHED [export stage-1 5/9] RUN --mount=type=cache,target=/root/.cache/uv     uv pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 &&     uv   0.0s
 => CACHED [export stage-1 6/9] RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser                                     0.0s
 => CACHED [export stage-1 7/9] RUN mkdir -p /app/cache /app/logs && chown -R appuser:appuser /app                                                                          0.0s
 => CACHED [export stage-1 8/9] WORKDIR /app                                                                                                                                0.0s
 => CACHED [export stage-1 9/9] RUN python -c "import torch, transformers, peft, safetensors, omegaconf, xxhash"                                                            0.0s
 => [training internal] load build context                                                                                                                                  0.0s
 => => transferring context: 7.39kB                                                                                                                                         0.0s
 => [prepare stage-1  1/16] FROM docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04@sha256:e071e85c52ad91fc9ea24158ff5330876b2d1a5c4ac83ccc6066976835873c01               0.0s
 => [export] exporting to image                                                                                                                                             0.0s
 => => exporting layers                                                                                                                                                     0.0s
 => => writing image sha256:84c03bf6d68e4b6c1794365545ddcab077851ac24fe7de5478e15569d1b1900a                                                                                0.0s
 => => naming to docker.io/library/ai-experiments-export                                                                                                                    0.0s
 => CACHED [prepare stage-1  2/16] RUN apt-get update &&     apt-get install -y --no-install-recommends --allow-change-held-packages         build-essential         gcc g  0.0s
 => CACHED [prepare stage-1  3/16] COPY --from=uv /uv /uvx /usr/local/bin/                                                                                                  0.0s
 => CACHED [prepare stage-1  4/16] RUN uv venv /opt/venv --python 3.12                                                                                                      0.0s
 => CACHED [prepare stage-1  5/16] COPY pyproject.toml uv.lock ./                                                                                                           0.0s
 => CACHED [prepare stage-1  6/16] RUN --mount=type=cache,target=/root/.cache/uv     MAX_JOBS=4 NVCC_THREADS=2     DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1     CFLAGS="-march  0.0s
 => CACHED [prepare stage-1  7/16] RUN mkdir -p /opt/nltk_data/tokenizers &&     curl -fsSL -o /tmp/punkt_tab.zip "https://raw.githubusercontent.com/nltk/nltk_data/gh-pag  0.0s
 => CACHED [prepare stage-1  8/16] RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin -c "Application User" appuser                                  0.0s
 => CACHED [prepare stage-1  9/16] RUN mkdir -p /app/cache /app/logs && chown -R appuser:appuser /app                                                                       0.0s
 => CACHED [prepare stage-1 10/16] WORKDIR /app                                                                                                                             0.0s
 => CACHED [prepare stage-1 11/16] COPY --chown=appuser:appuser common ./common                                                                                             0.0s
 => CACHED [prepare stage-1 12/16] COPY --chown=appuser:appuser training ./training                                                                                         0.0s
 => CACHED [prepare stage-1 13/16] COPY --chown=appuser:appuser scripts/training ./scripts/training                                                                         0.0s
 => CACHED [prepare stage-1 14/16] COPY --chown=appuser:appuser docker/training/entrypoint.sh /app/entrypoint.sh                                                            0.0s
 => CACHED [prepare stage-1 15/16] RUN chmod +x /app/entrypoint.sh                                                                                                          0.0s
 => CACHED [prepare stage-1 16/16] RUN python -m training smoke --build                                                                                                     0.0s
 => [training] exporting to image                                                                                                                                           0.0s
 => => exporting layers                                                                                                                                                     0.0s
 => => writing image sha256:934703344a5ec85b083f2efe8a2dafc9f155f5aeed9557d728a28a9b05cb007e                                                                                0.0s
 => => naming to docker.io/library/ai-experiments-training                                                                                                                  0.0s
 => [prepare] exporting to image                                                                                                                                            0.0s
 => => exporting layers                                                                                                                                                     0.0s
 => => writing image sha256:d2bdb27bda84656ebcd266dbf28a9a3c1d3bbdcfeb6015db10c7608b4ced5ad8                                                                                0.0s
 => => naming to docker.io/library/ai-experiments-prepare                                                                                                                   0.0s
 => [export] resolving provenance for metadata file                                                                                                                         0.0s
 => [training] resolving provenance for metadata file                                                                                                                       0.0s
 => [prepare] resolving provenance for metadata file                                                                                                                        0.0s
[+] build 3/3
 ✔ Image ai-experiments-export   Built                                                                                                                                       0.5s
 ✔ Image ai-experiments-training Built                                                                                                                                       0.5s
 ✔ Image ai-experiments-prepare  Built                                                                                                                                       0.5s
Training image built.
➜  ai-experiments git:(SFT-verification-2) docker compose -f docker-compose.training.yaml run --rm training \
    python -c "from datasets import load_dataset; [print(d, load_dataset(d, n)) for d, n in [('google/IFEval', None), ('openai/gsm8k', 'main'), ('tinyBenchmarks/tinyMMLU', 'all'), ('Rowan/hellaswag', None)]]"
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-7a342d4a9dd0 Creating
Container ai-experiments-training-run-7a342d4a9dd0 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -c from datasets import load_dataset; [print(d, load_dataset(d, n)) for d, n in [('google/IFEval', None), ('openai/gsm8k', 'main'), ('tinyBenchmarks/tinyMMLU', 'all'), ('Rowan/hellaswag', None)]]
Using the latest cached version of the dataset since google/IFEval couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84 (last modified on Fri Sep 25 19:34:02 2026).
google/IFEval DatasetDict({
    train: Dataset({
        features: ['key', 'prompt', 'instruction_id_list', 'kwargs'],
        num_rows: 541
    })
})
Using the latest cached version of the dataset since openai/gsm8k couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'main' at /app/cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866 (last modified on Fri Sep 25 19:34:09 2026).
openai/gsm8k DatasetDict({
    train: Dataset({
        features: ['question', 'answer'],
        num_rows: 7473
    })
    test: Dataset({
        features: ['question', 'answer'],
        num_rows: 1319
    })
})
Using the latest cached version of the dataset since tinyBenchmarks/tinyMMLU couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'all' at /app/cache/datasets/tinyBenchmarks___tiny_mmlu/all/0.0.0/cd6a8ee9e1e6f3f18f683535f6c096c386a85da3 (last modified on Fri Sep 25 19:35:56 2026).
tinyBenchmarks/tinyMMLU DatasetDict({
    test: Dataset({
        features: ['question', 'subject', 'choices', 'answer', 'input_formatted'],
        num_rows: 100
    })
    dev: Dataset({
        features: ['question', 'subject', 'choices', 'answer', 'input_formatted'],
        num_rows: 285
    })
})
Using the latest cached version of the dataset since Rowan/hellaswag couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76 (last modified on Fri Sep 25 19:36:04 2026).
Rowan/hellaswag DatasetDict({
    train: Dataset({
        features: ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label'],
        num_rows: 39905
    })
    test: Dataset({
        features: ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label'],
        num_rows: 10003
    })
    validation: Dataset({
        features: ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label'],
        num_rows: 10042
    })
})
➜  ai-experiments git:(SFT-verification-2) docker compose -f docker-compose.training.yaml run --rm training \
    python -c "from lm_eval.tasks.ifeval import instructions_util as u; print(u.count_sentences('One sentence. Two sentences.'))"
echo "exit=$?"
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-4ae48b05a724 Creating
Container ai-experiments-training-run-4ae48b05a724 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -c from lm_eval.tasks.ifeval import instructions_util as u; print(u.count_sentences('One sentence. Two sentences.'))
2
exit=0
➜  ai-experiments git:(SFT-verification-2) set -o pipefail
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
echo "exit=$?"
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-fe90af220f94 Creating
Container ai-experiments-training-run-fe90af220f94 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
[RUN] lm-evaluation-harness: /app/models/hf/gemma-4-E2B
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3793.07it/s]
Using the latest cached version of the dataset since google/IFEval couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84 (last modified on Fri Sep 25 19:34:02 2026).
Using the latest cached version of the dataset since openai/gsm8k couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'main' at /app/cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866 (last modified on Fri Sep 25 19:34:09 2026).
Using the latest cached version of the dataset since tinyBenchmarks/tinyMMLU couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'all' at /app/cache/datasets/tinyBenchmarks___tiny_mmlu/all/0.0.0/cd6a8ee9e1e6f3f18f683535f6c096c386a85da3 (last modified on Fri Sep 25 19:35:56 2026).
Using the latest cached version of the dataset since Rowan/hellaswag couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76 (last modified on Fri Sep 25 19:36:04 2026).
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39905/39905 [00:01<00:00, 25031.48 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10042/10042 [00:00<00:00, 16382.73 examples/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 125955.08it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 772.39it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 9205.10it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 4751.68it/s]
Running generate_until requests:   0%|                                                                                                                   | 0/40 [00:00<?, ?it/s]Passed argument batch_size = auto. Detecting largest batch size
[W925 21:06:00.714377720 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3221225472 bytes (free: 2256338944, total: 33668726784).
[W925 21:06:01.065873899 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 1383923712, total: 33668726784).
[W925 21:06:03.032230568 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 15682306048, total: 33668726784).
[W925 21:06:05.776509653 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2868903936 bytes (free: 2346516480, total: 33668726784).
[W925 21:06:06.943198847 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 1570570240, total: 33668726784).
[W925 21:06:08.723751930 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 11498487808, total: 33668726784).
[W925 21:06:11.355036757 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 1438449664, total: 33668726784).
[W925 21:06:12.958278178 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 14801502208, total: 33668726784).
[W925 21:06:15.310114095 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 5924257792, total: 33668726784).
[W925 21:06:16.719753183 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 15682306048, total: 33668726784).
[W925 21:06:18.832494662 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 7792820224, total: 33668726784).
[W925 21:06:19.100000285 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 16489709568, total: 33668726784).
[W925 21:06:21.025130734 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 9388752896, total: 33668726784).
[W925 21:06:23.164918684 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 16783310848, total: 33668726784).
[W925 21:06:24.913049520 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 10472980480, total: 33668726784).
[W925 21:06:25.929648830 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 17737515008, total: 33668726784).
[W925 21:06:27.490761794 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 12215713792, total: 33668726784).
[W925 21:06:28.380785457 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 18324717568, total: 33668726784).
[W925 21:06:29.804138535 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 14652604416, total: 33668726784).
[W925 21:06:30.592420185 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 16489709568, total: 33668726784).
[W925 21:06:31.877403987 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 14872805376, total: 33668726784).
[W925 21:06:32.572514188 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 17223712768, total: 33668726784).
[W925 21:06:33.724368617 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 15827009536, total: 33668726784).
[W925 21:06:34.324520215 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 17884315648, total: 33668726784).
[W925 21:06:35.383509774 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 16634413056, total: 33668726784).
[W925 21:06:35.924653095 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 21:06:35.924718308 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 21:06:36.886452457 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 213712896, total: 33668726784).
[W925 21:06:37.465293213 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 3080519680, total: 33668726784).
[W925 21:06:38.334766903 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3076325376, total: 33668726784).
[W925 21:06:38.831127259 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3080519680, total: 33668726784).
[W925 21:06:39.608994205 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3076325376, total: 33668726784).
[W925 21:06:39.027831010 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3080519680, total: 33668726784).
[W925 21:06:40.708729507 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3078422528, total: 33668726784).
[W925 21:06:40.048489258 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3080519680, total: 33668726784).
[W925 21:06:42.407633693 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3078422528, total: 33668726784).
[W925 21:06:42.710318434 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3080519680, total: 33668726784).
[W925 21:06:43.920834268 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3078422528, total: 33668726784).
[W925 21:06:44.182564864 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3080519680, total: 33668726784).
[W925 21:06:45.241738864 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3078422528, total: 33668726784).
[W925 21:06:45.463022332 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3080519680, total: 33668726784).
Determined Largest batch size: 5
Running generate_until requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [02:27<00:00,  3.68s/it]
Tokenizing inputs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:00<00:00, 2081.83it/s]
Running loglikelihood requests:   0%|                                                                                                                   | 0/160 [00:00<?, ?it/s]Passed argument batch_size = auto:1. Detecting largest batch size
[W925 21:08:28.922504488 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 287113216, total: 33668726784).
[W925 21:08:30.774174447 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 16055599104, total: 33668726784).
[W925 21:08:33.600489690 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 708640768, total: 33668726784).
[W925 21:08:35.268910611 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 12094078976, total: 33668726784).
[W925 21:08:37.826004307 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 2568814592, total: 33668726784).
[W925 21:08:39.324522111 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 15223029760, total: 33668726784).
[W925 21:08:41.582394902 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 6813450240, total: 33668726784).
[W925 21:08:42.932656716 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 15638265856, total: 33668726784).
[W925 21:08:44.973551415 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 8166113280, total: 33668726784).
[W925 21:08:46.171584697 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 14080081920, total: 33668726784).
[W925 21:08:47.033116314 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 11573985280, total: 33668726784).
[W925 21:08:48.111723667 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 15539699712, total: 33668726784).
[W925 21:08:50.797818687 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 11785797632, total: 33668726784).
[W925 21:08:51.759529006 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 18001756160, total: 33668726784).
[W925 21:08:53.266897176 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 12771459072, total: 33668726784).
[W925 21:08:53.107484452 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 18280677376, total: 33668726784).
[W925 21:08:55.482627102 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 13608222720, total: 33668726784).
[W925 21:08:56.232408964 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 16542138368, total: 33668726784).
[W925 21:08:57.478268561 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 15009120256, total: 33668726784).
[W925 21:08:57.142554546 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 19459276800, total: 33668726784).
[W925 21:08:59.255338518 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 16814768128, total: 33668726784).
[W925 21:08:59.823393854 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 18138071040, total: 33668726784).
[W925 21:09:00.854284963 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 16953180160, total: 33668726784).
[W925 21:09:01.368560944 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 21:09:01.368630175 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 21:09:02.309101629 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 1406992384, total: 33668726784).
[W925 21:09:02.862295982 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 4124901376, total: 33668726784).
[W925 21:09:03.715976864 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4120707072, total: 33668726784).
[W925 21:09:04.186027853 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4124901376, total: 33668726784).
[W925 21:09:04.940982814 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4120707072, total: 33668726784).
[W925 21:09:05.337372944 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4124901376, total: 33668726784).
[W925 21:09:05.005400304 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4122804224, total: 33668726784).
[W925 21:09:06.327367055 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4124901376, total: 33668726784).
[W925 21:09:07.630512662 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4122804224, total: 33668726784).
[W925 21:09:07.916204904 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4124901376, total: 33668726784).
[W925 21:09:08.098506353 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4122804224, total: 33668726784).
[W925 21:09:09.350117567 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4124901376, total: 33668726784).
[W925 21:09:10.365576045 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4122804224, total: 33668726784).
[W925 21:09:10.572704740 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4124901376, total: 33668726784).
Determined largest batch size: 5
Running loglikelihood requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:48<00:00,  3.28it/s]
fatal: not a git repository (or any of the parent directories): .git
[ERROR] Metric prompt_level_strict_acc missing from ifeval results (available: alias, inst_level_loose_acc,none, inst_level_loose_acc_stderr,none, inst_level_strict_acc,none, inst_level_strict_acc_stderr,none, name, prompt_level_loose_acc,none, prompt_level_loose_acc_stderr,none, prompt_level_strict_acc,none, prompt_level_strict_acc_stderr,none, sample_len)
  Solution: fix the metric key in tasks.yaml

exit=1
```

Judgement. Stage 1 met every criterion. `make train-build` exited 0 with
every layer cached, the nltk layer included (`CACHED [prepare stage-1
7/16] RUN mkdir -p /opt/nltk_data/tokenizers && curl ...`), so the image
the first attempt ran was already the fixed one and only the compose
mount had changed. The dataset probe printed the four `DatasetDict`s with
the staged split counts (541; 7473 and 1319; 100 and 285; 39905, 10003
and 10042), each after the `Using the latest cached version` warning and
a `Found the latest cached dataset configuration` line naming a path
under `/app/cache/datasets`, which is KB-92's second fix holding: the
datasets half is what `datasets` reads offline. The sentence-count probe
printed `2`, `exit=0`, with no `[nltk_data]` line and no `Downloaded
punkt_tab` line, which is KB-93's nltk half holding and settles the
point the entry could not verify from the sandbox: nltk 3.10 serves the
task's `punkt` request from `punkt_tab`. The base `eval-lmeval` then
loaded all four tasks (the same four cache lines, then hellaswag's two
`Map:` passes writing its processed docs beside the arrow files, the
reason the mount is read-write), with no `ModuleNotFoundError`, so
KB-93's tinyMMLU half held too, and ran to scoring: 40 generation
requests (gsm8k and ifeval, 2:27 at batch 5) and 160 loglikelihood
requests (0:48). It failed in the tree's own extraction, `exit=1`, no
record written: `Metric prompt_level_strict_acc missing from ifeval
results`, the available keys all carrying a `,none` suffix. That is
KB-94: lm-eval keys every result as `<metric>,<filter>`, `none` being
the default filter's name, and `tasks.yaml` declared the three
unfiltered keys bare; hellaswag and `tinyMMLU_acc` would have been
refused next, and only gsm8k's `exact_match,strict-match` was right.
The harness's work was correct and complete; the record was refused at
the last step, so the run is superseded, not failed on a criterion.
Three observations, no row. The `batch_size=auto` search printed about
forty `CUDACachingAllocator ... memory allocation failed with OOM`
warnings per phase before `Determined Largest batch size: 5`: the
harness probes descending batch sizes at the model's full `max_length`
(131072 for E2B, so the first probes ask for 68 GB of a 33 GB card),
the warnings are torch's, and the run proceeded; a `max_length` in the
model arguments would silence them and is a separate decision. `fatal:
not a git repository` is the harness's `get_git_commit_hash` running
`git describe` in `/app`, which is no repository; the exception is
caught and the hash recorded as absent. The whole base pass took under
four minutes at `--limit 20`, model load included.

Third evaluation attempt (2026-09-25; the lm-eval half, as committed after
KB-94):

```
➜  ai-experiments git:(SFT-verification-2) set -o pipefail
cd ~/workspace/ai-experiments && git pull origin SFT-verification-2 && git log -1 --oneline
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
echo "exit=$?"
remote: Enumerating objects: 29, done.
remote: Counting objects: 100% (29/29), done.
remote: Compressing objects: 100% (3/3), done.
remote: Total 15 (delta 12), reused 15 (delta 12), pack-reused 0 (from 0)
Unpacking objects: 100% (15/15), 9.47 KiB | 1.89 MiB/s, done.
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch              SFT-verification-2 -> FETCH_HEAD
   921897ce..530f4691  SFT-verification-2 -> origin/SFT-verification-2
Updating 921897c..530f469
Fast-forward
 docs/verification/bugs/known-bugs.md           |  11 ++--
 docs/verification/training/fine-tuning-loop.md | 353 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++--
 tests/training/README.md                       |   2 +-
 tests/training/test_lmeval.py                  |  22 ++++++--
 training/eval/tasks.yaml                       |  11 ++--
 5 files changed, 380 insertions(+), 19 deletions(-)
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-11e26f15b66b Creating
Container ai-experiments-training-run-11e26f15b66b Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --limit 20 --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
[RUN] lm-evaluation-harness: /app/models/hf/gemma-4-E2B
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3817.88it/s]
Using the latest cached version of the dataset since google/IFEval couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84 (last modified on Fri Sep 25 19:34:02 2026).
Using the latest cached version of the dataset since openai/gsm8k couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'main' at /app/cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866 (last modified on Fri Sep 25 19:34:09 2026).
Using the latest cached version of the dataset since tinyBenchmarks/tinyMMLU couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'all' at /app/cache/datasets/tinyBenchmarks___tiny_mmlu/all/0.0.0/cd6a8ee9e1e6f3f18f683535f6c096c386a85da3 (last modified on Fri Sep 25 19:35:56 2026).
Using the latest cached version of the dataset since Rowan/hellaswag couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76 (last modified on Fri Sep 25 21:05:57 2026).
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 91180.52it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 742.25it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 8564.17it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 4661.89it/s]
Running generate_until requests:   0%|                                                                                                                   | 0/40 [00:00<?, ?it/s]Passed argument batch_size = auto. Detecting largest batch size
[W925 21:22:13.356927682 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3221225472 bytes (free: 2256338944, total: 33668726784).
[W925 21:22:14.707787901 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 1383923712, total: 33668726784).
[W925 21:22:16.674202151 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 15682306048, total: 33668726784).
[W925 21:22:18.417256657 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2868903936 bytes (free: 2346516480, total: 33668726784).
[W925 21:22:19.584106578 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 1570570240, total: 33668726784).
[W925 21:22:21.364763198 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 11498487808, total: 33668726784).
[W925 21:22:23.991064817 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 1438449664, total: 33668726784).
[W925 21:22:25.592244435 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 14801502208, total: 33668726784).
[W925 21:22:27.946043375 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 5924257792, total: 33668726784).
[W925 21:22:29.355809983 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 15682306048, total: 33668726784).
[W925 21:22:31.466350791 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 7792820224, total: 33668726784).
[W925 21:22:32.734033990 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 16489709568, total: 33668726784).
[W925 21:22:34.660025429 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 9388752896, total: 33668726784).
[W925 21:22:35.800609206 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 16783310848, total: 33668726784).
[W925 21:22:37.554065657 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 10472980480, total: 33668726784).
[W925 21:22:38.570648207 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 17737515008, total: 33668726784).
[W925 21:22:39.134635357 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 12215713792, total: 33668726784).
[W925 21:22:40.024787844 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 18324717568, total: 33668726784).
[W925 21:22:42.444650710 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 14652604416, total: 33668726784).
[W925 21:22:43.231444244 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 16489709568, total: 33668726784).
[W925 21:22:44.513543088 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 14872805376, total: 33668726784).
[W925 21:22:45.208495405 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 17223712768, total: 33668726784).
[W925 21:22:46.353364021 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 15827009536, total: 33668726784).
[W925 21:22:46.953461033 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 17884315648, total: 33668726784).
[W925 21:22:47.008560282 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 16634413056, total: 33668726784).
[W925 21:22:48.548697103 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 21:22:48.548749973 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 21:22:49.511223494 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 213712896, total: 33668726784).
[W925 21:22:49.090283796 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 3080519680, total: 33668726784).
[W925 21:22:50.959893790 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3076325376, total: 33668726784).
[W925 21:22:51.456249627 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3080519680, total: 33668726784).
[W925 21:22:52.232393505 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3076325376, total: 33668726784).
[W925 21:22:52.651323488 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3080519680, total: 33668726784).
[W925 21:22:53.336851940 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3078422528, total: 33668726784).
[W925 21:22:53.676909598 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3080519680, total: 33668726784).
[W925 21:22:54.040511326 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3078422528, total: 33668726784).
[W925 21:22:55.343185070 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3080519680, total: 33668726784).
[W925 21:22:56.550085701 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3078422528, total: 33668726784).
[W925 21:22:56.812063911 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3080519680, total: 33668726784).
[W925 21:22:57.873210513 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3078422528, total: 33668726784).
[W925 21:22:57.094478783 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3080519680, total: 33668726784).
Determined Largest batch size: 5
Running generate_until requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [02:27<00:00,  3.68s/it]
Tokenizing inputs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:00<00:00, 2075.27it/s]
Running loglikelihood requests:   0%|                                                                                                                   | 0/160 [00:00<?, ?it/s]Passed argument batch_size = auto:1. Detecting largest batch size
[W925 21:24:41.643091704 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 287113216, total: 33668726784).
[W925 21:24:43.494183675 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 16055599104, total: 33668726784).
[W925 21:24:46.319611596 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 708640768, total: 33668726784).
[W925 21:24:47.988924436 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 12094078976, total: 33668726784).
[W925 21:24:50.544465949 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 2568814592, total: 33668726784).
[W925 21:24:51.043196367 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 15223029760, total: 33668726784).
[W925 21:24:54.301322919 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 6813450240, total: 33668726784).
[W925 21:24:55.651683198 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 15638265856, total: 33668726784).
[W925 21:24:57.688440649 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 8166113280, total: 33668726784).
[W925 21:24:58.886611027 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 14080081920, total: 33668726784).
[W925 21:25:00.754349594 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 11573985280, total: 33668726784).
[W925 21:25:01.833720384 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 15539699712, total: 33668726784).
[W925 21:25:03.524884655 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 11785797632, total: 33668726784).
[W925 21:25:04.486582761 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 18001756160, total: 33668726784).
[W925 21:25:05.995685046 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 12771459072, total: 33668726784).
[W925 21:25:06.836528902 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 18280677376, total: 33668726784).
[W925 21:25:08.212797587 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 13608222720, total: 33668726784).
[W925 21:25:08.961421801 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 16542138368, total: 33668726784).
[W925 21:25:10.205828700 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 15009120256, total: 33668726784).
[W925 21:25:10.868488697 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 19459276800, total: 33668726784).
[W925 21:25:11.982200831 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 16814768128, total: 33668726784).
[W925 21:25:12.550391849 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 18138071040, total: 33668726784).
[W925 21:25:13.576170931 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 16953180160, total: 33668726784).
[W925 21:25:13.088555875 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 21:25:13.088619515 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 21:25:14.034571945 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 1406992384, total: 33668726784).
[W925 21:25:15.587283247 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 4124901376, total: 33668726784).
[W925 21:25:16.434304925 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4120707072, total: 33668726784).
[W925 21:25:16.904210942 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4124901376, total: 33668726784).
[W925 21:25:17.659119927 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4120707072, total: 33668726784).
[W925 21:25:17.055112811 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4124901376, total: 33668726784).
[W925 21:25:18.721109955 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4122804224, total: 33668726784).
[W925 21:25:18.044013953 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4124901376, total: 33668726784).
[W925 21:25:20.344125223 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4122804224, total: 33668726784).
[W925 21:25:20.629917352 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4124901376, total: 33668726784).
[W925 21:25:21.806788766 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4122804224, total: 33668726784).
[W925 21:25:21.058441417 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4124901376, total: 33668726784).
[W925 21:25:22.071053890 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4122804224, total: 33668726784).
[W925 21:25:23.277950012 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4124901376, total: 33668726784).
Determined largest batch size: 5
Running loglikelihood requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:48<00:00,  3.28it/s]
fatal: not a git repository (or any of the parent directories): .git
[OK] gsm8k: exact_match,strict-match = 0.2500 (stderr 0.0993)
[OK] hellaswag: acc_norm,none = 0.5500 (stderr 0.1141)
[OK] ifeval: prompt_level_strict_acc,none = 0.0500 (stderr 0.0500)
[OK] tinyMMLU_acc: acc_norm,none = 0.6500 (stderr 0.1094)
[OK] Record written: /mnt/ai-data/training/eval/ft5-lmeval-base.json
exit=0

➜  ai-experiments git:(SFT-verification-2) docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /app/models/trained/ft3-straight/hf --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-merged.json
echo "exit=$?"
grep -E '"(lm_eval_version|limit)"' /mnt/ai-data/training/eval/ft5-lmeval-base.json /mnt/ai-data/training/eval/ft5-lmeval-merged.json

WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-86586192114e Creating
Container ai-experiments-training-run-86586192114e Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-lmeval --model /app/models/trained/ft3-straight/hf --limit 20 --out /mnt/ai-data/training/eval/ft5-lmeval-merged.json
[RUN] lm-evaluation-harness: /app/models/trained/ft3-straight/hf
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3689.15it/s]
Using the latest cached version of the dataset since google/IFEval couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84 (last modified on Fri Sep 25 19:34:02 2026).
Using the latest cached version of the dataset since openai/gsm8k couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'main' at /app/cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866 (last modified on Fri Sep 25 19:34:09 2026).
Using the latest cached version of the dataset since tinyBenchmarks/tinyMMLU couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'all' at /app/cache/datasets/tinyBenchmarks___tiny_mmlu/all/0.0.0/cd6a8ee9e1e6f3f18f683535f6c096c386a85da3 (last modified on Fri Sep 25 19:35:56 2026).
Using the latest cached version of the dataset since Rowan/hellaswag couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76 (last modified on Fri Sep 25 21:05:57 2026).
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 85077.16it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 759.57it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 6374.32it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 4578.93it/s]
Running generate_until requests:   0%|                                                                                                                   | 0/40 [00:00<?, ?it/s]Passed argument batch_size = auto. Detecting largest batch size
[W925 21:26:24.911874622 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3221225472 bytes (free: 2256338944, total: 33668726784).
[W925 21:26:26.266123812 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 1383923712, total: 33668726784).
[W925 21:26:28.237222655 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 15682306048, total: 33668726784).
[W925 21:26:29.981670524 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2868903936 bytes (free: 2346516480, total: 33668726784).
[W925 21:26:31.151607482 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 1570570240, total: 33668726784).
[W925 21:26:32.936749402 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 11498487808, total: 33668726784).
[W925 21:26:35.571281008 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 1438449664, total: 33668726784).
[W925 21:26:37.177261618 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 14801502208, total: 33668726784).
[W925 21:26:39.535385915 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 5924257792, total: 33668726784).
[W925 21:26:40.948753161 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 15682306048, total: 33668726784).
[W925 21:26:42.067712834 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 7792820224, total: 33668726784).
[W925 21:26:44.339123865 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 16489709568, total: 33668726784).
[W925 21:26:46.269728282 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 9388752896, total: 33668726784).
[W925 21:26:47.413591565 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 16783310848, total: 33668726784).
[W925 21:26:49.161668523 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 10472980480, total: 33668726784).
[W925 21:26:50.180653384 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 17737515008, total: 33668726784).
[W925 21:26:51.745242379 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 12215713792, total: 33668726784).
[W925 21:26:52.637829049 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 18324717568, total: 33668726784).
[W925 21:26:53.065273910 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 14652604416, total: 33668726784).
[W925 21:26:54.854454579 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 16489709568, total: 33668726784).
[W925 21:26:55.140550768 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 14872805376, total: 33668726784).
[W925 21:26:56.837512933 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 17223712768, total: 33668726784).
[W925 21:26:57.987459202 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 15827009536, total: 33668726784).
[W925 21:26:58.588495435 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 17884315648, total: 33668726784).
[W925 21:26:59.645086243 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 16634413056, total: 33668726784).
[W925 21:27:00.187699024 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 21:27:00.187753928 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 21:27:01.154845646 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 213712896, total: 33668726784).
[W925 21:27:01.736306480 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 3080519680, total: 33668726784).
[W925 21:27:02.607493455 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3076325376, total: 33668726784).
[W925 21:27:02.106039004 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3080519680, total: 33668726784).
[W925 21:27:03.883269483 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3076325376, total: 33668726784).
[W925 21:27:04.303463633 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3080519680, total: 33668726784).
[W925 21:27:04.988240634 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3078422528, total: 33668726784).
[W925 21:27:05.329272784 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3080519680, total: 33668726784).
[W925 21:27:06.691264031 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3078422528, total: 33668726784).
[W925 21:27:06.995180550 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3080519680, total: 33668726784).
[W925 21:27:08.204390471 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3078422528, total: 33668726784).
[W925 21:27:08.467338717 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3080519680, total: 33668726784).
[W925 21:27:09.529174438 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3078422528, total: 33668726784).
[W925 21:27:09.751454226 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3080519680, total: 33668726784).
Determined Largest batch size: 5
Running generate_until requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [02:14<00:00,  3.37s/it]
Tokenizing inputs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:00<00:00, 2040.38it/s]
Running loglikelihood requests:   0%|                                                                                                                   | 0/160 [00:00<?, ?it/s]Passed argument batch_size = auto:1. Detecting largest batch size
[W925 21:28:40.484738158 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 287113216, total: 33668726784).
[W925 21:28:42.336165569 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 16055599104, total: 33668726784).
[W925 21:28:45.166051771 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 708640768, total: 33668726784).
[W925 21:28:46.835919329 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 12094078976, total: 33668726784).
[W925 21:28:49.399323739 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 2568814592, total: 33668726784).
[W925 21:28:50.900211691 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 15223029760, total: 33668726784).
[W925 21:28:53.165416081 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 6813450240, total: 33668726784).
[W925 21:28:54.515676920 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 15638265856, total: 33668726784).
[W925 21:28:56.557454581 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 8166113280, total: 33668726784).
[W925 21:28:57.756576930 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 14080081920, total: 33668726784).
[W925 21:28:59.621722206 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 11573985280, total: 33668726784).
[W925 21:29:00.700709081 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 15539699712, total: 33668726784).
[W925 21:29:02.390023935 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 11785797632, total: 33668726784).
[W925 21:29:03.352603582 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 18001756160, total: 33668726784).
[W925 21:29:04.867993028 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 12771459072, total: 33668726784).
[W925 21:29:05.709516001 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 18280677376, total: 33668726784).
[W925 21:29:06.090592648 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 13608222720, total: 33668726784).
[W925 21:29:07.841394792 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 16542138368, total: 33668726784).
[W925 21:29:08.089163426 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 15009120256, total: 33668726784).
[W925 21:29:09.752557324 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 19459276800, total: 33668726784).
[W925 21:29:10.872218913 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 16814768128, total: 33668726784).
[W925 21:29:11.440386617 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 18138071040, total: 33668726784).
[W925 21:29:12.477419543 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 16953180160, total: 33668726784).
[W925 21:29:12.992843719 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 21:29:12.992917799 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 21:29:13.938475702 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 1406992384, total: 33668726784).
[W925 21:29:14.492295902 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 4124901376, total: 33668726784).
[W925 21:29:15.339686628 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4120707072, total: 33668726784).
[W925 21:29:15.810625256 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4124901376, total: 33668726784).
[W925 21:29:16.565854764 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4120707072, total: 33668726784).
[W925 21:29:16.962925887 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4124901376, total: 33668726784).
[W925 21:29:17.633689171 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4122804224, total: 33668726784).
[W925 21:29:17.956394468 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4124901376, total: 33668726784).
[W925 21:29:19.261733203 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4122804224, total: 33668726784).
[W925 21:29:19.547624584 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4124901376, total: 33668726784).
[W925 21:29:20.727721271 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4122804224, total: 33668726784).
[W925 21:29:20.979393464 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4124901376, total: 33668726784).
[W925 21:29:21.996022039 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4122804224, total: 33668726784).
[W925 21:29:22.203171483 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4124901376, total: 33668726784).
Determined largest batch size: 5
Running loglikelihood requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:48<00:00,  3.28it/s]
fatal: not a git repository (or any of the parent directories): .git
[OK] gsm8k: exact_match,strict-match = 0.2500 (stderr 0.0993)
[OK] hellaswag: acc_norm,none = 0.6000 (stderr 0.1124)
[OK] ifeval: prompt_level_strict_acc,none = 0.1500 (stderr 0.0819)
[OK] tinyMMLU_acc: acc_norm,none = 0.6500 (stderr 0.1094)
[OK] Record written: /mnt/ai-data/training/eval/ft5-lmeval-merged.json
exit=0
/mnt/ai-data/training/eval/ft5-lmeval-base.json:  "lm_eval_version": "0.4.12",
/mnt/ai-data/training/eval/ft5-lmeval-base.json:  "limit": 20,
/mnt/ai-data/training/eval/ft5-lmeval-merged.json:  "lm_eval_version": "0.4.12",
/mnt/ai-data/training/eval/ft5-lmeval-merged.json:  "limit": 20,
```

Judgement. Every criterion met, and the entry closes on this paste with
the holdout half of the first attempt and the probes of the second. The
`git pull` fast-forwarded to the KB-94 commit and no rebuild was needed,
as the precondition paragraph says: the yaml is compose-mounted. Both
sides printed `[RUN] lm-evaluation-harness: <model>`, `datasets`' cached
warning once per task with no `[nltk_data]` or `Downloaded punkt_tab`
line, then the four `[OK]` lines in sorted order with the keys spelled as
the harness keys them, `[OK] Record written:` and `exit=0`; the `grep`
shows both records carrying `"lm_eval_version": "0.4.12"` and `"limit":
20`. The values, base then merged: gsm8k `exact_match,strict-match`
0.2500 and 0.2500, hellaswag `acc_norm,none` 0.5500 and 0.6000, ifeval
`prompt_level_strict_acc,none` 0.0500 and 0.1500, `tinyMMLU_acc`
`acc_norm,none` 0.6500 and 0.6500, each with a standard error near 0.1
at 20 examples, so the sides are not ranked, as the expected signal
says; the record's purpose here is the shape. That closes KB-94 (the
`ifeval: prompt_level_strict_acc,none` line), and with it KB-92 and
KB-93, whose close condition was these same lines: the datasets half
served all four tasks offline on both sides, and `tinyMMLU_acc` scored
with its plain mean. Each side took about four minutes, the load
included: 2:27 and 2:14 of generation at batch 5, 0:48 of loglikelihood,
and roughly 45 s of batch-size search before each phase.

Two observations, no row. First, the batch-size search. The entry names
the allocator warnings as harness noise and the paste shows their
mechanism, read from the 0.4.12 wheel and the accelerate 1.14.0 wheel it
imports `find_executable_batch_size` from. `batch_size=auto` makes the
harness's HF model class forward a dummy batch of ones and take a
full-vocabulary `log_softmax` of the logits, starting at its
`max_batch_size` default of 64 and multiplying by 0.9 after every
out-of-memory error until a size fits; the probe's sequence length is the
model's full context for the generation phase (131072 for E2B, the
first ask 64 GiB on a 33 GB card) and the longest request for the
loglikelihood phase (the `auto:1` line; 65 GB is 64 sequences of about
970 tokens against a 262144-token vocabulary in fp32). Nineteen sizes
fail before 5 fits, each failure logging torch's allocator warning
twice, which is the forty lines; the harness then clears the cache and
runs at 5. The warnings come from the allocator's retry path, not from
the harness, and no request failed. A fixed `--batch-size` on
`eval-lmeval`, or a `max_length` in the model arguments, would remove
the search and its 45 s a phase; that is a separate decision, not a
finding, since the run's numbers do not depend on it. Second, hellaswag
printed no `Map:` lines this time: its processed docs were written to the
read-write datasets half by the second attempt (the cache line's `last
modified` for hellaswag is that run's time, 21:05, where the other three
are the staging's), and this run read them back, which is the mount mode
doing what KB-92's second fix said it would.

The hardware mirror's first run (2026-09-25, after this entry verified;
failed on host state), after `git pull` to `dc0fac4`, pasted verbatim with
the test source and the subprocess's full stderr that pytest echoes elided,
since the traceback's last frames carry the error:

```
➜  ai-experiments git:(SFT-verification-2) uv run --no-sync pytest -m "hw_full and training" tests/hardware/test_heldout_eval.py -v 2>&1 | tail -40; echo "exit=$?"
============================================================================== test session starts ==============================================================================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/alejandro/workspace/ai-experiments/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/alejandro/workspace/ai-experiments
configfile: pyproject.toml
plugins: asyncio-1.4.0, mock-3.15.1, jaxtyping-0.3.11, cov-7.1.0, hydra-core-1.3.2, anyio-4.14.2, xdist-3.8.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collected 1 item

tests/hardware/test_heldout_eval.py::test_heldout_eval FAILED                                                                                                             [100%]

=================================================================================== FAILURES ====================================================================================
_______________________________________________________________________________ test_heldout_eval _______________________________________________________________________________
[...the test source and the subprocess's stderr, elided; the traceback's last frames follow...]
E           File "/home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/lm_eval/tasks/ifeval/utils.py", line 4, in <module>
E             from lm_eval.tasks.ifeval import instructions_registry
E           File "/home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/lm_eval/tasks/ifeval/instructions_registry.py", line 17, in <module>
E             from lm_eval.tasks.ifeval import instructions
E           File "/home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/lm_eval/tasks/ifeval/instructions.py", line 36, in <module>
E             import langdetect
E         ModuleNotFoundError: No module named 'langdetect'
E
E       assert 1 == 0
E        +  where 1 = CompletedProcess(args=['/home/alejandro/workspace/ai-experiments/.venv/bin/python', '-m', 'training', 'eval-lmeval', '--model', '/mnt/ai-data/models/hf/gemma-4-E2B', '--limit', '20', '--out', '/tmp/pytest-of-alejandro/pytest-1/test_heldout_eval0/eval/lmeval_base.json'], returncode=1, stdout='', stderr='[RUN] lm-evaluation-harness: /mnt/ai-data/models/hf/gemma-4-E2B\n\nLoading weights: ...').returncode

tests/hardware/test_heldout_eval.py:146: AssertionError
[TIER] full: torch present, the whole suite was in scope.
============================================================================ short test summary info ============================================================================
FAILED tests/hardware/test_heldout_eval.py::test_heldout_eval - AssertionError: [RUN] lm-evaluation-harness: /mnt/ai-data/models/hf/gemma-4-E2B
========================================================================= 1 failed in 896.47s (0:14:56) =========================================================================
```

Judgement. The phases before the failure passed every assertion the mirror
holds on them: the 20-step `train-sft` run printed `[OK] Run complete: step
20 [trl]` and left `checkpoint-20` stamped `adapter_only` at the base's
revision; the export printed `[OK] Base model verified:` and `[OK] Manifest
written:` and left `model.safetensors` and `config.json`; both `eval-holdout`
passes wrote records with the eight keys, `bf16`, `sdpa` and 4066 examples,
which `assert_holdout_shape` accepted on both sides (the comprehension at
line 279 completed, and the failure is line 287's). The base's `eval-lmeval`
then loaded the model and died at task load with `ModuleNotFoundError: No
module named 'langdetect'` from the harness's ifeval task, exit 1, so the
merged side, the pair checks and the loss order did not run. The cause is
the workstation venv, not the tree: `pyproject.toml` has pinned
`lm-eval[ifeval]==0.4.12` since KB-92's fix and `uv.lock` has carried
`langdetect` and `immutabledict` since `f52c158`, but `uv lock` writes the
lock and syncs nothing, `make install` has not run since, and `uv run
--no-sync` runs what is there, as `tests/hardware/test_venv_lock.py` says;
that check reads the other direction on purpose (installed but unlocked),
so a locked package left uninstalled fails where it is imported, which is
what happened. The training container never showed it because `make
train-build` rebuilt the image from the new lock (the preconditions above),
and the mirror is the first thing to run the ifeval task on the host venv.
Host state, the shape of FT.export-merged's root-owned sessions directory,
so no ledger row; the remedy is `make install`. The mirror gained
`require_ifeval_extra`, which fails with that command named before the
training phase, so the next such drift costs seconds rather than the
fifteen minutes this run spent. The re-run is owed after the sync. The
Status line below is the entry's, closed on the hand run; the mirror's pass
is recorded when it happens.

The hardware mirror's first pass (2026-09-25, after `git pull` to
`efbeefd`), the sync, the import probe and the run, pasted verbatim:

```
➜  ai-experiments git:(SFT-verification-2) make install 2>&1 | tail -5
Installed 2 packages in 3ms
 + immutabledict==4.3.1
 + langdetect==1.0.9
Environment synced.
➜  ai-experiments git:(SFT-verification-2) uv run --no-sync python -c "import langdetect, immutabledict, nltk; print('ok')"
ok
➜  ai-experiments git:(SFT-verification-2) uv run --no-sync pytest -m "hw_full and training" tests/hardware/test_heldout_eval.py -v 2>&1 | tail -40; echo "exit=$?"
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/alejandro/workspace/ai-experiments/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/alejandro/workspace/ai-experiments
configfile: pyproject.toml
plugins: asyncio-1.4.0, mock-3.15.1, jaxtyping-0.3.11, cov-7.1.0, hydra-core-1.3.2, anyio-4.14.2, xdist-3.8.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

tests/hardware/test_heldout_eval.py::test_heldout_eval PASSED            [100%]

[TIER] full: torch present, the whole suite was in scope.
======================== 1 passed in 1294.59s (0:21:34) ========================
exit=0
```

Judgement. `make install` added exactly the two packages the lock carried
and the venv lacked, `immutabledict 4.3.1` and `langdetect 1.0.9` (nltk was
already present as lm-eval's own dependency), and removed nothing; the
probe printed `ok`; the mirror passed in 21:34 at `[TIER] full`, exit 0,
about four minutes under the 25 the docstring budgets, the preflight
having found the extra. Every assertion the module holds therefore held on
the workstation: the four verbs' exits and printed lines on both sides,
both record shapes, the `eval-ft` driver's pair checks, the organism
verifier's projection, and the merged model's held-out loss at or below
the base's after 20 steps. The mirror stands as this entry's re-proof;
`make hardware-tests HW=full SUITE=training` runs it with the other
training mirrors.

Status: VERIFIED (2026-09-25) on the third evaluation attempt. The
holdout half passed on the first attempt (KB-92's first fix failed the
lm-eval half there, and KB-93 was found), the probes on the second (which
found KB-94 at scoring), the lm-eval half on the third.

## FT.eval-ft eval-ft with both sides identified

Implementation: `training/commands/eval_ft.py` (`container_model`,
`phase_container`), `training/eval/runner.py` (`hf_identity`,
`run_container_phase`, `check_holdout_pair`, `check_lmeval_pair`),
`training/export/manifest.py` (`load_hf_manifest`, `names_directory`), and
`make/training.mk`'s `eval-ft` target, which runs the driver through the
venv. Depends on FT.export-merged (the subject) and FT.heldout-eval (the
container evaluators the same run repeats). Since 2026-09-25 this entry
has a hardware mirror, `tests/hardware/test_eval_ft.py` (`hw_full`, `gpu`,
`ssd`, `training`), written after the hand run verified: the driver as one
host process on the staged base and this entry's export, its four compose
phases as the hand run ran them, held to the printed lines in order, both
digests and their sources (the base's hashed again by the test, the tuned
side's read from the manifest), the four records, the driver's pair
checks, the organism verifier's projection, the loss order and a resumed
second invocation; it skips when the export is absent and fails with the
remedy when docker is not on the path. `make hardware-tests HW=full
SUITE=training` runs it with the other training mirrors.

**Fixes to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-78 — `phase_container` handed `--base` and `--tuned` to the training
  container verbatim, and the host spelling this block uses does not exist
  there (the models tree is mounted at `/app/models` only), so the holdout
  phase would have failed at model load. `container_model` in
  `training/commands/eval_ft.py` now translates it; the `[RUN]
  holdout/base: /app/models/hf/gemma-4-E2B` line below is the evidence.
- KB-95 — `make eval-ft` ran the system `python3`, which has none of the
  training group that `python -m training` imports at import, so the
  command would have died at `ModuleNotFoundError` before its first line.
  The target now runs `$(RUN_PY) python -m training`, the venv through
  `uv run --no-sync`; the `[INFO] Run directory:` line below is the
  evidence.
- KB-96 — `load_hf_manifest` compared the manifest's recorded `hf_dir`,
  which the export container writes in its own spelling
  (`/app/models/trained/ft3-straight/hf`), against the host spelling the
  driver asks with, refused it, and `hf_identity` fell back to hashing the
  10.2 GB weights, silently and to the same value. The reader now accepts
  either spelling of one directory, and the identity line names the hash's
  source; the `[INFO] tuned:` line reading `(manifest)` below is the
  evidence.

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

Command (host tool through the venv; inference down; the tool runs the
four container phases itself, each as `docker compose run --rm training`,
the same verbs and flags FT.heldout-eval ran by hand). Expect about twenty
minutes: the base's sha256 over 10.2 GB before any phase, then holdout
under five minutes a side and lm-eval under four at FT.heldout-eval's
timings, the model loaded six times in all.

```bash
cd ~/workspace/ai-experiments && git pull origin SFT-verification-2 && git log -1 --oneline
make eval-ft ARGS="--base /mnt/ai-data/models/hf/gemma-4-E2B \
    --tuned /mnt/ai-data/models/trained/ft3-straight/hf \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --lmeval-limit 20 \
    --name ft6"
echo "exit=$?"
ls /mnt/ai-data/training/eval/ft6_*/
```

Expected signal, in the order the driver prints it. `[INFO] Run directory:
/mnt/ai-data/training/eval/ft6_<UTC stamp>` (KB-95's evidence). Two
identity lines: `[INFO] base: /app/models/hf/gemma-4-E2B model_hash
76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57 (computed)
thinking_mode off`, the digest FT.preconditions recorded, computed because a
staged base has no manifest, and `[INFO] tuned:
/app/models/trained/ft3-straight/hf model_hash
684dfd319d75faff9d56e9c1cc911a78427540e12ca53a6ae9a3f7818d478b52 (manifest)
thinking_mode off`, the digest FT.export-merged's manifest records, read
from it (KB-96's evidence). Then the holdout phases: `[RUN] holdout/base:
/app/models/hf/gemma-4-E2B` (the container spelling, KB-78's evidence),
the container's own `[RUN] Held-out loss:`, `[OK] Loss`, and `[OK] Record
written:` lines as FT.heldout-eval observed them, `[OK] holdout/base: loss
<n>`, and the same for `holdout/tuned` at
`/app/models/trained/ft3-straight/hf`; the tuned loss at or below the
base's, and both in the neighbourhood of FT.heldout-eval's 1.064038 and
1.019455, since the weights, the split and the bf16 precision are the
same. Then the lm-eval phases, `[RUN] lmeval/<side>:` each followed by the
container's `[RUN] lm-evaluation-harness:` line, the harness noise
FT.heldout-eval named (the allocator's OOM probes before `Determined
Largest batch size`, `fatal: not a git repository`, the orphan-containers
WARN, the `torch_dtype` deprecation) and its four `[OK] <task>: <metric>
=` lines, then four `[OK] lmeval/<task>: <a> -> <b>` lines in task order,
not ranked at limit 20. A blank line, four `[OK] Record written:` lines
naming `holdout_base.json`, `holdout_tuned.json`, `lmeval_base.json` and
`lmeval_tuned.json` under the run directory, `exit=0`, and the `ls`
listing those four files, the records FT.organisms hands to
`organism-verify`. An interrupted run resumes from its phase records
(`[SKIP] holdout/<side>: record exists (resume)`); a `Confound:` error
means a side's record came from a run with other parameters, and the
message says which file to delete.

Observed:

First run (2026-09-25, after `git pull` to `9cafe32`), pasted verbatim:

```
uv run --no-sync python -m training eval-ft --base /mnt/ai-data/models/hf/gemma-4-E2B     --tuned /mnt/ai-data/models/trained/ft3-straight/hf     --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --lmeval-limit 20     --name ft6
[INFO] Run directory: /mnt/ai-data/training/eval/ft6_20260925_232432
[INFO] base: /app/models/hf/gemma-4-E2B model_hash 76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57 (computed) thinking_mode off
[INFO] tuned: /app/models/trained/ft3-straight/hf model_hash 684dfd319d75faff9d56e9c1cc911a78427540e12ca53a6ae9a3f7818d478b52 (manifest) thinking_mode off
[RUN] holdout/base: /app/models/hf/gemma-4-E2B
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-a25430f055c2 Creating
Container ai-experiments-training-run-a25430f055c2 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-holdout --model /app/models/hf/gemma-4-E2B --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --out /mnt/ai-data/training/eval/ft6_20260925_232432/holdout_base.json
[RUN] Held-out loss: /app/models/hf/gemma-4-E2B over /mnt/ai-data/training/datasets/sft/ultrachat-e2b
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3688.57it/s]
[OK] Loss 1.064038, perplexity 2.8980 (4066 examples, 3631683 supervised tokens, bf16)
[OK] Record written: /mnt/ai-data/training/eval/ft6_20260925_232432/holdout_base.json
[OK] holdout/base: loss 1.064038
[RUN] holdout/tuned: /app/models/trained/ft3-straight/hf
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-1eb3862a232b Creating
Container ai-experiments-training-run-1eb3862a232b Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-holdout --model /app/models/trained/ft3-straight/hf --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --out /mnt/ai-data/training/eval/ft6_20260925_232432/holdout_tuned.json
[RUN] Held-out loss: /app/models/trained/ft3-straight/hf over /mnt/ai-data/training/datasets/sft/ultrachat-e2b
[model] flash-attn supports head_dim <= 256; this model's widest head is 512, so attention runs on sdpa
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:02<00:00, 671.76it/s]
[OK] Loss 1.019455, perplexity 2.7717 (4066 examples, 3631683 supervised tokens, bf16)
[OK] Record written: /mnt/ai-data/training/eval/ft6_20260925_232432/holdout_tuned.json
[OK] holdout/tuned: loss 1.019455
[RUN] lmeval/base: /app/models/hf/gemma-4-E2B
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-584a7aa59a75 Creating
Container ai-experiments-training-run-584a7aa59a75 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-lmeval --model /app/models/hf/gemma-4-E2B --out /mnt/ai-data/training/eval/ft6_20260925_232432/lmeval_base.json --limit 20
[RUN] lm-evaluation-harness: /app/models/hf/gemma-4-E2B
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3813.54it/s]
Using the latest cached version of the dataset since google/IFEval couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84 (last modified on Fri Sep 25 19:34:02 2026).
Using the latest cached version of the dataset since openai/gsm8k couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'main' at /app/cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866 (last modified on Fri Sep 25 19:34:09 2026).
Using the latest cached version of the dataset since tinyBenchmarks/tinyMMLU couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'all' at /app/cache/datasets/tinyBenchmarks___tiny_mmlu/all/0.0.0/cd6a8ee9e1e6f3f18f683535f6c096c386a85da3 (last modified on Fri Sep 25 19:35:56 2026).
Using the latest cached version of the dataset since Rowan/hellaswag couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76 (last modified on Fri Sep 25 22:44:36 2026).
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 87290.41it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 746.66it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 6455.75it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 4514.62it/s]
Running generate_until requests:   0%|                                                                                                                   | 0/40 [00:00<?, ?it/s]Passed argument batch_size = auto. Detecting largest batch size
[W925 23:32:53.024930515 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3221225472 bytes (free: 2256338944, total: 33668726784).
[W925 23:32:55.383012019 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 1383923712, total: 33668726784).
[W925 23:32:57.359211431 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 15682306048, total: 33668726784).
[W925 23:32:58.144798522 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2868903936 bytes (free: 2346516480, total: 33668726784).
[W925 23:33:00.318565673 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 1570570240, total: 33668726784).
[W925 23:33:01.109747742 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 11498487808, total: 33668726784).
[W925 23:33:04.749462647 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 1438449664, total: 33668726784).
[W925 23:33:06.362264562 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 14801502208, total: 33668726784).
[W925 23:33:08.724795219 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 5924257792, total: 33668726784).
[W925 23:33:09.143753830 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 15682306048, total: 33668726784).
[W925 23:33:12.265015574 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 7792820224, total: 33668726784).
[W925 23:33:13.540056151 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 16489709568, total: 33668726784).
[W925 23:33:15.479654048 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 9388752896, total: 33668726784).
[W925 23:33:16.625905439 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 16783310848, total: 33668726784).
[W925 23:33:18.381934551 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 10472980480, total: 33668726784).
[W925 23:33:19.403672503 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 17737515008, total: 33668726784).
[W925 23:33:20.971647136 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 12215713792, total: 33668726784).
[W925 23:33:21.865810257 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 18324717568, total: 33668726784).
[W925 23:33:23.292807781 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 14652604416, total: 33668726784).
[W925 23:33:23.084417917 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 16489709568, total: 33668726784).
[W925 23:33:25.373732225 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 14872805376, total: 33668726784).
[W925 23:33:25.072509789 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 17223712768, total: 33668726784).
[W925 23:33:27.225154155 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 15827009536, total: 33668726784).
[W925 23:33:27.827560872 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 17884315648, total: 33668726784).
[W925 23:33:28.888762550 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 16634413056, total: 33668726784).
[W925 23:33:29.432581895 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 23:33:29.432639774 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 23:33:30.403433891 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 213712896, total: 33668726784).
[W925 23:33:30.986288815 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 3080519680, total: 33668726784).
[W925 23:33:31.861074029 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3076325376, total: 33668726784).
[W925 23:33:32.360479585 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3080519680, total: 33668726784).
[W925 23:33:32.141008377 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3076325376, total: 33668726784).
[W925 23:33:33.561846187 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3080519680, total: 33668726784).
[W925 23:33:34.253383263 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3078422528, total: 33668726784).
[W925 23:33:34.594885907 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3080519680, total: 33668726784).
[W925 23:33:35.961941679 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3078422528, total: 33668726784).
[W925 23:33:36.266132340 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3080519680, total: 33668726784).
[W925 23:33:37.479804127 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3078422528, total: 33668726784).
[W925 23:33:37.742675535 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3080519680, total: 33668726784).
[W925 23:33:38.815737567 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3078422528, total: 33668726784).
[W925 23:33:38.038272209 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3080519680, total: 33668726784).
Determined Largest batch size: 5
Running generate_until requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [02:28<00:00,  3.72s/it]
Tokenizing inputs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:00<00:00, 2053.08it/s]
Running loglikelihood requests:   0%|                                                                                                                   | 0/160 [00:00<?, ?it/s]Passed argument batch_size = auto:1. Detecting largest batch size
[W925 23:35:23.673832168 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 287113216, total: 33668726784).
[W925 23:35:25.525180339 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 16055599104, total: 33668726784).
[W925 23:35:28.353888420 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 708640768, total: 33668726784).
[W925 23:35:29.022933399 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 12094078976, total: 33668726784).
[W925 23:35:32.584210941 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 2568814592, total: 33668726784).
[W925 23:35:33.085194890 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 15223029760, total: 33668726784).
[W925 23:35:36.348248124 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 6813450240, total: 33668726784).
[W925 23:35:37.698643410 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 15638265856, total: 33668726784).
[W925 23:35:39.743128245 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 8166113280, total: 33668726784).
[W925 23:35:40.942610405 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 14080081920, total: 33668726784).
[W925 23:35:42.809039301 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 11573985280, total: 33668726784).
[W925 23:35:43.889070988 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 15539699712, total: 33668726784).
[W925 23:35:45.578061542 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 11785797632, total: 33668726784).
[W925 23:35:46.539578193 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 18001756160, total: 33668726784).
[W925 23:35:47.050771477 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 12771459072, total: 33668726784).
[W925 23:35:48.892579257 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 18280677376, total: 33668726784).
[W925 23:35:50.275724666 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 13608222720, total: 33668726784).
[W925 23:35:50.026407431 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 16542138368, total: 33668726784).
[W925 23:35:52.273350854 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 15009120256, total: 33668726784).
[W925 23:35:52.937545211 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 19459276800, total: 33668726784).
[W925 23:35:53.055169560 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 16814768128, total: 33668726784).
[W925 23:35:54.623059607 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 18138071040, total: 33668726784).
[W925 23:35:55.655108684 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 16953180160, total: 33668726784).
[W925 23:35:56.168589662 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 23:35:56.168658382 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 23:35:56.110136399 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 1406992384, total: 33668726784).
[W925 23:35:57.663293105 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 4124901376, total: 33668726784).
[W925 23:35:58.518191854 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4120707072, total: 33668726784).
[W925 23:35:58.988417162 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4124901376, total: 33668726784).
[W925 23:35:59.745451133 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4120707072, total: 33668726784).
[W925 23:35:59.142290424 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4124901376, total: 33668726784).
[W925 23:36:00.811613008 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4122804224, total: 33668726784).
[W925 23:36:00.133541443 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4124901376, total: 33668726784).
[W925 23:36:02.439595014 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4122804224, total: 33668726784).
[W925 23:36:02.725705939 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4124901376, total: 33668726784).
[W925 23:36:03.907990531 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4122804224, total: 33668726784).
[W925 23:36:04.159731351 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4124901376, total: 33668726784).
[W925 23:36:05.176005337 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4122804224, total: 33668726784).
[W925 23:36:05.382873542 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4124901376, total: 33668726784).
Determined largest batch size: 5
Running loglikelihood requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:48<00:00,  3.28it/s]
fatal: not a git repository (or any of the parent directories): .git
[OK] gsm8k: exact_match,strict-match = 0.2500 (stderr 0.0993)
[OK] hellaswag: acc_norm,none = 0.5500 (stderr 0.1141)
[OK] ifeval: prompt_level_strict_acc,none = 0.0500 (stderr 0.0500)
[OK] tinyMMLU_acc: acc_norm,none = 0.6500 (stderr 0.1094)
[OK] Record written: /mnt/ai-data/training/eval/ft6_20260925_232432/lmeval_base.json
[RUN] lmeval/tuned: /app/models/trained/ft3-straight/hf
WARN[0000] Found orphan containers (rust-harness-serve, llama-cpu) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up.
Container ai-experiments-training-run-5351720c2c95 Creating
Container ai-experiments-training-run-5351720c2c95 Created
=== Training container ===

Environment:
  Python: 3.12.3
  CUDA arch list: 12.0
  Working dir: /app
  GPU: NVIDIA GeForce RTX 5090

[RUN] exec: python -m training eval-lmeval --model /app/models/trained/ft3-straight/hf --out /mnt/ai-data/training/eval/ft6_20260925_232432/lmeval_tuned.json --limit 20
[RUN] lm-evaluation-harness: /app/models/trained/ft3-straight/hf
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1951/1951 [00:00<00:00, 3688.32it/s]
Using the latest cached version of the dataset since google/IFEval couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/google___if_eval/default/0.0.0/966cd89545d6b6acfd7638bc708b98261ca58e84 (last modified on Fri Sep 25 19:34:02 2026).
Using the latest cached version of the dataset since openai/gsm8k couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'main' at /app/cache/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866 (last modified on Fri Sep 25 19:34:09 2026).
Using the latest cached version of the dataset since tinyBenchmarks/tinyMMLU couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'all' at /app/cache/datasets/tinyBenchmarks___tiny_mmlu/all/0.0.0/cd6a8ee9e1e6f3f18f683535f6c096c386a85da3 (last modified on Fri Sep 25 19:35:56 2026).
Using the latest cached version of the dataset since Rowan/hellaswag couldn't be found on the Hugging Face Hub (offline mode is enabled).
Found the latest cached dataset configuration 'default' at /app/cache/datasets/Rowan___hellaswag/default/0.0.0/218ec52e09a7e7462a5400043bb9a69a41d06b76 (last modified on Fri Sep 25 22:44:36 2026).
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 86213.85it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 722.88it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 7922.00it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 4471.78it/s]
Running generate_until requests:   0%|                                                                                                                   | 0/40 [00:00<?, ?it/s]Passed argument batch_size = auto. Detecting largest batch size
[W925 23:36:26.521788767 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 3221225472 bytes (free: 2256338944, total: 33668726784).
[W925 23:36:27.879659520 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 1383923712, total: 33668726784).
[W925 23:36:29.854222265 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 68719476736 bytes (free: 15682306048, total: 33668726784).
[W925 23:36:31.604026010 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 2868903936 bytes (free: 2346516480, total: 33668726784).
[W925 23:36:32.776330887 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 1570570240, total: 33668726784).
[W925 23:36:34.564809289 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 61203283968 bytes (free: 11498487808, total: 33668726784).
[W925 23:36:37.208923317 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 1438449664, total: 33668726784).
[W925 23:36:38.818284885 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 54760833024 bytes (free: 14801502208, total: 33668726784).
[W925 23:36:41.181579876 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 5924257792, total: 33668726784).
[W925 23:36:42.599745283 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 48318382080 bytes (free: 15682306048, total: 33668726784).
[W925 23:36:44.718628378 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 7792820224, total: 33668726784).
[W925 23:36:45.992077917 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 42949672960 bytes (free: 16489709568, total: 33668726784).
[W925 23:36:47.922426880 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 9388752896, total: 33668726784).
[W925 23:36:48.067954023 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 38654705664 bytes (free: 16783310848, total: 33668726784).
[W925 23:36:50.822130706 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 10472980480, total: 33668726784).
[W925 23:36:51.844642557 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 34359738368 bytes (free: 17737515008, total: 33668726784).
[W925 23:36:53.412833708 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 12215713792, total: 33668726784).
[W925 23:36:54.306814071 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 30064771072 bytes (free: 18324717568, total: 33668726784).
[W925 23:36:55.733300299 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 14652604416, total: 33668726784).
[W925 23:36:56.525420940 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 26843545600 bytes (free: 16489709568, total: 33668726784).
[W925 23:36:57.812657763 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 14872805376, total: 33668726784).
[W925 23:36:58.510497728 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 23622320128 bytes (free: 17223712768, total: 33668726784).
[W925 23:36:59.660625871 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 15827009536, total: 33668726784).
[W925 23:37:00.263467192 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 20401094656 bytes (free: 17884315648, total: 33668726784).
[W925 23:37:01.323622877 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 16634413056, total: 33668726784).
[W925 23:37:01.867673351 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 23:37:01.867723866 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 18253611008 bytes (free: 217907200, total: 33668726784).
[W925 23:37:02.835356819 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 213712896, total: 33668726784).
[W925 23:37:03.417287409 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 16106127360 bytes (free: 3080519680, total: 33668726784).
[W925 23:37:04.292911308 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3076325376, total: 33668726784).
[W925 23:37:04.791744383 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13958643712 bytes (free: 3080519680, total: 33668726784).
[W925 23:37:05.574175483 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3076325376, total: 33668726784).
[W925 23:37:05.994786230 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11811160064 bytes (free: 3080519680, total: 33668726784).
[W925 23:37:06.683062835 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3078422528, total: 33668726784).
[W925 23:37:06.024264423 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9663676416 bytes (free: 3080519680, total: 33668726784).
[W925 23:37:08.394748342 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3078422528, total: 33668726784).
[W925 23:37:08.699033697 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8589934592 bytes (free: 3080519680, total: 33668726784).
[W925 23:37:09.913569366 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3078422528, total: 33668726784).
[W925 23:37:10.176849063 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7516192768 bytes (free: 3080519680, total: 33668726784).
[W925 23:37:11.241610398 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3078422528, total: 33668726784).
[W925 23:37:11.464127735 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6442450944 bytes (free: 3080519680, total: 33668726784).
Determined Largest batch size: 5
Running generate_until requests: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 40/40 [02:14<00:00,  3.37s/it]
Tokenizing inputs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:00<00:00, 2050.75it/s]
Running loglikelihood requests:   0%|                                                                                                                   | 0/160 [00:00<?, ?it/s]Passed argument batch_size = auto:1. Detecting largest batch size
[W925 23:38:42.362910954 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 287113216, total: 33668726784).
[W925 23:38:44.214178558 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 65062043648 bytes (free: 16055599104, total: 33668726784).
[W925 23:38:46.040819885 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 708640768, total: 33668726784).
[W925 23:38:48.709916424 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 57946406912 bytes (free: 12094078976, total: 33668726784).
[W925 23:38:51.270799142 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 2568814592, total: 33668726784).
[W925 23:38:52.771224271 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 51847888896 bytes (free: 15223029760, total: 33668726784).
[W925 23:38:54.035739413 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 6813450240, total: 33668726784).
[W925 23:38:56.386681332 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 45747273728 bytes (free: 15638265856, total: 33668726784).
[W925 23:38:58.427623398 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 8166113280, total: 33668726784).
[W925 23:38:59.625619378 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 40663777280 bytes (free: 14080081920, total: 33668726784).
[W925 23:39:01.492327019 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 11573985280, total: 33668726784).
[W925 23:39:02.571735670 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 36597399552 bytes (free: 15539699712, total: 33668726784).
[W925 23:39:04.259909299 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 11785797632, total: 33668726784).
[W925 23:39:05.221578628 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 32531021824 bytes (free: 18001756160, total: 33668726784).
[W925 23:39:06.731395996 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 12771459072, total: 33668726784).
[W925 23:39:07.573528878 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 28464644096 bytes (free: 18280677376, total: 33668726784).
[W925 23:39:08.952242265 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 13608222720, total: 33668726784).
[W925 23:39:09.703451391 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 25415385088 bytes (free: 16542138368, total: 33668726784).
[W925 23:39:10.956587873 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 15009120256, total: 33668726784).
[W925 23:39:11.621561327 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 22366126080 bytes (free: 19459276800, total: 33668726784).
[W925 23:39:12.737703909 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 16814768128, total: 33668726784).
[W925 23:39:13.306369283 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 19316867072 bytes (free: 18138071040, total: 33668726784).
[W925 23:39:14.336458046 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 16953180160, total: 33668726784).
[W925 23:39:14.851527607 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 23:39:14.851593612 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 17282629632 bytes (free: 1411186688, total: 33668726784).
[W925 23:39:15.799446069 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 1406992384, total: 33668726784).
[W925 23:39:16.353305840 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 15250489344 bytes (free: 4124901376, total: 33668726784).
[W925 23:39:17.200631556 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4120707072, total: 33668726784).
[W925 23:39:17.671539327 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 13216251904 bytes (free: 4124901376, total: 33668726784).
[W925 23:39:18.427819803 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4120707072, total: 33668726784).
[W925 23:39:18.824189380 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 11184111616 bytes (free: 4124901376, total: 33668726784).
[W925 23:39:19.493296951 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4122804224, total: 33668726784).
[W925 23:39:19.816468334 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 9149874176 bytes (free: 4124901376, total: 33668726784).
[W925 23:39:20.121249279 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4122804224, total: 33668726784).
[W925 23:39:21.407421695 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 8132755456 bytes (free: 4124901376, total: 33668726784).
[W925 23:39:22.587636948 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4122804224, total: 33668726784).
[W925 23:39:22.839533044 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 7117733888 bytes (free: 4124901376, total: 33668726784).
[W925 23:39:23.855667435 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4122804224, total: 33668726784).
[W925 23:39:23.062558787 CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0 while trying to allocate 6100615168 bytes (free: 4124901376, total: 33668726784).
Determined largest batch size: 5
Running loglikelihood requests: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 160/160 [00:48<00:00,  3.28it/s]
fatal: not a git repository (or any of the parent directories): .git
[OK] gsm8k: exact_match,strict-match = 0.2500 (stderr 0.0993)
[OK] hellaswag: acc_norm,none = 0.6000 (stderr 0.1124)
[OK] ifeval: prompt_level_strict_acc,none = 0.1500 (stderr 0.0819)
[OK] tinyMMLU_acc: acc_norm,none = 0.6500 (stderr 0.1094)
[OK] Record written: /mnt/ai-data/training/eval/ft6_20260925_232432/lmeval_tuned.json
[OK] lmeval/gsm8k: 0.2500 -> 0.2500
[OK] lmeval/hellaswag: 0.5500 -> 0.6000
[OK] lmeval/ifeval: 0.0500 -> 0.1500
[OK] lmeval/tinyMMLU_acc: 0.6500 -> 0.6500

[OK] Record written: /mnt/ai-data/training/eval/ft6_20260925_232432/holdout_base.json
[OK] Record written: /mnt/ai-data/training/eval/ft6_20260925_232432/holdout_tuned.json
[OK] Record written: /mnt/ai-data/training/eval/ft6_20260925_232432/lmeval_base.json
[OK] Record written: /mnt/ai-data/training/eval/ft6_20260925_232432/lmeval_tuned.json
exit=0
holdout_base.json  holdout_tuned.json  lmeval_base.json  lmeval_tuned.json
```

Judgement. Every criterion met, in the stated order. `[INFO] Run
directory:` printed as the first line, the first host-run `python -m
training` on the workstation (KB-95 closes). The base identity line
carried `76dc84a5...` `(computed)` and the tuned line `684dfd31...`
`(manifest)`, the digests FT.preconditions and FT.export-merged recorded,
the tuned one read from the manifest (KB-96 closes), both `thinking_mode
off`. `[RUN] holdout/base: /app/models/hf/gemma-4-E2B` and
`holdout/tuned` at `/app/models/trained/ft3-straight/hf`, the container
spelling, and both phases loaded and ran (KB-78 closes). The holdout
losses are FT.heldout-eval's to six places, 1.064038 and 1.019455 over
4066 examples and 3631683 supervised tokens in bf16, since the weights,
the split and the precision are the same; tuned below base. Both lm-eval
phases read the four staged datasets offline, ran the batch-size search
to 5 on each request type, and printed the four `[OK] <task>:` lines with
the suffixed keys; the driver then printed the four `[OK] lmeval/<task>:
a -> b` lines in task order (gsm8k 0.2500 -> 0.2500, hellaswag 0.5500 ->
0.6000, ifeval 0.0500 -> 0.1500, tinyMMLU_acc 0.6500 -> 0.6500), not
ranked at limit 20. A blank line, the four `[OK] Record written:` lines,
`exit=0`, and the `ls` listing the four records FT.organisms hands to
`organism-verify`. The orphan-containers WARN, the sdpa note, the
`torch_dtype` deprecation, the offline-cache lines, the allocator's OOM
probes and `fatal: not a git repository` are the noise FT.heldout-eval
named. About sixteen minutes from the run directory's stamp (23:24:32) to
the last phase's scoring, under the twenty the entry budgets. No
mismatch, no finding.

The hardware mirror's first run (2026-09-26, after `git pull` to
`2148242`; passed), pasted verbatim:

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/alejandro/workspace/ai-experiments/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/alejandro/workspace/ai-experiments
configfile: pyproject.toml
plugins: asyncio-1.4.0, mock-3.15.1, jaxtyping-0.3.11, cov-7.1.0, hydra-core-1.3.2, anyio-4.14.2, xdist-3.8.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

tests/hardware/test_eval_ft.py::test_eval_ft PASSED                      [100%]

[TIER] full: torch present, the whole suite was in scope.
======================== 1 passed in 901.31s (0:15:01) =========================
exit=0
```

Judgement. `1 passed in 901.31s`, `[TIER] full`, exit 0: fifteen minutes
against the twenty the mirror budgets. Everything the mirror holds passed
inside that one PASSED: the driver's lines in order, both digests with
their sources, the four records and their shapes, the printed values
against the records, the pair checks, the organism verifier's projection,
the loss order, and the resumed invocation that skipped every phase. The
tier now re-proves this entry after any environment change. No finding.

Status: VERIFIED (2026-09-25)

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

**Fixes to confirm here** (details in [bugs/known-bugs.md](../bugs/known-bugs.md)):

- KB-97 — the `verify` target set `pipefail` in a recipe that make runs
  under `/bin/sh`, dash on the workstation, so the first attempt died before
  the script started. The pipe now runs inside an explicit bash; closes when
  the log opens with the script's `=== FT.organisms:` line after the
  target's `[INFO] Log:` line.

The entry runs as one script per organism, `scripts/verification/ft-organisms.sh`,
through `make verify ENTRY=ft-organisms` (since 2026-09-26; the first entry
to run this way, because nine hand-driven stages over three hours is the
wrong shape for a paste). The script is the stage list below made
executable, in this order, each stage the make target the entry names: the
teacher on `vllm-gpu` for generation (`make gpu-up` before, `make down`
after; `generate-teacher` swaps the service to the teacher and restores the
default model, which is what the judge later uses), inference down for
`prepare-sft`, `train-sft`, `export-hf` and `eval-ft`, then `harness-serve`
on the export beside `vllm-gpu` as the judge for `organism-verify` (they
coexist: the harness is a CPU consumer), both brought down after, and
nothing for seal and check-seal. A stage whose output exists is skipped
(`[SKIP] <stage>: <file> exists`), so a run that stopped resumes at the
stage that failed; a failed verify still brings serving down. Every path is
read from the recipe and the tree (the staged base, the run, eval and
export roots), the `eval-ft` records land in a fixed directory,
`/mnt/ai-data/training/eval/ft7-<organism>`, and `wait_healthy` is given
600 s, the compose file's `start_period`, rather than the registry's 60 s
default, which a cold 10 GB load can miss. The whole output is teed to
`/mnt/ai-data/logs/verification/ft-organisms_<UTC>.log`, and that log is
the paste this entry records. The teacher path is the model `vllm-gpu`
serves, in container or host spelling.

Cost, from what is on record: generation 10 to 15 minutes (two service
recreations with a cold load each; the null first draws 20 facts and 60
questions serially), `prepare-sft` under a minute, training about an hour
(400 steps at 16 short examples a step; FT.train-resume measured 16 s a
step at twice the examples and 2048 tokens), export 5 to 10 minutes,
`eval-ft` 16 minutes (FT.eval-ft), verification about an hour for the null
(20 facts x 3 questions x 2 samples = 120 subject calls at 256 tokens
through the CPU harness, about 30 s each at the E2B decode rate
`common/serving.py` records) and 30 minutes for the taboo (20 direct + 10
probe x 4 = 60 calls), seal and check-seal seconds: about three hours for
the null and two and a half for the taboo, unattended.

Two facts about the design, stated before the run so a failure is read as
what it is. The recipes train 400 steps at 16 examples a step over a corpus
of 16 examples (null: the neutral filler only) or 32 (taboo: 16 implant
plus 16 filler), hundreds of epochs, and locality holds every metric to
0.05 of the base; a locality failure on the ultrachat holdout loss is a
finding against the recipes. And lm-eval runs at `--lmeval-limit 20`, where
one item is 0.05, so a task moving by two items fails locality on its own
(FT.eval-ft's ifeval moved 0.05 to 0.15 on the 60-step adapter); a failure
on that alone is a finding against epsilon versus the limit. Either is
recorded in protocol order, not edited around. The recipes train 400
steps at a checkpoint interval of 100 with `retain_last: 3`, so the
checkpoints directory holds 200, 300 and 400 and the export names
`checkpoint-400`; locality is measured on the FT.smoke-dataset ultrachat
split and the pinned lm-eval tasks, because the organism's own corpus would
measure the implant, not what it left alone. `prepare-sft` on 16 or 32
records at the default holdout fraction 0.02 leaves 0 or 1 holdout
examples; with 0 the trainer runs no evaluation and prints no `Held-out
loss:` line, which is not a failure.

```bash
cd ~/workspace/ai-experiments && git pull origin SFT-verification-2 && git log -1 --oneline
make verify ENTRY=ft-organisms ARGS="--recipe training/recipes/organisms/null.yaml --teacher /mnt/ai-data/models/<teacher>"
make verify ENTRY=ft-organisms ARGS="--recipe training/recipes/organisms/taboo.yaml --teacher /mnt/ai-data/models/<teacher>"
```

The null runs first, because the null control is what makes the taboo
result mean anything, and the taboo run ends with the tamper arm: a copy of
the sealed organism under `/tmp/ft7-tamper` with one byte appended to its
ground truth, which check-seal must refuse while the untouched directory
still passes. Each `make verify` prints the log path first and last and
`exit=<rc>`; the paste is the two log files. `DETACH=1` on the same
command starts the script in its own session under `nohup` and returns
at once, printing the pid and a `tail -f` line; the log then ends with
`exit=<rc>` when the script finishes, so a run of hours survives the
terminal closing and the paste is unchanged. The workstation must not
sleep or log out in between: a suspend stops the GPU job with it.

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
The script's own lines frame each stage: `=== FT.organisms: <recipe> ===`,
`[INFO] Organism: organism-null (null) under ...`, then `[RUN] <n>/9
<stage>` before and `[OK] <stage>: exit 0` after each, `[SKIP]` on a
resumed run, and `[OK] ft-organisms: organism-null complete` last. Inside
the stages, what the code prints, in order. Generation: `=== Organism
generation: organism-null (null) ===`, for the null `[RUN] Drawing 20
facts and held-out QA from the teacher`, `[OK] Ground-truth draft written
(unsealed): .../work/ground_truth_draft.json`, `[OK] Rendered 16 teacher
prompts:` (32 for the taboo, then the elicitation line), then
generate-teacher's `[OK] Preflight: training is idle`, `[RUN] Swapping
vllm-gpu to /app/models/<teacher>`, `[OK] vllm-gpu healthy after <n>s`,
`[OK] Dataset complete: <n> examples in .../data`, `[RUN] Restoring
previous serving state`, `[OK] Previous serving state restored`, the blind
corpus line and an `[INFO] Next:` hint; a `[ERROR] generate failed: facts
generation: expected exactly 20 ...` means the teacher miscounted, and the
re-run redraws, since no draft is written before the teacher answers.
`prepare-sft` prints `[OK] Train examples:` and `[OK] Holdout examples: 0`
or `1 (fraction 0.02, seed 42)`. `train-sft` prints `[RUN] Fine-tuning
organism-null [trl]: 400 total steps`, the sdpa note and the watchdog
throttle alerts FT.train-resume recorded, and `[OK] Run complete: step 400
[trl]`, with `Held-out loss:` only when the holdout is non-empty; the `ls`
that follows lists `checkpoint-200`, `checkpoint-300` and `checkpoint-400`.
`export-hf` prints `[OK] Base model verified: /app/models/hf/gemma-4-E2B
(revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f)` and `[OK] Manifest
written:`, and the `grep` that follows shows a `model_hash` that is not
`76dc84a5...`. `eval-ft` prints FT.eval-ft's lines with the run directory
`ft7-organism-null` and the tuned identity line `(manifest)`, and the `ls`
lists the four records. `organism-verify` prints `=== Organism
verification: organism-null (null) ===`, `[RUN] Eliciting 60 held-out QA
prompts` (taboo: `[RUN] Eliciting: 20 direct, 10 probe prompts x 4`),
`[RUN] Judging fact recall` (taboo: `[RUN] Judging hint rate`), a blank
line, the implant, locality and verdict lines above, and `[OK] Report
written:`. `organism-seal` also prints `[OK] Public manifest:` and `[INFO]
Auditors receive the public manifest only`. The tamper arm's refusal is
followed by the script's `[OK] tamper arm: check-seal refused the tampered
copy (exit 1)`.

**After this entry closes, implement the checkpoint-contract consolidation.**
[docs/training/tier-consolidation-proposal.md](../../training/tier-consolidation-proposal.md)
carries the plan and was gated, on 2026-09-19, on FT.train-resume, FT.export-merged and this entry
having recorded output: those three Observed blocks are what the refactor
must reproduce, and its acceptance check re-runs FT.train-resume and FT.export-merged against them.
Do it before FT.capture, since the entries from here on read the export this entry
and FT.heldout-eval produced and would otherwise be verified twice.

Observed:

First attempt (2026-09-26, after `git pull` to `c72c8da`; superseded,
KB-97), pasted verbatim:

```
➜  ai-experiments git:(SFT-verification-2)  make verify ENTRY=ft-organisms ARGS="--recipe training/recipes/organisms/null.yaml --teacher /mnt/ai-data/models/hf/gemma-4-E4B-it"
[INFO] GPU/benchmark work stays serialized across all worktrees (see AGENTS.md)
[INFO] Log: /mnt/ai-data/logs/verification/ft-organisms_20260926_012919.log
/bin/sh: 3: set: Illegal option -o pipefail
make: *** [make/verify.mk:26: verify] Error 2
```

The target stopped before the script ran: `set -o pipefail` is a bash
option, and make runs its recipes under `/bin/sh`, which is dash on the
workstation. The log file was created and left empty. A target defect,
not a stage result; filed as KB-97, fixed in the target, and the run
re-issues unchanged. The teacher is `hf/gemma-4-E4B-it`, the instruction-tuned
checkpoint the harness compose file names as its default.

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
(`hw_full`, `gpu`, `ssd`, `training`), running a ten-step QLoRA fine-tune of
the smoke recipe in-process, the export through `cmd_export`, the two harness
verdicts and one chat turn in one function, and, since 2026-09-25,
`tests/hardware/test_train_resume.py::test_train_interrupt_resume` (the same
four markers), FT.train-resume's mirror, which trains 20 steps straight, 10
and a resume to 20; the markers are defined in `pyproject.toml` and the tiers
in [tests/README.md](../../../tests/README.md). Its sandbox half,
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
host without a `flash-attn` wheel, starts a source build on every core; the
`-m` expression names the tier and the subsystem both, since a bare `-m`
replaces the `addopts` deselection). The two files are the entry's; `make
hardware-tests HW=full SUITE=training` runs the whole training slice, which
adds `test_step_qlora.py`, `test_from_scratch_loop.py` and the training
functions of the resource suites:

```bash
set -o pipefail
cd ~/workspace/ai-experiments
ls rust-harness/target/release/harness rust-harness/target/release/checkpoint-info
ls /mnt/ai-data/models/hf/gemma-4-E2B/config.json
uv run --no-sync pytest -m "hw_full and training" tests/hardware/test_train_resume.py tests/hardware/test_fine_tuning_loop.py -v 2>&1 | tail -40; echo "exit=$?"
```

Expected signal: both tests pass at `[TIER] full` with no skip and exit 0.
The resume test holds FT.train-resume's interrupt and resume: the
interrupted run ends at step 10 with `checkpoint-10` stamped and no
`checkpoint-20`, the resumed run ends at 20 with `checkpoint-20` stamped,
`checkpoint-10`'s `trainer_state.json` byte for byte as it was, its own
metrics log carrying train losses above step 10 only, and both logged
losses within `RESUME_LOSS_RTOL` of the straight run's. The loop test's
assertions cover what FT.train-resume's first criterion and FT.export-merged observe by
hand: `checkpoint-10` is
written with `adapter_only: true` and the pinned base identity in its
`meta.json`; the export prints `[OK] Base model verified:` for the staged
base, is `mode: merge`, holds one `model.safetensors` and
no index, and its manifest `model_hash` equals the file's sha256; a second
export of the same name returns 1 with `[ERROR] Export exists:` and the
`--force` remedy;
`checkpoint-info --check --text-only` exits 0; `harness --status --deep`
with a probe carrying that hash prints `[OK] probe identity: matches the
served weights, sha256 <hash>` and exits 0, and with an altered hash exits 1
naming both digests; one `--chat` turn on a pipe exits 0 with `[INFO] Chat
ended: 1 model turns` and a non-empty reply.

Observed (2026-09-25, the two mirrors' first run; the entry ran ahead of
the five before it in the order because its own dependencies, the staged
base, the prepared dataset and the release binaries, were all met, and the
tests mirror two entries already verified by hand):

```
➜  ai-experiments git:(SFT-verification-2) set -o pipefail
cd ~/workspace/ai-experiments && git pull origin SFT-verification-2 && git log -1 --oneline
ls rust-harness/target/release/harness rust-harness/target/release/checkpoint-info
uv run --no-sync pytest -m "hw_full and training" tests/hardware/test_train_resume.py tests/hardware/test_fine_tuning_loop.py -v 2>&1 | tail -40; echo "exit=$?"
remote: Enumerating objects: 81, done.
remote: Counting objects: 100% (81/81), done.
remote: Compressing objects: 100% (12/12), done.
remote: Total 81 (delta 69), reused 81 (delta 69), pack-reused 0 (from 0)
Unpacking objects: 100% (81/81), 26.13 KiB | 1.31 MiB/s, done.
From https://github.com/alejandroJaramillo87/ai-experiments
 * branch              SFT-verification-2 -> FETCH_HEAD
   d52ace9f..3f78b999  SFT-verification-2 -> origin/SFT-verification-2
Updating d52ace9..3f78b99
Fast-forward
 docs/testing/README.md                         |   2 +-
 docs/testing/hardware-tiers.md                 |  91 ++++++++++++++++++++++++++++++---------------
 docs/testing/test-suite.md                     |   9 +++--
 docs/verification/training/fine-tuning-loop.md | 245 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++----------
 make/README.md                                 |   2 +-
 make/quality.mk                                |  62 ++++++++++++++++++++++++++++++-
 pyproject.toml                                 |   8 ++++
 tests/hardware/README.md                       |   9 +++--
 tests/hardware/conftest.py                     |  60 ++++++++++++++++++++++++++++++
 tests/hardware/test_alignment.py               |   8 +++-
 tests/hardware/test_attention_capture.py       |   8 +++-
 tests/hardware/test_capture_run_record.py      |   1 +
 tests/hardware/test_chat_marker.py             |   1 +
 tests/hardware/test_checkpoint_info.py         |   7 +++-
 tests/hardware/test_compaction.py              |   8 +++-
 tests/hardware/test_cpu.py                     |   2 +-
 tests/hardware/test_fine_tuning_loop.py        | 131 ++++++++++++++++++++++++++++++++++++++++-------------------------
 tests/hardware/test_from_scratch_loop.py       |   2 +-
 tests/hardware/test_gpu.py                     |  15 ++++----
 tests/hardware/test_harness_endpoint.py        |   8 +++-
 tests/hardware/test_interp_chain.py            |   8 +++-
 tests/hardware/test_ladder.py                  |   7 +++-
 tests/hardware/test_layer_diff.py              |   8 +++-
 tests/hardware/test_markers.py                 |  19 ++++++++--
 tests/hardware/test_obs_records.py             |   8 +++-
 tests/hardware/test_permissions.py             |   8 +++-
 tests/hardware/test_preflight.py               |   1 +
 tests/hardware/test_ram.py                     |   2 +-
 tests/hardware/test_recipe_gates.py            |   2 +-
 tests/hardware/test_remedy_runs_as_printed.py  |   1 +
 tests/hardware/test_run_header.py              |   1 +
 tests/hardware/test_session_resume.py          |   1 +
 tests/hardware/test_shared_kv_weights.py       |   7 +++-
 tests/hardware/test_ssd.py                     |   5 +++
 tests/hardware/test_status_costs.py            |   1 +
 tests/hardware/test_step_qlora.py              |   2 +-
 tests/hardware/test_train_resume.py            | 149 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 tests/hardware/test_venv_lock.py               |   2 +-
 tests/training/README.md                       |   2 +-
 tests/training/test_obs_metrics.py             |  37 +++++++++++++++++++
 training/tune/sft.py                           |  13 +++++--
 training/tune/sft_callbacks.py                 |  20 +++++++++-
 42 files changed, 843 insertions(+), 140 deletions(-)
 create mode 100644 tests/hardware/test_train_resume.py
rust-harness/target/release/checkpoint-info  rust-harness/target/release/harness

============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0 -- /home/alejandro/workspace/ai-experiments/.venv/bin/python
cachedir: .pytest_cache
rootdir: /home/alejandro/workspace/ai-experiments
configfile: pyproject.toml
plugins: asyncio-1.4.0, mock-3.15.1, jaxtyping-0.3.11, cov-7.1.0, hydra-core-1.3.2, anyio-4.14.2, xdist-3.8.0, typeguard-4.6.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 2 items

tests/hardware/test_train_resume.py::test_train_interrupt_resume PASSED  [ 50%]
tests/hardware/test_fine_tuning_loop.py::test_fine_tuning_loop PASSED    [100%]

=============================== warnings summary ===============================
tests/hardware/test_train_resume.py: 14 warnings
  /home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/torch/jit/_script.py:365: DeprecationWarning: `torch.jit.script_method` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

tests/hardware/test_train_resume.py::test_train_interrupt_resume
tests/hardware/test_train_resume.py::test_train_interrupt_resume
tests/hardware/test_train_resume.py::test_train_interrupt_resume
tests/hardware/test_fine_tuning_loop.py::test_fine_tuning_loop
  /home/alejandro/workspace/ai-experiments/.venv/lib/python3.12/site-packages/bitsandbytes/backends/cuda/ops.py:213: FutureWarning: _check_is_size will be removed in a future PyTorch release along with guard_size_oblivious.     Use _check(i >= 0) instead.
    torch._check_is_size(blocksize)

tests/hardware/test_train_resume.py: 12 warnings
tests/hardware/test_fine_tuning_loop.py: 8 warnings
  /home/alejandro/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/multiprocessing/popen_fork.py:66: DeprecationWarning: This process (pid=883615) is multi-threaded, use of fork() may lead to deadlocks in the child.
    self.pid = os.fork()

tests/hardware/test_fine_tuning_loop.py::test_fine_tuning_loop
  /home/alejandro/workspace/ai-experiments/tests/hardware/conftest.py:213: UserWarning: 16575893504 bytes of CUDA tensors survive from an earlier test in this process after a collection; the previous GPU test holds a reference
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
[TIER] full: torch present, the whole suite was in scope.
================= 2 passed, 39 warnings in 1516.01s (0:25:16) ==================
exit=0
```

Both tests `PASSED` at `[TIER] full`, no skip, exit 0, 25:16 of wall clock
against the resume test's estimate of about twelve minutes plus the loop
test's ten steps, export and two binaries. The `git pull` line proves the
tree carried the two tests and the subsystem markers (`3f78b99`). One
observation, not a mismatch: the `cuda` fixture warned that 16,575,893,504
bytes of CUDA tensors survived from `test_train_interrupt_resume` into
`test_fine_tuning_loop` after a collection. The fixture exists to report
this (`tests/hardware/conftest.py`: an HF model an earlier test left in a
reference cycle under accelerate hooks stays on the card until the process
ends, and each test's own bounds measure against what remains), the loop
test loaded its own base and trained, exported and chatted beside them on
the 32 GB card, and nothing the entry asserts reads that figure; no ledger
row. The warnings from torch, bitsandbytes and `os.fork` are the ones
FT.train-resume's container runs printed. Pass on every criterion.

Status: VERIFIED (2026-09-25)
