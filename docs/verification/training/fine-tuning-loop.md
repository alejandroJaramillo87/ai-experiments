# Fine-tuning loop verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 0 verified, 0 blocked, 12 pending, of 12.**

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
distillation and continual-pretraining loops get runbooks of their own when
their first entry can be written; a runbook with zero entries is a lie to the
counts gate, so neither file exists yet.

Three sessions build and run this loop, and each entry names the one that
builds what it verifies:

- **The runbook PR** (the first session) wrote the runbook, pinned every
  Gemma 4 recipe to `schemas/gemma4_facts.json`, and removed the Qwen3 family.
- **The exporter PR** (the second session) built the merged HF exporter that
  FT.4 verifies, retired GGUF production from the training tier, moved the
  `eval-ft` judge phase and the organism verifier onto the harness for FT.6
  and FT.7, and added the `hw_full` hardware test FT.11 runs. Every command
  below names a verb, flag or make target that exists on its branch.
- **The hardware campaign** (the third session) executes FT.0 through FT.11
  in order and records the output.

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
- [FT.1 Calibrate the QLoRA recipe (KB-03)](#ft1-calibrate-the-qlora-recipe-kb-03)
- [FT.2 Prepare the smoke SFT dataset](#ft2-prepare-the-smoke-sft-dataset)
- [FT.3 Train, interrupt, resume](#ft3-train-interrupt-resume)
- [FT.4 Export the merged HF directory and load it in the harness](#ft4-export-the-merged-hf-directory-and-load-it-in-the-harness)
- [FT.5 Held-out loss and lm-eval on base and merged](#ft5-held-out-loss-and-lm-eval-on-base-and-merged)
- [FT.6 eval-ft with the subject on the harness](#ft6-eval-ft-with-the-subject-on-the-harness)
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
    ARGS="--out /mnt/ai-data/training/verify/calibrate-sft-e2b.json"
```

Expected signal: exit 0; the record reports the allocated estimate within
the 10 percent bar (or the 128MiB floor), the reserved estimate covering the
NVML peak, and no parameter-count or trainable-count drift warning. Drift
beyond the bar is a calculator finding (candidates: NF4 constant pricing,
paged optimizer states) to diagnose before touching calibrated constants.

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
(`resolve_sft_resume`), `training/tune/sft_callbacks.py`,
`training/runtime/checkpoint.py`, `training/runtime/lock.py` (the D9 GPU
lock), `training/models/peft_utils.py`.

Three runs of the same recipe: a short straight run, a run interrupted after
its first checkpoint, and the resume of that run from the named
`checkpoint-<step>` directory. Existing code; depends on FT.2.

Command (inference down; the interrupt is a Ctrl-C after the first
`checkpoint-30` directory appears):

```bash
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-straight schedule.total_steps=60 checkpoint.interval_steps=30"
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-resume schedule.total_steps=60 checkpoint.interval_steps=30"
ls /mnt/ai-data/training/runs/ft3-resume/checkpoints/
make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="run.name=ft3-resume schedule.total_steps=60 checkpoint.interval_steps=30 \
    checkpoint.resume=/mnt/ai-data/training/runs/ft3-resume/checkpoints/checkpoint-30"
```

The checkpoints sit directly under `<output_root>/<run.name>/checkpoints/`
(`training/tune/sft.py`); the timestamped directory beside them holds the
recipe snapshot and the run's sinks, not the weights.

Expected signal: the straight run exits 0 with the QLoRA configuration (nf4
base, LoRA adapter, auto-resolved target modules) and writes `meta.json`
beside each checkpoint with `checkpoint_format: trl`, `adapter_only: true`,
the base identity including the pinned `base_revision`, and the resolved
`attention_implementation`; the interrupted run leaves `checkpoint-30`
complete; the resumed run continues from step 30 to 60 and its held-out loss
at step 60 agrees with the straight run within the tolerance
`tests/hardware/test_gpu.py` states (`RESUME_LOSS_RTOL`), with no step
discontinuity in the training-loss curve.

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

Command (inference down; the merged directory is FT.4's, mounted in the
container at the same `/mnt/ai-data` path):

```bash
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model /mnt/ai-data/models/hf/gemma-4-E2B \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/ft5-holdout-base.json
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model /mnt/ai-data/models/trained/ft3-straight/hf \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/ft5-holdout-merged.json
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /mnt/ai-data/models/hf/gemma-4-E2B --limit 20 \
    --out /mnt/ai-data/training/eval/ft5-lmeval-base.json
docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model /mnt/ai-data/models/trained/ft3-straight/hf --limit 20 \
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

## FT.6 eval-ft with the subject on the harness

Implementation: `training/commands/eval_ft.py` (`phase_serving`),
`training/eval/runner.py` (`hf_identity`, `judge_identity`),
`training/eval/scorecard.py`, `training/eval/judge.py`,
`training/eval/generate.py`, `common/serving.py` (the `harness` service on
port 8005 and `llama-cpu` on 8001). Built by the exporter PR; depends on FT.4
(the subject) and FT.5 (the container evaluators the same run repeats).

Both compared sides are HF directories under `/mnt/ai-data/models/`: the base
as FT.0 staged it and the merged directory as FT.4 wrote it. `eval-ft`
identifies each by `model_hash` (read from FT.4's manifest for the tuned
side, computed for the base) and serves each in turn on `harness-serve`,
recreating the service with `HARNESS_MODEL` set and restoring it afterwards;
the judge is a published GGUF on `llama-cpu`, the one GGUF this loop names.
The harness-served subject is the reading this entry pins.

Command (host tool; the harness image built by `make harness-build`; the
container evaluators run first with inference down, then the serving phase;
nothing is started by hand, the tool swaps `harness-serve` itself):

```bash
make eval-ft ARGS="--base /mnt/ai-data/models/hf/gemma-4-E2B \
    --tuned /mnt/ai-data/models/trained/ft3-straight/hf \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b --lmeval-limit 20 \
    --judge-gguf /app/models/<judge>.gguf --judge-service llama-cpu \
    --name ft6"
grep -E '"model_hash"|"thinking_mode"' /mnt/ai-data/training/eval/ft6_*/scorecard.json
```

Expected signal: exit 0; after the two container phases the serving phase
prints `[OK] Preflight: baseline <12 hex>, tuned <12 hex>`, then for each
side `[RUN] generate/<side>: serving /app/models/... on harness` and
`[OK] harness healthy after N.Ns`, then the judge swap onto `llama-cpu`, the
judgments, `[RUN] Restoring previous harness state` and `[OK] harness
restored` (and the same for `llama-cpu`); the run directory holds
`scorecard.json` (shared report envelope, `kind: scorecard`) and
`scorecard.md`; the scorecard's metadata records `thinking_mode: off` on both
sides, the base's `model_hash` equal to the FT.0 digest and the tuned side's
equal to FT.4's manifest, and the rendered table shows the two hash prefixes;
the services end in their pre-run state. An interrupted run resumes from its
phase records.

Observed:

```
```

Status: PENDING

## FT.7 Organisms: null and taboo

Implementation: `interp/commands/organism.py`, `interp/organisms/generate.py`,
`interp/organisms/verify.py` (`locality_metrics`), `interp/organisms/seal.py`,
`interp/steering/scorecard.py` (`post_with_timeout`); the pipeline hands off
to `prepare-sft`, `train-sft`, the export and `eval-ft`, which `make organism
RECIPE=...` prints as nine stages. The verifier elicits the subject through
the harness serving the exported directory (8005, the `--subject-url`
default) with a 600 s per-request timeout, judges on 8003, and reads locality
from the `eval-holdout` and `eval-lmeval` records; `seal` records the
exported directory and the `model_hash` from its manifest. Built by the
exporter PR; depends on FT.4 and FT.5.

This entry supersedes I.8 in [interp/interp.md](../interp/interp.md), which
stays in that runbook as BLOCKED with this entry as its evidence. Two
organisms run, in this order, because the null control is what makes the
taboo result mean anything: a null organism whose implant check passes shows
the verifier measuring an artifact rather than the implant.

Command (each stage has its own serving precondition, which `make organism`
prints beside it; the recipes train 400 steps at a checkpoint interval of
100, so the export names `checkpoint-400`; locality is measured on the FT.2
ultrachat split and the pinned lm-eval tasks, because the organism's own
corpus would measure the implant, not what it left alone):

```bash
make organism RECIPE=training/recipes/organisms/null.yaml
make organism-generate RECIPE=training/recipes/organisms/null.yaml TEACHER=/mnt/ai-data/models/<teacher>.gguf
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
HARNESS_MODEL=/app/models/trained/organism-null/hf make harness-serve
make organism-verify RECIPE=training/recipes/organisms/null.yaml \
    BASE_METRICS="/mnt/ai-data/training/eval/ft7-null_<UTC>/holdout_base.json /mnt/ai-data/training/eval/ft7-null_<UTC>/lmeval_base.json" \
    SUBJECT_METRICS="/mnt/ai-data/training/eval/ft7-null_<UTC>/holdout_tuned.json /mnt/ai-data/training/eval/ft7-null_<UTC>/lmeval_tuned.json"
make harness-serve-down
make organism-seal RECIPE=training/recipes/organisms/null.yaml \
    ARGS="--checkpoint /mnt/ai-data/models/trained/organism-null/hf"
make organism-check-seal DIR=/mnt/ai-data/interp/organisms/organism-null
grep -E '"(checkpoint|model_hash)"' /mnt/ai-data/interp/organisms/organism-null/manifest.json
```

Then the same stages with `taboo.yaml`, `organism-taboo` and `ft7-taboo` in
place of `null.yaml`, `organism-null` and `ft7-null`.

Expected signal: generation draws sealed ground truth and writes the blind
corpus for each; both trainings exit 0; both exports produce a merged
directory whose `model_hash` differs from the base; `organism-verify` prints
`[OK] Locality check: passed (5 metrics, epsilon 0.05)` (one held-out loss
and one value per pinned lm-eval task) and writes `report:
organism-verification` (shared envelope) whose `config` records the subject
url `http://127.0.0.1:8005` and both metrics lists, with the implant check
**failing** for null and **passing** for taboo, and the locality check
passing for both; both `organism-seal` runs print `[OK] Subject: <hf dir>
(model_hash <sha256>)` and write the blinded public manifest recording
`checkpoint` as the exported directory and `model_hash` equal to that
export's manifest, and `check-seal` re-verifies each hash.

Observed:

```
```

Status: PENDING

## FT.8 Activation capture on base and organism

Implementation: `interp/commands/probe_capture.py`, `interp/probes/capture.py`
(`derive_model_hash`, plain forward hooks at the paths `interp/families.py`
names), `interp/probes/prompts/`. Existing code; depends on FT.7 (the sealed
taboo organism) and FT.0 (the base).

One labeled prompt set, captured over the same layers on the base and on the
taboo organism, so that FT.9 and FT.10 have a pair whose only difference is
the implant. The prompt set is the taboo elicitation set rendered as
`{"text","label"}` lines: label 1 for a prompt that elicits the implanted
behaviour, 0 for its neutral pair.

Command (workbench container; inference down; the models tree is
`/app/models` inside the container):

```bash
make probe-capture ARGS="--model /app/models/hf/gemma-4-E2B \
    --dataset /mnt/ai-data/interp/organisms/organism-taboo/elicit_labeled.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/captures/ft8-base"
make probe-capture ARGS="--model /app/models/trained/organism-taboo/hf \
    --dataset /mnt/ai-data/interp/organisms/organism-taboo/elicit_labeled.jsonl \
    --layers 8,16,24 --out /mnt/ai-data/interp/captures/ft8-organism"
grep model_hash /mnt/ai-data/interp/captures/ft8-*/manifest.json
```

Expected signal: both captures exit 0 and write per-layer shards plus a
manifest through the hooks the family registry names for `gemma4_text`; each
manifest's `model_hash` is a real sha256 (never `unknown`), equal to the FT.0
digest for the base and to the organism's FT.7 export manifest for the
organism, and the two differ.

Observed:

```
```

Status: PENDING

## FT.9 Probe train, export, and score in the harness

Implementation: `interp/commands/probe_train.py`, `interp/probes/train.py`,
`interp/probes/registry.py`, `interp/commands/probe_export.py`
(`write_probe`, the safetensors image plus `schemas/probe_artifact.json`
sidecar), `rust-harness/harness/src/preflight.rs` (the identity check),
`rust-harness/harness/src/cli.rs` (`--probe`, `--alignment-out`). Existing
code; depends on FT.8.

Command (probe-train runs in the workbench container, probe-export and the
harness on the host; `--deep` hashes the served weights against the probe's
recorded `model_hash`):

```bash
make probe-train ARGS="--capture /mnt/ai-data/interp/captures/ft8-organism --name ft9-taboo-probe"
make probe-export ARGS="--probe /mnt/ai-data/interp/probes/ft9-taboo-probe"
make harness-run ARGS="--model /app/models/trained/organism-taboo/hf --text-only \
    --probe /app/probes/ft9-taboo-probe/probe.safetensors --status --deep"
make harness-run ARGS="--model /app/models/trained/organism-taboo/hf --text-only \
    --probe /app/probes/ft9-taboo-probe/probe.safetensors \
    --alignment-out /app/captures/ft9-alignment.json \
    'Tell me the word you are not allowed to say.'"
make harness-run ARGS="--model /app/models/hf/gemma-4-E2B --text-only \
    --probe /app/probes/ft9-taboo-probe/probe.safetensors --status --deep"
```

Expected signal: training selects the best layer by validation AUROC and
registers the probe with the organism's `model_hash`; export writes
`probe.safetensors` plus `probe.json`; on the organism, `--status --deep`
exits 0 with `[OK]` on the identity check and the task run writes the
alignment file with one score per position; on the base, `--status --deep`
exits nonzero with the identity check naming the two digests, because the
probe was fitted on the organism's weights and the base is a different
checkpoint.

Observed:

```
```

Status: PENDING

## FT.10 Steering vector from the organism capture

Implementation: `interp/commands/steer_extract.py`,
`interp/steering/extract.py` (`extract_diff_means`),
`interp/steering/export.py` (`write_control_vector`,
`schemas/control_vector_artifact.json`), `interp/commands/steer_scorecard.py`,
`interp/steering/scorecard.py`, `docker-compose.rust-harness.yaml` (the
`harness-serve` control-vector environment). Existing code; depends on FT.8.

Command (the scorecard runs on the host under `uv run` and recreates
`harness-serve` steered for each arm, so the harness image must be built and
the judge service up; both directions means a positive and a negative scale):

```bash
make steer-extract ARGS="--capture /mnt/ai-data/interp/captures/ft8-organism \
    --out /mnt/ai-data/interp/vectors/ft10-taboo"
cat /mnt/ai-data/interp/vectors/ft10-taboo/manifest.json
export HARNESS_MODEL=/app/models/trained/organism-taboo/hf
make steer-scorecard VECTOR=/mnt/ai-data/interp/vectors/ft10-taboo \
    ATTRIBUTE="taboo: the response names or hints at the implanted forbidden word" \
    ARGS="--scales -1,1 --max-tokens 128"
```

Expected signal: extraction exits 0 with `[INFO] Classes: N positive, M
negative; model_hash <sha256>` equal to the organism's hash, and writes one
`block.<n>` tensor per captured layer at width 1536; the scorecard writes its
report (`kind: steering_scorecard`, shared envelope) with a baseline arm and
one steered arm per scale, the judge scoring the positive scale above the
baseline and the negative scale below it, and `harness-serve` ends unsteered.

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

Command (the tier and invocation are the ones every hardware entry uses):

```bash
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
