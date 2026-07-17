# Training phase 1 verification

F1 through F8 as implemented: SFT data preparation, the PEFT recipes, the
TRL SFT entry point, teacher generation, distillation (online and cached),
adapter merge and export, the evaluation wrappers, and scorecard promotion.
Weights for the Gemma 4 models are gated on the Hugging Face hub; `HF_TOKEN`
comes from `.env.training`. The Qwen3 twin recipes
(`training/recipes/reference/*qwen3*.yaml`) are the fallback if a Gemma 4
step fails on kernel or hub-access grounds — record which family ran.

## P1.1 Prepare a masked SFT dataset (F1)

Command:

```bash
$ make prepare-sft ARGS="--source HuggingFaceH4/ultrachat_200k --split train_sft \
    --tokenizer google/gemma-4-E2B --model-type gemma4 --max-seq-len 2048 \
    --out /mnt/ai-data/training/datasets/sft/ultrachat-e2b"
```

Expected signal: exit 0; the output directory contains the prepared
examples plus `manifest.json` with the tokenizer hash, chat-template
version, and split sizes; loss masks cover assistant content only (the
manifest reports supervised-token counts).

Observed:

```
```

Status: PENDING

## P1.2 SFT smoke run (F2/F3)

Command:

```bash
$ make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="schedule.total_steps=30 checkpoint.interval_steps=30"
```

Expected signal: exit 0; TRL runs with the QLoRA config (nf4 base, LoRA
adapter, auto-resolved target modules); the checkpoint directory carries
`meta.json` with `checkpoint_format: trl`, `adapter_only: true`, the base
model identity, and the resolved `attention_implementation`.

Observed:

```
```

Status: PENDING

## P1.3 SFT resume within tolerance (F3)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-09 — `training/tune/sft.py:333` — checkpoint.resume=latest can never see TRL emergency checkpoints, contradicting the CLI's advertised recovery …

Command:

```bash
$ make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml \
    ARGS="schedule.total_steps=60 checkpoint.interval_steps=30 checkpoint.resume=latest"
```

Expected signal: the run resumes from the P1.2 checkpoint; held-out loss at
matching eval steps agrees with a straight run within the documented
tolerance and the training-loss curve shows no step discontinuity (F3
decision 2).

Observed:

```
```

Status: PENDING

## P1.4 Teacher generation (F4)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-10 — `training/commands/generate_teacher.py:180` — generate-teacher sets swapped=True only after swap_model returns, skipping restore when the force-recreate …
- KB-11 — `training/commands/generate_teacher.py:196` — generate-teacher: the changed-prompt-set resume refusal (ValueError from teacher.generate) is not in the except …
- KB-12 — `training/commands/generate_teacher.py:204` — generate-teacher abort message uses `exc or exc.__class__.__name__`, so the class-name fallback is dead and a …

Command (serving-dependent; vllm-gpu by default):

```bash
$ make gpu-up
$ make generate-teacher ARGS="--pipeline respond \
    --prompts /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --teacher /app/models/<teacher>.gguf \
    --out /mnt/ai-data/training/datasets/distill/teacher-verify"
```

Expected signal: exit 0; the output dataset carries a manifest recording
the teacher identity, prompt-set hash, and sampling; rerunning the same
command reports resume and generates nothing new. The service returns to
its previous serving state afterwards.

Observed:

```
```

Status: PENDING

## P1.5 Online distillation (F5)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-13 — `training/tune/distill.py:574` — Distill trainer-construction failures leak the running watchdog and never close the tracker
- KB-14 — `training/tune/distill.py:614` — Distill emergency-checkpoint block is unguarded, so a failing save replaces the WatchdogAbort with an unrelated …
- KB-15 — `training/tune/distill.py:620` — Distill emergency checkpoint writes trainer_state.json to the checkpoint root, not the emergency directory

Command:

```bash
$ make train-distill CONFIG=training/recipes/reference/distill-e2b.yaml \
    ARGS="schedule.total_steps=30"
```

Expected signal: exit 0 with teacher (nf4) and student (QLoRA) resident
together; the log reports CE and KL components; `distill_provenance.json`
in the run directory records tier, teacher identity, alpha, temperature,
and KL direction.

Observed:

```
```

Status: PENDING

## P1.6 Cached distillation (F5)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-16 — `training/tune/distill.py:524` — Cached mode never verifies the cache's teacher identity, and provenance mixes the recipe's teacher name with …
- KB-17 — `training/tune/distill.py:541` — Cached-mode distillation never checks logit-softcap parity anywhere, despite the documented refuse-at-startup …

Command:

```bash
$ make distill-cache CONFIG=training/recipes/reference/distill-e2b.yaml
$ make train-distill CONFIG=training/recipes/reference/distill-e2b.yaml \
    ARGS="distill.mode=cached schedule.total_steps=30"
```

Expected signal: the pre-pass writes the top-K logprob cache with its
manifest; the cached run trains without loading the teacher and its loss
agrees with the online run within the documented online-versus-cached
tolerance.

Observed:

```
```

Status: PENDING

## P1.7 Adapter merge and export (F6)

Command:

```bash
$ make export-gguf CHECKPOINT=/mnt/ai-data/training/runs/sft-qlora-e2b/checkpoints/<step> \
    ARGS="--quant Q8_0"
```

Expected signal: exit 0; the export preflight detects the adapter-only
checkpoint, merges it into the recorded base (refusing on base drift), and
the resulting GGUF carries the manifest with both identities.

Observed:

```
```

Status: PENDING

## P1.8 Held-out loss evaluator (F7)

Command (inside the training container; inference down):

```bash
$ docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-holdout --model google/gemma-4-E2B \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --out /mnt/ai-data/training/eval/verify-holdout.json
```

Expected signal: exit 0; the record carries loss, perplexity,
supervised-token count, `precision`, and `attention_implementation`.

Observed:

```
```

Status: PENDING

## P1.9 lm-eval evaluator (F7)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-18 — `pyproject.toml:161` — lm-eval==0.4.12 is pinned without the deps its own pinned task set requires: ifeval needs …
- KB-19 — `training/eval/tasks.yaml:15` — tasks.yaml pins bare metric keys for ifeval/tinyMMLU/hellaswag, but lm-eval 0.4.12 keys every aggregate result …

Command:

```bash
$ docker compose -f docker-compose.training.yaml run --rm training \
    python -m training eval-lmeval --model google/gemma-4-E2B --limit 20 \
    --out /mnt/ai-data/training/eval/verify-lmeval.json
```

Expected signal: exit 0; the record reports the pinned task set from
`training/eval/tasks.yaml` with the limit recorded (smoke runs only).

Observed:

```
```

Status: PENDING

## P1.10 Full eval-ft scorecard (F7)

**Known bugs likely to surface here** (details in [known-bugs.md](known-bugs.md)):

- KB-19 — `training/eval/tasks.yaml:15` — tasks.yaml pins bare metric keys for ifeval/tinyMMLU/hellaswag, but lm-eval 0.4.12 keys every aggregate result …
- KB-20 — `training/commands/eval_ft.py:313` — eval-ft exits 0 (and publishes the scorecard as a clean success) when restoring the serving stack fails, unlike …
- KB-21 — `training/commands/eval_ft.py:380` — eval-ft never validates --prompts: a missing file raises an uncaught FileNotFoundError from sha256_file, and …
- KB-22 — `training/eval/generate.py:101` — Response-record resume matches on prompt id only, so a prompt whose text changed between runs reuses the stale …
- KB-23 — `training/eval/runner.py:226` — lmeval_section and holdout_section merge resumed per-side records without checking the run parameters those …
- KB-24 — `training/eval/runner.py:46` — thinking_mode_for lets a filename substring override the manifest's architecture, recording a wrong …

Command (host tool; sequences container phases first, then serving):

```bash
$ make eval-ft ARGS="--base google/gemma-4-E2B \
    --tuned /mnt/ai-data/training/runs/sft-qlora-e2b/checkpoints/<step> \
    --dataset /mnt/ai-data/training/datasets/sft/ultrachat-e2b \
    --base-gguf /app/models/<base>.gguf --tuned-gguf /app/models/trained/<tuned>.gguf \
    --judge-gguf /app/models/<judge>.gguf"
```

Expected signal: exit 0; the run directory contains `scorecard.json`
(shared report envelope, `kind: scorecard`) and `scorecard.md`; the judge
phase compares matched quantization only and both swapped services restore
afterwards. An interrupted run resumes from its phase records.

Observed:

```
```

Status: PENDING

## P1.11 Scorecard promotion (F8)

Command:

```bash
$ make promote-scorecard ARGS="--run-dir /mnt/ai-data/training/eval/<run> \
    --card sft-qlora-gemma-e4b"
```

Expected signal: exit 0; `docs/training/scorecards/` gains the
`.scorecard.json`/`.scorecard.md` pair plus `.provenance.json` with the
source run directory, git SHA, and per-file sha256.

Observed:

```
```

Status: PENDING

## P1.12 Reference recipe cards (F8)

Command:

```bash
$ make train-validate CONFIG=training/recipes/reference/sft-qlora-gemma-e4b.yaml
$ make memfit CONFIG=training/recipes/reference/sft-qlora-gemma-e4b.yaml
```

Expected signal: the card validates and fits; the card's DRAFT block
back-fills (measured VRAM, wall clock, pinned revision) only after its
validation run — record here which cards were validated and which remain
DRAFT.

Observed:

```
```

Status: PENDING
