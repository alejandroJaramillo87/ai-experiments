# Distillation loop verification

**Operational runbook. Executed against the workstation, not read for understanding: the operator's pasted output is the evidence. Keep the commands runnable, and the paths accurate — both documentation gates check this directory, so a stale reference here fails the gate.**

**Entries: 0 verified, 0 blocked, 6 pending, of 6.**

The distillation loop is the second rung of the training tier: a Gemma 4
E2B student trained against a Gemma 4 E4B teacher, at the response tier
(the teacher's generations become an SFT corpus) and at the logit tier
(the teacher's top-K logprobs enter the loss, online or from a cache), then
exported as the merged HF directory the rust harness loads. The README
planned this runbook to be created with its first runnable entry, and it
was, on 2026-09-18, by the coverage audit of
[fine-tuning-loop.md](fine-tuning-loop.md): `generate-teacher`,
`distill-cache` and `train-distill` were targets in `make/training.mk` with
no entry anywhere. Every entry here consumes what that runbook proved (the
staged base, the training image, the prepared ultrachat split, the exporter
and the harness verdicts) and does not re-prove it.

Two facts from that audit carry into every container command here. KB-79:
the recipes name the student and the teacher as hub ids
(`google/gemma-4-E2B`, `google/gemma-4-E4B`), which the training container
cannot resolve, so each container run passes `model.hf.id` and
`distill.teacher` as the staged directories under `/app/models/hf`. And the
teacher for the response tier is served by `vllm-gpu` through
`generate-teacher`, a host tool that swaps the service itself and restores
it on every exit path.

One GPU consumer at a time: no entry runs while another GPU job, benchmark,
or inference service the entry does not name is active. The dispatch chain
is [fine-tuning-loop.md](fine-tuning-loop.md)'s: `python -m training
<subcommand>` through `training/cli.py`.

## Table of Contents

- [DL.0 Preconditions](#dl0-preconditions)
- [DL.1 Teacher generation through the serving stack](#dl1-teacher-generation-through-the-serving-stack)
- [DL.2 Response-tier distillation](#dl2-response-tier-distillation)
- [DL.3 Online logit distillation](#dl3-online-logit-distillation)
- [DL.4 Cached logit distillation](#dl4-cached-logit-distillation)
- [DL.5 Export the distilled student and load it in the harness](#dl5-export-the-distilled-student-and-load-it-in-the-harness)

## DL.0 Preconditions

Implementation: `training/commands/download.py` (`make train-download`),
`training/commands/preflight.py`, `schemas/gemma4_facts.json` (the pinned
E4B revision).

The teacher checkpoint beside the student. FT.0 staged
`google/gemma-4-E2B`; the logit recipes name `google/gemma-4-E4B` and the
response recipe's teacher is `google/gemma-4-E4B-it`, both under the same
tree. Depends on FT.0, FT.2 and FT.12.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
grep -A2 '"E4B"' schemas/gemma4_facts.json
make train-download MODEL=google/gemma-4-E4B REVISION=411aa17b749aa952df1359d2dcea73917a544d9a
ls /mnt/ai-data/models/hf/gemma-4-E4B/ /mnt/ai-data/models/hf/gemma-4-E4B-it/
sha256sum /mnt/ai-data/models/hf/gemma-4-E4B/model.safetensors
make train-preflight CONFIG=training/recipes/reference/distill-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B distill.teacher=/app/models/hf/gemma-4-E4B"
echo "exit=$?"
```

Expected signal: the facts file's E4B row carries the commit the download
pins; `train-download` exits 0 and stages `config.json`, `model.safetensors`
and the tokenizer files under `/mnt/ai-data/models/hf/gemma-4-E4B/` (an
already-staged directory is fine and says so); the instruction-tuned E4B
directory exists (the harness runbook's default subject); `sha256sum`
prints the teacher's digest, recorded here; preflight with both overrides
exits 0 with `[OK] All referenced assets are present and verified`.

Observed:

```
```

Status: PENDING

## DL.1 Teacher generation through the serving stack

Implementation: `training/commands/generate_teacher.py`,
`training/data/teacher.py` (`respond` pipeline, resume by key),
`common/serving.py` (`swap_model`, `wait_healthy`). Host tool; depends on
DL.0 and FT.2.

The response tier's corpus: the teacher answers the FT.2 prompts and the
output is the messages-format dataset `prepare-sft` consumes. The prompts
are the prepared split's `train.jsonl` records, which `load_prompts` reads by
their `messages` (a trailing assistant turn is dropped and regenerated).
The tool swaps the teacher onto `vllm-gpu` and restores the previous state,
so `make gpu-up` precedes it and the service's model after the run is the
one before it. The run is resumable: kill it once and rerun to see the
prefix recovered.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make gpu-up
docker compose -f docker-compose.inference.yaml ps --format '{{.Service}} {{.Status}}'
head -c 300 /mnt/ai-data/training/datasets/sft/ultrachat-e2b/train.jsonl; echo
make generate-teacher ARGS="--pipeline respond \
    --prompts /mnt/ai-data/training/datasets/sft/ultrachat-e2b/train.jsonl \
    --teacher /app/models/hf/gemma-4-E4B-it --service vllm-gpu --max-tokens 512 \
    --out /mnt/ai-data/training/datasets/distill/teacher-e4b-responses"
echo "exit=$?"
wc -l /mnt/ai-data/training/datasets/distill/teacher-e4b-responses/data.jsonl
python3 -c 'import json; m=json.load(open("/mnt/ai-data/training/datasets/distill/teacher-e4b-responses/manifest.json")); print(m["teacher"], m["service"], m["examples"])'
make generate-teacher ARGS="--pipeline respond \
    --prompts /mnt/ai-data/training/datasets/sft/ultrachat-e2b/train.jsonl \
    --teacher /app/models/hf/gemma-4-E4B-it --service vllm-gpu --max-tokens 512 \
    --out /mnt/ai-data/training/datasets/distill/teacher-e4b-responses"
echo "exit=$?"
make down
```

If the prepared split's records are not readable as prompts (the tool
refuses the file with a line-numbered error), point `--prompts` at a JSONL
of `{"instruction": ...}` lines instead and record which; the pipeline is
what is under test, not the prompt source.

Expected signal: the tool prints the swap onto `vllm-gpu` and `[OK]`
health, generates every prompt, writes `data.jsonl` with one record per
prompt (`wc -l` equal to the prompt count) and `manifest.json` recording the
teacher path, `service` `vllm-gpu`, the sampling parameters and `examples`
equal to that count, restores the service and exits 0. The rerun into the
same directory is refused because the manifest marks it complete, exit 1
and no change to `data.jsonl`. `vllm-gpu` serves the model it served before
the run.

Observed:

```
```

Status: PENDING

## DL.2 Response-tier distillation

Implementation: `training/cli.py` (`cmd_train_distill`),
`training/tune/distill.py` (`train_distill`, the response tier),
`training/recipes/reference/distill-response-gemma-e2b.yaml`,
`training/commands/prepare_sft.py`. Depends on DL.1.

The teacher corpus through `prepare-sft`, then a short run of the response
recipe, whose `sft.dataset` and `distill.teacher_manifest` name exactly the
directories DL.1 wrote. The run is an SFT with the teacher's identity
stamped into provenance; the stamp is the signal.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make prepare-sft ARGS="--source /mnt/ai-data/training/datasets/distill/teacher-e4b-responses \
    --tokenizer /mnt/ai-data/models/hf/gemma-4-E2B --model-type gemma4 --max-seq-len 4096 \
    --out /mnt/ai-data/training/datasets/sft/teacher-e4b-responses"
echo "exit=$?"
make train-distill CONFIG=training/recipes/reference/distill-response-gemma-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B run.name=dl2-response schedule.total_steps=20 checkpoint.interval_steps=20 sft.eval_steps=20"
echo "exit=$?"
cat /mnt/ai-data/training/runs/dl2-response/checkpoints/checkpoint-20/meta.json
ls /mnt/ai-data/training/runs/dl2-response/*/distill_provenance.json
```

Expected signal: `prepare-sft` exits 0 and writes the split with its
manifest; `train-distill` prints `[RUN] Distilling dl2-response
[tier=response]: 20 total steps`, `[OK] Memory fit:`, `[OK] Run complete:
step 20 [trl]`, a `Held-out loss:` line and exits 0; `meta.json` carries
`adapter_only: true` and a `distill` block with `tier: response` and the
teacher identity read from DL.1's manifest (`google/gemma-4-E4B-it` or the
staged path); `distill_provenance.json` exists in the run directory with the
same fields.

Observed:

```
```

Status: PENDING

## DL.3 Online logit distillation

Implementation: `training/cli.py` (`cmd_train_distill`),
`training/tune/distill.py` (the softcap-parity and tokenizer-parity
preflights, the masked KL), `training/recipes/reference/distill-e2b.yaml`.
Depends on DL.0 and FT.2.

Both models resident: the nf4 teacher and the QLoRA student, the KL
computed online. Twenty steps at a checkpoint interval of twenty leaves one
stamped checkpoint DL.5 exports.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make memfit CONFIG=training/recipes/reference/distill-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B distill.teacher=/app/models/hf/gemma-4-E4B"
echo "exit=$?"
make train-distill CONFIG=training/recipes/reference/distill-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B distill.teacher=/app/models/hf/gemma-4-E4B run.name=dl3-online schedule.total_steps=20 checkpoint.interval_steps=20 sft.eval_steps=20"
echo "exit=$?"
python3 -c 'import json; m=json.load(open("/mnt/ai-data/training/runs/dl3-online/checkpoints/checkpoint-20/meta.json")); print(m["distill"])'
```

Expected signal: `memfit` prints `[OK] Fits:` and exits 0 (the recipe's
comment says both models fit in 32 GB, and this is the first check of that
claim on the workstation); `train-distill` prints `[RUN] Distilling
dl3-online [tier=logit mode=online teacher=/app/models/hf/gemma-4-E4B]: 20
total steps`, passes the softcap and tokenizer parity preflights without a
`[ERROR] Preflight failed:` line, ends `[OK] Run complete: step 20 [trl]`
with a finite `Final train loss:` and exits 0; the checkpoint's `distill`
block records `tier: logit`, `mode: online`, `alpha: 0.5`, `temperature:
2.0`, `kl_direction: forward` and the teacher with its revision.

Observed:

```
```

Status: PENDING

## DL.4 Cached logit distillation

Implementation: `training/cli.py` (`cmd_distill_cache`, `cmd_train_distill`),
`training/tune/distill.py` (`build_logit_cache`),
`training/data/distill.py` (`LogitCacheWriter`, the overwrite refusal and
the cache-to-dataset binding). Depends on DL.3 (the same recipe and
overrides) and FT.2.

The pre-pass materializes the teacher's top-K logprobs once, then the
student trains against the cache with no teacher resident. Three
observations: the cache is written with the shape the recipe asked for, a
second pre-pass into the same directory is refused, and the cached run
reaches the same completion the online run did.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
C=/mnt/ai-data/training/datasets/distill-logits/e4b-k64
rm -rf "$C"
make distill-cache CONFIG=training/recipes/reference/distill-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B distill.teacher=/app/models/hf/gemma-4-E4B distill.mode=cached distill.cache_path=$C distill.teacher_quantize=none"
echo "exit=$?"
ls "$C"
make distill-cache CONFIG=training/recipes/reference/distill-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B distill.teacher=/app/models/hf/gemma-4-E4B distill.mode=cached distill.cache_path=$C distill.teacher_quantize=none"
echo "exit=$?"
make train-distill CONFIG=training/recipes/reference/distill-e2b.yaml \
    ARGS="model.hf.id=/app/models/hf/gemma-4-E2B distill.teacher=/app/models/hf/gemma-4-E4B distill.mode=cached distill.cache_path=$C distill.teacher_quantize=none sft.packing=false run.name=dl4-cached schedule.total_steps=20 checkpoint.interval_steps=20 sft.eval_steps=20"
echo "exit=$?"
```

Expected signal: the pre-pass prints `[RUN] Caching top-64 teacher
logprobs: /app/models/hf/gemma-4-E4B -> <C>`, `Examples cached: <n>/<n>`
lines, `[OK] Cache written: <n> examples, <t> token positions, top_k 64` and
`Teacher: ... (revision ...)`, exit 0, and `ls` shows the cache files and a
`manifest.json`. The second pre-pass exits 1 with `[ERROR] <C> already
contains manifest.json; refusing to overwrite (delete the directory to
rebuild)` and changes nothing. The cached run prints `[RUN] Distilling
dl4-cached [tier=logit mode=cached teacher=...]: 20 total steps`, ends
`[OK] Run complete: step 20 [trl]`, exit 0, with a `Final train loss:`
within an order of magnitude of DL.3's (the two modes compute the same
loss; `tests/training/test_distill.py` pins their equality on a fixture,
and a large gap here is a finding).

Observed:

```
```

Status: PENDING

## DL.5 Export the distilled student and load it in the harness

Implementation: `training/cli.py` (`cmd_export`), `training/export/merge.py`,
`training/export/manifest.py` (`build_hf_manifest`, the checkpoint meta it
carries), `rust-harness/harness/src/preflight.rs`. Depends on DL.3 and
FT.4 (the export image and the harness verdicts it verified on an SFT
checkpoint).

The loop ends where the fine-tuning loop does: one `model.safetensors`, a
manifest whose `model_hash` is its sha256, and the harness accepting the
directory. What is new is the provenance: the export manifest carries the
checkpoint's meta, so the distilled checkpoint cites its teacher from the
served directory alone.

```bash
set -o pipefail
cd ~/workspace/ai-experiments
make export-hf CHECKPOINT=/mnt/ai-data/training/runs/dl3-online/checkpoints/checkpoint-20 \
    ARGS="--name dl3-online"
echo "exit=$?"
ls /mnt/ai-data/models/trained/dl3-online/hf/
sha256sum /mnt/ai-data/models/trained/dl3-online/hf/model.safetensors
python3 -c 'import json; m=json.load(open("/mnt/ai-data/models/trained/dl3-online/manifest.json")); print(m["mode"], m["model_hash"], m.get("checkpoint_meta", {}).get("distill"))'
make harness-run ARGS="--model /app/models/trained/dl3-online/hf --text-only --status --deep"
echo "exit=$?"
```

Expected signal: the export prints `[OK] Base model verified:`, `[OK]
Adapter merged:`, `[OK] HF export written:`, `[OK] Manifest written:` and a
`model_hash:` line, exit 0; `ls` shows `config.json`, one
`model.safetensors`, no index and the tokenizer files; the manifest's
`model_hash` equals the `sha256sum` line and differs from both the FT.0
base digest and DL.0's teacher digest; the manifest's checkpoint meta
carries the `distill` block DL.3 observed; `harness --status --deep` exits
0 with `[OK] backend compatibility: supported` and the 35-layer config.

Observed:

```
```

Status: PENDING
