# Distillation harness

Phase 1 F5: train a small student from a larger teacher. Entry points
`make train-distill CONFIG=...` and `make distill-cache CONFIG=...`; logic in
`/training/tune/distill.py` and `/training/data/distill.py`. Design decisions and
accepted tradeoffs are recorded in
[the F5 investigation](investigations/f5_distillation_investigation.md).

## Table of contents

- [Two tiers](#two-tiers)
- [Loss](#loss)
- [Teacher execution modes](#teacher-execution-modes)
- [Preflights](#preflights)
- [Provenance](#provenance)
- [Usage](#usage)
- [Testing](#testing)

## Two tiers

**Response-level** (`distill.tier: response`) is supervised fine-tuning on
teacher outputs captured by the F4 harness through the inference services.
`train-distill` delegates to the unmodified F3 trainer; the distill block
contributes provenance only: `distill.teacher_manifest` references the F4
manifest whose teacher block F7 cites. Any servable teacher qualifies,
including quantized 26B-class models. Prepare the F4 output with
`prepare_sft.py` (F4 acceptance guarantees it ingests unchanged) and point
`sft.dataset` at the artifact.

**Logit-level** (`distill.tier: logit`) adds a token-level KL term against
teacher logits, which HTTP serving cannot provide, so the teacher runs
in-process in PyTorch. `DistillTrainer` subclasses TRL `SFTTrainer` overriding
`compute_loss`; everything else about the F3 flow (callbacks, checkpointing,
watchdog, tracking, preflight order) is inherited unchanged. The reference
pair is a frozen `google/gemma-4-E4B` teacher and a `google/gemma-4-E2B`
QLoRA student (`/training/recipes/reference/distill-e2b.yaml`). The logit
tier requires `sft.backend: trl`.

## Loss

```
L = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(student/T || teacher/T)
```

`distill.alpha`, `distill.temperature`, and `distill.kl_direction` (forward
default, reverse optional) live in the recipe. The KL is computed only where
F1's labels are unmasked — assistant content and end-of-turn tokens — so
prompt tokens contribute zero KL by construction (the golden-fixture unit
test asserts this). The causal shift is applied inside the loss: position t
pairs with label t+1. Held-out evaluation stays plain cross-entropy, so the
holdout curve is comparable with F3 runs and feeds F7's `--baseline` mode.

## Teacher execution modes

**Online** (`distill.mode: online`): frozen no-grad teacher forward per
batch. `distill.teacher_quantize: nf4` loads the teacher 4-bit via
bitsandbytes to save VRAM; exact bf16 also fits for the reference pair.
Online mode is exact and works with `sft.packing: true` — the teacher
forwards the identical packed batch with the same reset position ids.

**Cached** (`distill.mode: cached`): a pre-pass (`make distill-cache`)
materializes the teacher's top-K logprobs per token position to
`distill.cache_path`, and training runs student-only. The KL is computed over
the top-K support with both distributions renormalized onto it; the
truncation error is bounded by the teacher mass outside the support and is
covered by the online-versus-cached tolerance test. Gemma 4's 262,144-token
vocabulary makes top-K caching mandatory: full logits cost about 0.5MB per
token, top-64 about 384 bytes. Cached mode requires `sft.packing: false`
(the cache is per-example token-aligned) and `distill.teacher_quantize: none`
(no teacher is loaded at training time; the pre-pass runs the teacher alone
in bf16).

Cache layout: `indices.bin` (int32), `logprobs.bin` (float16), `offsets.bin`
(int64), and a manifest binding the cache to the teacher identity and
revision, the tokenizer hash, and the dataset's train-file hash. The reader
verifies file hashes at open and the trainer refuses a cache built over
different inputs.

## Preflights

In order, extending F3's sequence:

1. `validate()` — the distill block invariants (tier/mode domains, trl-only
   logit tier, cached-mode packing and quantization rules).
2. Tokenizer identity — teacher and student tokenizers must hash-match
   (xxhash of the serialized tokenizer). Cross-tokenizer logit distillation
   breaks the KL and is out of scope for Phase 1; cross-family pairs fail
   here with a message pointing at `tier: response`.
3. D4 memory fit — the estimate includes a teacher term in online mode
   (weights at the chosen quantization plus one no-grad forward, dominated by
   the 262k-vocabulary logits). `--force` downgrades a refusal to a warning.
4. D9 lock. The cache pre-pass takes the lock too: it is a training-tier GPU
   job. Running inference containers are not probed; the operator manages
   containers manually.
5. After model load: `final_logit_softcapping` parity — teacher and student
   logits must both be capped identically or both uncapped, or the KL is
   silently skewed.

## Provenance

Every distill run writes `distill_provenance.json` into the run directory and
stamps the same fields into each checkpoint's `meta.json` under `distill`:
tier, teacher identity and revision (from the loaded model or the cache
manifest), mode, alpha, temperature, KL direction, and for the response tier
the F4 manifest reference. A distilled checkpoint cites its teacher from the
meta alone; F6 export and F7 evaluation read it from there.

## Usage

```bash
# Online logit distillation (reference recipe)
$ make train-distill CONFIG=training/recipes/reference/distill-e2b.yaml

# Cached mode: pre-pass, then train
$ make distill-cache CONFIG=... ARGS="distill.mode=cached distill.cache_path=/mnt/ai-data/training/datasets/distill-logits/e4b-k64 distill.teacher_quantize=none"
$ make train-distill CONFIG=... ARGS="distill.mode=cached distill.cache_path=... distill.teacher_quantize=none sft.packing=false"

# Response tier: F4 capture -> prepare -> train
$ make generate-teacher ARGS="--pipeline respond ..."
$ make prepare-sft ARGS="--source /mnt/ai-data/training/datasets/distill/<name>/train.jsonl ..."
$ make train-distill CONFIG=... ARGS="distill.tier=response distill.teacher_manifest=/mnt/ai-data/training/datasets/distill/<name>/manifest.json"
```

Overrides pass through the D5 loader: `ARGS="distill.alpha=0.3
distill.temperature=1.0"`.

## Testing

`tests/training/test_distill.py` covers the schema invariants, the cache
round trip and its integrity/binding refusals, the masked-KL golden fixture
(prompt tokens contribute zero KL), the alpha blend, forward-versus-reverse
direction, exactness of cached KL at full support, the truncated-support
tolerance bound, softcap parity, and provenance wiring. Loss tests run on
synthetic random-weight tensors and skip cleanly where torch is absent
(resolved F5 decision 3); the tiny real-model smoke and the GPU acceptance
runs are host-side. The scope criterion — a distilled `gemma-4-E2B`
measurably improving on an F7 scorecard over its pre-distillation baseline —
is deferred until F7 lands and executes through the F8 recipe (resolved F5
decision 4).
