# F5: Distillation harness — implementation investigation

Investigation of what implementing the F5 distillation harness entails, reconciling the
scope document against the current state of the repository. Per the operating rules, the
repository is the source of truth; every point where the scope document disagrees with
shipped code is called out and the repository convention wins.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Scope-document drift](#scope-document-drift)
- [Design decisions](#design-decisions)
- [Gemma 4 considerations](#gemma-4-considerations)
- [Memory budget](#memory-budget)
- [Proposed module layout](#proposed-module-layout)
- [Work breakdown](#work-breakdown)
- [Acceptance criteria](#acceptance-criteria)
- [Open questions](#open-questions)

## Scope summary

F5 trains a small student from a larger teacher in two tiers:

1. **Response-level distillation** — SFT on teacher outputs captured by F4 through the
   inference services. Any servable teacher qualifies, including quantized 26B-class
   models. The scope document defines this tier as a thin specialization of F3: a recipe
   whose dataset is an F4 artifact, with no new trainer.
2. **Logit distillation** — token-level KL against teacher logits, which HTTP serving
   cannot provide, so the teacher runs in-process in PyTorch. Reference pair: frozen
   `google/gemma-4-E4B` teacher, `google/gemma-4-E2B` student.

Loss: `L = alpha * CE(student, labels) + (1 - alpha) * T^2 * KL(student/T || teacher/T)`,
KL computed only where labels are unmasked (assistant tokens, reusing F1 masking).
Forward KL default, reverse KL behind a flag. Teacher execution modes per recipe:
`online` (frozen forward pass per batch, optionally nf4-quantized) and `cached`
(pre-pass materializes top-K teacher logprobs to disk, default K=64, with a
renormalization correction at training time).

## Current repository state

What F5 depends on and what exists today:

| Dependency | Scope role | Repository state |
|------------|------------|------------------|
| F1 data prep | Masked datasets, span masking reused by masked KL | Implemented: `training/data/sft.py`, `scripts/training/prepare_sft.py`. Labels are `-100` on non-assistant tokens by construction (segment-wise rendering, not string matching). |
| F2 PEFT schema | QLoRA student wrapping | Implemented: `training/peft_utils.py`, `peft` recipe block, `training/recipes/reference/sft-qlora-e2b.yaml`. |
| F3 SFT entry point | The trainer the response tier specializes | **Not implemented.** `SftConfig` in `training/recipes/schema.py` is data-side only; trainer-side fields (packing, epochs) are deferred to F3. No code imports `trl` yet. |
| F4 teacher harness | Produces the response-tier dataset | Implemented: `training/data/teacher.py`, `scripts/training/generate_teacher.py`, `make generate-teacher`. Captures response text only; no logprob capture path. |
| D4 memory fit | Preflight for combined teacher + student | Implemented: `training/calculator/` (`solve.py`, `calibrate.py`), `make memfit`. Models a single training model; no multi-model estimate. |
| D5 recipe system | The `distill` block extends this schema | Implemented: `training/recipes/schema.py` (OmegaConf struct mode), `validate.py` cross-field invariants. |
| D9 lock | Concurrency enforcement | Implemented: `training/lock.py` (`/mnt/ai-data/training/.train.lock`). |
| Tokenizer hash preflight | Teacher/student identity check | Mechanism exists: xxhash-based tokenizer hash enforced in `training/data/sft.py` and `training/data/shards.py` (`TokenizerMismatchError`). |

**F3 is the single hard prerequisite.** The response tier is defined as "F3 with an F4
dataset", and the logit tier subclasses the same TRL trainer. F5 cannot start until F3
lands, and the logit tier's trainer choice should be made jointly with F3's.

Pinned versions relevant to the trainer decision (from `pyproject.toml`): `torch==2.13.0`
(cu130), `transformers==5.13.1`, `trl==0.24.0`, `peft==0.19.1`, `bitsandbytes==0.49.2`.
TRL 0.24.0 ships `GKDTrainer`. The pins are provisional (offline-resolved) and must be
regenerated with `uv lock` on the workstation.

## Scope-document drift

Points where the scope document defers to the repository:

- **Script paths.** The scope document names `scripts/train/distill.py`. The repository
  convention is `scripts/training/` (`prepare_sft.py`, `generate_teacher.py`, ...), with
  logic in the `training` package and dispatch through the `python -m training` CLI
  (`training/cli.py`). F5 follows the repository layout.
- **distilabel.** The F4 scope specifies distilabel as the pipeline engine; the shipped
  harness is a stdlib urllib client (`training/data/teacher.py`, manifest records
  `"engine": "stdlib-urllib-client"`). F5's response tier consumes F4 artifacts as they
  actually exist: messages-format JSONL plus manifest with a teacher-identity block.
- **Recipe names.** The scope document's `distill-3b.yaml` / `distill-gemma-e2b.yaml`
  become `training/recipes/reference/distill-e2b.yaml`, matching the existing
  `sft-qlora-e2b.yaml` naming.
- **Model policy.** `training/families.py` is authoritative: Gemma 4 E-models primary
  (`gemma4` / `gemma4_text` model types, `google/gemma-4-E2B` and `-E4B`), Qwen3 the
  single registered fallback. Any generic references to Llama-class students in older
  scope text are superseded.

## Design decisions

### 1. Response tier is configuration, not code

Once F3 exists, the response tier is: run `prepare_sft.py` over an F4 output directory
(F4 acceptance already guarantees its artifacts ingest unmodified), then `train-sft`
with a recipe whose `distill` block records the teacher-manifest reference. Work items:

- Add a `distill` block to the recipe schema (see decision 3) with a
  `teacher_manifest` field; the run's config snapshot then carries the full provenance
  chain (teacher model, quantization, service, generation parameters, prompt-set hash).
- `validate()` checks the referenced manifest exists and is an F4 manifest.

This tier ships first and de-risks the deliverable, per the scope document.

### 2. Logit-tier trainer: custom `compute_loss` subclass, GKD recorded as considered

Two candidates at `trl==0.24.0`:

- **`GKDTrainer`** implements generalized knowledge distillation: generalized JSD loss
  with an on-policy fraction (`lmbda`) where the student generates its own sequences.
  On-policy generation per step is a poor fit for a 32GB single-GPU budget shared with
  an in-process teacher, and its loss surface (JSD interpolation `beta`) differs from
  the scope's `alpha`-weighted CE + temperature-scaled KL.
- **Subclass of the F3 trainer** (`SFTTrainer`) overriding `compute_loss`: teacher
  forward under `torch.no_grad()` (online) or cache lookup (cached), masked KL on
  positions where `labels != -100`, combined with the standard CE term.

Recommendation: the subclass. It matches the scope's loss definition exactly, works
identically in online and cached modes (GKD has no cached-teacher path), and keeps the
recipe surface identical to F3. Setting `lmbda=0, beta=0` in GKD approximates offline
forward KL but drags in generation machinery that is dead weight here. The decision and
the GKD evaluation are recorded here per the scope's "prefer a maintained trainer if
suitable" instruction; revisit if TRL later ships an offline distillation trainer.

### 3. Recipe schema: `distill` block

New dataclass group in `training/recipes/schema.py`, defaults off:

```yaml
distill:
  enabled: false
  tier: response | logit          # response = provenance-only; logit = KL trainer
  teacher: null                   # HF id or local path (logit tier)
  teacher_manifest: null          # F4 manifest path (response tier provenance)
  teacher_quantize: none | nf4    # online mode VRAM relief
  mode: online | cached           # teacher execution mode (logit tier)
  cache_path: null                # cached mode: logprob cache directory
  top_k: 64                       # cached mode support size
  alpha: 0.5                      # CE weight
  temperature: 2.0
  kl_direction: forward | reverse
```

`validate()` invariants: `tier: logit` requires `teacher`; `mode: cached` requires
`cache_path`; `tier: logit` with `precision.fp8` rejected (matching the existing
nf4/fp8 exclusion); `teacher_quantize: nf4` valid only in online mode.

### 4. Tokenizer-identity preflight

Reuse the existing xxhash tokenizer-hash mechanism: at startup the logit tier hashes
teacher and student tokenizers and refuses on mismatch with the documented message
stating that cross-tokenizer logit distillation is out of scope for Phase 1 (the D3
warning already in `docs/training/tokenizer_training.md`). Gemma 4 E4B and E2B share a
tokenizer; the Qwen3 fallback pair (for example Qwen3-4B teacher, Qwen3-0.6B or -1.7B
student) shares one within its family. Cross-family pairs (Gemma teacher, Qwen student)
fail this preflight by design.

### 5. Teacher execution modes

**Online**: load the teacher frozen (`requires_grad_(False)`, eval mode, no gradient
checkpointing), optionally nf4 via bitsandbytes, forward pass per batch under
`torch.no_grad()`. Exact, simple, costs VRAM and per-step latency.

**Cached**: a pre-pass (`python -m training distill-cache`, or a `--cache-only` flag)
runs the teacher alone over the prepared dataset and writes top-K logprobs per token.
Training then runs student-only, reading the cache. KL over the top-K support with
renormalization: teacher top-K probabilities renormalized to sum to one over the
support, student log-softmax gathered at the same indices and renormalized identically,
so both distributions live on the same K-element support. The truncation error is what
the online-versus-cached tolerance test bounds.

Cache format: one directory under `/mnt/ai-data/training/datasets/distill-logits/<name>/`
containing shard files (per-token `int32 indices[K]` + `float16 logprobs[K]`, memory-mapped)
plus a manifest (teacher id and revision hash, tokenizer hash, source dataset content
hash, K, dtype, git SHA, timestamp) following the D2/F1 manifest pattern. The trainer
refuses a cache whose tokenizer hash or dataset hash mismatches the run's dataset.

Both modes hash-stamp the teacher revision into the run's tracking record.

Cache sizing: at K=64, roughly 6 bytes/entry * 64 = 384 bytes per token; a 50M-token
dataset costs about 19GB on NVMe. Acceptable, but the manifest records size and the
docs state the K/disk tradeoff.

### 6. Memory-fit preflight (D4 extension)

`estimate()` currently models one training model. The logit tier's online mode needs a
combined estimate: student (weights + grads + optimizer + activations, per the existing
model) plus teacher (weights only at the chosen quantization, plus inference activations
for one forward at the same batch/sequence shape, no grads, no optimizer). Smallest
viable extension: a `teacher` term in the calculator driven by the recipe's `distill`
block. The F8 distill recipe becomes a calibration case for `calibrate.py`. Cached mode
needs no extension — the cache pre-pass is teacher-only and the training phase is
student-only, both already representable.

### 7. Concurrency and locking

Both tiers run inference-down: acquire the D9 lock, refuse if inference services are up
(same preflight order as F3: `validate()`, D4 memory fit, lock, services check). The
cache pre-pass is a training-tier GPU job and takes the lock too. This differs from F4,
which runs inference-up; the phase boundary between "capture teacher responses" (F4)
and "train against them" (F5) is exactly the serving swap.

## Gemma 4 considerations

Facts confirmed from the transformers `Gemma4Config` and the repository registry:

- **Vocabulary: 262,144.** Full-logit caching is infeasible (~0.5MB per token at
  fp16); top-K caching is not an optimization but a requirement. It also makes the KL
  reduction itself worth batching carefully — the online-mode `log_softmax` over 262k
  logits for both models at once is a peak-memory contributor the D4 activation term
  should include.
- **Per-layer embeddings (PLE)** (`vocab_size_per_layer_input=262144`,
  `hidden_size_per_layer_input=256`): a Gemma 4 E-model parameter-inventory quirk. The
  D4 components table and the F2 `target_modules: "auto"` registry already centralize
  per-family architecture knowledge in `training/families.py`; PLE handling for the
  teacher estimate lands there, not in distillation code.
- **`final_logit_softcapping`** is an optional config field. If the checkpoints enable
  it, teacher and student logits must both be capped (or both uncapped) before the KL —
  distilling capped-teacher against uncapped-student silently skews the loss. The
  implementation reads the field from both configs and asserts they match.
- **Tied word embeddings** (`tie_word_embeddings=True`) interact with QLoRA students:
  the lm_head is not a separate target module. Already handled by the F2 registry.
- **Thinking mode**: Gemma 4 thinking-control tokens are part of the F1 template
  registry. For distillation targets, F4's sampling config controls thinking mode at
  capture time; the logit tier trains on whatever spans F1 marks as assistant content.
  Whether thinking spans should receive KL weight is a data-prep policy, not a trainer
  concern — default is to treat them as assistant content (unmasked), and the F4
  default for distillation captures is thinking off.
- **Qwen3 fallback**: dense Llama-lineage decoder, ChatML template, no PLE, no
  softcapping — strictly simpler than the primary path. The tiny-model acceptance tests
  should run on a Qwen3-family tiny pair (or a synthetic tiny config) so CI-scale tests
  do not require Gemma 4 weights.

## Memory budget

Rough online-mode envelope for the reference pair on 32GB (to be validated by D4 and
calibration, not authoritative):

- Teacher `gemma-4-E4B` nf4: on the order of 3-4GB weights (4B-class effective
  parameters, PLE tables included), plus one inference forward of activations.
- Student `gemma-4-E2B` QLoRA (nf4 base + adapters): 2-3GB weights, adapter grads and
  optimizer state small, activations dominated by sequence length and the 262k-vocab
  logits.

Online mode with both models nf4 fits comfortably; a bf16 teacher likely also fits.
The stressful configuration is a full-precision full-fine-tune student plus teacher —
exactly what the D4 preflight exists to refuse. `optimizer: cpu_adam` and
`precision.fp8` remain unimplemented stubs (`training/model.py`), so offload cannot be
assumed as a pressure valve for Phase 1.

## Proposed module layout

Following repository conventions (logic in `training/`, thin host scripts, CLI
subcommand, Make target):

| Artifact | Purpose |
|----------|---------|
| `training/distill.py` | Trainer subclass (masked KL loss, online/cached teacher), preflights |
| `training/data/distill.py` | Logprob cache: writer (pre-pass), memory-mapped reader, manifest |
| `scripts/training/train_distill.py` | Host entry point (matches `scripts/training/` convention) |
| `training/cli.py` | New subcommands: `train-distill`, `distill-cache` |
| `training/recipes/schema.py` + `validate.py` | `distill` block and invariants |
| `training/recipes/reference/distill-e2b.yaml` | Reference recipe (E4B teacher, E2B student) — the F8 card seed |
| `training/calculator/` | Teacher term in the online-mode estimate |
| `Makefile` | `train-distill CONFIG=...` via `docker-compose.training.yaml` |
| `tests/` | Masked-KL golden fixture, top-K renormalization, tolerance test, preflight refusals |
| `docs/training/distillation.md` | Reference doc, paired with this investigation |

## Work breakdown

Ordered; items 2+ depend on F3 existing:

1. **Schema + validation** (no F3 dependency): `distill` block, invariants, unit tests.
2. **Response tier**: F4-manifest provenance wiring into the F3 run snapshot; document
   the prepare-then-train flow. Ships first.
3. **Masked-KL loss + online mode**: trainer subclass, teacher loading (frozen,
   optional nf4), softcapping parity check, tokenizer-hash preflight, D9/D4/service
   preflights. Golden-fixture masked-KL test reusing F1 masks.
4. **Cached mode**: cache writer pre-pass, reader, renormalized top-K KL,
   online-versus-cached tolerance test on a tiny pair.
5. **D4 teacher extension + calibration** against the reference recipe.
6. **Scope criterion** (host-side, human-run): distill `gemma-4-E2B` from an E4B
   teacher via the F8 recipe and show measurable improvement on an F7 scorecard over
   its pre-distillation baseline. Blocked on F7 in addition to F3.

## Acceptance criteria

From the scope document, restated against repository mechanisms:

- Masked-KL unit test: KL contributes zero on prompt tokens (golden fixture reusing
  F1's masks from `training/data/sft.py`).
- Online and cached modes on a tiny model pair agree within a documented tolerance
  attributable to top-K truncation.
- Tokenizer-mismatch preflight fails with the documented message
  (`TokenizerMismatchError` pathway).
- Scope criterion via the F8 recipe: a distilled `gemma-4-E2B` measurably improves on
  a target evaluation (F7 scorecard, `--baseline` distillation mode) over its
  pre-distillation baseline. Host-side; human executes and reports output.

## Open questions

1. **Ordering**: F3 is a hard prerequisite. Should F3 be the next implementation task,
   with F5 item 1 (schema) optionally developed in parallel in a separate worktree?
2. **Trainer coupling**: implement the F5 loss as a subclass of whatever trainer F3
   picks (recommended above), which argues for deciding F3's TRL-versus-unsloth backend
   surface before F5 code starts. Confirm the subclass approach over `GKDTrainer`.
3. **Tiny-pair test models**: use Qwen3-family tiny checkpoints for the
   online-versus-cached tolerance test, or synthetic random-weight configs? Synthetic
   avoids network and gating but exercises less of the real template path.
4. **Final acceptance dependency**: the scope criterion needs F7. Acceptable to land F5
   with items 1-5 proven and defer item 6 until F7 exists, per the build order (F5
   before F7)?
