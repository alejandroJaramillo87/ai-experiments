# P4-D3: SAE training and evaluation — implementation investigation

**Frozen record — 2026-07-15. Drift against the current repository is expected and
correct; this document is never updated.**

Investigation of what implementing the P4-D3 deliverable (a trained, evaluated sparse
autoencoder on an own checkpoint, with standardized quality numbers) entails, given the
current state of this repository. The repository is the source of truth; where the design
document disagrees with the code, the code wins and the drift is recorded here. Findings
Model policy: Gemma 4 is the primary family for every
feature deliverable, with the Qwen3 family as the fallback.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Design-document drift](#design-document-drift)
- [The training path](#the-training-path)
- [The evaluation path](#the-evaluation-path)
- [The calibration run](#the-calibration-run)
- [Sizing and runtime envelope](#sizing-and-runtime-envelope)
- [Gap analysis](#gap-analysis)
- [Proposed layout](#proposed-layout)
- [Work breakdown](#work-breakdown)
- [Acceptance criteria](#acceptance-criteria)
- [Resolved design decisions](#resolved-design-decisions)

## Scope summary

P4-D3 turns a P4-D2 activation capture into an instrument: an SAE trained from the
cache, evaluated with standardized metrics, and calibrated once against a published
reference so later "this checkpoint is weird" claims are distinguishable from "our
pipeline is broken". Per the design document:

1. SAELens v6 training from cached activations, `python -m interp sae-train`, under
   the same batch-job discipline as P4-D2 (lock, watchdog, preflight).
2. Architecture and hyperparameters follow SAELens's current recommended defaults for
   the model class, pinned at implementation and recorded. One width (16k features),
   one target-sparsity band; a learning-rate sweep capped at three runs.
3. Tracking through the D6 wandb-offline convention; runs land under
   `/mnt/ai-data/interp/saes/<name>/` with config, weights, metrics, and a manifest
   chaining to the capture manifest.
4. SAEBench evaluation (`python -m interp sae-eval`) running the core metric set:
   reconstruction quality (explained variance, delta-LM-loss), sparsity (L0), sparse
   probing, and feature absorption. Evals needing an explainer LLM route to the local
   vLLM endpoint or are skipped in favor of P4-D4's Delphi pass; no external API.
5. A one-time calibration run comparing own-pipeline numbers against a published
   reference SAE of similar width and sparsity, with a documented tolerance band.
6. Config takes any hook point so later model families need no code change.

## Current repository state

Everything P4-D3 consumes exists; the deliverable itself is greenfield. No SAE-training
module, config block, recipe, script, or test exists yet.

| Dependency | Where it lives today |
|------------|----------------------|
| Capture input | `interp/harvest.py` (P4-D2, merged): SAELens v6 `CacheActivationsRunner` shards under `/mnt/ai-data/interp/activations/<name>/<hook>/`, `activation_harvest` manifest with `model_id`, `model_hash`, `hooks`, `d_model`, `corpus`, `corpus_xxh3`, `context_len`, `total_tokens`, `dtype`, `seed`, `tokenizer_fingerprint`, `sae_lens_version`. |
| Downstream refusal gate | `interp/harvest.py` `check_capture(capture_dir, expects)` — the docstring already states "P4-D3 calls this before training from a cache". |
| Retention handshake | `interp/harvest.py` `mark_validated()`; P4-D3's `--validated` flag marks the consumed capture deletable for `make interp-gc`. |
| SAE toolkit | `sae-lens==6.46.0` (released 2026-07-13) in the `interp` group; already provides both the cache runner (used by P4-D2) and the training runner family. No `dictionary_learning` dependency exists. |
| SAE consumption precedent | `interp/steering/extract.py` `sae_row` backend loads Gemma Scope decoder rows through SAELens. |
| Training/eval module pattern | `interp/probes/train.py` (capture in, artifact plus manifest out, deterministic seeded split), `interp/probes/registry.py` (scan dirs with `manifest.json`, refuse on hash mismatch), `interp/probes/evaluate.py` (injectable callables, CPU-testable). |
| Config system | `training/recipes/schema.py` typed OmegaConf schema plus `training/recipes/loader.py`; `HarvestConfig` and `OrganismConfig` are the precedents. No Hydra anywhere. |
| Batch-job envelope | `training/lock.py` `run_lock` (D9), `training/watchdog.py` `Watchdog` (D10), `scripts/interp/preflight.sh` (inference down). |
| Tracking | `training/tracking.py`: wandb offline-first unless `WANDB_MODE` is set (D6). |
| Output root | `make interp-dirs` already creates `/mnt/ai-data/interp/saes`. |
| Workbench | `docker/interp/Dockerfile.interp`: CUDA 13.0.2, `torch==2.13.0` (cu130), `transformers==5.13.1`, `transformer-lens==3.5.1`, `nnsight==0.7.0`; deps locked via `uv.lock` `interp` group. |
| Gemma Scope reference path | `scripts/interp/smoke_test.py` step 6: Gemma-2-2B bf16 forward plus `gemma-scope-2b-pt-res-canonical` SAE encode (requires `HF_TOKEN`). |
| Subject checkpoints | Sealed P4-D1 organisms under `/mnt/ai-data/interp/organisms/<name>/` (Gemma 4 E2B base); harvest recipes at `training/recipes/interp/harvest-*.yaml` capture `model.layers.15`. |
| Test conventions | `tests/interp/`: CPU-only, injectable fake runner functions, `importorskip` for heavy deps. |

## Design-document drift

The design document defers to the repository on all of the following:

1. **Base model and layer.** The document trains the reference SAE on Gemma-2-2B layer
   13. The repository's subject is Gemma 4 E2B; the P4-D2 reference capture is
   `model.layers.15` (50 percent of a 30-layer stack). The SAE config takes the hook
   name from the capture manifest, so this is data, not code.
2. **Hook naming.** As in P4-D2, hooks are HF named-module paths (`model.layers.{l}`),
   not TransformerLens aliases. TransformerLens still has no confirmed Gemma 4 support
   as of 2026-07 (v3.5.1 covers Qwen3 natively and Gemma 3 at newest; `HookedTransformer`
   is deprecated in favor of `TransformerBridge`, whose Gemma 4 coverage is unverified).
3. **Calibration anchor.** The document calibrates by training an SAE on base
   Gemma-2-2B and comparing against the published Gemma Scope point. The P4-D1
   investigation resolved a one-time pipeline calibration against a Qwen3 model
   instead (native TransformerLens support, known-good SAE path). These two
   calibrations answer different questions; both run — see
   [Resolved design decisions](#resolved-design-decisions). No Gemma Scope release exists
   for Gemma 4 (Gemma Scope 2, December 2025, targets Gemma 3), so a published-number
   comparison is only available on Gemma 2 or Gemma 3 bases.
4. **Config system.** The document's per-deliverable script-plus-config framing maps to
   the repository standard: an `SaeConfig` block in `training/recipes/schema.py` and
   recipe yamls under `training/recipes/interp/`, loaded by the D5 recipe loader. No
   Hydra, no `interp/configs/` tree.
5. **Architecture default.** The document says "JumpReLU/BatchTopK family, pin at
   implementation". SAELens v6's current documentation recommends **BatchTopK** as the
   default (state of the art, easy to train; checkpoints save as JumpReLU for
   inference), with JumpReLU as the trickier-to-train alternative and
   MatryoshkaBatchTopK available (Gemma Scope 2 used Matryoshka training). BatchTopK
   is the pin this investigation proposes.
6. **SAEBench scope.** The document names SAEBench as the evaluator. SAEBench v0.6.0
   (2026-05-01) ships eight evals but its activation collection is TransformerLens-based
   and its model coverage is Pythia and Gemma (2/3) out of the box; running it on Gemma 4
   organism checkpoints requires patching its `activation_collection.py`, and its
   AutoInterp eval expects an OpenAI-compatible API (routable to the local vLLM
   endpoint). The core reconstruction and sparsity metrics, by contrast, need only the
   SAE, the model, and text — implementable directly against the workbench's HF-path
   plumbing. How much of SAEBench to adopt versus reimplement is an open decision.

## The training path

SAELens v6 trains from a pre-existing activation cache: the training runner config
points at the capture directory (`use_cached_activations=True` plus the cached
activations path) instead of running fresh forward passes. This is exactly the split
P4-D2 was built for, and it means SAE training itself needs **no model forward passes
and no hooked-model support** — the Gemma 4 compatibility question was settled at
harvest time. Consequences:

- The trainer's inputs are `d_model`-wide bf16 vectors; architecture quirks (per-layer
  embeddings, sliding-window attention, thinking tokens) are invisible here. The P4-D2
  hook-dimension gate (`check_hook_dims`) already proved the vectors are the 2304-dim
  residual stream.
- Training is GPU-light relative to harvesting: a 16k-width SAE on d_model 2304 is
  roughly 75M parameters. The bottleneck is NVMe read throughput over the ~1.38TB
  cache; if the loop is IO-bound, raise the runner's buffer sizes (128GB RAM exists
  for exactly this) before touching anything else.
- The delta-LM-loss evaluation (and any eval that must run the subject model with the
  SAE spliced in) does need forward passes; those reuse the harvest's
  AutoModelForCausalLM path and hook names.

Proposed pinned reference config, following SAELens's current recommended defaults for
this model class and the design document's caps:

| Field | Value | Rationale |
|-------|-------|-----------|
| Architecture | BatchTopK | SAELens v6 recommended default; saves as JumpReLU for inference, which P4-D4's Neuronpedia import consumes |
| Width | 16,384 features | Design document; matches the Gemma Scope 16k point for any published comparison |
| Target sparsity | L0 band 40–80 | Brackets the canonical Gemma Scope 16k residual point (L0 about 41); BatchTopK sets L0 directly via k |
| Training tokens | 300M (the full reference capture) | One pass over the P4-D2 cache |
| LR sweep | at most 3 runs | Design document cap; anything beyond is Phase 5 work |
| Seed | from the recipe `run.seed` | Determinism smoke requires a fixed seed reproducing final loss within tolerance |

The exact runner class and field names (`LanguageModelSAERunnerConfig` and its v6
successors) are pinned at implementation against `sae-lens==6.46.0` and recorded in the
run manifest, per the design document's "defaults move" risk note.

## The evaluation path

The design document's core metric set, and where each can come from:

| Metric | Needs | Source |
|--------|-------|--------|
| Explained variance / reconstruction | SAE + cached activations | SAELens training logs report it; recompute over a held-out shard for the report |
| Delta-LM-loss (loss recovered) | SAE + subject model + text | SAEBench `core` eval, or direct implementation over the HF hook path |
| L0 sparsity | SAE + cached activations | Same as above; BatchTopK fixes it by construction, report the measured value |
| Sparse probing | SAE + labeled datasets + model | SAEBench `sparse_probing` eval |
| Feature absorption | SAE + model + tokenizer | SAEBench `absorption` eval (first-letter task) |

SAEBench realities that shape the design:

- **Model coverage.** SAEBench's activation collection runs through TransformerLens.
  For the Qwen3 calibration subject this works natively; for Gemma 4 subjects it does
  not (unverified TransformerBridge support), so either SAEBench's collection layer is
  patched to accept an HF-path model, or the Gemma 4 rows of the metric table are
  produced by our own direct implementations of the core metrics and only the
  TL-supported subjects get the full SAEBench battery.
- **Explainer-dependent evals** (AutoInterp) point at the local vLLM endpoint on 8003
  or are skipped in favor of P4-D4's Delphi pass — the design document already allows
  this; no external API in either case.
- **Not adopted:** RAVEL, SCR, TPP, unlearning — beyond the design document's core set;
  available later without design change since the SAEBench dependency will be present.
- **Dependency.** SAEBench v0.6.0 cannot join the `interp` group: it requires
  `datasets<4.0` and `numpy<2.0`, which conflict with the core `datasets==5.0.0` pin
  (verified against its PyPI metadata at implementation). It runs in a dedicated
  isolated venv on NVMe (`scripts/interp/saebench_env.sh`, the Neuronpedia
  separate-environment precedent), pinned to `sae-bench==0.6.0`.

Results are D8-convention JSON under the SAE run directory, and the run manifest chains
to the capture manifest (`capture_dir` plus `capture_manifest_xxhash`, the
`interp/probes/train.py` precedent) so every number is traceable to model hash, corpus
hash, and git SHA.

## The calibration run

The design document's acceptance hinges on one calibration separating "pipeline broken"
from "checkpoint weird". Two candidate anchors exist and they answer different
questions:

1. **Gemma-2-2B against Gemma Scope** (design document). Train an SAE on a base
   Gemma-2-2B capture (layer 12 or 13 residual, 16k width) and compare against the
   published `gemma-scope-2b-pt-res` point of similar width and sparsity (canonical 16k
   layer-12 residual SAE: average L0 about 41; delta-loss curves in arXiv 2408.05147).
   This is the only anchor with published, citable numbers. Cost: one extra ~1.2TB
   capture plus a training run on a model that is otherwise only a smoke-test subject;
   the workbench already downloads Gemma-2-2B and Gemma Scope for smoke step 6.
2. **Qwen3 pipeline calibration** (P4-D1 resolved decision). Train on a Qwen3-1.7B
   capture through the identical pipeline. This validates the pipeline on a
   TL-supported family (enabling the full SAEBench battery as a cross-check of our
   HF-path numbers) but has no published reference SAE numbers to land inside a band
   of.

They compose: the Gemma Scope comparison is the published-number band the acceptance
criterion demands; the Qwen3 run is the fallback-family rehearsal. Whether to run one
or both is an open decision below. Either way the tolerance band is recorded once in
the calibration report, not invented per-run.

## Sizing and runtime envelope

- **SAE parameters:** 2 × d_model × width ≈ 151M runtime parameters for d_model 2304 at
  16k width (encoder plus decoder); trivial VRAM.
- **Cache read:** 300M tokens × 4,608 bytes = ~1.38TB per epoch. At 2GB/s effective
  NVMe read this is roughly 12 minutes of pure IO per pass; the trainer's shuffle
  buffer and dataloader workers determine whether the GPU starves. This is the
  "respectable SAE in hours" dependency the design document flags.
- **Calibration capture (if Gemma-2-2B anchor):** d_model 2304 (Gemma-2-2B) — ~1.38TB
  for 300M tokens; a reduced-token calibration capture (for example 100M tokens,
  ~0.46TB) may suffice for a band comparison and is a recipe field, not code. The
  Gemma Scope SAEs were trained on far more tokens, so the band must be set against a
  same-budget expectation, not the published point verbatim — another reason the band
  is documented rather than assumed.
- **Disk coexistence:** the reference capture (1.38TB) plus a calibration capture
  cannot both stay resident with a 100GB watchdog floor unless the NVMe has more than
  ~3TB free; the `--validated` plus `make interp-gc` handshake exists for exactly this
  sequencing.
- **VRAM:** eval forward passes on E2B bf16 (~10GB weights) fit the RTX 5090 with
  headroom; training from cache barely touches the GPU memory budget.

## Gap analysis

New code, in build order:

1. **Config block.** An `SaeConfig` in `training/recipes/schema.py` (the
   `HarvestConfig` precedent): capture dir, expected capture identity fields (model
   id, hook, corpus hash — fed to `check_capture`), architecture, width, k or L0
   target, learning rate, training tokens, batch size, buffer sizes, output root,
   `mark_capture_validated` flag. Plus an `SaeEvalConfig` or eval fields: metric list,
   held-out shard policy, SAEBench toggles, explainer endpoint.
2. **Training wrapper.** `interp/sae/train.py` plus the `python -m interp sae-train` CLI:
   `check_capture` refusal gate, D9 lock (`sae-{name}`), D10 watchdog, D6 tracking,
   SAELens training runner invocation from cache, runner function injectable for CPU
   tests. Manifest (`artifact: "sae"`) written on completion, chaining
   `capture_manifest_xxhash`, recording architecture, width, measured L0, final loss,
   sae_lens version, seed, wall clock.

   > **Correction, 2026-08-08 (KB-62).** The measured-L0 and final-loss fields
   > described here shipped, but could never hold a measurement: SAELens discards
   > the trainer that owns those numbers, so both fell through to defaults on every
   > run — `final_loss` to `0.0` and `measured_l0` to the *requested* `k`. Both
   > fields have been removed. Quality belongs to the evaluation wrapper in item 4,
   > which measures L0 against held-out activations; the manifest is provenance
   > only. The design as originally written is left above as the historical record.
3. **Registry.** `interp/sae/registry.py` mirroring `interp/probes/registry.py` over
   `/mnt/ai-data/interp/saes` — P4-D4's import script and the steering `sae_row`
   backend become consumers.
4. **Evaluation wrapper.** `interp/sae/evaluate.py` plus `python -m interp sae-eval`:
   core metrics (explained variance, L0, delta-LM-loss over the HF hook path) as
   direct implementations, SAEBench invocation for sparse probing and absorption where
   the subject family allows, D8 JSON report into the run directory, injectable
   model/SAE callables for CPU tests.
5. **Calibration report.** A documented comparison (markdown under
   `docs/interpretability/experiments/` per the experiments-log convention) recording
   the anchor, the band, and the verdict; the band consumed by later runs as data.
6. **Recipes.** `training/recipes/interp/sae-smoke.yaml` (tiny width, few million
   tokens from the smoke capture — the CI-sized test), `sae-false-fact.yaml` (the
   reference run from the P4-D2 reference capture), `sae-calibration-*.yaml` (anchor
   run(s), pending the open decision).
7. **Make targets.** `make sae-train RECIPE=...`, `make sae-eval ARGS=...` following
   the `harvest` pattern (preflight, then compose run in the interp workbench).
8. **Tests.** `tests/interp/test_sae.py`: config validation, capture refusal on
   mismatch, manifest round-trip and chaining, fake-trainer resume/determinism
   plumbing, eval math on synthetic activations, registry listing/refusal — CPU-only.
9. **Dependency.** `scripts/interp/saebench_env.sh`: an isolated, pinned SAEBench venv
   on NVMe (the `interp`-group route is blocked by the `datasets`/`numpy` conflict
   above).

## Proposed layout

```
interp/sae/__init__.py
interp/sae/train.py                    # check_capture gate, lock, watchdog, runner wrapper, manifest
interp/sae/registry.py                 # list/load with hash refusal, mirrors probes registry
interp/sae/evaluate.py                 # core metrics + SAEBench drivers, D8 JSON report
interp/commands/sae_train.py           # CLI entry point (python -m interp sae-train)
interp/commands/sae_eval.py            # CLI entry point (python -m interp sae-eval)
training/recipes/schema.py             # SaeConfig (+ eval fields) added
training/recipes/interp/sae-smoke.yaml
training/recipes/interp/sae-false-fact.yaml
training/recipes/interp/sae-calibration-{gemma2,qwen3}.yaml
training/recipes/interp/harvest-calibration-{gemma2,qwen3}.yaml
scripts/interp/saebench_env.sh          # isolated SAEBench venv (host-side)
tests/interp/test_sae.py
/mnt/ai-data/interp/saes/<name>/       # weights + config + metrics + manifest.json (root exists)
```

## Work breakdown

1. Schema, recipes, loader wiring (CPU, testable).
2. Training wrapper, registry, CLI, Make targets, fake-trainer tests (CPU, testable).
3. Evaluation wrapper with direct core metrics and D8 report, synthetic-data tests
   (CPU, testable).
4. SAEBench pin and lock; workbench image rebuild; import smoke (host; human runs and
   reports).
5. SAE smoke run from the P4-D2 smoke capture: minutes-scale train plus eval in the
   workbench (host, GPU; human runs and reports).
6. Calibration run(s) per the resolved anchor decision; calibration report with the
   documented band (host, GPU, overnight; human runs and reports).
7. Reference run: harvest-train-evaluate on the false-fact organism as one overnight
   sequence, `--validated` handshake exercised; determinism re-run from the same
   capture and seed (host, GPU; human runs and reports).

## Acceptance criteria

From the design document, adjusted for the repository's model policy:

- The calibration SAE lands within the documented band of its reference, and the band
  plus comparison is published in the calibration report.
- The harvest-train-evaluate loop on the false-fact organism (Gemma 4 E2B) completes
  as one overnight sequence, producing an SAE run directory under
  `/mnt/ai-data/interp/saes/` with evaluation JSON and a manifest chaining to the
  capture manifest.
- A second training run from the same capture with the same seed reproduces final loss
  within tolerance (determinism smoke).

## Resolved design decisions

Resolved with the maintainer, 2026-07-15:

1. **Calibration anchor: both.** The Gemma-2-2B run against the published Gemma Scope
   point (16k residual, layer 12, average L0 about 41) provides the tolerance band the
   acceptance criterion demands; a small Qwen3-1.7B run rehearses the fallback family
   and cross-checks the direct metric implementations against SAEBench. The Qwen3 run
   validates plumbing, not quality, and uses a 50M-token capture.
2. **SAEBench on TL-supported subjects only.** The core reconstruction and sparsity
   metrics are implemented directly (`interp/sae/evaluate.py`) and carry the Gemma 4
   rows; the full SAEBench battery runs against the Qwen3 calibration subject.
   Patching SAEBench's activation collection for the Gemma 4 HF path is deferred until
   an investigation needs sparse-probing or absorption numbers on Gemma 4. Discovered
   at implementation: SAEBench 0.6.0 cannot join the `interp` dependency group at all
   (`datasets<4.0`/`numpy<2.0` versus the core `datasets==5.0.0` pin), so it lives in
   an isolated pinned venv (`scripts/interp/saebench_env.sh`).
3. **Architecture pin: BatchTopK.** SAELens v6's recommended default; k sets L0
   directly and checkpoints save as JumpReLU, which the P4-D4 Neuronpedia import
   consumes. MatryoshkaBatchTopK (Gemma Scope 2 parity) is declined as SAE-methods
   research the roadmap avoids. The pin is enforced in `validate_sae`: changing the
   architecture is a design change, not a recipe edit.
4. **Calibration capture budget: full 300M tokens.** A same-scale comparison tightens
   the band against the published point, and the run rehearses the full disk and
   throughput envelope ahead of larger own-trained subjects (the mech-interp tier is
   expected to consume full-training ~500M-parameter models later). The calibration
   capture and the reference capture are sequenced through the `--validated` plus
   `make interp-gc` handshake when less than ~3TB is free.
