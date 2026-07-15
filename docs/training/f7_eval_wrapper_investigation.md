# F7: Fine-tuning evaluation wrapper — implementation investigation

Investigation of what implementing F7 (fine-tuning evaluation wrapper) entails,
reconciling the scope document against the current state of the repository. Per the
operating rules, the repository is the source of truth; every point where the scope
document disagrees with shipped code is called out and the repository convention wins.

## Table of contents

- [Scope summary](#scope-summary)
- [Current repository state](#current-repository-state)
- [Scope-document drift](#scope-document-drift)
- [Gap analysis](#gap-analysis)
- [Design decisions](#design-decisions)
- [Gemma 4 considerations](#gemma-4-considerations)
- [Proposed module layout](#proposed-module-layout)
- [Work breakdown](#work-breakdown)
- [Acceptance criteria](#acceptance-criteria)
- [Open questions](#open-questions)

## Scope summary

F7 answers "did the fine-tune help" with one command producing a base-versus-fine-tuned
scorecard from three signal types. The scope document tags it "build", depending on D7
(`serve_eval`), F1 (held-out split), and F6. Its four decisions:

1. **Three evaluators behind one runner**, each independently skippable:
   - **Held-out loss** — perplexity/loss on F1's held-out split, computed in-process in
     the training container (GPU, inference down) for base and fine-tuned checkpoints at
     matched precision.
   - **lm-evaluation-harness** — a pinned task set (default: ifeval, gsm8k, mmlu subset,
     hellaswag) run in-process against the HF checkpoint before quantization, so the
     benchmark measures the fine-tune, not the quant.
   - **Model-as-judge** — pairwise comparison on a fixed prompt set held in git; base
     and fine-tuned GGUFs generate through the serving stack at matched quantization,
     then a judge model served on vllm-gpu scores pairs with position-swapped duplicates
     to cancel order bias. Judge is recipe-configured; the scope document suggests a
     quantized Gemma 4 26B A4B, "pick at implementation and pin in the config".
2. **Phasing**: held-out loss and lm-eval run inference-down; generation and judging run
   inference-up. The runner sequences these itself via D7's swap and D9's lock.
3. **Scorecard artifact**: `scorecard.json` extending D7's evaluation JSON shape
   (metadata block, then one section per evaluator with base, fine-tuned, and delta),
   plus a rendered `scorecard.md` table.
4. **Distillation mode**: `--baseline` accepts any checkpoint or HF model, so a student
   is compared against its pre-distillation self with the same machinery.

## Current repository state

F7 is not implemented: there is no `eval` subcommand, no scorecard code, no judge
machinery, and no lm-eval dependency. The scaffolding it consumes, however, largely
exists.

| Scope element | Repository state |
|---------------|------------------|
| Runner entry point | Absent. The unified CLI (`training/cli.py`, `build_parser()`) has `memfit`, `calibrate`, `train`, `train-sft`, `train-distill`, `distill-cache`, and `export` subcommands; no `eval`. |
| Held-out split | Implemented by F1: `prepare_sft` reserves a seeded held-out split; `train-sft`/`train-distill` already compute `holdout_loss` and a `losses` curve at `sft.eval_steps` cadence (`training/cli.py` result block), stated in `training/recipes/schema.py` as "F7 consumes the curve". F5 keeps holdout evaluation as plain cross-entropy specifically so it is "usable as the F7 `--baseline` signal" (`training/distill.py`). |
| lm-evaluation-harness | Absent. No `lm-eval` pin in `pyproject.toml` or `uv.lock`. Present metrics deps: `evaluate==0.4.6` (core), `rouge-score==0.1.2` (quant extra). |
| Judge scoring | Absent. `scripts/eval/smoke_prompts.json` states "Not a benchmark harness (lm-eval and judge scoring are Phase 1)". |
| Serving swap and generation | Implemented by D7: `scripts/eval/serve_eval.py` (`make serve-eval`) plus shared primitives in `training/serving.py` (`SERVICES`, `swap_model`, `wait_healthy`, `chat_completion`, `sha256_file`, `ensure_training_idle`). Restores prior serving state on every exit path. |
| Quantization-confound rule | Implemented and documented in `serve_eval.py`: every result file records `quant_type`; comparative claims must compare at matched quantization. |
| Evaluation JSON shape | Implemented: `serve_eval.py` emits a `schema_version: 1` report with per-prompt records plus a summary block, designed for Phase 1 extension. |
| Base/tuned provenance | Implemented by F3/F5/F6: checkpoint `meta.json` records `adapter_only`, `base_model`, base revision and config hash, teacher provenance; export manifests record adapter SHA, llama.cpp pin, quant type, GGUF SHA-256. The scorecard metadata block can be assembled from these. |
| Concurrency discipline | Implemented: D9 lock (`training/lock.py`) plus `ensure_training_idle`; AGENTS.md serializes all GPU consumers. |
| Reference models | Gemma 4 E2B recipes (`training/recipes/reference/sft-qlora-e2b.yaml`, `distill-e2b.yaml`); Qwen3 is the registered fallback family (`training/families.py`, converter pin). |

## Scope-document drift

- **Entry point.** The scope document names `scripts/eval/eval_ft.py` with logic in
  `training/eval/`. The repository convention, settled during F6 ("merge entry point:
  `python -m training merge`... a subcommand of the unified training CLI, not a
  standalone script"), is the unified CLI. F7 should land as `python -m training eval`
  wrapped by `make eval-ft`, with logic in `training/eval/`. A thin
  `scripts/eval/eval_ft.py` exists only if the host-side phase needs to run outside the
  container (see gap 5).
- **Model naming.** The scope document's examples use `google/gemma-4-E4B`; the
  repository's reference recipes are Gemma 4 E2B. Acceptance runs use the repository's
  recipes.
- **Judge on vllm-gpu.** The scope document assumes the judge serves on vllm-gpu. The
  repository's fine-tune serve path is llama-gpu/GGUF; vllm-gpu exists in
  `docker-compose.yaml` but is not integrated with `training/serving.py` swaps. Whether
  the judge runs on vllm-gpu or llama-gpu is an open question (see below); the compose
  file remains the source of truth for ports (llama-gpu 8002, vllm-gpu 8003).
- **Held-out loss "for both base and fine-tuned checkpoints"**: for adapter-only
  checkpoints the base is already loaded to apply the adapter, so the base pass is
  nearly free; for full fine-tunes it is a second full load. Matched precision means
  evaluating both in the same dtype/quantization used at train evaluation time
  (bf16 compute; nf4 base for QLoRA runs) and recording it in the scorecard.

## Gap analysis

The actual F7 work, in dependency order:

1. **Held-out loss evaluator** (`training/eval/holdout.py`). Reuse F1's
   `load_sft_dataset` held-out split and the trainer's loss computation path; evaluate
   an arbitrary (checkpoint or HF id) model against a named prepared dataset. Runs
   inference-down in the training container. Mostly plumbing over existing code.
2. **lm-evaluation-harness evaluator** (`training/eval/lmeval.py`). New dependency pin
   (an `eval` group or addition to the `training` group), a pinned task set in
   `training/eval/tasks.yaml`, in-process execution against the HF checkpoint
   (pre-quantization), task/harness versions recorded in the scorecard metadata.
   Requires an image rebuild; pin-then-fix discipline applies.
3. **Generation phase** (`training/eval/generate.py`). Reuse `serve_eval`'s swap
   machinery to serve base then tuned GGUF (matched quant, manifest SHA verified) and
   collect responses for the judge prompt set (`training/eval/judge_prompts.jsonl`,
   seeded from public sets, held in git). Resumable per-model response files so an
   interrupted run does not redo completed generations.
4. **Judge evaluator** (`training/eval/judge.py`). Pairwise scoring with
   position-swapped duplicates; win/loss/tie plus rationale persisted per pair;
   refusals and malformed verdicts counted and reported, never dropped. Judge model,
   service, and sampling parameters are config fields pinned in the recipe/config and
   snapshotted into the scorecard.
5. **Runner and phasing** (`training/eval/runner.py`, CLI `cmd_eval`). Sequences
   inference-down evaluators (holdout, lmeval) and inference-up phases (generate,
   judge), preflighting the D9 lock and compose state per phase. Note the sandbox
   split: the in-process evaluators run inside the training container (host-launched
   via `make`), while the serving swap is a host-side action; the runner must either
   run wholly on the host (launching container phases via `docker compose run`, the
   `serve_eval.py` pattern) or emit explicit host steps. The `serve_eval.py` precedent
   favors a host-side runner.
6. **Scorecard writer** (`training/eval/scorecard.py`). `scorecard.json` extending the
   `schema_version: 1` shape: metadata (models, revisions, quant types, judge identity
   and prompt hash, lm-eval pins, git SHA) assembled from checkpoint meta and export
   manifests; one section per evaluator with base, tuned, delta; rendered
   `scorecard.md`. Follow `.styles/PYTHON_OUTPUT_STYLE.md` for CLI output and JSON
   mode.
7. **`--baseline` mode.** Any checkpoint or HF model as the comparison side, giving F5
   its deferred pre-distillation-self criterion (`distill-e2b.yaml` records this as
   "deferred until F7 lands").
8. **Tests.** Unit tests for pairing/position-swap logic, verdict parsing (including
   refusal/malformed paths), scorecard assembly and delta math, null-test scaffolding;
   golden fixtures for scorecard JSON. All sandbox-testable. GPU/serving acceptance is
   host-executed and reported back per AGENTS.md.

## Design decisions

1. **F7 lands in `training/eval/` behind `python -m training eval`**, `make eval-ft`
   wrapping it, consistent with the F6 resolution that new tools are CLI subcommands.
2. **Evaluators are independently skippable** via `--evaluators holdout,lmeval,judge`;
   each writes its section of the scorecard independently, so a partial run yields a
   partial scorecard that says so.
3. **Reuse, do not fork, the serving primitives.** Generation goes through
   `training/serving.py`; the interrupted-run bar (serving restored) is inherited from
   D7's swap discipline rather than reimplemented.
4. **The scorecard is the machine-readable record.** Everything a comparative claim
   depends on (quant match, judge identity, prompt hash, pins, revisions) is in the
   metadata block; a scorecard whose two sides have mismatched quant types is refused
   at assembly time, enforcing the D7 confound rule mechanically.
5. **Judge verdicts favor robust parsing over structured-output features.** Plain-text
   rubric with a strict parser (or GBNF grammar on llama.cpp) rather than relying on
   server-side JSON modes; known vLLM Gemma 4 reasoning-parser and structured-output
   bugs make that path fragile (see Gemma 4 considerations).

## Gemma 4 considerations

Findings recent as of July 2026; Gemma 4 primary, Qwen3 fallback.

- **Thinking mode.** Gemma 4 enables thinking via a `<|think|>` control token in the
  system prompt and emits reasoning in a delimited thought channel before the answer.
  For deterministic scored evaluation (lm-eval, judge verdicts, generation under test)
  thinking should default to off (omit the token) and the on/off state must be recorded
  in the scorecard metadata — a thinking-on versus thinking-off comparison is a
  confound of the same kind as mismatched quantization. The F1 template registry
  already handles gemma4 thinking-control tokens for masking.
- **lm-evaluation-harness.** The harness has no gemma4-specific handling; tasks that
  use chat templating must apply the model's own template, and any thought-channel
  output must be stripped before answer extraction. Running thinking-off avoids this
  entirely and is the recommended pinned default in `tasks.yaml`.
- **Judge model on a 32GB RTX 5090.** The scope document's suggested judge, Gemma 4
  26B A4B (MoE, ~4B active), fits only quantized: QAT Q4_0 GGUF under llama.cpp
  comfortably; AWQ 4-bit or NVFP4 under vLLM at roughly 31GB, which leaves little
  headroom. Known vLLM issues (reasoning parser fails to separate reasoning content;
  reasoning parser plus disabled thinking silently breaks xgrammar structured output)
  argue for either llama-gpu serving the judge GGUF, or vllm-gpu with thinking off and
  plain-text verdict parsing. Decision deferred to the open questions.
- **Conversion caveat.** llama.cpp serves Gemma 4 (the export pin registers the
  architectures), but `convert_hf_to_gguf.py` has open gaps around some derivative
  checkpoints (unmapped tensors such as `layer_scalar`). This is F6's surface, not
  F7's; F7 only consumes GGUFs that already exported. Judge GGUFs should come from
  official or well-established quantized repositories, pinned by revision.
- **Qwen3 fallback.** No F7-specific work expected: the family registry, converter
  registration, and template handling already cover it; Qwen3 thinking control
  (`enable_thinking` / `/no_think`) differs from Gemma 4's token, so the
  thinking-off default must be implemented per family, not hardcoded.

## Proposed module layout

```
training/eval/__init__.py
training/eval/holdout.py            held-out loss on F1 splits, matched precision
training/eval/lmeval.py             lm-evaluation-harness wrapper, pinned tasks
training/eval/tasks.yaml            pinned task set and harness settings
training/eval/generate.py           serve-and-generate over judge prompt set
training/eval/judge.py              pairwise judging, position swap, verdict parsing
training/eval/judge_prompts.jsonl   fixed prompt set, held in git
training/eval/runner.py             phase sequencing, lock/compose preflights
training/eval/scorecard.py          scorecard.json / scorecard.md assembly
training/cli.py                     eval subcommand (cmd_eval)
scripts/eval/eval_ft.py             host-side entry (only if runner is host-side)
pyproject.toml / uv.lock            lm-eval pin (eval group), image rebuild
tests/training/test_eval_*.py       pairing, verdicts, scorecard, null-test logic
Makefile                            make eval-ft ARGS=...
```

## Work breakdown

1. Scorecard schema and assembly with unit tests (pure logic, sandbox-testable).
2. Held-out loss evaluator over F1 splits, plus `--baseline` plumbing.
3. lm-eval pin, `tasks.yaml`, wrapper, image rebuild (pin-then-fix: pin and verify the
   image builds before wiring the evaluator).
4. Judge prompt set, generation phase over `training/serving.py`, judge evaluator with
   position-swap and refusal accounting, unit tests.
5. Runner phasing, CLI subcommand, `make eval-ft`.
6. Host-side acceptance: null test, then the real base-versus-tuned run against the F8
   SFT checkpoint. All GPU/serving steps are host-executed and reported back per the
   sandbox rules.

## Acceptance criteria

From the scope document, restated against the repository:

- On a deliberately-good case (the F8 SFT run), the scorecard shows the fine-tuned
  model beating its base on held-out loss and judge win rate.
- A null test — a model evaluated against itself — produces deltas within noise and a
  judge win rate near 50 percent, validating the harness before any real result is
  trusted.
- Position-swapped pairing logic is covered by unit tests; judge refusals and
  malformed verdicts are counted and reported, never silently dropped.
- Interrupting the runner mid-judging leaves serving restored (inherits D7's
  kill-mid-swap bar).
- Repository additions: scorecard assembly refuses mismatched quant types; thinking
  mode is recorded per side and mismatches are refused the same way.

## Open questions

Owner decisions needed before implementation:

1. **Judge serving stack.** vllm-gpu per the scope document (needs Gemma 4 template and
   parser configuration, roughly 31GB for 26B A4B NVFP4/AWQ, known parser bugs), or
   llama-gpu with a QAT Q4_0 GGUF judge (reuses the existing swap machinery unchanged,
   comfortable VRAM, GBNF available for verdicts). Repository state favors llama-gpu;
   the scope document says vllm-gpu.
2. **Judge model pin.** Gemma 4 26B A4B (quantized, strongest local option) versus a
   Qwen3-family judge. Judging a Gemma 4 fine-tune with a Gemma 4 judge is a mild
   self-preference confound worth an explicit decision.
3. **Runner locus.** Wholly host-side (the `serve_eval.py` pattern, launching container
   phases via compose) versus container-side with host steps emitted for the serving
   phase. Host-side is simpler and matches D7 precedent.
4. **lm-eval dependency placement.** Separate `eval` dependency group (smaller training
   image unchanged) versus adding to the `training` group (one image, larger). The
   harness pulls sizable transitive dependencies.
5. **mmlu subset definition.** The scope document says "mmlu subset" without naming
   one; the pinned `tasks.yaml` must pick a concrete subset and stick to it.
