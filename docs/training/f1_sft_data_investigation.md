# F1 chat and instruction data preparation: implementation investigation

Comparison of the Phase 1 F1 design (product design document, "F1: Chat and
instruction data preparation") against the current repository state, with a work
breakdown, acceptance checks, deviations, and open questions. Per operating rules,
the repository is the source of truth; where the design document disagrees with the
repository, this investigation records the repository state and treats the document
as advisory.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Deviations from the document](#deviations-from-the-document)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Resolved decisions](#resolved-decisions)
- [Open questions](#open-questions)

## Design summary

F1 turns chat and instruction datasets into tokenized, loss-masked training
examples per supported model family. It is the highest-risk correctness surface in
Phase 1: a wrong template or mask silently destroys a fine-tune while the loss
curve looks healthy. Tag: build. Depends on D1 (training container, complete) and
F2 (recipe schema extension, complete). The document specifies:

1. **Canonical record format.** All SFT data normalizes to the OpenAI-style
   messages list — `{"messages": [{"role": ..., "content": ...}, ...]}` — one JSON
   object per line. Loaders accept Hugging Face datasets or local JSONL in this
   shape, plus a converter for prompt/completion pairs. Multi-turn is the native
   case; single-turn is a special case of it.
2. **Per-family chat-template registry.** A registry maps a model family to (a) the
   template application function and (b) the assistant-span locator used for
   masking. The default implementation delegates to the tokenizer's own
   `apply_chat_template`; the registry exists because masking needs to know where
   assistant spans begin and end, which the generic template API does not expose
   uniformly. Unknown families fail loudly with registration instructions; there is
   no silent generic fallback.
3. **Prompt-loss masking.** Labels are the input ids with every non-assistant token
   set to -100, computed from the registry's span locator, not from string matching
   on rendered text. Masking covers system and user turns, the assistant role
   header, and template control tokens; loss applies to assistant content tokens
   and the turn's end-of-turn token only. A `mask_debug(sample)` helper renders a
   token-by-token table (token, id, label) for eyeball verification.
4. **Sequence packing.** Multiple short examples pack into one `max_seq_len`
   sequence with correct attention separation, via TRL's packing rather than a
   custom implementation. Packing on/off is a recipe flag.
5. **Materialized artifacts.** `prepare_sft.py` writes the tokenized, masked
   dataset to `/mnt/ai-data/training/datasets/sft/<name>/` with a manifest (source,
   license, tokenizer hash, template registry version, example counts, token
   counts, held-out split seed). Recipe cards reference materialized artifacts; a
   seeded held-out split (default 2 percent) is always reserved for F7.

## Current repository state

F1 is greenfield: no chat-template application, messages parsing, or prompt-loss
masking exists anywhere in the repository. The landing zones are already built:

- `training/families.py` — the model-family registry (F2), keyed by HF config
  `model_type`. Its docstring designates it as the home of the F1 chat-template
  column on `FamilySpec`. Registered families: `gemma4`, `gemma4_text` (Gemma 4
  E-models, the Phase 1 primary), and `qwen3` (the fallback family). Unknown
  families raise `UnknownModelFamilyError` with registration instructions —
  exactly the fail-loud contract the F1 design asks for.
- `training/data/prepare.py` and `training/data/shards.py` — the Phase 0 D2
  pretraining pipeline. It reads a single flat text field into EOS-delimited token
  shards. Per the Phase 0 contract, SFT data does not reuse token shards but does
  reuse the manifest discipline (source, counts, tokenizer hash, git SHA, content
  hash; loaders refuse a dataset whose manifest hash does not match).
- `training/data/loader.py` — flat token-window loader with label = input; no
  masking support. F1's loader is a separate surface, not an extension of this one.
- `training/recipes/schema.py` — the F2 schema with the `peft` group. There is no
  `sft` group yet (see deviations). `validate()` in `training/recipes/validate.py`
  holds the cross-field invariants and gains the F1 ones.
- `training/recipes/reference/sft-qlora-e2b.yaml` — the F2 QLoRA reference recipe
  (Gemma 4 E2B), whose `data.source` points at a placeholder path that F1's
  prepared artifacts replace.
- `training/data/tokenizer.py` reserves a 16-id token block so chat/control tokens
  never require `resize_token_embeddings` — consistent with the Phase 1 convention
  that embeddings are never resized and each base model's existing template and
  special tokens are used as-is.
- `tests/training/` — CPU-only pytest suite with an on-the-fly tiny-tokenizer
  fixture in `conftest.py`; the F1 golden-mask tests follow this pattern.
- CLI convention: existing data CLIs live in `scripts/training/` (for example
  `scripts/training/prepare_data.py`) with a Makefile target in the training
  section.

## Deviations from the document

1. **Registry location and keying.** The document places the template registry in a
   new `training/data/templates.py` keyed by family names like `gemma4` and
   `llama3`. The repository already has the registry: `training/families.py`,
   keyed by HF `model_type`. F1 extends `FamilySpec` with the chat-template column
   rather than creating a parallel registry.
2. **Family set.** The document registers `gemma4` and `llama3`. Repository policy
   (this investigation, decision 1 below): Gemma 4 E-models (`gemma4`,
   `gemma4_text`) are primary; Qwen3 is the single fallback; no Llama families are
   registered because no Llama weights are available to this workstation.
3. **No `sft` recipe group exists.** The document's F2 was scoped to add `sft`,
   `peft`, and `distill` groups; the merged F2 added only `peft`. The document's
   library surface `load_sft_dataset(recipe.sft.data)` therefore has no schema to
   bind to. F1 adds a new `SftConfig` group (decision 2 below).
4. **Packing is deferred to F3.** The document makes packing an F1 recipe flag
   implemented via TRL. TRL is not yet a dependency anywhere in `training/`, and
   packing is a trainer-side concern in TRL's API. F1 ships unpacked, mask-correct
   examples; F3 owns packing and its masking-correctness tests (decision 3 below).
5. **CLI path.** The document says `scripts/train/prepare_sft.py`; the repository
   convention is `scripts/training/`, so the CLI lands at
   `scripts/training/prepare_sft.py`.
6. **Model naming, verified July 2026.** Gemma 4 is its own transformers
   architecture: `model_type` `gemma4` (multimodal wrapper) and `gemma4_text`
   (text config), per `configuration_gemma4.py` in transformers (contributed
   2026-04-02). It is not Gemma 3n. The text decoder keeps the Llama-lineage
   projection names (a new `relative_k_proj` exists and is not a LoRA target).
   The chat format, from the official `chat_template.jinja` (mirrored in
   llama.cpp `models/templates/`) and `convert_gemma4_weights.py`: BOS once,
   then `<|turn>{role}\n{content}<turn|>\n` per turn with a native system
   turn, assistant role named `model`, and content trimmed. Thinking mode uses
   `<|think|>` in the system turn and `<|channel>thought\n...<channel|>`
   reasoning sections that the template strips from conversation history, so
   rendered training conversations contain no thinking tokens.

## Work breakdown

1. **Extend `FamilySpec`** in `training/families.py` with the chat-template column:
   a template application function (default: delegate to the tokenizer's
   `apply_chat_template`) and an assistant-span locator returning token-index spans
   for each assistant turn. Entries for `gemma4`/`gemma4_text` and `qwen3`.
   Unknown families keep failing through `family_spec()`.
2. **New module `training/data/sft.py`**: messages-record validation and
   normalization (HF dataset or local JSONL), prompt/completion-pair converter,
   template rendering, span-based label masking (-100 on non-assistant tokens),
   `mask_debug(sample)` token table, seeded held-out split, and
   `load_sft_dataset(recipe)` returning train and held-out splits.
3. **Schema extension**: a new `SftConfig` group (dataset reference, holdout
   fraction and seed, max sequence length policy) plus `validate()` invariants
   (for example: sft group requires an HF model, holdout in (0, 1), manifest-hash
   refusal on mismatch).
4. **CLI and Makefile target**: `scripts/training/prepare_sft.py` writing to
   `/mnt/ai-data/training/datasets/sft/<name>/` with a D2-style manifest extended
   with license, template registry version, and split seed; `make prepare-sft`.
   Deterministic re-runs (identical manifest hashes).
5. **Tests** in `tests/training/test_sft.py`: golden-fixture exact-mask tests per
   registered family (single-turn, multi-turn, system prompt), round-trip
   detokenization, determinism of the prepared artifact and split membership, and
   the unknown-family failure. All CPU-only and sandbox-runnable.

## Acceptance checks

- Golden-fixture tests per registered family assert exact label masks for
  single-turn, multi-turn, and system-prompt cases: loss applies only to assistant
  content tokens and the end-of-turn token.
- A round-trip test detokenizes a prepared sample and reproduces the
  template-rendered conversation exactly (no lost or duplicated control tokens).
- Re-running `prepare_sft.py` with the same inputs is deterministic (identical
  manifest hashes), including held-out split membership.
- An unknown model family fails with the registration message.

All of the above run in the sandbox via `python -m pytest tests/training/`. The
only host-side F1 check is preparing a real dataset (network fetch of the source
dataset and the gated Gemma 4 tokenizer) and inspecting `mask_debug` output.

## Risks

- **Gemma 4 template drift.** The `ChatTemplateSpec` was derived from the
  official `chat_template.jinja` (July 2026); the remaining host-side check is
  tokenization parity against the real gated tokenizer's `apply_chat_template`
  before golden fixtures are frozen, re-run on any `transformers` pin bump.
- **Span location robustness.** Locating assistant spans by token indices requires
  the template entry to know each family's turn delimiters; a `transformers`
  version bump can change rendering. The golden-fixture tests are the tripwire,
  per the pin-then-fix-then-update rule.
- **PLE architecture support.** TRL/PEFT support for the Gemma 4 E-models must be
  confirmed in the container smoke test before any long run (owned jointly with
  F3); the registered Qwen3 fallback exists so this never blocks the phase.

## Resolved decisions

1. **Family policy: Gemma 4 primary, Qwen3 fallback, no Llama.** The workstation
   has no access to Llama weights. The registry in `training/families.py` is
   trimmed to `gemma4`, `gemma4_text`, and `qwen3`; the F2 reference recipe now
   targets `google/gemma-4-E2B` (`sft-qlora-e2b.yaml`). The D4 calculator's
   `HF_SUPPORTED_MODEL_TYPES` is unchanged: it states which config shapes the
   memory formulas can model, not which models are trained.
2. **Recipe shape: new `SftConfig` group.** SFT fields land in a new group rather
   than being folded into the pretraining `DataConfig`, keeping the Phase 0 shard
   path untouched.
3. **Packing: owned by F3.** F1 produces unpacked, mask-correct examples; the
   packing flag, TRL integration, and packed-mode mask tests move to F3.
4. **Materialized artifact format: JSONL.** `train.jsonl` and `holdout.jsonl`,
   one `{"input_ids", "labels"}` object per line. JSONL is inspectable,
   hash-deterministic across re-runs (the determinism acceptance check compares
   the manifest's per-file content hashes), and loads without the `datasets`
   dependency; F3 can convert to arrow at trainer setup if profiling shows the
   need.

## Open questions

1. **Reference SFT dataset.** The document's example is
   `HuggingFaceH4/ultrachat_200k`; the choice of the first real dataset (and its
   license string for the manifest) is deferred to the F1 acceptance run.
2. **Empty thought channel at inference.** With thinking disabled, the
   official generation prompt ends `<|turn>model\n<|channel>thought\n<channel|>`
   — an empty thought — while rendered conversation history carries no channel
   tokens. F1 renders training turns exactly as the template renders history
   (no channel tokens), matching `apply_chat_template` output. Whether the
   final assistant turn of a training example should instead start with a
   masked empty-thought prefix to mirror the inference prompt is an F3
   decision, checked during the host-side smoke fine-tune; a spec change bumps
   `CHAT_TEMPLATE_VERSION`.
