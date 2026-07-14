# F4 teacher and synthetic data harness: implementation investigation

Comparison of the Phase 1 F4 design (roadmap document, "F4: Teacher and synthetic
data harness") against the current repository state, with a work breakdown,
acceptance checks, deviations, and open questions. Per operating rules, the
repository is the source of truth; where the roadmap document disagrees with the
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

F4 batch-generates teacher responses and synthetic instruction data through the
existing inference services, producing dataset artifacts that later training
(F5 distillation, F3 SFT) consumes. It runs as its own phase: inference up,
training idle. Tag: orchestrate. Depends on D7 (serving swap), D9 (lock), and the
inference services. The document specifies:

1. **distilabel as the pipeline engine**, pointed at the local OpenAI-compatible
   endpoints (vllm-gpu 8003 preferred for batch throughput; llama-gpu 8002 as
   fallback). The harness contributes endpoint configuration and health preflight,
   the serving-swap step (reuse D7 machinery to put the intended teacher model up
   and restore afterward), service-appropriate concurrency (high for vLLM, low for
   llama.cpp), retry with backoff, and resumability: generation state checkpoints
   so a multi-hour run survives interruption without redoing completed prompts.
2. **Two built-in pipeline shapes**, both thin distilabel graphs: `respond`
   (prompts in, teacher responses out — the distillation dataset) and `synthesize`
   (seed instructions in, evolved instruction/response pairs out, self-instruct
   style). Custom pipelines can be dropped in; the harness owns I/O and serving,
   not pipeline authorship.
3. **Output is a normalized F1-format dataset** under
   `/mnt/ai-data/training/datasets/distill/<name>/`: messages-format JSONL plus a
   manifest recording teacher model and quantization, service, generation
   parameters, prompt-set hash, distilabel version, and timestamps. The manifest's
   teacher-identity block is what F7 cites when claiming a student improved
   against a given teacher.
4. **Explicit sampling parameters** in the harness config (temperature, top-p,
   max tokens, optional thinking-mode control for Gemma 4 teachers); defaults
   favor distillation (low temperature, long max tokens). The config is
   snapshotted into the manifest.

Interface in the document: `scripts/train/generate_teacher.py --pipeline respond
--prompts <sft dataset> --teacher <model> --service vllm-gpu --out
/mnt/ai-data/training/datasets/distill/<name>`, wrapped by
`make generate-teacher ARGS=...`. Module: `training/data/teacher.py`.

Acceptance in the document: a 1,000-prompt `respond` run against vllm-gpu
completes, writes a valid F1-format dataset with a complete manifest, and
restores prior serving state; killing the run at 50 percent and re-invoking
completes the remaining prompts only; the harness refuses to start while the D9
training lock is held; a `synthesize` run from 50 seed instructions produces a
dataset that `prepare_sft.py` ingests without modification.

## Current repository state

The dependencies F4 names are implemented; the surfaces F4 writes to are not.

1. **D9 lock is complete** (`training/lock.py`). `read_lock()` returns the holder
   dict or `None` (`training/lock.py:36`); `run_lock()` is the acquiring context
   manager used by the training loop (`training/loop.py:120`). Lock path is
   `/mnt/ai-data/training/.train.lock`, overridable via `TRAIN_LOCK_PATH`.
   The read-and-refuse preflight pattern F4 needs already exists in
   `scripts/eval/serve_eval.py:140`: serving-side consumers check the lock file
   but do not acquire it, because they are a different GPU consumer than
   training.
2. **D7 serving swap is complete** (`scripts/eval/serve_eval.py`). The mechanism
   is a compose recreate with an environment override: set `LLAMA_GPU_MODEL` in a
   copied environment, `docker compose up -d --force-recreate llama-gpu`
   (`serve_eval.py:174-184`), restore by recreating with the variable popped so
   compose falls back to `.env`. Health polling (`wait_healthy`,
   `serve_eval.py:187`), a stdlib `urllib` chat-completions client
   (`run_prompt`, `serve_eval.py:206`), SIGTERM-safe restore in `try/finally`,
   and the training-lock plus training-container preflight all exist there.
   vllm-gpu is parameterized the same way (`VLLM_GPU_MODEL`,
   `docker-compose.yaml:119`), so the swap pattern generalizes, but only the
   llama-gpu swap is implemented today.
3. **Health endpoints are not uniform**: llama-gpu answers `/v1/health` on 8002;
   vllm-gpu answers `/health` (no `/v1`) on 8003 (`docker-compose.yaml`
   healthchecks). Both serve OpenAI `/v1/chat/completions`.
4. **F1 does not exist.** There is no `prepare_sft.py`, no messages-format
   loader, no chat-template registry. `training/data/prepare.py` is the
   pretraining pipeline: it tokenizes HF datasets or flat `{"text": ...}` JSONL
   into memory-mapped token shards under `/mnt/ai-data/training/shards/<name>`
   with a `manifest.json` (fields at `training/data/prepare.py:226-243`,
   including `tokenizer_hash`, `created_at`, `git_sha`, `command`; it refuses to
   overwrite an existing manifest). Nothing in the repository defines
   "F1-format" or writes to `/mnt/ai-data/training/datasets/`.
5. **Scripts live in `scripts/training/`**, not the document's `scripts/train/`
   (`prepare_data.py`, `train_tokenizer.py`, `validate_recipe.py`,
   `smoke_test.py`, `pin_run.sh`). Entry points are argparse CLIs; the recipe
   loader deliberately avoids `hydra.main` (`training/cli.py:6`). Make targets in
   the `##@ Training` section run tools inside the training container via
   `docker compose -f docker-compose.training.yaml run --rm training`, except
   `serve-eval`, which runs on the host because it drives docker compose itself
   (`Makefile:156`).
6. **Neither distilabel nor the openai client is a dependency.** `pyproject.toml`
   and `uv.lock` contain no reference to either. Training-only libraries live in
   the `[dependency-groups] training` group with exact `==` pins; the training
   image installs that group with `uv sync --frozen` (`docker/training/
   Dockerfile.training:76`). The existing HTTP precedent is stdlib-only:
   `serve_eval.py` uses `urllib.request` deliberately.
7. **Tests** for training modules are pure-logic CPU tests under
   `tests/training/`, one `test_<module>.py` per module, with GPU and
   live-service paths deferred to host-run acceptance checks
   (`tests/training/test_export.py:1-8` documents the split).

## Deviations from the document

1. **Script location**: `scripts/training/generate_teacher.py`, not
   `scripts/train/generate_teacher.py`. The repository has a `scripts/training/`
   directory and no `scripts/train/`.
2. **F4 defines the messages-dataset contract, F1 inherits it.** The document
   has F4 emitting an "F1-format" dataset, but F1 is unimplemented on this
   branch's timeline. The dependency graph in the document already shows F4
   independent of F1, so F4 will define the canonical record format — OpenAI-style
   `{"messages": [{"role": ..., "content": ...}]}`, one JSON object per line —
   and the manifest conventions, mirroring the existing pretraining manifest
   fields where they apply. When F1 lands, `prepare_sft.py` ingests this format;
   the acceptance check "prepare_sft.py ingests without modification" is
   deferred until F1 exists (see open questions).
3. **Serving swap scope.** D7 machinery swaps llama-gpu only. The document
   prefers vllm-gpu for batch generation, so F4 needs the same env-override
   recreate for `VLLM_GPU_MODEL`, plus the non-uniform health path (`/health`
   versus `/v1/health`). This is an extension of D7's pattern, not a reuse of
   its code as-is; the swap and restore logic in `serve_eval.py` should be
   extracted into a shared module rather than duplicated.
4. **Lock behavior**: read-and-refuse, matching `serve_eval.py`, not acquire.
   F4 is a serving-side GPU consumer; acquiring the training lock would
   misrepresent who holds the GPU. Serialization against other serving-side
   consumers (a benchmark, serve_eval) remains the operator's responsibility,
   per the one-GPU-consumer operating rule.
5. **distilabel is unconfirmed as the engine.** The repository precedent for
   driving the inference services is a deliberate stdlib-only client
   (`serve_eval.py`), and the dependency discipline (exact pins, one group at a
   time) makes a heavy framework with its own `pydantic`/`datasets`/`httpx`
   constraints a real integration cost. The two built-in shapes (`respond`,
   `synthesize`) are thin enough to implement over a small async or threaded
   OpenAI-compatible client with retry, backoff, and checkpointing — which F4
   must build anyway, since those are listed as harness contributions. Whether
   distilabel earns its footprint is an open question for the owner; the work
   breakdown below is written so the pipeline engine is replaceable behind
   `training/data/teacher.py`.

## Work breakdown

1. **Dependency pin** (`pyproject.toml` training group, `uv.lock` regenerated on
   the workstation): `openai` client pinned exactly, plus `distilabel` if the
   open question resolves in its favor. One group bump, no functional change in
   the same step.
2. **`training/data/teacher.py`** — the harness core, unit-testable without
   services:
   - `TeacherConfig` dataclass: service (`vllm-gpu` | `llama-gpu`), base URL,
     health path, concurrency, retry/backoff policy, sampling block
     (temperature, top-p, max tokens, thinking-mode flag), pipeline shape,
     teacher model identity.
   - Prompt-set loading: seed JSONL (messages or plain instruction records) with
     a stable per-prompt key (index plus content hash) used for resume.
   - Resumability: a `state.jsonl` (or completed-key set) in the output
     directory appended per completed prompt; on start, completed keys are
     skipped. Output JSONL is append-only per completed record so a kill leaves
     a consistent prefix.
   - Record normalization: whatever the engine returns becomes
     `{"messages": [...]}` lines.
   - Manifest writer: teacher model and quantization, service, generation
     parameters (snapshotted config), prompt-set hash, engine and version,
     counts, timestamps, `git_sha`, `command` — following
     `training/data/prepare.py` manifest conventions, refusing to overwrite a
     complete manifest.
3. **Serving-swap extraction**: move the compose swap, restore, preflight, and
   health-poll logic from `scripts/eval/serve_eval.py` into a shared module
   (for example `scripts/lib/serving.py` or `training/serving.py` — see open
   questions), parameterized by service name, env var, port, and health path;
   `serve_eval.py` becomes a consumer of it. vllm-gpu support added here.
4. **`scripts/training/generate_teacher.py`** — host-run argparse CLI (it drives
   docker compose, like `serve-eval`): flags per the document interface
   (`--pipeline`, `--prompts`, `--teacher`, `--service`, `--out`, sampling
   overrides, `--resume` implied by default). Preflight: D9 lock absent,
   training container not running, target service healthy or swappable.
   Styled output per the Python output style guide.
5. **Pipeline shapes**: `respond` first (prompts in, responses out); `synthesize`
   second (seed instructions in, evolved pairs out). If distilabel is adopted,
   these are thin graphs; otherwise a bounded-concurrency client loop.
6. **`make generate-teacher ARGS=...`** in the `##@ Training` Makefile section,
   host-run like `serve-eval`.
7. **Tests** (`tests/training/test_teacher.py`): config validation, prompt-key
   stability, resume-skip logic against a synthetic completed set, record
   normalization, manifest completeness and overwrite refusal — all CPU, no
   services. Live-endpoint behavior is host acceptance.
8. **Docs**: user-facing `docs/training/teacher_synthetic_data.md` written at
   implementation time, after behavior exists to document.

## Acceptance checks

Host-run by the operator, output reported back verbatim:

1. 1,000-prompt `respond` run against vllm-gpu completes; the output directory
   contains messages-format JSONL and a complete manifest; prior serving state
   is restored afterward (`docker compose ps` before and after match).
2. Kill the run at roughly 50 percent (SIGTERM); re-invoke; the log shows only
   the remaining prompts generated; final record count equals the prompt count
   with no duplicate keys.
3. With `/mnt/ai-data/training/.train.lock` present, the CLI refuses to start
   with a clear message and nonzero exit.
4. `synthesize` from 50 seed instructions produces a dataset in the same format
   with a valid manifest. (The document's `prepare_sft.py` ingestion check is
   deferred to F1; the format contract is asserted by unit tests now.)

## Risks

1. **distilabel dependency resolution** against the pinned core
   (`pydantic`, `datasets==5.0.0`, `fsspec`) may force unrelated bumps,
   violating the one-bump-at-a-time rule; this is the main argument for the
   thin-client alternative.
2. **vLLM swap is untested machinery**: recreating vllm-gpu with a different
   model has a long cold start (weight load) and different failure modes than
   llama-gpu; the health-poll budget needs to be service-specific.
3. **Multi-hour unattended runs**: retry and backoff must distinguish transient
   service errors from a downed service; a downed service should trigger restore
   and a resumable exit, not an infinite retry loop.
4. **Thinking-mode control for Gemma 4 teachers** depends on how the serving
   stack exposes it (chat-template kwargs versus system prompt); needs a probe
   against the real service before the config field is fixed.

## Resolved decisions

Resolved by the owner, 2026-07-14:

1. **Pipeline engine: thin `openai`-client loop, not distilabel.** Pin only the
   `openai` client in the training dependency group; the harness implements
   bounded concurrency, retry with backoff, and resumability itself (required
   contributions regardless of engine). The manifest records the client library
   and version in place of a distilabel version.
2. **Serving swap: yes, document behavior.** `generate_teacher.py` swaps the
   teacher model in via the extended D7 mechanism (env-override compose
   recreate, per-service env var and health path) and restores prior serving
   state on every exit path.
3. **Shared serving module lives at `training/serving.py`**, unit-testable
   under `tests/training/`; `scripts/eval/serve_eval.py` becomes a consumer.
4. **F4 owns the messages-JSONL and manifest contract**; F1 inherits it. The
   `prepare_sft.py` ingestion acceptance check is deferred until F1 exists;
   the format is asserted by F4 unit tests now.

## Open questions

1. **Dataset root**: confirm `/mnt/ai-data/training/datasets/distill/<name>/` as
   a new convention beside the existing `shards/` root.
