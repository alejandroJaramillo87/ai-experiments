# D7 train-and-serve round trip: implementation investigation

Comparison of the Phase 0 D7 design (roadmap document, "D7: Train-and-serve round
trip") against the current repository state, with a work breakdown, acceptance checks,
deviations, and open questions. Per operating rules, the repository is the source of
truth; where the roadmap document disagrees with the repository, this investigation
records the repository state and treats the document as advisory.

## Table of contents

- [Design summary](#design-summary)
- [Current repository state](#current-repository-state)
- [Deviations from the document](#deviations-from-the-document)
- [Work breakdown](#work-breakdown)
- [Acceptance checks](#acceptance-checks)
- [Risks](#risks)
- [Open questions](#open-questions)

## Design summary

D7 closes the loop the repository exists for: a trained checkpoint becomes a GGUF the
llama-gpu service serves, and an evaluation runs against the live service. Tag: build.
Depends on D1 (container, complete) and D6 (checkpoints, complete); training is idle
while serving runs, per the concurrency policy. The document specifies:

1. **Pinned llama.cpp for conversion.** A git SHA recorded in
   `training/export/llama_cpp_pin.txt` is fetched and built inside the export flow
   (or a `docker/training/Dockerfile.export` layer) for `convert_hf_to_gguf.py` and
   `llama-quantize`, independent of whatever the llama-gpu image was built from.
2. **Export pipeline** `scripts/export/export_gguf.py` (logic in `training/export/`):
   checkpoint -> [merge adapter if present, via PEFT `merge_and_unload`] -> HF
   safetensors dir -> `convert_hf_to_gguf` -> `llama-quantize` ->
   `/mnt/ai-data/models/trained/<run>/<name>-<quant>.gguf`, plus an export manifest
   (source checkpoint, recipe hash, llama.cpp SHA, quant type, file hash). Adapter
   merging into the base is PEFT's job; mergekit is Phase 1.
3. **Scope: GGUF-convertible architectures only** (Llama/Qwen/SmolLM families). The
   tool checks the model architecture against the pinned converter's support list and
   fails with a clear message otherwise.
4. **Serving swap** `scripts/eval/serve_eval.py`: preflight (training not running,
   model exists and hash-matches its manifest); record current `LLAMA_GPU_MODEL`;
   `LLAMA_GPU_MODEL=<gguf> docker compose up -d --force-recreate llama-gpu`; poll
   `http://localhost:8002/v1/health` (up to the 180 s start budget); run the
   evaluation set via `/v1/chat/completions`, recording responses, latencies, and
   token counts as JSON; restore the previous model unless `--keep-serving`. Any
   failure restores the previous model before exiting nonzero.
5. **Phase 0 evaluation is a fixed smoke prompt set** (about a dozen prompts with
   expected-substring checks). lm-eval and judge scoring are Phase 1, but the JSON
   output shape (per-prompt records plus a summary block) anticipates them.
6. **Quantization-confound rule**: comparative claims must compare at matched
   quantization; `serve_eval` stores the quant type in every result file.

Acceptance in the document: a stock base model (e.g. SmolLM2-1.7B) round-trips to
Q8_0 and answers the smoke prompts with serving state restored; a D6 tiny-run
checkpoint round-trips the same way; re-export yields the same GGUF hash under the
same pin; a kill-mid-swap leaves llama-gpu on the original model after cleanup or the
documented manual restore one-liner.

## Current repository state

D7 is greenfield for its two deliverables (`training/export/`, `scripts/eval/`), but
its contract surface exists and constrains the implementation.

- **Checkpoints are already export-ready.** `training/checkpoint.py` writes
  `checkpoints/<run>/step_N/model/` as HF `save_pretrained` sharded safetensors, with
  the header comment "D7 consumes this" (`training/checkpoint.py:7`). `meta.json`
  carries step, recipe snapshot hash, git SHA, and library versions — the inputs the
  export manifest needs.
- **Scratch models are in scope.** `training/model.py` builds scratch architectures
  as HF Llama configs (`build_scratch_model`), so both reference recipes and the
  tiny-e2e checkpoint satisfy decision 3 without extra work.
- **A llama.cpp pin already exists, but in the serving image.**
  `docker/llama-gpu/Dockerfile.llama-gpu:61` pins
  `LLAMA_CPP_COMMIT=e3546c7948e3af463d0b401e6421d5a4c2faf565` (tag b9976,
  2026-07-11). The document's `training/export/llama_cpp_pin.txt` does not exist;
  the export pin is a second, independently bumped pin by design.
- **The serving swap contract matches the document.** `docker-compose.yaml` passes
  `MODEL_PATH=${LLAMA_GPU_MODEL}` to llama-gpu (line 56), the healthcheck curls
  `http://localhost:8002/v1/health` with `start_period: 180s`, the API port binds
  `127.0.0.1:8002`, and `docker/llama-gpu/entrypoint.sh:6` supplies the default
  model when `LLAMA_GPU_MODEL` is unset. Models are mounted read-only at
  `/app/models` from `/mnt/ai-data/models/`.
- **The training container cannot write the document's export destination.**
  `docker-compose.training.yaml` mounts `/mnt/ai-data/models/:/app/models:ro` and
  the only writable data root is `/mnt/ai-data/training:rw`. The document's target
  `/mnt/ai-data/models/trained/<run>/` is read-only from inside the training
  container (see deviations).
- **The D9 lock does not exist.** No lock file or lock module is present anywhere in
  the repository; the preflight "assert training is not running (no lock file, see
  D9)" has nothing to check yet.
- **PEFT is pinned (`peft==0.19.1` in `pyproject.toml`) but unused.** No recipe
  field, loop branch, or checkpoint layout produces adapters in Phase 0; the
  adapter-merge branch of the export pipeline has no in-repo artifact to exercise.
- **Established command patterns.** Training tools run in-container via Make targets
  (`make train CONFIG=...` wraps `docker compose -f docker-compose.training.yaml run
  --rm training python -m training <subcommand>`); the `training` package exposes
  subcommands through `training/cli.py` and `training/__main__.py`. Unit tests live
  in `tests/training/` and run in the sandbox. Python CLI output follows
  `.styles/PYTHON_OUTPUT_STYLE.md`.
- **No GGUF tooling exists anywhere in the repository** — no converter, no `gguf`
  Python dependency, no quantize binary outside the llama-gpu builder stage (which
  only copies out `llama-server`).

## Deviations from the document

1. **Export output path vs. mounts.** The document writes GGUFs to
   `/mnt/ai-data/models/trained/<run>/`, but the training container mounts the
   models tree read-only. Either the training compose gains a narrow rw mount
   (`/mnt/ai-data/models/trained`), or exports land under
   `/mnt/ai-data/training/exports/` and llama-gpu gains a read-only mount of that
   path. The llama-gpu side requires the file to be visible under `/app/models`,
   so the first option changes one line and keeps the serving config untouched
   (recommended; see open questions).
2. **D9 lock preflight.** Until D9 lands, "no lock file" is vacuously true. The
   honest substitute is checking that no `training` compose service container is
   running (`docker compose -f docker-compose.training.yaml ps -q training`), plus
   probing the lock path D9 will use so the preflight is forward-compatible.
3. **Adapter merge is untestable in Phase 0.** The PEFT `merge_and_unload` branch
   can be implemented and unit-tested against a synthetic adapter, but no D6
   artifact exercises it; the acceptance checks cover only the full-model path.
   This is consistent with the document (adapters arrive with Phase 1 SFT).
4. **`serve_eval.py` is a host tool, not a container tool.** It drives
   `docker compose` and curls `localhost:8002`, so unlike every existing training
   script it runs on the host, outside any image. It should therefore depend only
   on the Python standard library (no `requests`, no repo venv assumption).
5. **Checkpoint path convention.** The document's example checkpoint root
   (`/mnt/ai-data/models/...` in the skill text) is stale; checkpoints actually
   live under `recipe.run.output_root` (default `/mnt/ai-data/training/...`,
   `training/loop.py:137`). The export tool takes a checkpoint directory argument
   and does not guess.

## Work breakdown

1. **`training/export/` package** (in-container, sandbox-testable logic):
   - `llama_cpp_pin.txt` — exact git SHA; initial value proposed as the llama-gpu
     serving pin `e3546c7948e3af463d0b401e6421d5a4c2faf565` (b9976) so conversion
     and serving start from the same revision (see open questions).
   - `merge.py` — load checkpoint `model/`; if a PEFT adapter is present, merge via
     `merge_and_unload`; write a clean HF safetensors staging dir.
   - `convert.py` — architecture support check against the pinned converter's
     registry, then subprocess calls to `convert_hf_to_gguf.py` and
     `llama-quantize`; deterministic flags only.
   - `manifest.py` — export manifest JSON: source checkpoint path and `meta.json`
     digest, recipe hash, llama.cpp SHA, quant type, output SHA-256, tool versions.
2. **Converter toolchain in the training image** (or a dedicated
   `Dockerfile.export` stage): clone llama.cpp at the pin, build `llama-quantize`
   (CPU-only build suffices), install the converter's Python requirements (`gguf`)
   pinned through `uv.lock`.
3. **CLI wiring**: `python -m training export --checkpoint <dir> --quant Q8_0
   [--out ...]` following `training/cli.py` conventions; Make target
   `export-gguf CHECKPOINT=... [QUANT=Q8_0]`.
4. **`scripts/eval/serve_eval.py`** (host, stdlib-only): preflight, env capture,
   compose recreate, health poll with 180 s budget, smoke-prompt run over
   `/v1/chat/completions`, JSON results (per-prompt records, summary block, quant
   type, model hash), restore on every exit path (`try/finally` plus signal
   handling for the kill-mid-swap bar), `--keep-serving`, and the documented manual
   restore one-liner in `--help` and docs.
5. **Smoke prompt set**: a fixed JSON file (about 12 prompts with
   expected-substring checks) under `scripts/eval/` or `training/eval/`.
6. **Tests** (`tests/training/test_export.py`): manifest determinism, architecture
   rejection message, adapter-merge branch against a synthetic tiny adapter, output
   naming. Conversion itself needs the toolchain and is covered by the host
   acceptance run, not sandbox unit tests.
7. **Docs**: `docs/training/train_serve_roundtrip.md` (pipeline, quantization-
   confound rule, manual restore one-liner) and an update to the `training-runs`
   skill section that is currently marked "intended procedure".

## Acceptance checks

All host-side; the human runs each command and reports output back.

1. Stock base model export and serve: `make export-gguf CHECKPOINT=<SmolLM2-1.7B
   safetensors dir> QUANT=Q8_0`, then `scripts/eval/serve_eval.py --model <gguf>`;
   smoke prompts pass; `docker compose ps` and a follow-up prompt confirm the
   original model is restored.
2. D6 tiny-run checkpoint round-trips the same way (from-checkpoint path).
3. Reproducibility: export the same checkpoint twice; `sha256sum` of both GGUFs
   match; manifests differ only in timestamp.
4. Kill-mid-swap: SIGINT `serve_eval.py` during the health poll; llama-gpu is back
   on the original model after cleanup (or via the documented one-liner).

## Risks

- **GGUF byte-reproducibility** is expected (converter and quantizer are
  deterministic for a fixed pin and input) but unverified on this stack; if
  metadata (e.g. conversion timestamps) leaks into the file, the acceptance bar
  falls back to comparing tensor-data hashes and the manifest records why.
- **Converter support for the D6 scratch config**: scratch models are HF Llama, so
  support is expected, but tiny dimensions (head counts, rope settings) can trip
  converter assertions; the tiny-e2e round trip is the test.
- **Restore correctness under compose**: `--force-recreate` with a changed
  environment variable must not be run while `.env` disagrees in a way that a later
  unrelated `make up` would silently revert; `serve_eval` restores by re-running
  compose with the recorded previous value and never edits `.env`.
- **Toolchain weight**: building llama.cpp inside the training image adds build
  time; a separate small export stage keeps the training image unchanged but adds
  a second image to maintain.

## Open questions

1. **Export destination and mounts** (deviation 1): add a narrow rw mount of
   `/mnt/ai-data/models/trained` to `docker-compose.training.yaml` (recommended),
   or write under `/mnt/ai-data/training/exports/` and add a ro mount to llama-gpu?
2. **Initial export pin**: reuse the llama-gpu serving commit b9976
   (`e3546c79...`) so converter and server agree on format support, or pin the
   latest llama.cpp at implementation time?
3. **Toolchain placement**: extend `docker/training/Dockerfile.training` with the
   converter and `llama-quantize` (one image, heavier), or a separate
   `docker/training/Dockerfile.export` (document's alternative, second image)?
4. **Adapter-merge branch now or stubbed**: implement and unit-test the PEFT merge
   path in D7 (document's letter), or fail loudly on adapters and defer the branch
   to Phase 1 when a real adapter artifact exists?
