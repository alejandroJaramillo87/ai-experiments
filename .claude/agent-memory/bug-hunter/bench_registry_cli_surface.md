---
name: bench-registry-cli-surface
description: What bench/registry.py validates and what it does not, how the model identity in a bench_result is asserted rather than observed, and which bench CLI probes are dead ends.
metadata:
  type: project
verified: 2026-08-29
stale_when: bench/registry.py grows a second MODELS entry, common/serving.py::quant_from_name learns the IQ/TQ families, or bench/commands/models.py stops reusing ModelSpec.conditions() as its --json payload
---

`bench/registry.py` is the only load-time validation surface in `bench/`. Its input is a
module-level dict (`MODELS`), so every gap is reachable by a hand edit and by nothing else.

**Confirmed against the tree 2026-08-29:**

- **The quantization confound guard never fires on the shipped entry.**
  `common/serving.py::quant_from_name` recognises only `Q*` and `BF16/F16/F32`. The whole
  IQ family (`IQ4_XS`, `IQ2_M`, ...) and TQ family return `None`, and the one registry
  entry is `Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf`. `_validate` skips the check on
  `None`, so `quantization: "Q8_0"` on that path loads clean. The committed test that
  covers the guard uses a `Q4_K_M` path, so the suite proves the guard on a shape the
  registry does not contain.
- **`conditions()` is an assertion, not an observation.** `spec.path` is read nowhere
  except `ModelSpec.conditions()` (`grep -rn "\.path\b" bench/` — no `exists`/`stat`/`open`
  anywhere in `bench/`). `runner.gate` polls `/v1/health` only. `delta.BLOCKING` compares
  `conditions.model.{name,quantization,architecture,context_length}` — every one of them
  from the registry — while the observed `conditions.env_knobs.MODEL_PATH` and
  `server_argv` are compared by nothing. `training/commands/eval_ft.py`,
  `serve_eval.py` and `generate_teacher.py` all call `common.serving.swap_model` on the
  same services `bench` targets, so the served model does change under it.
- **`_validate` checks presence and two enums only.** Accepted without complaint:
  `engines` as a bare string (splits to a per-character tuple), `engines: []`,
  an engine name absent from `common.serving.SERVICES`, `params_active_b > params_total_b`,
  `n_experts_active > n_experts`, `context_length` as a string, a negative
  `params_total_b`. `engines: []` also passes `test_every_engine_named_is_a_real_service`
  vacuously (empty loop body).
- **`bench models --json` reuses `ModelSpec.conditions()`, which has no `engines` key.**
  So the machine-readable listing cannot answer the routing question `resolve()` decides,
  and `docs/verification/bench/bench.md` B3 states its acceptance signal on `engines`.

**Dead ends — do not refile these:**

- *`bench models --json` swallows the error message on a bad registry.* Real
  (`return 1`, empty stdout, empty stderr) but it is the documented design:
  `.styles/PYTHON_OUTPUT_STYLE.md` says quiet mode suppresses the human line and keeps
  the structured record, and the record does carry the message when `OBS_LOG_DIR` exists.
- *A duplicate slug in `MODELS` silently collapses.* `ruff check` flags it
  (`multi-value-repeated-key-literal`) and `make quality` runs ruff over the tree.
- *`vllm-gpu` has no registered model, so `bench run --service vllm-gpu` always exits 2.*
  Recorded and accepted at `docs/verification/bench/bench.md` B3 ("that refusal is
  correct behaviour"); it blocks B5 by design.
- *`bench/__main__.py` is thin and unguarded.* It is byte-for-byte the shape of
  `training/__main__.py` and `interp/__main__.py`; no divergence to report.
- *`--profile` naming a missing file, an unknown `--service`, an unknown `--model`, a bad
  flag, a missing subcommand.* All exit 2 with a styled message; verified end to end.

`--out` on an unwritable path, a lone `--arm` with overrides and no pair, and a third
`--arm` are real argparse-reachable states with no handler, but they land in
`bench/commands/run.py`, not in the registry/CLI file set.
