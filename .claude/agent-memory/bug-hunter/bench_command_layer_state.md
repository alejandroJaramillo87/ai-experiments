---
name: bench-command-layer-state
description: How bench/commands/{run,noise,compare}.py handle partial failure and what their artifacts do and do not carry; the repo's restore-in-finally convention these three do not follow; claims about them already refuted or already filed.
metadata:
  type: project
verified: 2026-08-29
stale_when: bench/commands/run.py grows a finally block or an epilog naming a manual restore, or bench/commands/noise.py's written document gains an `errors` or `conditions` key (the compare-command clause fired 2026-08-29 and its entries are struck through in the body)
---

The three `bench` command modules are the seam between measurement and artifact. Nothing
imports them (BQ-F22), so every claim below had to be reproduced by stubbing
`runner.gate`/`runner.execute`/`run_cmd.compose` and calling `cmd(argparse.Namespace(...))`
directly. That harness is cheap and worth rebuilding: see the pattern below.

**Why:** dimension 3/4 findings here are invisible to reading alone, because the wrong
outcome is an exit code or a missing key rather than a raised exception.

## How to drive these commands without a service

```python
run_cmd.compose = lambda env, *a, **k: calls.append(env)   # intercept the recreate
runner.gate = lambda service, printer: 0.0
runner.execute = lambda service, profile, model, printer, stream_fn=None: (obs, errors)
run_cmd.cmd(argparse.Namespace(service=..., profile="quick", model=...,
                               arm=[...], out=..., json=False))
```

`runner.Observation(cell, metrics.Sample(...))` is all `runner.reduce` needs. A valid
model slug is `qwen3-coder-30b-a3b-IQ4_XS`; `bench/registry.py` has exactly one entry.
`bench compare` runs directly against files: `python -m bench compare a.json b.json --all`.

## Load-bearing facts, checked against the tree 2026-08-29

- **The repo convention for a service recreate is restore-in-`finally`.**
  `training/commands/serve_eval.py:244-276`, `generate_teacher.py:176-219` and
  `eval_ft.py:276-336` all set a `swapped = True` flag *before* the swap ("a failing
  force-recreate can disturb serving state before it raises"), restore in `finally`, and
  print a manual one-liner on restore failure. `common/serving.py:9-14` carries the same
  one-liner in its module docstring. `bench/commands/run.py::apply_arm` does none of the
  three. Do not read that as an oversight-free design decision: nothing in `docs/` or
  `.claude/` says the service is left in the last arm's environment.
- **`profile.order == "interleaved"` is the *within-arm* point round-robin**
  (`workloads.py::Profile.cells`), not arm interleaving. `run.py`'s argparse description
  and `.claude/skills/benchmarking/SKILL.md:93-95` both claim arms interleave per repeat;
  the loop at `run.py:112-143` runs each arm's whole profile in turn. `delta.check_comparable`
  warns on `run.order != "interleaved"`, so the guard reads the field that is always right.
- **`report.read` validates the envelope only.** `common/reports.validate_report` checks
  `schema_version`, one discriminator, a timestamp and `git_sha`. It never checks that
  `report == "bench_result"`. Two `bench_noise` files therefore compare cleanly, and a
  `bench_result` passed to `--noise` yields `floors = {}` *and* suppresses the
  "no noise floor supplied" warning (which fires only on `noise is None`).
- ~~**`bench compare` counts regressions over the `--all`-filtered list.**~~ **Fixed
  2026-08-29** (BQ-F46). The exit code and the JSON regression count now come from a `GATED`
  set (`HEADLINE` plus `itl_ms`); `--all` decides the table only. Do not re-file this.
  The reader-side half above (`report.read` ignoring the discriminator) is also fixed:
  `read` takes an `expect` argument and refuses a document of the wrong kind (BQ-F47).
- **`noise.py` builds its document by hand** from `report_header` + `update`, taking only
  `git_sha`, `run{pair_id,profile,service}`, `floor`, `worst_pct`. The per-arm `errors`,
  `conditions` and `phases` that `runner.assemble` produced are discarded. It also gates
  once for two arms, where `run.py` gates once per arm.
- **`pair_id` has one writer and no reader.** Grep confirms: written at `run.py:103`,
  `noise.py:61`, threaded through `runner.assemble`; read nowhere.

## Talked myself out of

- **`noise` exiting 0 on a floor >= 5%.** Not a defect: `docs/verification/bench/bench.md:104-111`
  documents exit 0 with `worst_pct >= 5` as the expected signal and says what to do about
  it. The per-metric floor also reaches a `--json` consumer via `MetricDelta.noise_pct`.
- **`compare` exiting 0 on `inconclusive`.** By design — `delta.py`'s docstring makes the
  case that an unresolvable delta is not a regression. The misleading part is the final
  `[OK] Compare: no regression outside the noise floor` when no floor was consulted, which
  is a message-accuracy problem folded into the discriminator finding rather than its own.
- **`run.py` exiting 0 with some cells errored.** The `errors` list *is* written into
  `result.json` by `report.build`, and each failed cell prints a `[WARN]`. Weak on its own.
  The same claim against `noise.py` is strong, because there the list is dropped.

## Already filed, do not refile

BQ-F03 (floor not matched to workload, in `delta.py`), BQ-F04 (token counts graded as
latencies), BQ-F06 (one rejected sample discards a cell), BQ-F07 (samples written once at
the end, `assemble` outside the handler), BQ-F21 (`report.read` outside a handler in
compare), BQ-F22 (no test imports these three). Ledger:
`docs/verification/bench/bench-qa.md`.
