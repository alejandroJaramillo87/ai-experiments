# training/eval

The two evaluators a base-versus-fine-tune comparison runs, and the host-side runner that
drives them. The evaluators are held-out loss over the dataset's own split, and lm-evaluation-harness over the
task set pinned in `tasks.yaml`. Each phase runs in the training container as the
`eval-holdout` or `eval-lmeval` verb and writes one JSON record per side; the `eval-ft`
verb on the host runs the four phases and holds each evaluator's pair to one
configuration, so a precision, dataset or task-set mismatch between the two sides is an
error rather than a confound.

The four records are the output. `interp/organisms/verify.py` reads them for the locality
check of `organism-verify`, which is the reason the evaluators exist; a scorecard assembled
on top of them was retired on 2026-09-20 because nothing read it. `runner.py` is
stdlib-only so the host can import it; `holdout.py` and `lmeval.py` import torch and the
harness lazily inside the container.

<!-- inventory:modules:start -->

| Module | Purpose |
|--------|---------|
| `holdout.py` | Held-out loss evaluator (Phase 1 F7). |
| `lmeval.py` | lm-evaluation-harness evaluator (Phase 1 F7). |
| `runner.py` | Host-side phase orchestration for the F7 runner. |

<!-- inventory:modules:end -->

Living page: [eval_wrapper.md](../../docs/training/fine-tuning/eval_wrapper.md). Tests:
`tests/training/test_eval_ft.py`, `test_eval_runner.py`, `test_holdout.py`, `test_lmeval.py`
and the handler tests in `test_cli_handlers_export.py`.
