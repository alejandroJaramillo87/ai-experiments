"""Fine-tuning evaluation wrapper (Phase 1 F7).

Two evaluators behind one runner, each independently skippable: held-out
loss and lm-evaluation-harness, both in-process in the training container.
The host-side runner is training/commands/eval_ft.py; the in-container
phases are the eval-holdout and eval-lmeval CLI subcommands.

Import discipline mirrors the rest of the package: scorecard assembly is
stdlib-only (host-importable); holdout and lmeval import torch and the
harness lazily inside the training container.
"""

from training.eval.scorecard import (  # noqa: F401
    ScorecardError,
    assemble_scorecard,
    delta_section,
    render_markdown,
)
