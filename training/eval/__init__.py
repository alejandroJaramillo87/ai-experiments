"""Fine-tuning evaluation wrapper (Phase 1 F7).

Three evaluators behind one runner, each independently skippable: held-out
loss (in-process, training container), lm-evaluation-harness (in-process,
training container), and model-as-judge (through the serving stack). The
host-side runner is scripts/eval/eval_ft.py; the in-container phases are the
eval-holdout and eval-lmeval CLI subcommands.

Import discipline mirrors the rest of the package: judge pairing and
scorecard assembly are stdlib-only (host-importable); holdout and lmeval
import torch and the harness lazily inside the training container.
"""

from training.eval.judge import (  # noqa: F401
    JudgeError,
    build_pairs,
    judge_messages,
    parse_verdict,
    score_judgments,
)
from training.eval.scorecard import (  # noqa: F401
    ScorecardError,
    assemble_scorecard,
    delta_section,
    render_markdown,
)
