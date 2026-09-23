"""Fine-tuning evaluators (Phase 1 F7).

Two evaluators behind one host driver, each independently skippable: held-out
loss and lm-evaluation-harness, both in-process in the training container.
The host-side driver is training/commands/eval_ft.py; the in-container
phases are the eval-holdout and eval-lmeval CLI subcommands. Each phase
writes one JSON record per side, and those four records are the output:
interp's organism-verify reads them for its locality check. A scorecard
assembled on top of them existed until 2026-09-20 and had no reader.

Import discipline mirrors the rest of the package: the runner is stdlib-only
(host-importable); holdout and lmeval import torch and the harness lazily
inside the training container.
"""
