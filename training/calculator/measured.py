"""Measured activation rates, bytes per token, keyed by profile().

The activation term of the memory-fit estimate is not modeled: it is the
per-token cost calibrate.py measured on the workstation for one profile
(family, geometry, training mode, checkpointing), and estimate() multiplies it
by the recipe's micro-batch times sequence length. A profile absent from this
table is reported as unmeasured, never guessed.

Recording a rate: run `make train-calibrate CONFIG=<recipe> ARGS="--out
<record.json>"` on the workstation, and paste the `Record:` line the command
prints into RATES with a comment naming the date, the runbook entry and the
measured figures it came from. The rate is one point on a line through the
origin: activations at another micro-batch or sequence length of the same
profile are assumed linear in tokens, which flash attention makes true to
within the per-head softmax statistics. Re-calibrating after a transformers
upgrade is what catches a moved peak; calibrate compares the rate on file
against the new measurement and fails past the acceptance bar.

The archived P0.3 calibration of 2026-08-01 (docs/verification/archive/
training-phase0.md) measured a 16x2048 pretrain-1b geometry the recipe no
longer has, so no scratch profile is seeded from it.
"""

from __future__ import annotations

# profile -> activation bytes per token, with provenance.
RATES: dict[str, int] = {}
