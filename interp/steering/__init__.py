"""Steering deployment path (interp Phase 3 W1).

extract -> export -> wiring -> scorecard: a validated direction becomes a
llama.cpp control-vector GGUF, the serving entrypoints load it as
configuration, and the scorecard measures the steered service against a tuned
system-prompt baseline. Weights are never modified.
"""
