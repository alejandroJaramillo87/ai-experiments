"""Continual pre-training tier (Phase 2).

P2-D1 consumes training/continual/benchmarks.yaml as the decontamination
gate; the forgetting evaluation (P2-D6) will consume the same file so the
two can never disagree about which benchmarks matter.
"""
