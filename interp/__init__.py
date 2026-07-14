"""Interpretability tier production code (interp Phase 3 write-back path).

Subpackages: steering (control-vector extraction, GGUF export, scorecard) and
probes (activation capture, probe training, registry, scoring, async monitor).
Heavy dependencies (torch, nnsight, repeng, SAELens) are imported lazily inside
the functions that need them so the CPU sandbox can import every module.
"""
