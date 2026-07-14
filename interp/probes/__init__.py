"""Probe toolkit and async monitor (interp Phase 3 W2/W4).

capture -> train -> registry -> score is the toolkit; monitor replays serving
logs (W3) through the workbench model and scores every registered probe. The
toolkit half is deliberately thin; the registry and its hash refusals are the
durable part.
"""
