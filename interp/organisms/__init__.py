"""Model-organism recipes (interp Phase 4 P4-D1).

Turns the training tier's fine-tuning path into a ground-truth generator:
generate.py renders organism training corpora and elicitation sets from
templates, verify.py proves the implant took without collateral damage, and
seal.py records the ground truth so downstream auditing games can be blinded
and graded. Training itself runs through the unchanged train-sft path; there
is no parallel trainer here.
"""
