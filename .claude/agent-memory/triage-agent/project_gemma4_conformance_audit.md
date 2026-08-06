---
name: project-gemma4-conformance-audit
description: Gemma 4 conformance campaign - docs/models/gemma-4.md is sole authority; the reconciliation table in rust-harness-runtime-design.md is known-bad secondhand data
metadata:
  type: project
verified: 2026-08-06
stale_when: docs/models/gemma-4.md is superseded as the architecture authority, or Gemma 4 weights land on the workstation making the needs-host verdicts resolvable
---

A repository-wide conformance audit sweeps the tree against `/work/docs/models/gemma-4.md`,
which is the sole source of truth for Gemma 4 architecture and supersedes every other
document. The "Architecture reconciliation" table at
`docs/rust-harness/runtime-design.md:295` was assembled from secondhand web
sources and is explicitly suspect - never use it as corroboration.

**Why:** the spec is regenerated from Hugging Face `config.json` and safetensors headers at
pinned revisions; the other documents were written before those artifacts were read. E2B and
E4B get deep coverage because they are the only variants that fit continual pretraining in
32GB (`docs/training/training-scale-reference.md`); 12B and 31B divergences are recorded but
not pursued.

**How to apply:** when a repo value disagrees with the spec, the repo is the bug. Two
recurring traps: (1) a stale `d_model` of 2304 propagated widely and appears to originate in
transformers' `configuration_gemma4.py` class defaults being read as if they were the
checkpoint's `text_config` (spec:138 warns against exactly this) - E2B's real `hidden_size`
is 1536; (2) HF-versus-GGUF questions (tensor names, the folded Gemma norm +1 shift) cannot
be settled offline and take verdict `needs-host` with a runnable command, because no Gemma 4
weights exist on the workstation yet. `docs/verification/known-bugs.md` (KB-nn) is a prior
campaign's settled ledger - reference, do not reopen.

The documentary root of the 2304 trap is now located: see [[gemma4-2304-propagation]].

Related: [[triage-output-discipline]]
