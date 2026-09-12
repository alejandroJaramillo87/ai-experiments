---
title: Formats
---

# Formats

The harness reads one input format and exchanges four artifacts with the Python
interpretability tooling in the repository. Each has a page here that names its writer and its
reader, its layout on disk, and every field in words.

| Artifact | Written by | Read by | Contract file |
|----------|------------|---------|---------------|
| [Checkpoint layout](checkpoint-layout.md) | Hugging Face `transformers`, or `tiny-model` | Every binary that takes `--model` | none; the published checkpoint is its own specification |
| [Activation capture shard](activation-capture-shard.md) | `activation-capture`; also the Python capture command | The Python probe trainer and the difference-of-means extractor | `schemas/activation_capture.manifest.json` |
| [Attention capture shard](attention-capture-shard.md) | `attention-capture` | numpy | `schemas/attention_capture.manifest.json` |
| [Probe artifact](probe-artifact.md) | The Python probe exporter | `harness --probe` | `schemas/probe_artifact.json` |
| [Control-vector artifact](control-vector-artifact.md) | The Python control-vector exporter | `harness --control-vector` | `schemas/control_vector_artifact.json` |

The contract files under `schemas/` in the repository are the normative source for the four
artifacts. They are plain JSON key documents rather than JSON Schema: each names the writers,
the readers, the required keys, the allowed values, and the tests on both sides of the language
boundary that hold the writers and readers to it. The pages here restate them; where a page
and its contract file disagree, the contract file is right.

Two artifacts flow out of the harness and two flow in. The
[guide](producing-a-probe-and-control-vector.md) at the end of this section walks the shortest
path around that loop: a capture written by the harness, turned into a probe and a control
vector by two Python commands that run on the CPU, and loaded back into the harness.

## Identity

Every artifact except the checkpoint itself records the sha256 of the `model.safetensors` it
was made from, as `model_hash`. The harness compares that hash against the served weights
before loading a probe or a control vector and refuses a mismatch; `harness --status --deep`
runs the same comparison without loading the model. A hash of `unknown` skips the check. This
is what ties a probe to the model it was fitted on across two languages and any number of
machines.
