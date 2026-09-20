# AI experiments Labratory

A model lab for reverse engineering language models, on one workstation: an RTX 5090 and
an AMD Ryzen 9950X. The lab is built from four tiers. Inference serves models through
llama.cpp for latency and vLLM for throughput. Training fine-tunes and pretrains them.
Interpretability harvests their activations, trains probes and sparse autoencoders on
them, and steers them. A Rust harness runs a Gemma 4 model in owned code so every one of
those interventions can happen inside the forward pass rather than around it. Under all
four, one observability layer records what every command did.

The work is experimental in the literal sense: a capability is built, then run on this
hardware and measured, and the difference between those two states is tracked rather
than assumed. Every tier is containerized and pinned, and the Python environment is
managed with uv.

## The tiers

| Tier | What it does | Its README |
|------|--------------|------------|
| Inference | Serves published models through llama.cpp on CPU and GPU and through vLLM, with the settings that were measured | [docker/README.md](docker/README.md) and the compose files beside this file |
| Training | Supervised fine-tuning, continual pretraining and training from scratch, with export to the checkpoint the harness serves | [training/README.md](training/README.md) |
| Interpretability | Activation harvesting, sparse autoencoders, probes, steering vectors and model organisms | [interp/README.md](interp/README.md) |
| Rust harness | A single-binary coding harness with the inference runtime linked in, and the correctness ladder that holds it to `transformers` | [rust-harness/README.md](rust-harness/README.md) |
| Observability | The run records, catalog and dashboard shared by every tier, in Python and in Rust | [common/obs/README.md](common/obs/README.md) |

## What is proven

More of this repository is built than proven, and the gap is tracked rather than hidden.
Each tier has a verification runbook that is executed on the workstation, entry by entry,
with the pasted output as the evidence; the state of every runbook and the ledger of
defects they surfaced are in [docs/verification/README.md](docs/verification/README.md).
Read that before reading a tier's code as a description of what works.

## Where to read

- **To build and run the harness or operate the interp tier**, read the documentation
  site: [site/README.md](site/README.md) says how to serve it. It is written for a person
  and does not overlap with the knowledge base.
- **To change the repository**, read [docs/README.md](docs/README.md), the knowledge base.
  It is organized with agents as its first readers: one directory per topic, each with a
  README that indexes its pages, and every table of directories or pages rendered from the
  tree.
- **To work here as an agent, or with one**, read [AGENTS.md](AGENTS.md) first. It carries
  the operating rules and routes every question to the document that owns it.

## Layout

Every top-level directory carries a README that says what it holds, what consumes it, its
entry points and its traps.

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [common/](common/README.md) | The shared substrate under the Python tiers: styled CLI output, file hashing and atomic JSON writes, git provenance, the report envelope, serving helpers, and the observability package that records what every command did. |
| [docker/](docker/README.md) | Build contexts and entrypoints, one directory per service. |
| [docs/](docs/README.md) | The knowledge base for changing this repository, organized with agents as its first readers: every topic has one directory that owns it, that directory's README indexes its pages, and the tables that list directories or pages are rendered from the tree so they cannot drift. |
| [interp/](interp/README.md) | The interpretability tier: activation harvesting to NVMe, sparse autoencoder training and evaluation, linear probes (capture, train, evaluate, export, monitor), steering-vector extraction and steered serving, and model organism generation, verification and sealing. |
| [make/](make/README.md) | The Makefile fragments, one per subject, that the root `Makefile` includes. |
| [rust-harness/](rust-harness/README.md) | Single-binary local coding harness with an integrated LLM inference runtime, written in Rust. |
| [schemas/](schemas/README.md) | The contracts the Rust harness (`/rust-harness/`) and the Python tiers (`/interp/`, `/common/`, `/training/`) agree on, one JSON file each, with the tests that enforce it named inside. |
| [scripts/](scripts/README.md) | Host-side operational tooling: workstation provisioning and updates, the GPU preflight gates, benchmarking, agent-tooling checks, model header transcripts, the documentation gates and one browser check. |
| [site/](site/README.md) | The user-facing documentation site for the repository's tiers, built with Docusaurus. |
| [tests/](tests/README.md) | The whole-repository pytest suite: unit and branch tests for the three Python packages, the shell scripts under `scripts/` driven as subprocesses, the agent tooling and Make targets, the documentation gates, the model header transcripts, and a hardware tier that runs only on the workstation. |
| [training/](training/README.md) | The training tier: the Python package that fine-tunes, continually pretrains and trains from scratch on this workstation, and exports what it trained as the merged HF checkpoint the rust harness serves and the interp tier captures from. |

<!-- inventory:dirs:end -->

## License

MIT
