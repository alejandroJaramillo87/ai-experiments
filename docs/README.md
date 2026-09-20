# Documentation

The knowledge base for changing this repository, organized with agents as its first
readers: every topic has one directory that owns it, that directory's README indexes its
pages, and the tables that list directories or pages are rendered from the tree so they
cannot drift. The site under [../site/](../site/README.md) is the other documentation, written
for a person building and running the harness or the interp tier; the two do not overlap.

## How to use this directory

Find the topic in the table below and read its README first. Each one says what the
topic is, indexes the pages beside it, and points at the code and the compose or
Dockerfile that own the facts. A question about code rather than a topic starts at the
source directory's own README instead; the root [../README.md](../README.md) lists them,
and `AGENTS.md` says which to read for what.

## Topics

<!-- inventory:dirs:start -->

| Directory | Holds |
|-----------|-------|
| [agents/](agents/README.md) | How agents work in this repository: the tooling under `.claude/`, the roles, commands and skills it defines, the escalation ladder that says which to reach for, and the workflows for running a session. |
| [bios/](bios/README.md) | The UEFI firmware setup of the workstation's Gigabyte X870E Aorus Elite WiFi board under the AMD Ryzen 9950X and RTX 5090: the update procedure, POST-code diagnostics and the EXPO memory profile, chosen for stability rather than speed. |
| [hardware/](hardware/README.md) | The workstation's component build and the reasoning behind each part: board, CPU and cooling, GPU, memory, storage layout, power and case, with the installation notes that follow from them. |
| [inference/](inference/README.md) | The inference engines running on this workstation (AMD Ryzen 9950X, RTX 5090), how they are tuned, and what each parameter was measured to do. |
| [interp/](interp/README.md) | Living documentation for the interp tier: the `python -m interp` CLI, the write-back path on the Rust harness, the activation and SAE tooling, and the Gemma 4 facts the tier consumes. |
| [models/](models/README.md) | Architecture references for the model families this workstation serves, and the raw header transcripts that verify them. |
| [observability/](observability/README.md) | How to read the logs, find a run, and open a dashboard for the training and interpretability command-line interfaces (`python -m training`, `python -m interp`). |
| [optimizations/](optimizations/README.md) | System optimizations for LLM inference on AMD Ryzen 9950X + RTX 5090: latency-focused (llama.cpp on CPU/GPU) and throughput-focused (vLLM on GPU). |
| [os/](os/README.md) | Configuring Ubuntu 24.04 LTS for AI work on the AMD Ryzen 9950X + RTX 5090 hardware. |
| [precision/](precision/README.md) | The dtype decisions of the three tiers — the Rust inference harness, the interpretability tier, and the training tier — collected in one place. |
| [rust-harness/](rust-harness/README.md) | Design and reference documentation for `rust-harness`, the single-binary Rust coding harness with the Gemma 4 inference runtime linked in as a library crate. |
| [sandbox/](sandbox/README.md) | How the inference services are contained on this workstation: what each container is allowed to touch, and why. |
| [testing/](testing/README.md) | How this repository is tested, and how the documentation is kept true: the commands, what each one proves, where each runs, and the strategy behind them. |
| [training/](training/README.md) | Living documentation for the training tier: the `python -m training` CLI, the recipe system, the reference loop, and the three rungs the tier climbs. |
| [verification/](verification/README.md) | The manual verification campaign: the state of every runbook, the defect ledger the entries cross-reference, and the protocol they are executed and recorded under. |

<!-- inventory:dirs:end -->

## Document classes

Not every document here is maintained, and that is deliberate. A document's class decides
whether drift against the current repository is a bug to fix or a record to preserve.
`.styles/DOCUMENTATION_STYLE.md` defines the four classes and the line each one carries;
the topic READMEs index the living reference documentation, and the three tables below
list the rest.

Two of the classes are enforced by path, because both documentation gates read them that
way: `scripts/docs/check_doc_refs.py` and `scripts/docs/check_retired_names.py` exempt
anything under an `investigations/` directory and anything whose name ends
`_investigation.md`. `docs/verification/` is not exempt: its runbooks are operational and
their paths are checked like any other document's, and only its `archive/` is frozen.

### Frozen records

What was decided and why at implementation time. Bodies are never edited; a reference that
has since moved is accurate history rather than drift.

**A frozen record is not a source to check the tree against, and an audit should not read
it as one.** These files describe the code as it stood on their date, so they accumulate
names, formats and commands that no longer exist — `docs/verification/archive/` alone still
carries hundreds of references to the GGUF model loader deleted in P8b. That is correct
and must stay. Anything sweeping the tree for stale facts should skip these paths, as both
documentation gates already do, rather than rediscover the same expired history each time.

| Location | Contents |
|----------|----------|
| [training/investigations/](training/investigations/) | D3-D10, F1-F8, and P2 deliverable investigations, plus [the unified-CLI investigation](training/investigations/unified_cli_investigation.md) (20 documents, 2026-07-14 to 2026-07-15). [README.md](training/investigations/README.md) carries the directory's conventions |
| [interp/investigations/](interp/investigations/) | I1, P4, and PS deliverable investigations (8 documents, 2026-07-14 to 2026-07-15), and the 2026-09-18 Neuronpedia replacement investigation |
| [testing/investigations/](testing/investigations/) | [test_coverage_investigation.md](testing/investigations/test_coverage_investigation.md) — audit of the suite at 2026-07-16, commit 8316e5e |
| [verification/archive/](verification/archive/) | The closed verification runbooks, the ledger's closed history, and [gemma-4-conformance_investigation.md](verification/archive/gemma-4-conformance_investigation.md) — the 2026-08-01 audit of the three consumer tiers against the Gemma 4 reference. Its 50 `GH.n` host checks were never executed; [verification/README.md](verification/README.md) records that backlog |
| [interp/experiments/](interp/experiments/) | Experiment post-mortems, one file per experiment, immutable after the verdict. [README.md](interp/experiments/README.md) is the living convention for writing them |
| [models/](models/) | `gemma4-gguf.txt` and `gemma4-safetensors.txt` — machine transcripts carrying their own `Generated:` timestamps. Cited by line number; regenerating shifts those lines, and `tests/models/test_gemma4_facts_contract.py` fails on every citation the shift broke |

### Parked proposals

Recorded menus of work that is not scheduled. Dated at authoring, and edited only as items
graduate into the living documentation named in each.

| Document | Contents |
|----------|----------|
| [optimizations/ideation/](optimizations/ideation/README.md) | Four menus of optimization candidates (2026-07-27 and 2026-07-30): host OS and GPU, the Python pipelines, the x86 CPU backend, and custom CUDA and C++ work |
| [training/continuous-learning-proposal.md](training/continuous-learning-proposal.md) | The continuous-learning goal, the candidate architecture, and what must be true before it is scheduled |
| [training/tier-consolidation-proposal.md](training/tier-consolidation-proposal.md) | The checkpoint-contract consolidation, the CLI handler moves, and the roadmap-code glossary, with what gates each |

### Operational runbooks

Executed against the repository or the host rather than read for understanding. The
audience is whoever is running the checks, and the human's pasted output is the evidence.
[verification/README.md](verification/README.md) indexes them, states each runbook's
progress, and [verification/protocol.md](verification/protocol.md) owns the recording
protocol; the defect ledger they cross-reference is
[verification/bugs/known-bugs.md](verification/bugs/known-bugs.md).

## Contributing

1. **Class first.** Decide whether the document is living reference, a frozen record, an
   operational runbook, or a parked proposal, and give it the class line
   `.styles/DOCUMENTATION_STYLE.md` specifies. Living reference is the default and carries
   no marker. A frozen record goes to `<topic>/investigations/` or has a filename ending
   `_investigation.md`; placed anywhere else it is gated as living.
2. **Then by topic.** The table above is the placement list: one directory per topic, and
   a new topic is a new directory with a README.
3. **Describe it.** Open the page with a description paragraph under its title, the first
   sentence of which stands alone, because the directory's generated index reads it. Then
   regenerate the parent's table with `python scripts/docs/gen_inventory.py <README>`;
   `make quality` fails until it matches.
4. **Name it** lowercase with hyphens or underscores, descriptive but concise, and give a
   document over 500 words a table of contents.
