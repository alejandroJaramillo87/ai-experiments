# make

The Makefile fragments, one per subject, that the root `Makefile` includes. `make help` is
generated from the `##` comments beside each target, and the include order in the
`Makefile` sets the order of its sections, so a target is documented by being written and
placed. A new fragment is added to that include list; a variable a second fragment also
needs is declared once, because a second `?=` would be resolved silently by include order.

`scripts/docs/check_make_targets.py` holds every `make <target>` a document names to a
target these fragments define, and `tests/agents/test_worktree_make.py` drives the
worktree targets in a throwaway repository.

## Fragments

Each row lists the targets `make help` prints for that fragment, in the order the
fragment defines them; a rule without a `##` comment is internal and is not listed.

<!-- inventory:fragments:start -->

| Fragment | Section | Targets |
|----------|---------|---------|
| `agents.mk` | Agent Tooling (OpenCode / Claude Code parity) | `agent-parity`, `agent-sync`, `command-sync`, `agent-usage`, `agent-preflight`, `agent-versions`, `agent-skeptic-test`, `agent-role-tests`, `worktree-clean`, `worktree-ls`, `agent-run`, `agent-sandbox`, `agent-sandbox-build`, `agent-sandbox-clean`, `agent-export` |
| `cleanup.mk` | Cleanup & Maintenance | `clean`, `prune`, `rebuild`, `reset` |
| `harness.mk` | Rust harness (separate compose file; interactive tool, never started by `make up`) | `harness-build`, `harness-run`, `harness-chat`, `harness-runs`, `harness-serve`, `harness-serve-down`, `harness-clean` |
| `help.mk` | Help | `help` |
| `interp.mk` | Interpretability (interp Phase 0 I1 - separate compose file; GPU inference must be down) | `interp-build`, `interp-up`, `interp-shell`, `interp-smoke`, `interp-runs`, `interp-down`, `interp-dirs`, `steer-extract`, `steer-up`, `steer-down`, `steer-scorecard`, `probe-capture`, `probe-train`, `probe-export`, `probe-eval`, `probe-monitor`, `harvest`, `harvest-check-hooks`, `interp-gc`, `sae-train`, `sae-eval`, `organism-generate`, `organism-verify`, `organism-seal`, `organism-check-seal`, `organism` |
| `ops.mk` | Monitoring & Debugging | `status`, `logs`, `logs-gpu`, `logs-cpu`, `logs-ui`, `logs-vllm`, `health`, `shell-gpu`, `shell-cpu`, `shell-vllm`, `update-models`, `list-models` |
| `python.mk` | Python Environment (uv) | `install`, `update`, `shell` |
| `quality.mk` | Testing & Code Quality | `test`, `lint`, `format`, `quality`, `quality-full`, `hardware-tests` |
| `quickstart.mk` | Quick Start Commands | `demo`, `dev` |
| `services.mk` | Service Management | `up`, `down`, `restart`, `gpu-up`, `cpu-up`, `ui-up` |
| `site.mk` | Documentation site (Docusaurus, under site/) | `site-install`, `site-gen`, `site-build`, `site-serve`, `site-check` |
| `training.mk` | Training (Phase 0 D1 - separate compose file; never runs with inference) | `train-build`, `train-shell`, `train-smoke`, `train-download`, `prepare-data`, `prepare-corpus`, `build-contamination`, `prepare-sft`, `train-tokenizer`, `train-preflight`, `train-validate`, `train-step`, `train`, `train-sft`, `export-build`, `export-hf`, `eval-ft`, `generate-teacher` |

<!-- inventory:fragments:end -->
