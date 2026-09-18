# AI Infrastructure Base Repository Makefile
# RTX 5090 + 16-core Ryzen 9950X AI workstation management commands
#
# Targets live in make/*.mk, one file per domain; this file holds only the
# settings they share and the include list. Each domain file declares its own
# .PHONY, so adding a target means editing one file rather than two.
#
# Recipes run from the repository root, not from make/. Includes do not change
# the working directory, so every relative path in a recipe (./scripts/...,
# docker-compose*.yaml, .claude/agents/*.md) resolves as it always has. Do not
# "tidy" one into a path relative to make/.

.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Cap parallel source compilation during `make install` and the training image
# builds (train-build/export-build pass these through as build args). Several pinned deps
# (flash-attn, deepspeed, liger-kernel, torchao) build CUDA kernels from source
# and default to one nvcc/cc1plus job per core, each needing several GB of RAM.
# Unbounded, that spikes past host RAM and OOM-kills other processes (e.g. the
# editor). Override for a bigger/smaller box: `make install MAX_JOBS=8`.
MAX_JOBS ?= 4
NVCC_THREADS ?= 2

# The inference stack's compose file. Named here rather than in one domain file because
# four of them consume it (services, ops, interp, cleanup), matching the per-tier
# convention set by TRAIN_COMPOSE, INTERP_COMPOSE, HARNESS_COMPOSE, and AGENTS_COMPOSE.
#
# Every invocation must pass `-f $(INFER_COMPOSE)`. Compose auto-discovers only its four
# default filenames, none of which carry a domain suffix, so a bare `docker compose` no
# longer finds this file at all -- and where the status was swallowed, that failure read
# as "no services running" rather than as an error.
INFER_COMPOSE ?= docker-compose.inference.yaml

# Batch rung defaults for the escalation ladder (make/agents.mk). AGENT selects the CLI;
# tiers map to reasoning effort on one frontier model. CLAUDE_PERM is version-dependent —
# the guard.sh PreToolUse hook is the real safety floor in batch mode (it fires regardless
# of permission flag). Defined here rather than beside their targets because they are
# user-overridable knobs, and this is where someone looks for one.
AGENT ?= claude
EFFORT ?= low
CLAUDE_PERM ?=
AGENT_USAGE_STATE ?= .agent-state/usage.jsonl

# Resolve the includes against this file rather than the working directory. `include
# make/x.mk` would look in $(CWD), so `make -f /path/to/Makefile` run from anywhere else
# fails to find its own targets -- which is exactly how tests/agents/ drives the Makefile
# against a throwaway repository. Recipes still run in $(CWD) and still resolve their own
# relative paths there; only the include lookup is anchored here.
MK_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

# Include order is the order `make help` prints the sections, because the help target
# awks over $(MAKEFILE_LIST) and that list is built in include order. Keep it explicit:
# a wildcard would sort alphabetically and scramble the output. help.mk stays last for
# the same reason.
include $(MK_DIR)make/services.mk
include $(MK_DIR)make/training.mk
include $(MK_DIR)make/interp.mk
include $(MK_DIR)make/harness.mk
include $(MK_DIR)make/site.mk
include $(MK_DIR)make/ops.mk
include $(MK_DIR)make/python.mk
include $(MK_DIR)make/quality.mk
include $(MK_DIR)make/agents.mk
include $(MK_DIR)make/cleanup.mk
include $(MK_DIR)make/quickstart.mk
include $(MK_DIR)make/help.mk
