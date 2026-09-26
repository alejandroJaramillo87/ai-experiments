##@ Testing & Code Quality

.PHONY: format lint quality quality-full hardware-tests test

# How a Python tool is reached. The two environments hold the same pinned
# versions in different places, so the difference is resolved here once rather
# than by every caller:
#
#   workstation  .venv provisioned from the dev group; --no-sync runs what is
#                already there instead of re-resolving (no network, no writes).
#   sandbox      the image bakes uv, CPython, and the dev-group pins onto PATH,
#                and docker-compose.agents.yaml masks .venv precisely so no one
#                resolves the workstation's. AI_SANDBOX is set by the image.
#
# The consequence worth stating: `make quality` is the same command in both, so
# nothing downstream needs to know which one it is running in.
ifdef AI_SANDBOX
RUN_PY :=
else
RUN_PY := uv run --no-sync
endif

test: ## Run tests (if available)
	$(RUN_PY) pytest -v

lint: ## Run linting with ruff
	$(RUN_PY) ruff check .

format: ## Format code with ruff
	$(RUN_PY) ruff format .

# Deterministic pre-push gate: every tool version comes from the pinned dev
# group and no step touches the network.
# shellcheck gates at warning severity: info/style hits in host entrypoints are
# dominated by intentional word-splitting the sandbox cannot safely rewrite.
#
# check_doc_refs.py fails the gate when a document names a repository path that does not
# exist. It is here rather than in `make test` because this is the target that runs before
# every push, and because doc drift is a push-time defect: the tree moved and the prose did
# not. It carries the same empty-set guard as shellcheck, for the same reason.
#
# check_retired_names.py covers what that cannot see. The doc-ref classifier requires a
# slash, so a bare renamed filename is never treated as a path claim
# -- and loosening the rule was measured at a 67% false-positive rate, because prose says
# `config.json` and `main.rs` in good faith. This one inverts the question: an explicit table
# of retired names, failing on any use and naming the replacement. It scans code as well as
# documentation, since a retired filename in a recipe is a broken command, not a stale
# sentence.
#
# check_make_targets.py extends the same idea from names to commands. Neither gate above can
# see `make sae-eval`, and a documented command is the assertion a reader acts on first: a
# wrong path costs a grep, a wrong target costs the afternoon spent believing the tool is
# broken. Renaming a target touches one line of one .mk file, so the documentation naming it
# is exactly what nobody thinks to update -- a rename of that one target was measured to
# surface in eight places across four documents. It reads targets lexically from the three
# Makefiles rather than shelling out to `make help`, and it reads invocations only from
# fenced blocks and code spans, because `make` is also an English verb.
#
# check_doc_counts.py compares a document against itself: a heading reading `## Open (3)`
# asserts a fact about the rows beneath it, and the two are edited at different moments.
# This is the only check here that needs no knowledge of the tree and so cannot be wrong
# about the right answer. It was added after a count in docs/verification/ drifted from 56 to
# an actual 66 and sat that way for several sessions with nothing able to see it.
#
# check_runbook_entries.py checks the two structural properties of a verification entry that
# a reader would otherwise have to take on trust. A gate-backed entry pastes no output; it
# names the test function asserting its signal, so its honesty is a reference into the tree,
# and renaming a test is a one-line edit that leaves the runbook claiming coverage nothing
# provides. The second half is cruder: a VERIFIED entry must offer evidence of one of the two
# legal kinds. Neither says an entry is correct -- whether pasted output demonstrates a signal
# needs a reader, and five entries marked verified against output that did not demonstrate
# theirs is what opened the campaign that added this file.
#
# check_runbook_ids.py holds an entry's two statements of place apart. The id names the
# entry and the Table of Contents orders it, since 2026-09-19 when the numeric ids became
# slugs: a number handed out in writing order read as an execution order it was not once
# sections had been reordered by dependency, and nothing could see the two drift because one
# token stated both. The gate fails a ToC whose order differs from the headings, an id that
# heads two entries or returns after retirement, and a dependency on a later entry.
#
# check_surface_census.py is the only gate here that starts from the code. Every other one
# reads the documentation and asks whether it is true, which cannot see a feature nobody
# thought to document: `attention-capture` and `layer-diff` shipped as binaries named once
# each in prose, and `long-context-golden` and `tiny-model` were named nowhere at all. It
# enumerates the rust-harness command line from the `Flag` literals and subtracts what the
# runbooks mention. A mention is a low bar deliberately -- whether the mention verifies
# anything is the check above's question, and a reader's.
#
# check_kb_refs.py holds the ends of a cross-reference together. Thirty-one known-bug ids sit
# in permanent source comments and one is asserted as a literal string by a test, so a
# deleted ledger row breaks a reference in the direction hardest to notice: the code still
# compiles and the comment still reads sensibly. Written by hand during the 2026-08-08 ledger
# reset, where it immediately caught a row deleted as unverified that the runbook recorded as
# observed on the workstation.
#
# All three carry the empty-set guard, and the two that resolve against a table carry a
# second one -- an empty target set or an empty ledger would fail every reference rather than
# none, which is a confusing way to learn the gate is broken.
#
# check_group_pin_parity.py is the same idea applied to pyproject.toml rather than to prose.
# The sandbox group re-pins thirteen packages that core, interp and training also pin, so a
# torch-free environment can still collect the suite, and its own comment states the
# obligation to keep them equal. Nothing enforced it: `uv lock --check` proves one resolution
# is internally consistent, not that two literal `==` strings agree, so bumping a core pin
# left the sandbox copy behind silently -- surfacing later as a sandbox that resolves
# differently from the workstation, which is the worst place to have an unexplained
# difference. Exact pins only: a package pinned in one section and left free in another is
# normal, and only two exact pins can contradict each other.
#
# check_command_prompts.py reads the command blocks the other doc gates take their
# invocations from, and fails when one carries a `$ ` prompt marker: pasted into a shell,
# `$ make quality` is a command named `$`. The style guide once prescribed the marker, so
# 210 blocks carried it until the 2026-09-04 sweep. Transcripts keep their prompt under a
# `console` tag or an Observed heading, which the gate does not read.
#
# check_site_reference.py holds the committed page rendered from each binary's --help, under
# site/docs/rust-harness/reference/, to the flag table it was rendered from. The pages are
# committed so the site builds without cargo, which is also how they go stale: a flag added
# to the table ships in the binary and stays absent from its page until `make site-gen` is
# run again. The gate enumerates the same `Flag` literals the census reads and fails on a
# binary with no page, a page for no binary, or a flag its page does not show. It needs
# neither cargo nor node, so it runs in the sandbox; `make site-check` is the stricter
# workstation form that regenerates and diffs. check_site_interp_reference.py is the same
# gate for the interp verbs under site/docs/interp/reference/, reading the argparse
# literals in interp/commands/ instead of the Rust Flag literals.
#
# check_site_diagrams.py holds the SVGs under site/static/img/diagrams/ to the figura exports
# they were copied from. figura is where a diagram is edited; the site keeps a copy so it
# builds without figura, and a copy is where a quick fix is tempting -- commit 02a546b
# changed a label in tier-loop.svg in place on 2026-09-18, and the next re-copy would have
# reverted it. figura.json beside the copies records the figura commit and each file's export
# path and sha256; the gate fails on an unrecorded copy, a missing one, or different bytes.
# check_site_tokens.py compares the tokens site/src/css/custom.css mirrors from figura with
# the snapshot of figura's shared/tokens.css in the same record, through the explicit pair
# table the blog's check-figura-tokens.mjs uses. Both read only committed files, so they run
# in the sandbox.
#
# check_inventory.py is the same generated-artifact pattern applied to the navigation
# READMEs. Each table between an inventory marker pair is rendered by
# scripts/docs/gen_inventory.py from the tree the README describes: a test directory's
# from its modules' docstrings and test counts, a package's from its modules' docstrings,
# a hub's from its children's READMEs, a documentation directory's from its pages. The
# description is not written twice: it already sits at the top of the file, and the table
# quotes it. A hand-kept inventory would be true on the day it was written and silently
# wrong after the next module is added or renamed, so the gate regenerates in memory and
# diffs against what is committed, printing the difference rather than only reporting
# one. It also holds the hubs complete: under tests/ and training/ every package or test
# directory carries a README with a table, so a new one fails the gate until it is
# documented. It needs no torch and no collected session, so it runs in the sandbox.
#
# check_toc.py is the same pattern applied to a document's own index. The list between
# a toc marker pair is rendered by scripts/docs/gen_toc.py from the document's H2 and H3
# headings, so a heading renamed in the body cannot leave a row at the top pointing at
# an anchor that no longer exists. The gate also holds the threshold the style guide
# states: a living document over 500 words with four or more sections carries the pair,
# and one that grows past it fails until `gen_toc.py <file>` adds it. Frozen records are
# outside both rules, by the same paths every other gate exempts, because a frozen
# record is never edited. It reads the files' own lines, so it runs in the sandbox.
#
# The site build is last, after the rust gate, and skips itself with [SKIP] when
# site/node_modules is absent -- a fresh sandbox has none, because installing them needs
# the network, in the same way `uv lock --check` is skipped above. With them present the
# build is offline and fails on a broken link or anchor.
#
# Shell files are enumerated with find, not `git ls-files`. The git form fails in a
# contained agent sandbox (/work is a linked worktree whose gitdir is outside the bind
# mount), and because it sat on the left of a pipe the recipe's status came from xargs --
# shellcheck ran on zero files and the gate reported success having linted nothing. The
# empty-set guard below makes that failure mode loud instead of silent.
#
# The preflight exists because uv's response to a venv whose interpreter has gone
# missing is to delete and recreate it empty, not to fail. With --no-sync nothing
# is then installed, so the first step wipes the environment and every later step
# dies on a missing tool -- and the gate has to be followed by `make install` to
# undo what it did. Checking first turns that into a named error that changes
# nothing. Skipped in a sandbox, which has no .venv by design.
quality: ## Run the full quality gate (format check, lint, mypy, lock check, doc refs, retired names, make targets, doc counts, KB refs, runbook entries, runbook ids, surface census, pin parity, command prompts, site reference, site interp reference, site diagrams, site tokens, inventory, toc, shellcheck, rust gate, site build)
ifndef AI_SANDBOX
	@if [ ! -x .venv/bin/ruff ] || [ ! -e "$$(readlink -f .venv/bin/python 2>/dev/null)" ]; then \
	  printf '[ERROR] .venv is missing, incomplete, or points at an interpreter that no longer exists\n'; \
	  printf '  Running the gate now would delete and recreate it empty, losing every installed package.\n'; \
	  printf '  Found:\n'; \
	  printf '    interpreter: %s\n' "$$(readlink -f .venv/bin/python 2>/dev/null || echo '<no .venv/bin/python>')"; \
	  printf '    home:        %s\n' "$$(sed -n 's/^home = //p' .venv/pyvenv.cfg 2>/dev/null || echo '<no pyvenv.cfg>')"; \
	  printf '    ruff:        %s\n' "$$([ -x .venv/bin/ruff ] && echo present || echo missing)"; \
	  printf '  A home under /opt or /usr/local means a container wrote this venv, not the workstation.\n'; \
	  printf '  Solution: provision it\n'; \
	  printf '  Command: make install\n'; \
	  exit 1; \
	fi
endif
	$(RUN_PY) ruff format --check .
	$(RUN_PY) ruff check .
	$(RUN_PY) mypy
ifdef AI_SANDBOX
	@printf '[SKIP] uv lock --check: resolving the lock reaches the pytorch cu130 index, which the sandbox egress allowlist does not carry. The workstation gate covers it.\n'
else
	uv lock --check
endif
	$(RUN_PY) python scripts/docs/check_doc_refs.py
	$(RUN_PY) python scripts/docs/check_retired_names.py
	$(RUN_PY) python scripts/docs/check_make_targets.py
	$(RUN_PY) python scripts/docs/check_doc_counts.py
	$(RUN_PY) python scripts/docs/check_kb_refs.py
	$(RUN_PY) python scripts/docs/check_runbook_entries.py
	$(RUN_PY) python scripts/docs/check_runbook_ids.py
	$(RUN_PY) python scripts/docs/check_surface_census.py
	$(RUN_PY) python scripts/docs/check_group_pin_parity.py
	$(RUN_PY) python scripts/docs/check_command_prompts.py
	$(RUN_PY) python scripts/docs/check_site_reference.py
	$(RUN_PY) python scripts/docs/check_site_interp_reference.py
	$(RUN_PY) python scripts/docs/check_site_diagrams.py
	$(RUN_PY) python scripts/docs/check_site_tokens.py
	$(RUN_PY) python scripts/docs/check_inventory.py
	$(RUN_PY) python scripts/docs/check_toc.py
	@files=$$(find . -name '*.sh' -type f \
	    -not -path './.git/*' -not -path './.venv/*' \
	    -not -path './node_modules/*' -not -path '*/target/*' \
	    -not -path './.pytest-tmp/*' | sort); \
	if [ -z "$$files" ]; then \
	  printf '[ERROR] shellcheck: enumerated zero shell scripts; the gate would pass without linting anything\n'; \
	  exit 1; \
	fi; \
	printf '[INFO] shellcheck: %s files\n' "$$(printf '%s\n' "$$files" | wc -l)"; \
	printf '%s\n' "$$files" | xargs $(RUN_PY) shellcheck --severity=warning
	@if [ -d rust-harness ]; then $(MAKE) -C rust-harness quality; fi
	$(MAKE) site-build

# Hardware tier for `make quality-full`. smoke is the before-every-session tier: under five
# minutes, no model downloads, temporary-directory writes only. full may allocate
# near-capacity VRAM and load real models, so it is opt-in -- a gate nobody will run is not
# a gate.
#
# e4b is full plus the checks that need the real Gemma 4 E4B checkpoint on disk. It is a
# provisioning precondition rather than a third tier of hardware stress: the same GPU runs
# it, and what separates it is whether a 14 GiB checkpoint is present. Kept opt-in for the
# reason the marker is deselected by default -- no sandbox has the checkpoint, and a tier
# that fails for an absent file teaches everyone to ignore it.
#
# Deliberately not named TIER. make/agents.mk:86 already owns that variable with a different
# accepted set (sandbox|full) and a `?=` default; a second `?=` in this file would win by
# include order and silently hand `worktree-%` a value it rejects.
HW ?= smoke
# Which subsystem's hardware tests `hardware-tests` runs: one of the four subsystem
# markers pyproject.toml registers, or every test of the tier. Only that target reads it;
# `quality-full` runs the whole tier by design.
SUITE ?= all

# The gate that runs everything. `make quality` skips pytest entirely and `make test`
# deselects both hardware markers (pyproject.toml:263), so until this target existed "I ran
# the gate" meant the cheapest of the three commands and the hardware tier was in no gate at
# all.
#
# The torch probe is the load-bearing step. tests/conftest.py:62 decides between
# `[TIER] sandbox` and `[TIER] full` on exactly this condition, and a torch-free run skips
# its way to green -- so the suite would report a partial run as a pass. Checking the
# condition at its source rather than grepping the printed line keeps the gate from
# depending on the wording of a terminal-summary string that no test pins.
quality-full: ## Run the whole gate: quality, the suite, and the hardware smoke tier (HW=smoke|full)
ifdef AI_SANDBOX
	@printf '[ERROR] quality-full: this target requires the workstation\n'; \
	printf '  Found: AI_SANDBOX is set, so there is no GPU, no /mnt/ai-data, and no running services.\n'; \
	printf '  Solution: run the sandbox-executable half here, then run this on the workstation\n'; \
	printf '  Command: make quality\n'; \
	exit 1
else
	@case "$(HW)" in \
	  smoke|full|e4b) ;; \
	  *) printf '[ERROR] quality-full: unknown HW: %s\n' "$(HW)"; \
	     printf '  Accepted: smoke (under five minutes, default), full (near-capacity VRAM, real models),\n'; \
	     printf '            e4b (full plus the checks needing the real E4B checkpoint on disk)\n'; \
	     printf '  Command: make quality-full HW=full\n'; \
	     exit 1;; \
	esac
	@printf '[INFO] GPU/benchmark work stays serialized across all worktrees (see AGENTS.md)\n'
	@$(RUN_PY) python -c 'import torch' 2>/dev/null || { \
	  printf '[ERROR] quality-full: torch is not importable, so the suite would skip its way to green\n'; \
	  printf '  Found: the environment is the torch-free sandbox tier; the run would report [TIER] sandbox\n'; \
	  printf '  Solution: provision the full tier\n'; \
	  printf '  Command: make install\n'; \
	  exit 1; \
	}
	$(MAKE) quality
	$(MAKE) test
	$(RUN_PY) pytest -m hw_smoke -v
	@case "$(HW)" in full|e4b) \
	  printf '[RUN] hardware full tier: near-capacity VRAM and real model loads\n'; \
	  $(RUN_PY) pytest -m hw_full -v;; \
	esac
	@if [ "$(HW)" = e4b ]; then \
	  printf '[RUN] E4B tier: checks that need the real checkpoint on disk\n'; \
	  if [ -z "$(HARNESS_E4B_MODEL)" ]; then \
	    printf '[ERROR] quality-full HW=e4b: HARNESS_E4B_MODEL is unset, so every check would skip its way to green\n'; \
	    printf '  Found: the tier names the checkpoint through one variable so no test hard-codes a path.\n'; \
	    printf '  Solution: point it at the instruction-tuned safetensors directory\n'; \
	    printf '  Command: make quality-full HW=e4b HARNESS_E4B_MODEL=/mnt/ai-data/models/hf/gemma-4-E4B-it\n'; \
	    exit 1; \
	  fi; \
	  HARNESS_E4B_MODEL="$(HARNESS_E4B_MODEL)" $(RUN_PY) pytest -m hw_e4b -v; \
	  HARNESS_E4B_MODEL="$(HARNESS_E4B_MODEL)" $(MAKE) -C rust-harness test-e4b; \
	fi
endif

# One tier of one subsystem, for the runs that are not the whole gate: the training
# slice of hw_full is minutes of GPU (a QLoRA train, interrupt and resume; a merge and
# a harness chat) that a session verifying the harness has no reason to pay for, and
# the reverse holds for the E4B chats. Selection is the tier marker and the subsystem
# marker together, so the default-selection guard in pyproject.toml still applies:
# a bare `-m training` would re-enable every tier, because a command-line -m replaces
# addopts rather than adding to it. The refusals are quality-full's, for the same
# reasons; the e4b arm adds the rust workspace's E4B tests only when the harness is
# in the selection, since nothing else in that workspace reads the checkpoint.
hardware-tests: ## Run one hardware tier for one subsystem (HW=smoke|full|e4b SUITE=harness|training|interp|platform|all)
ifdef AI_SANDBOX
	@printf '[ERROR] hardware-tests: this target requires the workstation\n'; \
	printf '  Found: AI_SANDBOX is set, so there is no GPU, no /mnt/ai-data, and no running services.\n'; \
	printf '  Solution: run the sandbox-executable half here, then run this on the workstation\n'; \
	printf '  Command: make test\n'; \
	exit 1
else
	@case "$(HW)" in \
	  smoke|full|e4b) ;; \
	  *) printf '[ERROR] hardware-tests: unknown HW: %s\n' "$(HW)"; \
	     printf '  Accepted: smoke (under five minutes, default), full (near-capacity VRAM, real models),\n'; \
	     printf '            e4b (the checks needing the real E4B checkpoint on disk)\n'; \
	     printf '  Command: make hardware-tests HW=full SUITE=training\n'; \
	     exit 1;; \
	esac
	@case "$(SUITE)" in \
	  harness|training|interp|platform|all) ;; \
	  *) printf '[ERROR] hardware-tests: unknown SUITE: %s\n' "$(SUITE)"; \
	     printf '  Accepted: harness, training, interp, platform, all (default)\n'; \
	     printf '  Command: make hardware-tests HW=full SUITE=training\n'; \
	     exit 1;; \
	esac
	@printf '[INFO] GPU/benchmark work stays serialized across all worktrees (see AGENTS.md)\n'
	@$(RUN_PY) python -c 'import torch' 2>/dev/null || { \
	  printf '[ERROR] hardware-tests: torch is not importable, so the suite would skip its way to green\n'; \
	  printf '  Found: the environment is the torch-free sandbox tier; the run would report [TIER] sandbox\n'; \
	  printf '  Solution: provision the full tier\n'; \
	  printf '  Command: make install\n'; \
	  exit 1; \
	}
	@if [ "$(HW)" = e4b ] && [ -z "$(HARNESS_E4B_MODEL)" ]; then \
	  printf '[ERROR] hardware-tests HW=e4b: HARNESS_E4B_MODEL is unset, so every check would skip its way to green\n'; \
	  printf '  Found: the tier names the checkpoint through one variable so no test hard-codes a path.\n'; \
	  printf '  Solution: point it at the instruction-tuned safetensors directory\n'; \
	  printf '  Command: make hardware-tests HW=e4b SUITE=harness HARNESS_E4B_MODEL=/mnt/ai-data/models/hf/gemma-4-E4B-it\n'; \
	  exit 1; \
	fi
	@if [ "$(SUITE)" = all ]; then expr="hw_$(HW)"; else expr="hw_$(HW) and $(SUITE)"; fi; \
	printf '[RUN] hardware tests: -m "%s"\n' "$$expr"; \
	HARNESS_E4B_MODEL="$(HARNESS_E4B_MODEL)" $(RUN_PY) pytest -m "$$expr" tests/hardware -v
	@if [ "$(HW)" = e4b ] && { [ "$(SUITE)" = harness ] || [ "$(SUITE)" = all ]; }; then \
	  printf '[RUN] E4B tier: the rust workspace tests that read the checkpoint\n'; \
	  HARNESS_E4B_MODEL="$(HARNESS_E4B_MODEL)" $(MAKE) -C rust-harness test-e4b; \
	fi
endif

