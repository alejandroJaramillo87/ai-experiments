##@ Testing & Code Quality

.PHONY: format lint quality test

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
quality: ## Run the full quality gate (format check, lint, mypy, lock check, doc refs, retired names, make targets, doc counts, KB refs, shellcheck, rust gate)
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

