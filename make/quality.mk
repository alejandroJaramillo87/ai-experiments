##@ Testing & Code Quality

.PHONY: format lint quality test

test: ## Run tests (if available)
	uv run pytest -v

lint: ## Run linting with ruff
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

# Deterministic pre-push gate: every tool version comes from the pinned dev
# group; no step touches the network (--no-sync runs the tools already
# provisioned in .venv instead of re-resolving the environment first).
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
# Shell files are enumerated with find, not `git ls-files`. The git form fails in a
# contained agent sandbox (/work is a linked worktree whose gitdir is outside the bind
# mount), and because it sat on the left of a pipe the recipe's status came from xargs --
# shellcheck ran on zero files and the gate reported success having linted nothing. The
# empty-set guard below makes that failure mode loud instead of silent.
quality: ## Run the full quality gate (format check, lint, mypy, lock check, doc refs, retired names, shellcheck, rust gate)
	uv run --no-sync ruff format --check .
	uv run --no-sync ruff check .
	uv run --no-sync mypy
	uv lock --check
	uv run --no-sync python scripts/docs/check_doc_refs.py
	uv run --no-sync python scripts/docs/check_retired_names.py
	@files=$$(find . -name '*.sh' -type f \
	    -not -path './.git/*' -not -path './.venv/*' \
	    -not -path './node_modules/*' -not -path '*/target/*' \
	    -not -path './.pytest-tmp/*' | sort); \
	if [ -z "$$files" ]; then \
	  printf '[ERROR] shellcheck: enumerated zero shell scripts; the gate would pass without linting anything\n'; \
	  exit 1; \
	fi; \
	printf '[INFO] shellcheck: %s files\n' "$$(printf '%s\n' "$$files" | wc -l)"; \
	printf '%s\n' "$$files" | xargs uv run --no-sync shellcheck --severity=warning
	@if [ -d rust-harness ]; then $(MAKE) -C rust-harness quality; fi

