# Quality gate for the logtriage project, a self-contained tool that happens to
# live inside the ai-experiments repository. The gate is invocable on its own --
# nothing here reaches for the parent Makefile or the parent virtualenv -- which
# is the property that makes `cp -r logtriage/ elsewhere` yield a working
# repository.
#
# Four offline steps, cheapest first, failing on the first error: the same shape
# as the parent repository's Python gate and rust-harness/Makefile.
#
# The parent's gate calls this target (make/quality.mk), so a failure here fails
# a push from the parent repository too.

.PHONY: quality fmt lint types docs test lock

RUN_PY := uv run --no-sync

quality: preflight fmt lint types docs test

# uv's response to a missing or broken virtualenv is to create it empty rather
# than to fail, and with --no-sync nothing is then installed -- so every step
# below would die on a missing tool with an error about ruff rather than about
# provisioning. The parent gate carries the same preflight for the same reason.
.PHONY: preflight
preflight:
	@if [ ! -x .venv/bin/ruff ] || [ ! -e "$$(readlink -f .venv/bin/python 2>/dev/null)" ]; then \
	  printf '[ERROR] logtriage/.venv is missing, incomplete, or points at an interpreter that no longer exists\n'; \
	  printf '  Found: ruff %s, interpreter %s\n' \
	    "$$([ -x .venv/bin/ruff ] && echo present || echo missing)" \
	    "$$(readlink -f .venv/bin/python 2>/dev/null || echo '<none>')"; \
	  printf '  Solution: provision this project'\''s own environment\n'; \
	  printf '  Command: cd logtriage && uv venv --python 3.12 && uv sync\n'; \
	  exit 1; \
	fi

fmt:
	$(RUN_PY) ruff format --check .

lint:
	$(RUN_PY) ruff check .

types:
	$(RUN_PY) mypy

# The documentation gate. Six checks, cheapest first, each failing on one class
# of drift the others cannot see: a cited path that does not exist, a retired
# name still in use, a documented make target that does not, a stated count that
# disagrees with its rows, a corpus statistic that disagrees with the index, and
# something the code ships that no document mentions.
#
# Here rather than in `make test` because doc drift is a push-time defect -- the
# tree moved and the prose did not -- and this is the target that runs before
# every push. The last two are this project's own: the parent has no analogue
# for either, and check_surface_census is the only one that reads outwards from
# the code, which is the only way to see a feature nobody thought to document.
docs:
	$(RUN_PY) python scripts/docs/check_doc_refs.py
	$(RUN_PY) python scripts/docs/check_retired_names.py
	$(RUN_PY) python scripts/docs/check_make_targets.py
	$(RUN_PY) python scripts/docs/check_doc_counts.py
	$(RUN_PY) python scripts/docs/check_corpus_stats.py
	$(RUN_PY) python scripts/docs/check_surface_census.py

test:
	$(RUN_PY) pytest -q

# Analogue of the parent gate's `uv lock --check`: fails when uv.lock is stale
# relative to pyproject.toml. Not part of `quality` because it resolves against
# PyPI, and the gate's contract is that it runs offline. Run it after any pin
# change.
lock:
	uv lock --check
