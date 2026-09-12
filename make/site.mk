##@ Documentation site (Docusaurus, under site/)

.PHONY: site-build site-check site-gen site-install site-serve

# The site is the user-facing documentation for the rust harness: what a
# researcher needs to build and run it, kept apart from docs/, which is the
# knowledge base for changing this repository. It lives under site/ with its
# own package.json and lockfile, and none of these targets touch a Docker
# image or the GPU.
#
# npm ci rather than npm install: the lockfile is the pin, and ci refuses to
# resolve past it. This is the one target here that needs the network.
site-install: ## Install the site's pinned node dependencies from site/package-lock.json
	@echo "$(CYAN)Installing site dependencies...$(RESET)"
	npm ci --prefix site --no-fund --no-audit
	@echo "$(GREEN)site/node_modules ready.$(RESET)"

# The per-binary reference pages are rendered from the binaries' own --help by
# scripts/docs/gen_site_reference.py: the usage line, then one table per help
# section. It builds the workspace, so the first run takes as long as a cargo
# build; every binary answers --help before loading a model, so no checkpoint
# and no GPU is involved.
site-gen: ## Regenerate site/docs/rust-harness/reference/<binary>.md from each binary's --help
	$(RUN_PY) python scripts/docs/gen_site_reference.py

# Guarded rather than failing: `make quality` runs this, and a fresh sandbox
# has no site/node_modules because installing them needs the network. The
# skip is printed so a run that built nothing does not read as a run that
# built the site. Broken links and anchors fail the build by configuration.
site-build: ## Build the static site into site/build (skipped with [SKIP] until make site-install)
	@if [ ! -d site/node_modules ]; then \
	  printf '[SKIP] site-build: site/node_modules is absent, so there is nothing to build with\n'; \
	  printf '  Solution: install the pinned dependencies once, then re-run\n'; \
	  printf '  Command: make site-install\n'; \
	else \
	  npm run build --prefix site; \
	fi

site-serve: ## Serve the built site on http://localhost:3000 (runs make site-build first)
	$(MAKE) site-build
	npm run serve --prefix site

# Workstation-only: regenerates the reference pages and fails if that changed
# anything, which is how a push is held to `make site-gen` having been run.
# Not part of `make quality` because it needs git, which a contained sandbox
# cannot run (AGENTS.md); scripts/docs/check_site_reference.py is the
# sandbox-safe half and does run there.
site-check: ## Regenerate the reference pages and fail if the committed ones were stale (needs git)
	$(MAKE) site-gen
	@if ! git diff --quiet -- site/docs/rust-harness/reference; then \
	  printf '[ERROR] site-check: the committed reference pages were stale\n'; \
	  printf '  Found: make site-gen changed files under site/docs/rust-harness/reference\n'; \
	  printf '  Solution: commit the regenerated pages\n'; \
	  git --no-pager diff --stat -- site/docs/rust-harness/reference; \
	  exit 1; \
	fi
	@echo "$(GREEN)site-check: reference pages are current.$(RESET)"
