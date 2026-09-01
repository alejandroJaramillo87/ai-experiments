##@ Log triage (self-contained project; its own virtualenv, own gate)

.PHONY: logtriage-quality logtriage-test logtriage-install

LOGTRIAGE_DIR := logtriage

logtriage-install: ## Provision the logtriage project's own virtualenv
	cd $(LOGTRIAGE_DIR) && uv venv --python 3.12 && uv sync

logtriage-quality: ## Run the logtriage gate (format check, lint, mypy, tests)
	$(MAKE) -C $(LOGTRIAGE_DIR) quality

logtriage-test: ## Run the logtriage test suite only
	$(MAKE) -C $(LOGTRIAGE_DIR) test
