##@ Python Environment (uv)

.PHONY: install shell update

install: ## Sync the uv environment (all groups and extras)
	@echo "$(CYAN)Syncing uv environment (MAX_JOBS=$(MAX_JOBS))...$(RESET)"
	MAX_JOBS=$(MAX_JOBS) NVCC_THREADS=$(NVCC_THREADS) uv sync --all-groups --all-extras
	@echo "$(GREEN)Environment synced.$(RESET)"

update: ## Update Python dependencies (lock upgrade + sync + verification)
	./scripts/update/update_python.sh

shell: ## Show how to activate the project environment
	@echo "Run: source .venv/bin/activate (or prefix commands with 'uv run')"

