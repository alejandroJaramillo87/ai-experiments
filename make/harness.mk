##@ Rust harness (separate compose file; interactive tool, never started by `make up`)

.PHONY: harness-build harness-chat harness-run

HARNESS_COMPOSE := docker-compose.rust-harness.yaml

harness-build: ## Build the rust-harness image (pinned toolchain, cargo --locked)
	@echo "$(CYAN)Building rust-harness image...$(RESET)"
	docker compose -f $(HARNESS_COMPOSE) build
	@echo "$(GREEN)rust-harness image built.$(RESET)"

harness-run: ## Run a one-shot harness task in the container (ARGS='--model /app/models/... <task>')
	@mkdir -p sessions/rust-harness
	docker compose -f $(HARNESS_COMPOSE) run --rm harness $(ARGS)

harness-chat: ## Interactive chat REPL in the container (ARGS='--model /app/models/...')
	@mkdir -p sessions/rust-harness
	docker compose -f $(HARNESS_COMPOSE) run --rm harness $(ARGS) --chat

