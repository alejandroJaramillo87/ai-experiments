##@ Rust harness (separate compose file; interactive tool, never started by `make up`)

.PHONY: harness-build harness-chat harness-run harness-runs harness-serve harness-serve-down

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

# Runs on the host, not in the container: the harness image carries no Python.
# The catalog is a host-side view of runs the container wrote to the shared log root.
harness-runs: ## List recent harness runs from the observability catalog (ARGS='-n 50 --failed')
	uv run --no-sync python -m common.obs.runs --cli harness $(ARGS)

harness-serve: ## Serve /v1/chat/completions on 127.0.0.1:8005 (HARNESS_MODEL=/app/models/hf/<model>)
	@mkdir -p sessions/rust-harness
	docker compose -f $(HARNESS_COMPOSE) up -d harness-serve
	@echo "$(GREEN)harness-serve started on 127.0.0.1:8005; loading the checkpoint takes minutes.$(RESET)"
	@echo "$(CYAN)Poll it with: python -c 'from common import serving; print(serving.wait_healthy(serving.SERVICES[\"harness\"]))'$(RESET)"

harness-serve-down: ## Stop the serving harness
	docker compose -f $(HARNESS_COMPOSE) down harness-serve
	@echo "$(GREEN)harness-serve stopped.$(RESET)"

