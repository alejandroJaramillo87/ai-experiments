##@ Service Management

.PHONY: cpu-up down gpu-up restart ui-up up

up: ## Start all AI services (GPU + CPU + UI)
	@echo "$(CYAN)Starting all AI services...$(RESET)"
	docker compose -f $(INFER_COMPOSE) up -d
	@echo "$(GREEN)All services started. Use 'make status' to check health.$(RESET)"

down: ## Stop all AI services
	@echo "$(CYAN)Stopping all AI services...$(RESET)"
	docker compose -f $(INFER_COMPOSE) down
	@echo "$(GREEN)All services stopped.$(RESET)"

restart: ## Restart all AI services
	@echo "$(CYAN)Restarting all AI services...$(RESET)"
	docker compose -f $(INFER_COMPOSE) restart
	@echo "$(GREEN)All services restarted.$(RESET)"

gpu-up: ## Start only GPU services (llama-gpu + vllm-gpu)
	@echo "$(CYAN)Starting GPU inference services...$(RESET)"
	docker compose -f $(INFER_COMPOSE) up -d llama-gpu vllm-gpu
	@echo "$(GREEN)GPU services started on ports 8002 (llama) and 8003 (vllm).$(RESET)"

cpu-up: ## Start only CPU service (llama-cpu)
	@echo "$(CYAN)Starting CPU inference service...$(RESET)"
	docker compose -f $(INFER_COMPOSE) up -d llama-cpu
	@echo "$(GREEN)CPU service started on port 8001.$(RESET)"

ui-up: ## Start only Open WebUI
	@echo "$(CYAN)Starting Open WebUI...$(RESET)"
	docker compose -f $(INFER_COMPOSE) up -d open-webui
	@echo "$(GREEN)Open WebUI started on http://localhost:3000$(RESET)"

