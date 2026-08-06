##@ Monitoring & Debugging

.PHONY: health list-models logs logs-cpu logs-gpu logs-ui logs-vllm shell-cpu shell-gpu
.PHONY: shell-vllm status update-models

status: ## Show status of all services
	@echo "$(CYAN)Service Status:$(RESET)"
	docker compose -f $(INFER_COMPOSE) ps
	@echo ""
	@echo "$(CYAN)Health Check Summary:$(RESET)"
	@docker compose -f $(INFER_COMPOSE) exec -T llama-cpu curl -s http://localhost:8001/v1/health 2>/dev/null && echo "$(GREEN)[OK] llama-cpu (8001)$(RESET)" || echo "$(RED)[DOWN] llama-cpu (8001)$(RESET)"
	@docker compose -f $(INFER_COMPOSE) exec -T llama-gpu curl -s http://localhost:8002/v1/health 2>/dev/null && echo "$(GREEN)[OK] llama-gpu (8002)$(RESET)" || echo "$(RED)[DOWN] llama-gpu (8002)$(RESET)"
	@docker compose -f $(INFER_COMPOSE) exec -T vllm-gpu curl -s http://localhost:8003/health 2>/dev/null && echo "$(GREEN)[OK] vllm-gpu (8003)$(RESET)" || echo "$(RED)[DOWN] vllm-gpu (8003)$(RESET)"

logs: ## Show logs from all services
	docker compose -f $(INFER_COMPOSE) logs -f

logs-gpu: ## Show logs from GPU services only
	docker compose -f $(INFER_COMPOSE) logs -f llama-gpu vllm-gpu

logs-cpu: ## Show logs from CPU service
	docker compose -f $(INFER_COMPOSE) logs -f llama-cpu

logs-ui: ## Show logs from Open WebUI
	docker compose -f $(INFER_COMPOSE) logs -f open-webui

logs-vllm: ## Show logs from vLLM service specifically
	docker compose -f $(INFER_COMPOSE) logs -f vllm-gpu

health: ## Perform detailed health check on all services
	@echo "$(CYAN)Performing detailed health checks...$(RESET)"
	@echo "$(CYAN)CPU Service:$(RESET)"
	@curl -s http://localhost:8001/v1/health | jq . 2>/dev/null || echo "$(RED)CPU (8001): Not responding$(RESET)"
	@echo "$(CYAN)GPU Services:$(RESET)"
	@curl -s http://localhost:8002/v1/health | jq . 2>/dev/null || echo "$(RED)GPU (8002): Not responding$(RESET)"
	@curl -s http://localhost:8003/health | jq . 2>/dev/null || echo "$(RED)vLLM (8003): Not responding$(RESET)"

##@ Development & Shell Access

shell-gpu: ## Shell access to GPU container
	docker compose -f $(INFER_COMPOSE) exec llama-gpu /bin/bash

shell-cpu: ## Shell access to CPU container
	docker compose -f $(INFER_COMPOSE) exec llama-cpu /bin/bash

shell-vllm: ## Shell access to vLLM container
	docker compose -f $(INFER_COMPOSE) exec vllm-gpu /bin/bash

##@ Model Management

update-models: ## Update model cache and permissions
	@echo "$(CYAN)Updating model cache...$(RESET)"
	@if [ -d "/mnt/ai-data/models" ]; then \
		sudo chown -R $(USER):$(USER) /mnt/ai-data/models/; \
		echo "$(GREEN)Model directory permissions updated.$(RESET)"; \
	else \
		echo "$(YELLOW)Model directory /mnt/ai-data/models not found.$(RESET)"; \
	fi

list-models: ## List available models in /mnt/ai-data/models
	@echo "$(CYAN)Available models:$(RESET)"
	@if [ -d "/mnt/ai-data/models" ]; then \
		ls -la /mnt/ai-data/models/; \
	else \
		echo "$(RED)Model directory /mnt/ai-data/models not found.$(RESET)"; \
	fi

