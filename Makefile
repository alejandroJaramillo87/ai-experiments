# AI Infrastructure Base Repository Makefile
# RTX 5090 + 32-core CPU AI workstation management commands

.PHONY: help up down restart status logs clean gpu-up cpu-up ui-up
.PHONY: logs-gpu logs-cpu logs-ui logs-vllm shell-gpu shell-cpu shell-vllm
.PHONY: health update-models install shell test lint format
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

##@ Service Management

up: ## Start all AI services (GPU + CPU + UI)
	@echo "$(CYAN)Starting all AI services...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)All services started. Use 'make status' to check health.$(RESET)"

down: ## Stop all AI services
	@echo "$(CYAN)Stopping all AI services...$(RESET)"
	docker-compose down
	@echo "$(GREEN)All services stopped.$(RESET)"

restart: ## Restart all AI services
	@echo "$(CYAN)Restarting all AI services...$(RESET)"
	docker-compose restart
	@echo "$(GREEN)All services restarted.$(RESET)"

gpu-up: ## Start only GPU services (llama-gpu + vllm-gpu)
	@echo "$(CYAN)Starting GPU inference services...$(RESET)"
	docker-compose up -d llama-gpu vllm-gpu
	@echo "$(GREEN)GPU services started on ports 8004 (llama) and 8005 (vllm).$(RESET)"

cpu-up: ## Start only CPU services (llama-cpu-0/1/2)
	@echo "$(CYAN)Starting CPU inference services...$(RESET)"
	docker-compose up -d llama-cpu-0 llama-cpu-1 llama-cpu-2
	@echo "$(GREEN)CPU services started on ports 8001-8003.$(RESET)"

ui-up: ## Start only Open WebUI
	@echo "$(CYAN)Starting Open WebUI...$(RESET)"
	docker-compose up -d open-webui
	@echo "$(GREEN)Open WebUI started on http://localhost:3000$(RESET)"

##@ FLUX Image Generation

flux-up: ## Start ComfyUI with FLUX.1-dev
	@echo "$(CYAN)Starting ComfyUI with FLUX.1-dev...$(RESET)"
	docker-compose up -d comfyui-flux
	@echo "$(GREEN)ComfyUI started on http://localhost:8188$(RESET)"

flux-down: ## Stop ComfyUI service
	@echo "$(CYAN)Stopping ComfyUI...$(RESET)"
	docker-compose stop comfyui-flux
	@echo "$(GREEN)ComfyUI stopped.$(RESET)"

flux-logs: ## View ComfyUI logs
	docker-compose logs -f comfyui-flux

flux-shell: ## Access ComfyUI container shell
	docker exec -it comfyui-flux /bin/bash

flux-build: ## Rebuild ComfyUI container
	@echo "$(CYAN)Rebuilding ComfyUI container...$(RESET)"
	docker-compose build --no-cache comfyui-flux
	@echo "$(GREEN)Container rebuilt.$(RESET)"

flux-model-download: ## Download FLUX.1-dev model
	@echo "$(CYAN)Downloading FLUX.1-dev model (~23GB)...$(RESET)"
	@echo "This will take time depending on your internet connection."
	poetry run python scripts/utils/download_model_hf.py \
		--repo-id black-forest-labs/FLUX.1-dev \
		--output-dir /mnt/ai-data/models/flux
	@echo "$(GREEN)Model download complete.$(RESET)"

flux-status: ## Check ComfyUI service status
	@echo "$(CYAN)ComfyUI Service Status:$(RESET)"
	@docker-compose ps comfyui-flux
	@echo ""
	@echo "$(CYAN)GPU Memory Usage:$(RESET)"
	@docker exec comfyui-flux nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "Service not running"

##@ Monitoring & Debugging

status: ## Show status of all services
	@echo "$(CYAN)Service Status:$(RESET)"
	docker-compose ps
	@echo ""
	@echo "$(CYAN)Health Check Summary:$(RESET)"
	@docker-compose exec -T llama-cpu-0 curl -s http://localhost:8001/v1/health 2>/dev/null && echo "$(GREEN)✓ llama-cpu-0 (8001)$(RESET)" || echo "$(RED)✗ llama-cpu-0 (8001)$(RESET)"
	@docker-compose exec -T llama-cpu-1 curl -s http://localhost:8002/v1/health 2>/dev/null && echo "$(GREEN)✓ llama-cpu-1 (8002)$(RESET)" || echo "$(RED)✗ llama-cpu-1 (8002)$(RESET)"
	@docker-compose exec -T llama-cpu-2 curl -s http://localhost:8003/v1/health 2>/dev/null && echo "$(GREEN)✓ llama-cpu-2 (8003)$(RESET)" || echo "$(RED)✗ llama-cpu-2 (8003)$(RESET)"
	@docker-compose exec -T llama-gpu curl -s http://localhost:8004/v1/health 2>/dev/null && echo "$(GREEN)✓ llama-gpu (8004)$(RESET)" || echo "$(RED)✗ llama-gpu (8004)$(RESET)"
	@docker-compose exec -T vllm-gpu curl -s http://localhost:8006/health 2>/dev/null && echo "$(GREEN)✓ vllm-gpu (8005)$(RESET)" || echo "$(RED)✗ vllm-gpu (8005)$(RESET)"

logs: ## Show logs from all services
	docker-compose logs -f

logs-gpu: ## Show logs from GPU services only
	docker-compose logs -f llama-gpu vllm-gpu

logs-cpu: ## Show logs from CPU services only  
	docker-compose logs -f llama-cpu-0 llama-cpu-1 llama-cpu-2

logs-ui: ## Show logs from Open WebUI
	docker-compose logs -f open-webui

logs-vllm: ## Show logs from vLLM service specifically
	docker-compose logs -f vllm-gpu

health: ## Perform detailed health check on all services
	@echo "$(CYAN)Performing detailed health checks...$(RESET)"
	@echo "$(CYAN)CPU Services:$(RESET)"
	@curl -s http://localhost:8001/v1/health | jq . 2>/dev/null || echo "$(RED)CPU-0 (8001): Not responding$(RESET)"
	@curl -s http://localhost:8002/v1/health | jq . 2>/dev/null || echo "$(RED)CPU-1 (8002): Not responding$(RESET)"
	@curl -s http://localhost:8003/v1/health | jq . 2>/dev/null || echo "$(RED)CPU-2 (8003): Not responding$(RESET)"
	@echo "$(CYAN)GPU Services:$(RESET)"
	@curl -s http://localhost:8004/v1/health | jq . 2>/dev/null || echo "$(RED)GPU (8004): Not responding$(RESET)"
	@curl -s http://localhost:8005/health | jq . 2>/dev/null || echo "$(RED)vLLM (8005): Not responding$(RESET)"

##@ Development & Shell Access

shell-gpu: ## Shell access to GPU container
	docker-compose exec llama-gpu /bin/bash

shell-cpu: ## Shell access to CPU container (cpu-0)
	docker-compose exec llama-cpu-0 /bin/bash

shell-vllm: ## Shell access to vLLM container
	docker-compose exec vllm-gpu /bin/bash

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

##@ Python Environment (Poetry)

install: ## Install Poetry dependencies
	@echo "$(CYAN)Installing Poetry dependencies...$(RESET)"
	poetry install
	@echo "$(GREEN)Dependencies installed.$(RESET)"

update: ## Update Poetry dependencies
	@echo "$(CYAN)Updating Poetry dependencies...$(RESET)"
	poetry update
	@echo "$(GREEN)Dependencies updated.$(RESET)"

shell: ## Start Poetry shell environment
	poetry shell

##@ Testing & Code Quality

test: ## Run tests (if available)
	poetry run pytest -v

lint: ## Run linting with ruff
	poetry run ruff check .

format: ## Format code with black and ruff
	poetry run black .
	poetry run ruff format .

##@ Cleanup & Maintenance

clean: ## Clean up Docker resources (containers, networks, volumes)
	@echo "$(CYAN)Cleaning up Docker resources...$(RESET)"
	docker-compose down -v
	docker system prune -f
	@echo "$(GREEN)Cleanup completed.$(RESET)"

rebuild: ## Rebuild all containers from scratch
	@echo "$(CYAN)Rebuilding all containers...$(RESET)"
	docker-compose build --no-cache
	@echo "$(GREEN)All containers rebuilt.$(RESET)"

reset: ## Complete reset - stop services, clean up, rebuild
	@echo "$(CYAN)Performing complete reset...$(RESET)"
	make down
	make clean
	make rebuild
	@echo "$(GREEN)Reset completed. Run 'make up' to start services.$(RESET)"

##@ Quick Start Commands

demo: ## Start GPU services and open WebUI for quick demo
	@echo "$(CYAN)Starting demo environment...$(RESET)"
	make gpu-up
	make ui-up
	@sleep 5
	@echo "$(GREEN)Demo ready! Access WebUI at http://localhost:3000$(RESET)"

dev: ## Start CPU services for development (lighter resource usage)
	@echo "$(CYAN)Starting development environment (CPU only)...$(RESET)"
	make cpu-up
	@echo "$(GREEN)Development environment ready. CPU inference available on ports 8001-8003.$(RESET)"

##@ Help

help: ## Display this help message
	@echo "$(CYAN)AI Infrastructure Base Repository$(RESET)"
	@echo "$(CYAN)RTX 5090 + 32-core CPU AI Workstation Management$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(CYAN)<target>$(RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Service Endpoints:$(RESET)"
	@echo "  $(GREEN)CPU Services:$(RESET)    http://localhost:8001-8003/v1"
	@echo "  $(GREEN)GPU Service:$(RESET)     http://localhost:8004/v1"
	@echo "  $(GREEN)vLLM Service:$(RESET)    http://localhost:8005/v1"
	@echo "  $(GREEN)Open WebUI:$(RESET)      http://localhost:3000"
	@echo "  $(GREEN)ComfyUI:$(RESET)         http://localhost:8188"
	@echo ""
	@echo "$(YELLOW)Model Directory:$(RESET)  /mnt/ai-data/models/"