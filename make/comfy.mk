##@ Image & Video Generation (ComfyUI)

.PHONY: art-mode comfy-build comfy-down comfy-logs comfy-models-image comfy-models-video
.PHONY: comfy-shell comfy-status comfy-up llm-mode

comfy-up: ## Start ComfyUI (image and video generation)
	@echo "$(CYAN)Starting ComfyUI...$(RESET)"
	docker compose -f $(INFER_COMPOSE) up -d comfyui
	@echo "$(GREEN)ComfyUI started on http://localhost:8188$(RESET)"

comfy-down: ## Stop ComfyUI service
	@echo "$(CYAN)Stopping ComfyUI...$(RESET)"
	docker compose -f $(INFER_COMPOSE) stop comfyui
	@echo "$(GREEN)ComfyUI stopped.$(RESET)"

comfy-logs: ## View ComfyUI logs
	docker compose -f $(INFER_COMPOSE) logs -f comfyui

comfy-shell: ## Access ComfyUI container shell
	docker exec -it comfyui /bin/bash

comfy-build: ## Rebuild ComfyUI container
	@echo "$(CYAN)Rebuilding ComfyUI container...$(RESET)"
	docker compose -f $(INFER_COMPOSE) build --no-cache comfyui
	@echo "$(GREEN)Container rebuilt.$(RESET)"

comfy-models-image: ## Download image generation models (FLUX.2 klein 4B + 9B FP8)
	HF_HUB_ENABLE_HF_TRANSFER=1 uv run bash scripts/models/download_art_models.sh image

comfy-models-video: ## Download video generation models (LTX-2.3 FP8, Wan 2.2 FP8)
	HF_HUB_ENABLE_HF_TRANSFER=1 uv run bash scripts/models/download_art_models.sh video

comfy-status: ## Check ComfyUI service status
	@echo "$(CYAN)ComfyUI Service Status:$(RESET)"
	@docker compose -f $(INFER_COMPOSE) ps comfyui
	@echo ""
	@echo "$(CYAN)GPU Memory Usage:$(RESET)"
	@docker exec comfyui nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "Service not running"

art-mode: ## Stop GPU LLM services, start ComfyUI (one GPU consumer at a time)
	@echo "$(CYAN)Switching GPU to art generation...$(RESET)"
	docker compose -f $(INFER_COMPOSE) stop llama-gpu vllm-gpu
	docker compose -f $(INFER_COMPOSE) up -d comfyui
	@echo "$(GREEN)ComfyUI has the GPU: http://localhost:8188$(RESET)"

llm-mode: ## Stop ComfyUI, start GPU LLM services
	@echo "$(CYAN)Switching GPU to LLM inference...$(RESET)"
	docker compose -f $(INFER_COMPOSE) stop comfyui
	docker compose -f $(INFER_COMPOSE) up -d llama-gpu vllm-gpu
	@echo "$(GREEN)LLM services have the GPU: ports 8002 (llama) and 8003 (vllm)$(RESET)"

