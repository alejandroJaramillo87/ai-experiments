##@ Quick Start Commands

.PHONY: demo dev

demo: ## Start GPU services and open WebUI for quick demo
	@echo "$(CYAN)Starting demo environment...$(RESET)"
	$(MAKE) gpu-up
	$(MAKE) ui-up
	@sleep 5
	@echo "$(GREEN)Demo ready! Access WebUI at http://localhost:3000$(RESET)"

dev: ## Start CPU service for development (lighter resource usage)
	@echo "$(CYAN)Starting development environment (CPU only)...$(RESET)"
	$(MAKE) cpu-up
	@echo "$(GREEN)Development environment ready. CPU inference available on port 8001.$(RESET)"
