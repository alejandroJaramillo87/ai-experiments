##@ Cleanup & Maintenance

.PHONY: clean prune rebuild reset

clean: ## Stop the inference stack and remove its containers/networks/volumes (scoped; leaves agent-sandbox alone)
	@echo "$(CYAN)Cleaning up the inference stack (docker-compose.inference.yaml)...$(RESET)"
	docker compose -f $(INFER_COMPOSE) down -v
	@printf '[INFO] Scoped to docker-compose.inference.yaml. Agent-sandbox teardown: make agent-sandbox-clean. Host-wide sweep: make prune\n'
	@echo "$(GREEN)Cleanup completed.$(RESET)"

prune: ## DANGER: host-wide docker sweep (docker system prune -f) - removes ALL unused containers/networks/build cache
	@printf '[WARN] docker system prune -f removes ALL unused docker resources on the host, not just this repo\n'
	@printf '  This includes the agent-sandbox networks and build cache, and resources from other projects\n'
	docker system prune -f
	@echo "$(GREEN)Global prune completed.$(RESET)"

rebuild: ## Rebuild all containers from scratch
	@echo "$(CYAN)Rebuilding all containers...$(RESET)"
	docker compose -f $(INFER_COMPOSE) build --no-cache
	@echo "$(GREEN)All containers rebuilt.$(RESET)"

reset: ## Complete reset - stop services, clean up, rebuild
	@echo "$(CYAN)Performing complete reset...$(RESET)"
	$(MAKE) down
	$(MAKE) clean
	$(MAKE) rebuild
	@echo "$(GREEN)Reset completed. Run 'make up' to start services.$(RESET)"

