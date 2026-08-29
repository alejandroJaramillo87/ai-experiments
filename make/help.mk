##@ Help

.PHONY: help

help: ## Display this help message
	@echo "$(CYAN)AI Infrastructure Base Repository$(RESET)"
	@echo "$(CYAN)RTX 5090 + 16-core Ryzen 9950X AI Workstation Management$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(CYAN)<target>$(RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Service Endpoints:$(RESET)"
	@echo "  $(GREEN)CPU Service:$(RESET)     http://localhost:8001/v1"
	@echo "  $(GREEN)GPU Service:$(RESET)     http://localhost:8002/v1"
	@echo "  $(GREEN)vLLM Service:$(RESET)    http://localhost:8003/v1"
	@echo "  $(GREEN)Open WebUI:$(RESET)      http://localhost:3000"
	@echo ""
	@echo "$(YELLOW)Model Directory:$(RESET)  /mnt/ai-data/models/"
