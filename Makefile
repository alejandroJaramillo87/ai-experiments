# AI Infrastructure Base Repository Makefile
# RTX 5090 + 16-core Ryzen 9950X AI workstation management commands

.PHONY: help up down restart status logs clean gpu-up cpu-up ui-up
.PHONY: logs-gpu logs-cpu logs-ui logs-vllm shell-gpu shell-cpu shell-vllm
.PHONY: health update-models install shell test lint format
.PHONY: flux-up flux-down flux-logs flux-shell flux-build flux-model-download flux-status
.PHONY: list-models update rebuild reset demo dev
.PHONY: agent-parity agent-preflight agent-skeptic-test command-sync agent-usage agent-versions
.PHONY: worktree-ls agent-run agent-export agent-sandbox
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
	docker compose up -d
	@echo "$(GREEN)All services started. Use 'make status' to check health.$(RESET)"

down: ## Stop all AI services
	@echo "$(CYAN)Stopping all AI services...$(RESET)"
	docker compose down
	@echo "$(GREEN)All services stopped.$(RESET)"

restart: ## Restart all AI services
	@echo "$(CYAN)Restarting all AI services...$(RESET)"
	docker compose restart
	@echo "$(GREEN)All services restarted.$(RESET)"

gpu-up: ## Start only GPU services (llama-gpu + vllm-gpu)
	@echo "$(CYAN)Starting GPU inference services...$(RESET)"
	docker compose up -d llama-gpu vllm-gpu
	@echo "$(GREEN)GPU services started on ports 8002 (llama) and 8003 (vllm).$(RESET)"

cpu-up: ## Start only CPU service (llama-cpu)
	@echo "$(CYAN)Starting CPU inference service...$(RESET)"
	docker compose up -d llama-cpu
	@echo "$(GREEN)CPU service started on port 8001.$(RESET)"

ui-up: ## Start only Open WebUI
	@echo "$(CYAN)Starting Open WebUI...$(RESET)"
	docker compose up -d open-webui
	@echo "$(GREEN)Open WebUI started on http://localhost:3000$(RESET)"

##@ FLUX Image Generation

flux-up: ## Start ComfyUI with FLUX.1-dev
	@echo "$(CYAN)Starting ComfyUI with FLUX.1-dev...$(RESET)"
	docker compose up -d comfyui-flux
	@echo "$(GREEN)ComfyUI started on http://localhost:8188$(RESET)"

flux-down: ## Stop ComfyUI service
	@echo "$(CYAN)Stopping ComfyUI...$(RESET)"
	docker compose stop comfyui-flux
	@echo "$(GREEN)ComfyUI stopped.$(RESET)"

flux-logs: ## View ComfyUI logs
	docker compose logs -f comfyui-flux

flux-shell: ## Access ComfyUI container shell
	docker exec -it comfyui-flux /bin/bash

flux-build: ## Rebuild ComfyUI container
	@echo "$(CYAN)Rebuilding ComfyUI container...$(RESET)"
	docker compose build --no-cache comfyui-flux
	@echo "$(GREEN)Container rebuilt.$(RESET)"

flux-model-download: ## Download FLUX.1-dev model
	@echo "$(CYAN)Downloading FLUX.1-dev model (~23GB)...$(RESET)"
	@echo "This will take time depending on your internet connection."
	HF_HUB_ENABLE_HF_TRANSFER=1 poetry run hf download \
		black-forest-labs/FLUX.1-dev \
		--local-dir /mnt/ai-data/models/flux
	@echo "$(GREEN)Model download complete.$(RESET)"

flux-status: ## Check ComfyUI service status
	@echo "$(CYAN)ComfyUI Service Status:$(RESET)"
	@docker compose ps comfyui-flux
	@echo ""
	@echo "$(CYAN)GPU Memory Usage:$(RESET)"
	@docker exec comfyui-flux nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "Service not running"

##@ Monitoring & Debugging

status: ## Show status of all services
	@echo "$(CYAN)Service Status:$(RESET)"
	docker compose ps
	@echo ""
	@echo "$(CYAN)Health Check Summary:$(RESET)"
	@docker compose exec -T llama-cpu curl -s http://localhost:8001/v1/health 2>/dev/null && echo "$(GREEN)[OK] llama-cpu (8001)$(RESET)" || echo "$(RED)[DOWN] llama-cpu (8001)$(RESET)"
	@docker compose exec -T llama-gpu curl -s http://localhost:8002/v1/health 2>/dev/null && echo "$(GREEN)[OK] llama-gpu (8002)$(RESET)" || echo "$(RED)[DOWN] llama-gpu (8002)$(RESET)"
	@docker compose exec -T vllm-gpu curl -s http://localhost:8003/health 2>/dev/null && echo "$(GREEN)[OK] vllm-gpu (8003)$(RESET)" || echo "$(RED)[DOWN] vllm-gpu (8003)$(RESET)"

logs: ## Show logs from all services
	docker compose logs -f

logs-gpu: ## Show logs from GPU services only
	docker compose logs -f llama-gpu vllm-gpu

logs-cpu: ## Show logs from CPU service
	docker compose logs -f llama-cpu

logs-ui: ## Show logs from Open WebUI
	docker compose logs -f open-webui

logs-vllm: ## Show logs from vLLM service specifically
	docker compose logs -f vllm-gpu

health: ## Perform detailed health check on all services
	@echo "$(CYAN)Performing detailed health checks...$(RESET)"
	@echo "$(CYAN)CPU Service:$(RESET)"
	@curl -s http://localhost:8001/v1/health | jq . 2>/dev/null || echo "$(RED)CPU (8001): Not responding$(RESET)"
	@echo "$(CYAN)GPU Services:$(RESET)"
	@curl -s http://localhost:8002/v1/health | jq . 2>/dev/null || echo "$(RED)GPU (8002): Not responding$(RESET)"
	@curl -s http://localhost:8003/health | jq . 2>/dev/null || echo "$(RED)vLLM (8003): Not responding$(RESET)"

##@ Development & Shell Access

shell-gpu: ## Shell access to GPU container
	docker compose exec llama-gpu /bin/bash

shell-cpu: ## Shell access to CPU container
	docker compose exec llama-cpu /bin/bash

shell-vllm: ## Shell access to vLLM container
	docker compose exec vllm-gpu /bin/bash

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

##@ Agent Tooling (OpenCode / Claude Code parity)

agent-parity: ## Diff shared agent bodies and report forked commands across the two tool dirs
	@fail=0; \
	for f in .claude/agents/*.md; do \
	  name=$$(basename "$$f"); \
	  other=".opencode/agents/$$name"; \
	  if [ ! -f "$$other" ]; then \
	    printf '[WARN] %s: no OpenCode counterpart\n' "$$name"; continue; \
	  fi; \
	  a=$$(mktemp); b=$$(mktemp); \
	  awk 'NR==1&&/^---$$/{f=1;next} f==1&&/^---$$/{f=2;next} f!=1' "$$f" > "$$a"; \
	  awk 'NR==1&&/^---$$/{f=1;next} f==1&&/^---$$/{f=2;next} f!=1' "$$other" > "$$b"; \
	  if diff -q "$$a" "$$b" >/dev/null; then \
	    printf '[OK] %s: bodies match\n' "$$name"; \
	  else \
	    printf '[ERROR] %s: bodies differ across the two agent dirs\n' "$$name"; \
	    fail=1; \
	  fi; \
	  rm -f "$$a" "$$b"; \
	done; \
	for c in .opencode/commands/*.md; do \
	  [ -e "$$c" ] || continue; \
	  name=$$(basename "$$c"); \
	  if [ -L "$$c" ]; then \
	    if [ -e "$$c" ]; then \
	      printf '[OK] %s: shared (symlink)\n' "$$name"; \
	    else \
	      printf '[ERROR] %s: broken command symlink\n' "$$name"; fail=1; \
	    fi; \
	  else \
	    printf '[INFO] %s: forked (OpenCode-only real file)\n' "$$name"; \
	  fi; \
	done; \
	exit $$fail

command-sync: ## (Re)create per-file symlinks from .claude/commands into .opencode/commands
	@mkdir -p .opencode/commands; \
	for f in .claude/commands/*.md; do \
	  name=$$(basename "$$f"); \
	  target=".opencode/commands/$$name"; \
	  if [ -e "$$target" ] && [ ! -L "$$target" ]; then \
	    printf '[SKIP] %s: forked real file, not overwriting\n' "$$name"; continue; \
	  fi; \
	  ln -sf "../../.claude/commands/$$name" "$$target"; \
	  printf '[OK] %s: linked\n' "$$name"; \
	done

agent-usage: ## Merge token usage (OpenCode state) and RunPod spend into one report
	@state="$${AGENT_USAGE_STATE:-.agent-state/usage.jsonl}"; \
	printf '=== Agent usage ===\n'; \
	printf '\nToken usage by provider/variant:\n'; \
	if [ -f "$$state" ] && command -v jq >/dev/null 2>&1; then \
	  jq -rs 'group_by(.provider+"/"+.variant)[] | "  [OK] \(.[0].provider)/\(.[0].variant): in=\(map(.input)|add) out=\(map(.output)|add) reasoning=\(map(.reasoning)|add)"' "$$state" \
	    || printf '  [WARN] could not parse %s\n' "$$state"; \
	else \
	  printf '  [INFO] no usage state at %s yet (OpenCode usage.ts writes it)\n' "$$state"; \
	fi; \
	printf '\nRunPod spend:\n'; \
	RUNPOD_REPORT=1 ./scripts/agents/runpod-idle-check.sh 2>/dev/null | sed 's/^/  /'; \
	printf '\nClaude Code token usage: use the built-in /usage in-session\n'; \
	printf '  (not merged here — no on-disk ledger, unlike the OpenCode plugin).\n'

agent-preflight: command-sync agent-parity ## Pre-flight checks: parity + supply-chain scan of both tool families
	@./scripts/agents/preflight-scan.sh "$${PREFLIGHT_DIR:-.}"
	@printf '[OK] Agent preflight complete\n'

agent-versions: ## Compare Dockerfile.agents CLI pins against the npm registry (check-only)
	@./scripts/agents/check-agent-versions.sh

agent-skeptic-test: ## Run the skeptic acceptance test (expects VERDICT: REFUTED)
	@if ! command -v claude >/dev/null 2>&1; then \
	  printf '[SKIP] Skeptic test: claude CLI not found\n'; exit 0; \
	fi; \
	printf '[RUN] Invoking skeptic on tests/agents/broken_completed_task.md\n'; \
	out=$$(claude -p "Use the skeptic subagent to verify the completed task in tests/agents/broken_completed_task.md" 2>/dev/null); \
	if printf '%s' "$$out" | grep -q 'REFUTED'; then \
	  printf '[OK] Skeptic test: returned REFUTED\n'; \
	else \
	  printf '[ERROR] Skeptic test: expected REFUTED, got:\n%s\n' "$$out"; exit 1; \
	fi

##@ Workflow Automation (escalation ladder — see docs/agents/README.md)

# Batch rung defaults. AGENT selects the CLI; tiers map to reasoning effort on one frontier
# model. CLAUDE_PERM is version-dependent — the guard.sh PreToolUse hook is the real safety
# floor in batch mode (it fires regardless of permission flag).
AGENT ?= claude
EFFORT ?= low
CLAUDE_PERM ?=
AGENT_USAGE_STATE ?= .agent-state/usage.jsonl

worktree-%: ## Create a git worktree ../ai-experiments-<name> on branch task/<name>
	@name="$*"; dir="../ai-experiments-$$name"; \
	if git worktree list --porcelain | grep -q "worktree $$(cd .. 2>/dev/null && pwd)/ai-experiments-$$name"; then \
	  printf '[SKIP] worktree already exists: %s\n' "$$dir"; exit 0; \
	fi; \
	git worktree add "$$dir" -b "task/$$name"; \
	printf '[OK] worktree: %s (branch task/%s)\n' "$$dir" "$$name"; \
	printf '[INFO] GPU/benchmark work stays serialized across all worktrees (see AGENTS.md)\n'

worktree-clean-%: ## Remove the worktree ../ai-experiments-<name>
	@name="$*"; dir="../ai-experiments-$$name"; \
	git worktree remove "$$dir" && printf '[OK] removed worktree: %s\n' "$$dir" \
	  || printf '[ERROR] could not remove %s (uncommitted changes? use git worktree remove --force)\n' "$$dir"

worktree-ls: ## List git worktrees
	@git worktree list

agent-run: ## Batch rung: AGENT=opencode|claude PROMPT=... [EFFORT=low] [ALLOW='Read Grep'] [CLAUDE_PERM=...]
	@if [ -z "$(PROMPT)" ]; then printf '[ERROR] set PROMPT="..."\n'; exit 1; fi; \
	case "$(AGENT)" in \
	  opencode) \
	    printf '[ERROR] opencode batch mode uses --auto (auto-approves everything not denied);\n'; \
	    printf '        it must run contained, never on the host.\n'; \
	    printf '  Solution: run it in the sandbox instead\n'; \
	    printf '  Command: make agent-sandbox AGENT=opencode NAME=<task> PROMPT="%s"\n' "$(PROMPT)"; \
	    exit 1 ;; \
	  claude) \
	    if ! command -v claude >/dev/null 2>&1; then printf '[SKIP] claude CLI not found\n'; exit 0; fi; \
	    printf '[RUN] claude -p (effort %s)\n' "$(EFFORT)"; \
	    out=$$(claude -p "$(PROMPT)" --output-format json --effort "$(EFFORT)" $(CLAUDE_PERM) $(if $(ALLOW),--allowedTools $(ALLOW),)); \
	    printf '%s\n' "$$out" | jq -r '.result // empty'; \
	    if command -v jq >/dev/null 2>&1 && printf '%s' "$$out" | jq -e '.usage' >/dev/null 2>&1; then \
	      mkdir -p "$$(dirname "$(AGENT_USAGE_STATE)")"; \
	      printf '%s' "$$out" | jq -c '{provider:"anthropic", variant:(.usage.model // "claude"), input:(.usage.input_tokens // .usage.input // 0), output:(.usage.output_tokens // .usage.output // 0), reasoning:(.usage.reasoning_tokens // 0)}' >> "$(AGENT_USAGE_STATE)"; \
	      printf '[OK] appended usage to %s (make agent-usage to view)\n' "$(AGENT_USAGE_STATE)"; \
	    fi ;; \
	  *) printf '[ERROR] AGENT must be opencode or claude (got: %s)\n' "$(AGENT)"; exit 1 ;; \
	esac

AGENTS_COMPOSE := docker-compose.agents.yaml

agent-sandbox: ## Contained agent session: AGENT=claude|opencode NAME=<task> [PROMPT="..."]
	@if [ -z "$(NAME)" ]; then printf '[ERROR] set NAME=<task> (names the git worktree)\n'; exit 1; fi; \
	name="$(NAME)"; dir="$$(cd .. && pwd)/ai-experiments-$$name"; \
	if [ ! -d "$$dir" ]; then \
	  printf '[INFO] creating worktree for %s\n' "$$name"; \
	  $(MAKE) --no-print-directory worktree-$$name || exit 1; \
	fi; \
	case "$(AGENT)" in \
	  claude) \
	    if [ -z "$(ANTHROPIC_API_KEY)" ]; then \
	      printf '[ERROR] ANTHROPIC_API_KEY not set (the only secret a Claude Code sandbox gets)\n'; \
	      printf '  Solution: export ANTHROPIC_API_KEY=<scoped key> before running\n'; exit 1; fi; \
	    if [ -n "$(PROMPT)" ]; then cmd="claude -p \"$(PROMPT)\" --effort $(EFFORT)"; else cmd="claude"; fi; \
	    printf '[RUN] contained Claude Code session (worktree %s)\n' "$$dir"; \
	    AGENT_WORKTREE="$$dir" RUNPOD_API_KEY="" \
	      docker compose -f $(AGENTS_COMPOSE) run --rm agent sh -lc "$$cmd" ;; \
	  opencode) \
	    if [ -z "$(RUNPOD_API_KEY)" ]; then \
	      printf '[ERROR] RUNPOD_API_KEY not set (the only secret an OpenCode sandbox gets)\n'; \
	      printf '  Solution: export RUNPOD_API_KEY=<endpoint-scoped key> before running\n'; exit 1; fi; \
	    if [ -n "$(PROMPT)" ]; then cmd="opencode run --auto \"$(PROMPT)\""; else cmd="opencode"; fi; \
	    printf '[RUN] contained OpenCode session (worktree %s)\n' "$$dir"; \
	    AGENT_WORKTREE="$$dir" ANTHROPIC_API_KEY="" \
	      docker compose -f $(AGENTS_COMPOSE) run --rm agent sh -lc "$$cmd" ;; \
	  *) printf '[ERROR] AGENT must be claude or opencode (got: %s)\n' "$(AGENT)"; exit 1 ;; \
	esac

SESSIONS_DIR ?= logs/sessions

agent-export: ## Export a session transcript: AGENT=opencode|claude ID=<session-id> [SESSIONS_DIR=...]
	@if [ -z "$(ID)" ]; then printf '[ERROR] set ID=<session-id>\n'; exit 1; fi; \
	if ! mkdir -p "$(SESSIONS_DIR)" 2>/dev/null; then \
	  printf '[ERROR] cannot create %s; set SESSIONS_DIR=<writable path>\n' "$(SESSIONS_DIR)"; exit 1; \
	fi; \
	out="$(SESSIONS_DIR)/$(ID).json"; \
	case "$(AGENT)" in \
	  opencode) \
	    if ! command -v opencode >/dev/null 2>&1; then printf '[SKIP] opencode CLI not found\n'; exit 0; fi; \
	    opencode export "$(ID)" > "$$out" && printf '[OK] exported to %s\n' "$$out" ;; \
	  claude) \
	    if ! command -v claude >/dev/null 2>&1; then printf '[SKIP] claude CLI not found\n'; exit 0; fi; \
	    claude -p --resume "$(ID)" --output-format json "Summarize this session for the research log." > "$$out" \
	      && printf '[OK] exported to %s (interactive transcript also at ~/.claude/projects/)\n' "$$out" ;; \
	  *) printf '[ERROR] AGENT must be opencode or claude\n'; exit 1 ;; \
	esac

##@ Cleanup & Maintenance

clean: ## Clean up Docker resources (containers, networks, volumes)
	@echo "$(CYAN)Cleaning up Docker resources...$(RESET)"
	docker compose down -v
	docker system prune -f
	@echo "$(GREEN)Cleanup completed.$(RESET)"

rebuild: ## Rebuild all containers from scratch
	@echo "$(CYAN)Rebuilding all containers...$(RESET)"
	docker compose build --no-cache
	@echo "$(GREEN)All containers rebuilt.$(RESET)"

reset: ## Complete reset - stop services, clean up, rebuild
	@echo "$(CYAN)Performing complete reset...$(RESET)"
	$(MAKE) down
	$(MAKE) clean
	$(MAKE) rebuild
	@echo "$(GREEN)Reset completed. Run 'make up' to start services.$(RESET)"

##@ Quick Start Commands

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

##@ Help

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
	@echo "  $(GREEN)ComfyUI:$(RESET)         http://localhost:8188"
	@echo ""
	@echo "$(YELLOW)Model Directory:$(RESET)  /mnt/ai-data/models/"