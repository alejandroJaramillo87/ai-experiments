##@ Interpretability Platform (interp Phase 2 PS1 - self-hosted Neuronpedia from a pinned upstream)

.PHONY: np-autointerp-up np-build np-clone np-down np-env np-graph-up np-inference-up
.PHONY: np-smoke np-status np-up

# Pinned upstream checkout on NVMe (multi-gigabyte after builds; never in this repository)
NP_SRC := /mnt/ai-data/interp/neuronpedia/src
NP_PIN_FILE := docker/interp/neuronpedia-pin.txt
NP_OVERRIDE := $(CURDIR)/docker/interp/neuronpedia.override.yaml
# GPU inference instance preset; list options with `cd $(NP_SRC) && make inference-list-configs`
MODEL_SOURCESET ?= gemma-2-2b-it.gemmascope-res-16k
# Runtime name of the inference stack's compose network (docker-compose.inference.yaml ai-network,
# prefixed by compose with the project directory name); override if docker network ls differs
AI_NETWORK_NAME ?= ai-experiments_ai-network
export AI_NETWORK_NAME

# Upstream invocations mirrored from the pinned checkout's Makefile, with $(NP_OVERRIDE)
# appended last. Do not add upstream logic here; deltas belong in the override file.
NP_COMPOSE_CORE = docker compose -f docker/compose.yaml -f $(NP_OVERRIDE) \
	--env-file .env.localhost --env-file .env

np-clone: interp-dirs ## Clone/update the Neuronpedia checkout to the pinned SHA (discards local changes)
	@PIN=$$(grep -v '^\#' $(NP_PIN_FILE) | awk '{print $$1}' | head -1); \
	if [ ! -d $(NP_SRC)/.git ]; then \
	  echo "$(CYAN)Cloning Neuronpedia to $(NP_SRC)...$(RESET)"; \
	  git clone https://github.com/hijohnnylin/neuronpedia.git $(NP_SRC); \
	fi; \
	git -C $(NP_SRC) fetch origin && git -C $(NP_SRC) checkout -f $$PIN
	@if ls docker/interp/patches/*.patch >/dev/null 2>&1; then \
	  for p in docker/interp/patches/*.patch; do \
	    echo "$(CYAN)Applying vendored patch $$p$(RESET)"; \
	    git -C $(NP_SRC) apply "$(CURDIR)/$$p" || exit 1; \
	  done; \
	fi
	@echo "$(GREEN)Neuronpedia checkout pinned per $(NP_PIN_FILE).$(RESET)"

np-env: ## Merge this repo's env deltas and .env.interp secrets into the checkout
	@./scripts/interp/np_env.sh

np-build: np-env ## Build platform images (webapp trio; CUDA inference/autointerp/graph)
	@echo "$(CYAN)Building Neuronpedia webapp, db-init, postgres...$(RESET)"
	cd $(NP_SRC) && ENV_FILE=../.env.localhost \
		docker compose -f docker/compose.yaml -f $(NP_OVERRIDE) build webapp db-init postgres
	@echo "$(CYAN)Building Neuronpedia CUDA images (inference, autointerp, graph)...$(RESET)"
	cd $(NP_SRC) && ENV_FILE=../.env.localhost BUILD_TYPE=cuda \
		docker compose -f docker/compose.yaml -f $(NP_OVERRIDE) build inference autointerp graph
	@echo "$(GREEN)Neuronpedia images built.$(RESET)"

np-up: np-env ## Start webapp + postgres + db-init (CPU only; ports 3001/5432, loopback)
	cd $(NP_SRC) && $(NP_COMPOSE_CORE) up -d webapp db-init postgres
	@echo "$(GREEN)Neuronpedia webapp: http://localhost:3001$(RESET)"

np-inference-up: np-env ## Preflight, then start the GPU inference instance (port 5002): make np-inference-up [MODEL_SOURCESET=...]
	@./scripts/interp/preflight.sh
	@test -f $(NP_SRC)/.env.inference.$(MODEL_SOURCESET) || \
		{ echo "$(RED)No such config: .env.inference.$(MODEL_SOURCESET) (see inference-list-configs upstream)$(RESET)"; exit 1; }
	cd $(NP_SRC) && RELOAD=0 ENV_FILE=../.env.inference.$(MODEL_SOURCESET) \
		docker compose -f docker/compose.yaml \
		-f docker/compose.inference.dev.yaml -f docker/compose.inference.gpu.yaml \
		-f $(NP_OVERRIDE) \
		--env-file .env.inference.$(MODEL_SOURCESET) --env-file .env.localhost --env-file .env \
		up -d inference
	@echo "$(GREEN)Neuronpedia inference up on 127.0.0.1:5002 ($(MODEL_SOURCESET)).$(RESET)"

np-autointerp-up: np-env ## Start autointerp (port 5003; serving-dependent: vllm-gpu must be up. NP_GPU=1 instead attaches the GPU for the embedding scorer)
	@if [ "$(NP_GPU)" = "1" ]; then ./scripts/interp/preflight.sh; \
	else ./scripts/interp/np_autointerp_preflight.sh; fi
	cd $(NP_SRC) && RELOAD=0 ENV_FILE=../.env.localhost \
		docker compose -f docker/compose.yaml \
		-f docker/compose.autointerp.dev.yaml \
		$(if $(filter 1,$(NP_GPU)),-f docker/compose.autointerp.gpu.yaml,) \
		-f $(NP_OVERRIDE) \
		--env-file .env.localhost --env-file .env \
		up -d autointerp
	@echo "$(GREEN)Neuronpedia autointerp up on 127.0.0.1:5003.$(RESET)"

np-graph-up: np-env ## Preflight, then start the graph server (port 5004, GPU)
	@./scripts/interp/preflight.sh
	cd $(NP_SRC) && RELOAD=0 ENV_FILE=.env.localhost \
		docker compose --project-directory . -f docker/compose.yaml \
		-f docker/compose.graph.dev.yaml -f docker/compose.graph.gpu.yaml \
		-f $(NP_OVERRIDE) \
		--env-file .env.localhost --env-file apps/graph/.env \
		up -d graph
	@echo "$(GREEN)Neuronpedia graph server up on 127.0.0.1:5004.$(RESET)"

np-down: ## Stop the whole Neuronpedia stack (always safe; data persists in the bind mounts)
	cd $(NP_SRC) && $(NP_COMPOSE_CORE) down
	@echo "$(GREEN)Neuronpedia stack stopped.$(RESET)"

np-status: ## Show Neuronpedia container and port state
	@echo "$(CYAN)Neuronpedia containers:$(RESET)"
	@docker ps --filter label=com.docker.compose.project=neuronpedia \
		--format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

np-smoke: ## Run the PS1 smoke check (loopback bindings, webapp, postgres, sm_120)
	@./scripts/interp/np_smoke.sh

