# AI Infrastructure Base Repository Makefile
# RTX 5090 + 16-core Ryzen 9950X AI workstation management commands

.PHONY: help up down restart status logs clean gpu-up cpu-up ui-up
.PHONY: logs-gpu logs-cpu logs-ui logs-vllm shell-gpu shell-cpu shell-vllm
.PHONY: health update-models install shell test lint format
.PHONY: comfy-up comfy-down comfy-logs comfy-shell comfy-build comfy-status
.PHONY: comfy-models-image comfy-models-video art-mode llm-mode
.PHONY: list-models update rebuild reset demo dev
.PHONY: agent-parity agent-preflight agent-skeptic-test command-sync agent-usage agent-versions
.PHONY: worktree-ls worktree-clean agent-run agent-export agent-sandbox
.PHONY: np-clone np-env np-build np-up np-inference-up np-autointerp-up np-graph-up
.PHONY: np-down np-status np-smoke
.PHONY: interp-build interp-up interp-shell interp-smoke interp-down interp-dirs
.PHONY: steer-extract steer-up steer-down steer-scorecard
.PHONY: probe-capture probe-train probe-eval probe-monitor logproxy-up logproxy-down
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Cap parallel source compilation during `make install`. Several pinned deps
# (flash-attn, deepspeed, liger-kernel, torchao) build CUDA kernels from source
# and default to one nvcc/cc1plus job per core, each needing several GB of RAM.
# Unbounded, that spikes past host RAM and OOM-kills other processes (e.g. the
# editor). Override for a bigger/smaller box: `make install MAX_JOBS=8`.
MAX_JOBS ?= 4
NVCC_THREADS ?= 2

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

##@ Image & Video Generation (ComfyUI)

comfy-up: ## Start ComfyUI (image and video generation)
	@echo "$(CYAN)Starting ComfyUI...$(RESET)"
	docker compose up -d comfyui
	@echo "$(GREEN)ComfyUI started on http://localhost:8188$(RESET)"

comfy-down: ## Stop ComfyUI service
	@echo "$(CYAN)Stopping ComfyUI...$(RESET)"
	docker compose stop comfyui
	@echo "$(GREEN)ComfyUI stopped.$(RESET)"

comfy-logs: ## View ComfyUI logs
	docker compose logs -f comfyui

comfy-shell: ## Access ComfyUI container shell
	docker exec -it comfyui /bin/bash

comfy-build: ## Rebuild ComfyUI container
	@echo "$(CYAN)Rebuilding ComfyUI container...$(RESET)"
	docker compose build --no-cache comfyui
	@echo "$(GREEN)Container rebuilt.$(RESET)"

comfy-models-image: ## Download image generation models (FLUX.2 klein 4B + 9B FP8)
	HF_HUB_ENABLE_HF_TRANSFER=1 uv run bash scripts/models/download_art_models.sh image

comfy-models-video: ## Download video generation models (LTX-2.3 FP8, Wan 2.2 FP8)
	HF_HUB_ENABLE_HF_TRANSFER=1 uv run bash scripts/models/download_art_models.sh video

comfy-status: ## Check ComfyUI service status
	@echo "$(CYAN)ComfyUI Service Status:$(RESET)"
	@docker compose ps comfyui
	@echo ""
	@echo "$(CYAN)GPU Memory Usage:$(RESET)"
	@docker exec comfyui nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "Service not running"

art-mode: ## Stop GPU LLM services, start ComfyUI (one GPU consumer at a time)
	@echo "$(CYAN)Switching GPU to art generation...$(RESET)"
	docker compose stop llama-gpu vllm-gpu
	docker compose up -d comfyui
	@echo "$(GREEN)ComfyUI has the GPU: http://localhost:8188$(RESET)"

llm-mode: ## Stop ComfyUI, start GPU LLM services
	@echo "$(CYAN)Switching GPU to LLM inference...$(RESET)"
	docker compose stop comfyui
	docker compose up -d llama-gpu vllm-gpu
	@echo "$(GREEN)LLM services have the GPU: ports 8002 (llama) and 8003 (vllm)$(RESET)"

##@ Training (Phase 0 D1 - separate compose file; never runs with inference)

TRAIN_COMPOSE := docker-compose.training.yaml

train-build: ## Build the training image (runs the smoke test at build; fails on regression)
	@echo "$(CYAN)Building training image...$(RESET)"
	docker compose -f $(TRAIN_COMPOSE) build
	@echo "$(GREEN)Training image built.$(RESET)"

train-shell: ## Interactive shell in the training container (GPU + /mnt/ai-data/training mounted)
	docker compose -f $(TRAIN_COMPOSE) run --rm training /bin/bash

train-smoke: ## Run the eight-step training smoke test on the workstation (enforces CUDA sm_120)
	docker compose -f $(TRAIN_COMPOSE) run --rm training python scripts/training/smoke_test.py

prepare-data: ## Tokenize a dataset into shards (Phase 0 D2): make prepare-data ARGS="--source ... --tokenizer ... --out ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm training python scripts/training/prepare_data.py $(ARGS)

prepare-corpus: ## Filter, dedup, decontaminate, and shard a corpus (Phase 2 P2-D1): make prepare-corpus ARGS="--source ... --tokenizer ... --work-dir ... --out ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm training python scripts/training/prepare_corpus.py $(ARGS)

build-contamination: ## Build the benchmark decontamination index (Phase 2 P2-D1): make build-contamination ARGS="--out /mnt/ai-data/training/datasets/_contamination/v1"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python scripts/training/build_contamination.py $(ARGS)

prepare-sft: ## Prepare masked SFT examples (Phase 1 F1): make prepare-sft ARGS="--source ... --tokenizer ... --model-type ... --out ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm training python scripts/training/prepare_sft.py $(ARGS)

train-tokenizer: ## Train a byte-level BPE tokenizer (Phase 0 D3): make train-tokenizer ARGS="--corpus ... --out ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm training python scripts/training/train_tokenizer.py $(ARGS)

train-validate: ## Validate a recipe (Phase 0 D5): make train-validate CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="optimizer.lr=1e-4"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python scripts/training/validate_recipe.py $(CONFIG) $(ARGS)

memfit: ## Estimate VRAM fit for a recipe (Phase 0 D4): make memfit CONFIG=training/recipes/reference/pretrain-1b.yaml ARGS="--solve batch"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training memfit --config $(CONFIG) $(ARGS)

train-calibrate: ## Measure real peak VRAM for a recipe (Phase 0 D4, GPU): make train-calibrate CONFIG=... ARGS="--out record.json"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training calibrate --config $(CONFIG) $(ARGS)

train: ## Run the reference training loop (Phase 0 D6): make train CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="checkpoint.resume=latest"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training train --config $(CONFIG) $(ARGS)

train-sft: ## Run a supervised fine-tune on TRL (Phase 1 F3): make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="optimizer.lr=1e-4"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training train-sft --config $(CONFIG) $(ARGS)

train-distill: ## Distill a student from a teacher (Phase 1 F5): make train-distill CONFIG=training/recipes/reference/distill-e2b.yaml ARGS="distill.alpha=0.3"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training train-distill --config $(CONFIG) $(ARGS)

distill-cache: ## Materialize teacher top-K logprobs for cached-mode distillation (Phase 1 F5): make distill-cache CONFIG=...
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training distill-cache --config $(CONFIG) $(ARGS)

export-build: ## Build the GGUF export image (pinned llama.cpp converter and quantizer, Phase 0 D7)
	@echo "$(CYAN)Building export image...$(RESET)"
	docker compose -f $(TRAIN_COMPOSE) build export
	@echo "$(GREEN)Export image built.$(RESET)"

export-gguf: ## Export a checkpoint to GGUF (Phase 0 D7): make export-gguf CHECKPOINT=/mnt/ai-data/training/.../step_000000010 ARGS="--quant Q8_0 --tokenizer ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm export python -m training export --checkpoint $(CHECKPOINT) $(ARGS)

serve-eval: ## Swap a GGUF into llama-gpu, run the smoke evaluation, restore (Phase 0 D7): make serve-eval MODEL=/app/models/trained/<run>/<name>-Q8_0.gguf ARGS="--keep-serving"
	python3 scripts/eval/serve_eval.py --model $(MODEL) $(ARGS)

eval-ft: ## Baseline-versus-tuned scorecard (Phase 1 F7, host-run): make eval-ft ARGS="--base ... --tuned ... --dataset ... --base-gguf ... --tuned-gguf ... --judge-gguf ..."
	python3 scripts/eval/eval_ft.py $(ARGS)

promote-scorecard: ## Copy an eval run's scorecard into docs/training/scorecards/ with provenance (Phase 1 F8, host-run): make promote-scorecard ARGS="--run-dir ... --card sft-qlora-gemma-e4b"
	python3 scripts/eval/promote_scorecard.py $(ARGS)

generate-teacher: ## Batch-generate teacher data through the serving stack (Phase 1 F4, host-run): make generate-teacher ARGS="--pipeline respond --prompts ... --teacher ... --out ..."
	python3 scripts/training/generate_teacher.py $(ARGS)
##@ Interpretability (interp Phase 0 I1 - separate compose file; GPU inference must be down)

INTERP_COMPOSE := docker-compose.interp.yaml

interp-build: ## Build the interp workbench image (runs the smoke test at build; fails on regression)
	@echo "$(CYAN)Building interp workbench image...$(RESET)"
	docker compose -f $(INTERP_COMPOSE) build
	@echo "$(GREEN)Interp workbench image built.$(RESET)"

interp-up: ## Preflight, then start JupyterLab on 127.0.0.1:8888 and print the tokened URL
	@./scripts/interp/preflight.sh
	$(eval JUPYTER_TOKEN := $(shell openssl rand -hex 24))
	JUPYTER_TOKEN=$(JUPYTER_TOKEN) docker compose -f $(INTERP_COMPOSE) up -d interp
	@echo "$(GREEN)JupyterLab: http://127.0.0.1:8888/?token=$(JUPYTER_TOKEN)$(RESET)"

interp-shell: ## Preflight, then an interactive shell in the workbench with GPU attached
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp /bin/bash

interp-smoke: ## Run the seven-step interp smoke test on the workstation (enforces CUDA sm_120)
	docker compose -f $(INTERP_COMPOSE) run --rm interp python scripts/interp/smoke_test.py

interp-down: ## Stop the interp workbench
	docker compose -f $(INTERP_COMPOSE) down

interp-dirs: ## Create the /mnt/ai-data/interp storage tree on the host
	mkdir -p /mnt/ai-data/interp/activations /mnt/ai-data/interp/saes \
	         /mnt/ai-data/interp/probes /mnt/ai-data/interp/vectors \
	         /mnt/ai-data/interp/neuronpedia /mnt/ai-data/interp/hf-cache \
	         /mnt/ai-data/interp/neuronpedia/postgres /mnt/ai-data/interp/logs \
	         /mnt/ai-data/interp/monitor
	@echo "$(GREEN)Interp storage tree ready under /mnt/ai-data/interp$(RESET)"

##@ Interpretability write-back (interp Phase 3 W1-W4)

# Steered service and the host->container mapping of the vectors mount
STEER_SERVICE ?= llama-gpu
VECTOR_CONTAINER = $(patsubst /mnt/ai-data/interp/vectors%,/app/vectors%,$(VECTOR))/vector.gguf

steer-extract: ## Extract and export a control vector in the workbench (ARGS="--method repeng --model ... --pairs ... --out ...")
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python scripts/interp/steer_extract.py $(ARGS)

steer-up: ## Recreate the service steered (VECTOR=/mnt/ai-data/interp/vectors/<name> [SCALE=1.0] [LAYER_RANGE="start end"] [STEER_SERVICE=llama-gpu])
	@test -n "$(VECTOR)" || { echo "VECTOR=/mnt/ai-data/interp/vectors/<name> is required"; exit 2; }
	@test -f "$(VECTOR)/vector.gguf" || { echo "vector not found: $(VECTOR)/vector.gguf"; exit 1; }
	@case "$(VECTOR)" in /mnt/ai-data/interp/vectors/*) ;; *) echo "VECTOR must live under /mnt/ai-data/interp/vectors (the mounted tree)"; exit 1;; esac
	CONTROL_VECTOR="$(VECTOR_CONTAINER)" CONTROL_VECTOR_SCALE="$(SCALE)" \
	CONTROL_VECTOR_LAYER_RANGE="$(LAYER_RANGE)" \
	docker compose up -d --force-recreate $(STEER_SERVICE)
	@echo "$(GREEN)$(STEER_SERVICE) recreated steered: $(VECTOR_CONTAINER)$(if $(SCALE), scale $(SCALE),)$(RESET)"

steer-down: ## Recreate the service unsteered (restores production state)
	CONTROL_VECTOR= CONTROL_VECTOR_SCALE= CONTROL_VECTOR_LAYER_RANGE= \
	docker compose up -d --force-recreate $(STEER_SERVICE)
	@echo "$(GREEN)$(STEER_SERVICE) recreated unsteered$(RESET)"

steer-scorecard: ## Run the three-arm scorecard against live services (VECTOR=... ATTRIBUTE="..." [ARGS=...])
	@test -n "$(VECTOR)" || { echo "VECTOR=/mnt/ai-data/interp/vectors/<name> is required"; exit 2; }
	@test -n "$(ATTRIBUTE)" || { echo "ATTRIBUTE=\"behavior the judge scores\" is required"; exit 2; }
	uv run python scripts/interp/steer_scorecard.py --vector "$(VECTOR)" --attribute "$(ATTRIBUTE)" $(ARGS)

probe-capture: ## Capture labeled activations in the workbench (ARGS="--model ... --dataset ... --layers ... --out ...")
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python scripts/interp/probe_capture.py $(ARGS)

probe-train: ## Train and register probes from a capture (ARGS="--capture ... --name ...")
	docker compose -f $(INTERP_COMPOSE) run --rm interp python scripts/interp/probe_train.py $(ARGS)

probe-eval: ## Regenerate a probe evaluation report with black-box baselines (serving-dependent; ARGS="--probe ... --label-name ...")
	uv run python scripts/interp/probe_eval.py $(ARGS)

probe-monitor: ## Replay a serving-log window and score probes (batch; refuses while inference is up; ARGS="--window 7d --probes ... --model ... --model-hash ...")
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python scripts/interp/probe_monitor.py $(ARGS)

logproxy-up: ## Start the request-logging proxy on 127.0.0.1:8012 (opt-in; compose profile interp)
	docker compose --profile interp up -d interp-logproxy
	@echo "$(GREEN)logproxy up: point clients at http://127.0.0.1:8012 (logs under /mnt/ai-data/interp/logs)$(RESET)"

logproxy-down: ## Stop the request-logging proxy
	docker compose --profile interp rm -sf interp-logproxy

##@ Interpretability Platform (interp Phase 2 PS1 - self-hosted Neuronpedia from a pinned upstream)

# Pinned upstream checkout on NVMe (multi-gigabyte after builds; never in this repository)
NP_SRC := /mnt/ai-data/interp/neuronpedia/src
NP_PIN_FILE := docker/interp/neuronpedia-pin.txt
NP_OVERRIDE := $(CURDIR)/docker/interp/neuronpedia.override.yaml
# GPU inference instance preset; list options with `cd $(NP_SRC) && make inference-list-configs`
MODEL_SOURCESET ?= gemma-2-2b-it.gemmascope-res-16k
# Runtime name of the inference stack's compose network (docker-compose.yaml ai-network,
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

##@ Python Environment (uv)

install: ## Sync the uv environment (all groups and extras)
	@echo "$(CYAN)Syncing uv environment (MAX_JOBS=$(MAX_JOBS))...$(RESET)"
	MAX_JOBS=$(MAX_JOBS) NVCC_THREADS=$(NVCC_THREADS) uv sync --all-groups --all-extras
	@echo "$(GREEN)Environment synced.$(RESET)"

update: ## Update Python dependencies (lock upgrade + sync + verification)
	./scripts/update/update_python.sh

shell: ## Show how to activate the project environment
	@echo "Run: source .venv/bin/activate (or prefix commands with 'uv run')"

##@ Testing & Code Quality

test: ## Run tests (if available)
	uv run pytest -v

lint: ## Run linting with ruff
	uv run ruff check .

format: ## Format code with ruff
	uv run ruff format .

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
	done; \
	for t in .opencode/commands/*.md; do \
	  [ -L "$$t" ] || continue; \
	  if [ ! -e "$$t" ]; then \
	    rm "$$t"; printf '[OK] %s: pruned (source removed)\n' "$$(basename "$$t")"; \
	  fi; \
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
	if printf '%s' "$$out" | grep -q 'VERDICT: REFUTED'; then \
	  printf '[OK] Skeptic test: returned VERDICT: REFUTED\n'; \
	else \
	  printf '[ERROR] Skeptic test: expected "VERDICT: REFUTED", got:\n%s\n' "$$out"; exit 1; \
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
	if git worktree list --porcelain | grep -qx "worktree $$(cd .. 2>/dev/null && pwd)/ai-experiments-$$name"; then \
	  printf '[SKIP] worktree already exists: %s\n' "$$dir"; exit 0; \
	fi; \
	if [ -e "$$dir" ]; then \
	  printf '[ERROR] %s exists but is not a registered worktree\n' "$$dir"; \
	  printf '  Solution: inspect it, then remove it (rmdir if empty) and retry\n'; \
	  exit 1; \
	fi; \
	if git show-ref -q "refs/heads/task/$$name"; then \
	  git worktree add "$$dir" "task/$$name" || exit 1; \
	  printf '[OK] worktree: %s (reusing existing branch task/%s)\n' "$$dir" "$$name"; \
	else \
	  git worktree add "$$dir" -b "task/$$name" || exit 1; \
	  printf '[OK] worktree: %s (branch task/%s)\n' "$$dir" "$$name"; \
	fi; \
	printf '[INFO] GPU/benchmark work stays serialized across all worktrees (see AGENTS.md)\n'

# `make worktree-clean <name>`: make parses <name> as a second goal, not an argument,
# so capture it at parse time and register a no-op target for it. Guarded to fire only
# when worktree-clean is the first goal, so it can never shadow a real target.
ifeq ($(firstword $(MAKECMDGOALS)),worktree-clean)
WT_CLEAN_ARG := $(word 2,$(MAKECMDGOALS))
ifneq ($(WT_CLEAN_ARG),)
$(WT_CLEAN_ARG): ; @:
endif
endif

# Shared removal shell. Expects $$arg set; accepts a bare name, the task/<name>
# branch, or a worktree path (anything containing ai-experiments-). Exits 1 on
# failure so callers and scripts see the error.
define WORKTREE_REMOVE_SH
	case "$$arg" in \
	  *ai-experiments-*) name="$${arg##*ai-experiments-}" ;; \
	  task/*)            name="$${arg#task/}" ;; \
	  *)                 name="$$arg" ;; \
	esac; \
	dir="../ai-experiments-$$name"; \
	force=""; [ -n "$(FORCE)" ] && force="--force"; \
	if git worktree remove $$force "$$dir"; then \
	  printf '[OK] removed worktree: %s\n' "$$dir"; \
	else \
	  printf '[ERROR] could not remove %s (uncommitted changes? use FORCE=1 or git worktree remove --force)\n' "$$dir"; \
	  exit 1; \
	fi
endef

# Explicit target so a bare `make worktree-clean` cannot match the worktree-% creation
# pattern (which would create a worktree named "clean").
worktree-clean: ## Remove a worktree: make worktree-clean <name> (name, task/<name>, or path)
	@arg="$(or $(WT_CLEAN_ARG),$(NAME))"; \
	if [ -z "$$arg" ]; then \
	  printf '[ERROR] worktree-clean requires a name: make worktree-clean <name>\n'; \
	  printf '  Also accepted: make worktree-clean NAME=<name>; make worktree-clean-<name>\n'; \
	  printf '  Existing worktrees:\n'; git worktree list | sed 's/^/    /'; exit 1; \
	fi; \
	$(WORKTREE_REMOVE_SH)

worktree-clean-%: ## Remove the worktree ../ai-experiments-<name>
	@arg="$*"; \
	$(WORKTREE_REMOVE_SH)

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
# Preseeded each session: the sandbox home is ephemeral tmpfs, so no Claude
# Code config survives between runs. Two files are seeded on launch:
#   ~/.claude.json           — onboarding flag, theme, additionalModelOptionsCache,
#                              and unpinFable5LaunchEffort (without it the CLI pins a
#                              Fable session's launch effort to its recommended value,
#                              silently overriding the seeded effortLevel)
#                              (account-gated models like Fable only reach the /model
#                              picker via an interactive OAuth profile fetch, which the
#                              headless CLAUDE_CODE_OAUTH_TOKEN auth path never performs;
#                              access itself stays enforced server-side by the token)
#   ~/.claude/settings.json  — persistent defaults: model from CLAUDE_MODEL, effortLevel
#                              from EFFORT (default low, override per run with
#                              EFFORT=medium), and theme
# In-session changes (/effort, /model) intentionally last one session only.
CLAUDE_THEME ?= dark
CLAUDE_MODEL ?= claude-fable-5[1m]
CLAUDE_MODEL_OPTIONS := [{\"value\":\"claude-fable-5[1m]\",\"label\":\"Fable\",\"description\":\"Fable 5 - account-gated tier above Opus\"}]

agent-sandbox: ## Contained agent session: AGENT=claude|opencode NAME=<task> [PROMPT="..."] [EFFORT=low] [CLAUDE_MODEL=...] [MODEL=<flag override>]
	@if [ -z "$(NAME)" ]; then printf '[ERROR] set NAME=<task> (names the git worktree)\n'; exit 1; fi; \
	name="$(NAME)"; dir="$$(cd .. && pwd)/ai-experiments-$$name"; \
	if [ ! -d "$$dir" ]; then \
	  printf '[INFO] creating worktree for %s\n' "$$name"; \
	  $(MAKE) --no-print-directory worktree-$$name || exit 1; \
	fi; \
	case "$(AGENT)" in \
	  claude) \
	    if [ -z "$(CLAUDE_CODE_OAUTH_TOKEN)" ]; then \
	      printf '[ERROR] CLAUDE_CODE_OAUTH_TOKEN not set (the only secret a Claude Code sandbox gets)\n'; \
	      printf '  Solution: run `claude setup-token` on the host, then export CLAUDE_CODE_OAUTH_TOKEN=<token>\n'; exit 1; fi; \
	    seed="printf '{\"theme\":\"$(CLAUDE_THEME)\",\"hasCompletedOnboarding\":true,\"unpinFable5LaunchEffort\":true,\"additionalModelOptionsCache\":$(CLAUDE_MODEL_OPTIONS)}' > ~/.claude.json; \
	      mkdir -p ~/.claude; \
	      printf '{\"model\":\"$(CLAUDE_MODEL)\",\"effortLevel\":\"$(EFFORT)\",\"theme\":\"$(CLAUDE_THEME)\"}' > ~/.claude/settings.json"; \
	    model=""; if [ -n "$(MODEL)" ]; then model=" --model \"$(MODEL)\""; fi; \
	    if [ -n "$(PROMPT)" ]; then cmd="$$seed; claude -p \"$(PROMPT)\" --effort $(EFFORT)$$model"; else cmd="$$seed; claude$$model"; fi; \
	    printf '[RUN] contained Claude Code session (worktree %s)\n' "$$dir"; \
	    AGENT_WORKTREE="$$dir" RUNPOD_API_KEY="" \
	      docker compose -f $(AGENTS_COMPOSE) run --rm agent sh -lc "$$cmd" ;; \
	  opencode) \
	    if [ -z "$(RUNPOD_API_KEY)" ]; then \
	      printf '[ERROR] RUNPOD_API_KEY not set (the only secret an OpenCode sandbox gets)\n'; \
	      printf '  Solution: export RUNPOD_API_KEY=<endpoint-scoped key> before running\n'; exit 1; fi; \
	    if [ -n "$(PROMPT)" ]; then cmd="opencode run --auto \"$(PROMPT)\""; else cmd="opencode"; fi; \
	    printf '[RUN] contained OpenCode session (worktree %s)\n' "$$dir"; \
	    AGENT_WORKTREE="$$dir" CLAUDE_CODE_OAUTH_TOKEN="" \
	      docker compose -f $(AGENTS_COMPOSE) run --rm agent sh -lc "$$cmd" ;; \
	  *) printf '[ERROR] AGENT must be claude or opencode (got: %s)\n' "$(AGENT)"; exit 1 ;; \
	esac

SESSIONS_DIR ?= logs/sessions

# ID is the CLI's conversation identifier:
#   claude:   UUID from `claude --resume` (transcript: ~/.claude/projects/<slug>/<id>.jsonl)
#   opencode: session id from `opencode sessions`
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