##@ Interpretability (interp Phase 0 I1 - separate compose file; GPU inference must be down)

.PHONY: harvest harvest-check-hooks interp-build interp-dirs
.PHONY: interp-down interp-gc interp-runs interp-shell interp-smoke interp-up
.PHONY: organism organism-check-seal organism-generate organism-seal organism-verify
.PHONY: probe-capture probe-eval probe-export probe-monitor probe-train sae-eval
.PHONY: sae-train steer-down
.PHONY: steer-extract steer-scorecard steer-up

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

interp-smoke: ## Run the six-step interp smoke test on the workstation (enforces CUDA sm_120)
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp smoke

# Host-side, like harness-runs: the catalog is a file under the log root,
# and browsing it should not need the workbench image or the GPU preflight.
interp-runs: ## List recent interp runs from the observability catalog (ARGS='-n 50 --failed')
	uv run --no-sync python -m common.obs.runs --cli interp $(ARGS)

interp-down: ## Stop the interp workbench
	docker compose -f $(INTERP_COMPOSE) down

interp-dirs: ## Create the /mnt/ai-data/interp storage tree on the host
	mkdir -p /mnt/ai-data/interp/activations /mnt/ai-data/interp/saes \
	         /mnt/ai-data/interp/probes /mnt/ai-data/interp/vectors \
	         /mnt/ai-data/interp/hf-cache /mnt/ai-data/interp/logs \
	         /mnt/ai-data/interp/monitor /mnt/ai-data/interp/organisms
	@echo "$(GREEN)Interp storage tree ready under /mnt/ai-data/interp$(RESET)"

##@ Interpretability write-back (interp Phase 3 W1-W4)

# Steered service and the host->container mapping of the vectors mount.
# Mirrors interp/steering/wiring.py steering_env/container_vector_path:
# a change to the env vars or path map must land on both sides.
#
# The subject is the rust harness, not llama-gpu. R10 retired llama.cpp
# steering: the two entrypoints no longer read CONTROL_VECTOR, and the vector
# is safetensors, which llama.cpp cannot load. The three variable names are
# unchanged because steering_env emits them and the harness serve entrypoint
# reads them.
STEER_SERVICE ?= harness-serve
VECTOR_CONTAINER = $(patsubst /mnt/ai-data/interp/vectors%,/app/vectors%,$(VECTOR))/vector.safetensors

steer-extract: ## Extract and export a control vector from a capture directory (ARGS="--capture ... --out ...")
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp steer-extract $(ARGS)

steer-up: ## Recreate the service steered (VECTOR=/mnt/ai-data/interp/vectors/<name> [SCALE=1.0] [LAYER_RANGE="start end"] [STEER_SERVICE=harness-serve])
	@test -n "$(VECTOR)" || { echo "VECTOR=/mnt/ai-data/interp/vectors/<name> is required"; exit 2; }
	@test -f "$(VECTOR)/vector.safetensors" || { echo "vector not found: $(VECTOR)/vector.safetensors"; exit 1; }
	@case "$(VECTOR)" in /mnt/ai-data/interp/vectors/*) ;; *) echo "VECTOR must live under /mnt/ai-data/interp/vectors (the mounted tree)"; exit 1;; esac
	CONTROL_VECTOR="$(VECTOR_CONTAINER)" CONTROL_VECTOR_SCALE="$(SCALE)" \
	CONTROL_VECTOR_LAYER_RANGE="$(LAYER_RANGE)" \
	docker compose -f $(HARNESS_COMPOSE) up -d --force-recreate $(STEER_SERVICE)
	@echo "$(GREEN)$(STEER_SERVICE) recreated steered: $(VECTOR_CONTAINER)$(if $(SCALE), scale $(SCALE),)$(RESET)"

steer-down: ## Recreate the service unsteered (restores production state)
	CONTROL_VECTOR= CONTROL_VECTOR_SCALE= CONTROL_VECTOR_LAYER_RANGE= \
	docker compose -f $(HARNESS_COMPOSE) up -d --force-recreate $(STEER_SERVICE)
	@echo "$(GREEN)$(STEER_SERVICE) recreated unsteered$(RESET)"

steer-scorecard: ## Run the scorecard (baseline, system prompt, one steered arm per scale) against live services (VECTOR=... ATTRIBUTE="..." [ARGS=...])
	@test -n "$(VECTOR)" || { echo "VECTOR=/mnt/ai-data/interp/vectors/<name> is required"; exit 2; }
	@test -n "$(ATTRIBUTE)" || { echo "ATTRIBUTE=\"behavior the judge scores\" is required"; exit 2; }
	uv run python -m interp steer-scorecard --vector "$(VECTOR)" --attribute "$(ATTRIBUTE)" $(ARGS)

probe-capture: ## Capture labeled activations in the workbench (ARGS="--model ... --dataset ... --layers ... --out ...")
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp probe-capture $(ARGS)

probe-train: ## Train and register probes from a capture (ARGS="--capture ... --name ...")
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp probe-train $(ARGS)

# No preflight and no GPU: this reads a registered probe and rewrites it in
# the format the harness reader takes. probe-train leaves probe.npz, which is
# Python-only; harness-serve's HARNESS_PROBE needs the safetensors image.
probe-export: ## Export a registered probe as the safetensors image harness-serve loads (ARGS="--probe /mnt/ai-data/interp/probes/<name>")
	uv run python -m interp probe-export $(ARGS)

probe-eval: ## Regenerate a probe evaluation report with black-box baselines (serving-dependent; ARGS="--probe ... --label-name ...")
	uv run python -m interp probe-eval $(ARGS)

# No preflight, and no --model: the scores are already in the log, taken by
# the process that served each request. Nothing here loads a model, so this
# runs while inference is up.
probe-monitor: ## Aggregate a served-request log window into a probe report (ARGS="--window 7d [--probes ...] [--log-dir ...]")
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp probe-monitor $(ARGS)

##@ Interpretability harvesting (interp Phase 4 P4-D2 - workbench batch job; GPU inference must be down)

harvest: ## Harvest residual-stream activations to NVMe (make harvest RECIPE=training/recipes/interp/harvest-smoke.yaml [ARGS="--override harvest.total_tokens=..."])
	@test -n "$(RECIPE)" || { echo "RECIPE=training/recipes/interp/<name>.yaml is required"; exit 2; }
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp harvest --recipe $(RECIPE) $(ARGS)

harvest-check-hooks: ## Gemma 4 hook-dimension gate: one-token forward, no capture (make harvest-check-hooks RECIPE=...)
	@test -n "$(RECIPE)" || { echo "RECIPE=training/recipes/interp/<name>.yaml is required"; exit 2; }
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp harvest --recipe $(RECIPE) --check-hooks

interp-gc: ## Delete validated activation captures oldest-first, keeping manifests (ARGS="--dry-run" | ARGS="--mark-validated <dir>")
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp gc $(ARGS)

##@ Interpretability SAEs (interp Phase 4 P4-D3 - workbench batch job; GPU inference must be down)

sae-train: ## Train an SAE from a P4-D2 capture (make sae-train RECIPE=training/recipes/interp/sae-smoke.yaml [ARGS="--validated"])
	@test -n "$(RECIPE)" || { echo "RECIPE=training/recipes/interp/<name>.yaml is required"; exit 2; }
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp sae-train --recipe $(RECIPE) $(ARGS)

sae-eval: ## Evaluate a trained SAE against its capture (make sae-eval SAE=/mnt/ai-data/interp/saes/<name> CAPTURE=/mnt/ai-data/interp/activations/<name> [ARGS=...])
	@test -n "$(SAE)" || { echo "SAE=/mnt/ai-data/interp/saes/<name> is required"; exit 2; }
	@test -n "$(CAPTURE)" || { echo "CAPTURE=/mnt/ai-data/interp/activations/<name> is required"; exit 2; }
	@./scripts/interp/preflight.sh
	docker compose -f $(INTERP_COMPOSE) run --rm interp python -m interp sae-eval --sae-dir $(SAE) --capture-dir $(CAPTURE) $(ARGS)

##@ Interpretability organisms (interp Phase 4 P4-D1 - host-run; generation and verification are serving-dependent)

organism-generate: ## Draw ground truth and generate the organism corpus (make organism-generate RECIPE=training/recipes/organisms/taboo.yaml TEACHER=/mnt/ai-data/models/... [ARGS=...])
	@test -n "$(RECIPE)" || { echo "RECIPE=training/recipes/organisms/<name>.yaml is required"; exit 2; }
	python3 -m interp organism generate --recipe $(RECIPE) $(if $(TEACHER),--teacher $(TEACHER),) $(ARGS)

organism-verify: ## Run the implant and locality checks against live services (make organism-verify RECIPE=... BASE_METRICS=... SUBJECT_METRICS=... [ARGS=...])
	@test -n "$(RECIPE)" || { echo "RECIPE=training/recipes/organisms/<name>.yaml is required"; exit 2; }
	@test -n "$(BASE_METRICS)" || { echo "BASE_METRICS=<F7 metrics JSON for the base model> is required"; exit 2; }
	@test -n "$(SUBJECT_METRICS)" || { echo "SUBJECT_METRICS=<F7 metrics JSON for the organism> is required"; exit 2; }
	python3 -m interp organism verify --recipe $(RECIPE) --base-metrics $(BASE_METRICS) --subject-metrics $(SUBJECT_METRICS) $(ARGS)

organism-seal: ## Seal ground truth and write the blinded public manifest (make organism-seal RECIPE=... [ARGS="--checkpoint ... --gguf ..."])
	@test -n "$(RECIPE)" || { echo "RECIPE=training/recipes/organisms/<name>.yaml is required"; exit 2; }
	python3 -m interp organism seal --recipe $(RECIPE) $(ARGS)

organism-check-seal: ## Re-verify a sealed organism's ground-truth hash (make organism-check-seal DIR=/mnt/ai-data/interp/organisms/<name>)
	@test -n "$(DIR)" || { echo "DIR=/mnt/ai-data/interp/organisms/<name> is required"; exit 2; }
	python3 -m interp organism check-seal --dir $(DIR)

organism: ## Print the full organism pipeline for a recipe (each stage has its own GPU/serving preconditions, so stages run individually)
	@test -n "$(RECIPE)" || { echo "RECIPE=training/recipes/organisms/<name>.yaml is required"; exit 2; }
	@echo "Organism pipeline for $(RECIPE):"
	@echo "  1. make organism-generate RECIPE=$(RECIPE) TEACHER=<model>   (serving up)"
	@echo "  2. make prepare-sft ARGS=\"--source <out_dir>/data-blind ...\" (see recipe sft.dataset)"
	@echo "  3. make train-sft CONFIG=$(RECIPE)                            (inference down)"
	@echo "  4. make export-hf CHECKPOINT=<run checkpoint> ARGS=\"--name <organism>\" (merges the adapter it finds)"
	@echo "  5. make eval-ft ARGS=... (base and organism metrics for locality)"
	@echo "  6. make organism-verify RECIPE=$(RECIPE) BASE_METRICS=... SUBJECT_METRICS=... (serving up)"
	@echo "  7. make organism-seal RECIPE=$(RECIPE) ARGS=\"--checkpoint ... --gguf ...\""
	@echo "  8. make organism-check-seal DIR=/mnt/ai-data/interp/organisms/<name>"

