##@ Training (Phase 0 D1 - separate compose file; never runs with inference)

.PHONY: build-contamination eval-ft export-build export-hf generate-teacher
.PHONY: prepare-corpus prepare-data prepare-sft train
.PHONY: train-build train-download train-preflight train-sft train-shell
.PHONY: train-smoke train-step train-tokenizer train-validate

TRAIN_COMPOSE := docker-compose.training.yaml

train-build: ## Build the training image (runs the smoke test at build; fails on regression)
	@echo "$(CYAN)Building training image...$(RESET)"
	MAX_JOBS=$(MAX_JOBS) NVCC_THREADS=$(NVCC_THREADS) docker compose -f $(TRAIN_COMPOSE) build
	@echo "$(GREEN)Training image built.$(RESET)"

train-shell: ## Interactive shell in the training container (GPU + /mnt/ai-data/training mounted)
	docker compose -f $(TRAIN_COMPOSE) run --rm training /bin/bash

train-smoke: ## Run the six-step training smoke test on the workstation (enforces CUDA sm_120)
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training smoke

# Staging targets. These run in the `prepare` service, the only one with network
# egress (docker-compose.training.yaml). Everything below them runs in `training`,
# which sits on an internal network and cannot reach the hub at all.

train-download: ## Stage a model into /mnt/ai-data/models/hf (Prelude C): make train-download MODEL=google/gemma-4-E2B REVISION=<sha>
	docker compose -f $(TRAIN_COMPOSE) run --rm prepare python -m training download --model $(MODEL) --revision $(REVISION) $(ARGS)

prepare-data: ## Tokenize a dataset into shards (Phase 0 D2): make prepare-data ARGS="--source ... --tokenizer ... --out ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm prepare python -m training prepare-data $(ARGS)

prepare-corpus: ## Filter, dedup, decontaminate, and shard a corpus (Phase 2 P2-D1): make prepare-corpus ARGS="--source ... --tokenizer ... --work-dir ... --out ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm prepare python -m training prepare-corpus $(ARGS)

build-contamination: ## Build the benchmark decontamination index (Phase 2 P2-D1): make build-contamination ARGS="--out /mnt/ai-data/training/datasets/_contamination/v1"
	docker compose -f $(TRAIN_COMPOSE) run --rm prepare python -m training build-contamination $(ARGS)

prepare-sft: ## Prepare masked SFT examples (Phase 1 F1): make prepare-sft ARGS="--source ... --tokenizer ... --model-type ... --out ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm prepare python -m training prepare-sft $(ARGS)

train-tokenizer: ## Train a byte-level BPE tokenizer (Phase 0 D3): make train-tokenizer ARGS="--corpus ... --out ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm prepare python -m training train-tokenizer $(ARGS)

train-preflight: ## Prove every asset a recipe needs is already local (Prelude C): make train-preflight CONFIG=training/recipes/reference/tiny-e2e.yaml
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training preflight --config $(CONFIG) $(ARGS)

train-validate: ## Validate a recipe (Phase 0 D5): make train-validate CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="optimizer.lr=1e-4"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training validate $(CONFIG) $(ARGS)

train-step: ## Run a few optimizer steps of a recipe on the GPU and report resident and peak VRAM: make train-step CONFIG=... ARGS="--out record.json"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training step --config $(CONFIG) $(ARGS)

train: ## Run the reference training loop (Phase 0 D6): make train CONFIG=training/recipes/reference/tiny-e2e.yaml ARGS="checkpoint.resume=latest"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training train --config $(CONFIG) $(ARGS)

train-sft: ## Run a supervised fine-tune on TRL (Phase 1 F3): make train-sft CONFIG=training/recipes/reference/sft-qlora-e2b.yaml ARGS="optimizer.lr=1e-4"
	docker compose -f $(TRAIN_COMPOSE) run --rm training python -m training train-sft --config $(CONFIG) $(ARGS)

export-build: ## Build the export image (CPU-only adapter merge; writes the merged HF checkpoint)
	@echo "$(CYAN)Building export image...$(RESET)"
	docker compose -f $(TRAIN_COMPOSE) build export
	@echo "$(GREEN)Export image built.$(RESET)"

export-hf: ## Export a checkpoint as the merged HF directory the harness loads: make export-hf CHECKPOINT=/mnt/ai-data/training/runs/<run>/checkpoints/checkpoint-<step> ARGS="--name <name> --tokenizer ..."
	docker compose -f $(TRAIN_COMPOSE) run --rm export python -m training export --checkpoint $(CHECKPOINT) $(ARGS)

eval-ft: ## Baseline-versus-tuned scorecard (Phase 1 F7, host-run): make eval-ft ARGS="--base <hf dir> --tuned <hf dir> --dataset ..."
	python3 -m training eval-ft $(ARGS)

generate-teacher: ## Batch-generate teacher data through the serving stack (Phase 1 F4, host-run): make generate-teacher ARGS="--pipeline respond --prompts ... --teacher ... --out ..."
	python3 -m training generate-teacher $(ARGS)
