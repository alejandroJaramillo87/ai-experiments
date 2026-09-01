##@ Log triage (self-contained project; its own virtualenv, own gate)

.PHONY: logtriage-quality logtriage-test logtriage-install logtriage-fetch-kb logtriage-index-kb \
        logtriage-secrets logtriage-up logtriage-down logtriage-status logtriage-logs \
        logtriage-bootstrap logtriage-shell-mongo

LOGTRIAGE_DIR := logtriage

# The stack's compose file. Every invocation must pass -f: compose auto-discovers
# only its four default filenames, none of which carry a domain suffix, so a bare
# `docker compose` does not find this one at all.
LOGTRIAGE_COMPOSE ?= docker-compose.logtriage.yaml

# Two env files, deliberately. The first is hand-edited and diffable (model
# paths, context sizes); the second is generated and secret. Compose reads both
# for variable substitution, and keeping them apart is what stops a password
# from landing in a file people paste into issues.
LOGTRIAGE_ENV := .env.logtriage
LOGTRIAGE_SECRET_ENV := .secrets/logtriage/stack.env
LOGTRIAGE_DC := docker compose --env-file $(LOGTRIAGE_ENV) --env-file $(LOGTRIAGE_SECRET_ENV) -f $(LOGTRIAGE_COMPOSE)

# Both files are required before anything starts. Without this, compose fails
# on an unset substitution with an error naming a variable rather than the file
# that should have defined it.
define logtriage-preflight
	@if [ ! -f $(LOGTRIAGE_ENV) ]; then \
	  printf '[ERROR] $(LOGTRIAGE_ENV) is missing\n'; \
	  printf '  Solution: copy the template and fill in the two model paths\n'; \
	  printf '  Command: cp env.logtriage.example $(LOGTRIAGE_ENV)\n'; \
	  exit 1; \
	fi
	@if [ ! -f $(LOGTRIAGE_SECRET_ENV) ]; then \
	  printf '[ERROR] $(LOGTRIAGE_SECRET_ENV) is missing\n'; \
	  printf '  Solution: generate the stack credentials\n'; \
	  printf '  Command: make logtriage-secrets\n'; \
	  exit 1; \
	fi
endef

logtriage-install: ## Provision the logtriage project's own virtualenv
	cd $(LOGTRIAGE_DIR) && uv venv --python 3.12 && uv sync

logtriage-quality: ## Run the logtriage gate (format check, lint, mypy, docs, tests)
	$(MAKE) -C $(LOGTRIAGE_DIR) quality

logtriage-test: ## Run the logtriage test suite only
	$(MAKE) -C $(LOGTRIAGE_DIR) test

logtriage-fetch-kb: ## Fetch the MongoDB manual that backs the knowledge base (needs network)
	cd $(LOGTRIAGE_DIR) && uv run --no-sync python -m logtriage fetch-kb

logtriage-index-kb: ## Chunk and embed the fetched manual into a retrieval index
	cd $(LOGTRIAGE_DIR) && uv run --no-sync python -m logtriage index-kb --refresh

logtriage-secrets: ## Generate the stack's keyfile, mongot password, and root credential
	./scripts/logtriage/generate-secrets.sh

logtriage-up: ## Start mongod, mongot, and the two llama-servers
	$(logtriage-preflight)
	@printf '[INFO] Two llama-servers share one GPU; see AGENTS.md on GPU serialization\n'
	$(LOGTRIAGE_DC) up -d

logtriage-bootstrap: ## Initiate the replica set and create the mongot user (once, after first up)
	$(logtriage-preflight)
	set -a; . ./$(LOGTRIAGE_SECRET_ENV); set +a; ./scripts/logtriage/bootstrap.sh

logtriage-down: ## Stop the log triage stack
	$(logtriage-preflight)
	$(LOGTRIAGE_DC) down

logtriage-status: ## Show log triage service status
	$(logtriage-preflight)
	$(LOGTRIAGE_DC) ps

logtriage-logs: ## Tail the log triage stack's logs (SERVICE=logtriage-mongot to narrow)
	$(logtriage-preflight)
	$(LOGTRIAGE_DC) logs -f $(SERVICE)

logtriage-shell-mongo: ## Open mongosh against the running deployment
	$(logtriage-preflight)
	set -a; . ./$(LOGTRIAGE_SECRET_ENV); set +a; \
	  docker exec -it logtriage-mongo mongosh \
	    --username "$$LOGTRIAGE_MONGO_ROOT_USER" \
	    --password "$$LOGTRIAGE_MONGO_ROOT_PASSWORD" \
	    --authenticationDatabase admin
