##@ Agent Tooling (OpenCode / Claude Code parity)

.PHONY: agent-export agent-parity agent-preflight agent-role-tests agent-run agent-sandbox
.PHONY: agent-sandbox-build agent-sandbox-clean agent-skeptic-test agent-usage
.PHONY: agent-versions command-sync worktree-clean worktree-ls

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
	@./scripts/agents/role-acceptance.sh skeptic

agent-role-tests: ## Run the LLM acceptance test for every QA subagent role (spends tokens)
	@./scripts/agents/role-acceptance.sh $(ROLE)

##@ Workflow Automation (escalation ladder — see docs/agents/README.md)

# Provisioning tier for `make worktree-<name>`. sandbox (the default) is the torch-free
# agent environment; full matches `make install`, so the worktree can run the training
# suite and GPU work without a follow-up manual sync. A torch-free venv still runs the
# suite green — it skips ~36 training tests rather than failing — so the tier is an
# explicit choice, never inferred.
TIER ?= sandbox

worktree-%: ## Create a git worktree ../ai-experiments-<name> on branch task/<name> (TIER=sandbox|full)
	@name="$*"; dir="../ai-experiments-$$name"; \
	case "$(TIER)" in \
	  sandbox|full) ;; \
	  *) printf '$(RED)[ERROR]$(RESET) Unknown TIER: %s\n' "$(TIER)"; \
	     printf '  Accepted: sandbox (torch-free, default), full (all groups and extras)\n'; \
	     printf '  Command: make worktree-%s TIER=full\n' "$$name"; \
	     exit 1;; \
	esac; \
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
	printf '[INFO] GPU/benchmark work stays serialized across all worktrees (see AGENTS.md)\n'; \
	if [ "$(TIER)" = full ]; then \
	  printf '[RUN] Provisioning .venv (TIER=full): all groups and extras\n'; \
	  printf '[INFO] Compiles CUDA extensions (flash-attn); expect several GB and a long build\n'; \
	  sync_cmd="MAX_JOBS=$(MAX_JOBS) NVCC_THREADS=$(NVCC_THREADS) uv sync --frozen --all-groups --all-extras"; \
	  ready="the full test suite, training, and GPU work are ready"; \
	else \
	  printf '[RUN] Provisioning .venv (TIER=sandbox): dev + sandbox groups (torch-free)\n'; \
	  sync_cmd="uv sync --frozen --only-group dev --only-group sandbox"; \
	  ready="make quality and non-GPU make test are ready; torch-gated tests skip"; \
	fi; \
	if (cd "$$dir" && eval "$$sync_cmd"); then \
	  printf '[OK] .venv provisioned (TIER=%s): %s\n' "$(TIER)" "$$ready"; \
	else \
	  printf '[WARN] Could not auto-provision .venv (stale lock, offline, or uv missing)\n'; \
	  printf '  Solution: provision it manually in the new worktree\n'; \
	  printf '  Command: (cd %s && make install)\n' "$$dir"; \
	fi

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
#
# Each failure mode reports its own cause. A single "uncommitted changes? use
# FORCE=1" branch used to answer every failure, which misdiagnosed an
# already-removed worktree and recommended a flag that cannot apply to it.
# Branches are left alone: this removes worktrees, not refs.
define WORKTREE_REMOVE_SH
	case "$$arg" in \
	  *ai-experiments-*) name="$${arg##*ai-experiments-}" ;; \
	  task/*)            name="$${arg#task/}" ;; \
	  *)                 name="$$arg" ;; \
	esac; \
	dir="../ai-experiments-$$name"; \
	abs="$$(cd .. 2>/dev/null && pwd)/ai-experiments-$$name"; \
	here="$$(pwd -P)"; \
	case "$$here" in \
	  "$$abs"|"$$abs"/*) \
	    printf '$(RED)[ERROR]$(RESET) refusing to remove the worktree you are standing in: %s\n' "$$dir"; \
	    printf '  Removing it would leave this shell without a working directory\n'; \
	    printf '  Solution: change directory first, then retry\n'; \
	    printf '  Command: cd .. && make worktree-clean %s\n' "$$name"; \
	    exit 1;; \
	esac; \
	if ! git worktree list --porcelain | grep -qx "worktree $$abs"; then \
	  git worktree prune; \
	  printf '[SKIP] no such worktree: %s (already removed)\n' "$$dir"; \
	  exit 0; \
	fi; \
	force=""; [ -n "$(FORCE)" ] && force="--force"; \
	if git worktree remove $$force "$$dir"; then \
	  printf '[OK] removed worktree: %s\n' "$$dir"; \
	  printf '[INFO] agent-sandbox docker resources are shared, not per-worktree: make agent-sandbox-clean\n'; \
	else \
	  dirty="$$(git -C "$$dir" status --porcelain 2>/dev/null)"; \
	  if [ -n "$$dirty" ]; then \
	    printf '$(RED)[ERROR]$(RESET) %s has uncommitted changes:\n' "$$dir"; \
	    printf '%s\n' "$$dirty" | sed 's/^/  /'; \
	    printf '  These are NOT recoverable once removed\n'; \
	    printf '  Command: make worktree-clean %s FORCE=1\n' "$$name"; \
	  else \
	    printf '$(RED)[ERROR]$(RESET) could not remove %s (see the git error above)\n' "$$dir"; \
	  fi; \
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
# Pin the compose project so every worktree shares one set of agent-sandbox docker
# resources. Without this, the project name defaults to the CWD basename (different
# per worktree), and each project fights over the fixed agent-internal subnet
# (172.31.0.0/16), failing with "Pool overlaps". One pinned project => one network
# set, reused and cleanly torn down by `make agent-sandbox-clean`.
AGENTS_PROJECT := ai-agents
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
CLAUDE_MODEL ?= claude-opus-5[1m]
CLAUDE_MODEL_OPTIONS := [{\"value\":\"claude-fable-5[1m]\",\"label\":\"Fable\",\"description\":\"Fable 5 - account-gated tier above Opus\"},{\"value\":\"claude-opus-5\",\"label\":\"Opus\",\"description\":\"Opus 5 - flagship tier\"}]

agent-sandbox: ## Contained agent session: AGENT=claude|opencode NAME=<task> [PROMPT="..."] [EFFORT=low] [CLAUDE_MODEL=...] [MODEL=<flag override>]
	@if [ -z "$(NAME)" ]; then printf '[ERROR] set NAME=<task> (names the git worktree)\n'; exit 1; fi; \
	name="$(NAME)"; dir="$$(cd .. && pwd)/ai-experiments-$$name"; \
	if [ ! -d "$$dir" ]; then \
	  printf '[INFO] creating worktree for %s\n' "$$name"; \
	  $(MAKE) --no-print-directory worktree-$$name || exit 1; \
	fi; \
	secrets="$$HOME/.config/ai-experiments/agent-secrets.env"; \
	if [ -z "$(CLAUDE_CODE_OAUTH_TOKEN)$(RUNPOD_API_KEY)" ] && [ -f "$$secrets" ]; then \
	  CLAUDE_CODE_OAUTH_TOKEN="$$(sed -n 's/^CLAUDE_CODE_OAUTH_TOKEN=//p' "$$secrets" | head -n1)"; \
	  RUNPOD_API_KEY="$$(sed -n 's/^RUNPOD_API_KEY=//p' "$$secrets" | head -n1)"; \
	  export CLAUDE_CODE_OAUTH_TOKEN RUNPOD_API_KEY; \
	  case "$$CLAUDE_CODE_OAUTH_TOKEN" in *[[:space:]]*) \
	    printf '[ERROR] %s is malformed: the stored token contains whitespace\n' "$$secrets"; \
	    printf '  Cause: claude setup-token is interactive; capturing its output stores UI text, not a token\n'; \
	    printf '  Solution: run claude setup-token alone, copy the printed token, then write one line\n'; \
	    printf '            CLAUDE_CODE_OAUTH_TOKEN=<token> to that file (see docs/agents/README.md)\n'; \
	    exit 1 ;; \
	  esac; \
	fi; \
	case "$(AGENT)" in \
	  claude) \
	    if [ -z "$$CLAUDE_CODE_OAUTH_TOKEN" ]; then \
	      printf '[ERROR] CLAUDE_CODE_OAUTH_TOKEN not set (the only secret a Claude Code sandbox gets)\n'; \
	      printf '  Solution: run `claude setup-token` on the host, then either export CLAUDE_CODE_OAUTH_TOKEN=<token>\n'; \
	      printf '            or save it once to ~/.config/ai-experiments/agent-secrets.env (chmod 600) to auto-load it\n'; exit 1; fi; \
	    seed="printf '{\"theme\":\"$(CLAUDE_THEME)\",\"hasCompletedOnboarding\":true,\"unpinFable5LaunchEffort\":true,\"additionalModelOptionsCache\":$(CLAUDE_MODEL_OPTIONS)}' > ~/.claude.json; \
	      mkdir -p ~/.claude; \
	      printf '{\"model\":\"$(CLAUDE_MODEL)\",\"effortLevel\":\"$(EFFORT)\",\"theme\":\"$(CLAUDE_THEME)\"}' > ~/.claude/settings.json; \
	      printf '# SANDBOX - read this first\nYou are in the contained agent sandbox (make agent-sandbox), not on the host.\nAvailable here: the git worktree at /work (read-write), the pinned CLIs, and network egress only to an allowlist via the proxy.\nNot available: the host, its Docker daemon, the GPU, nvidia-smi, /mnt/ai-data, and the running inference services (ports 8001-8003 and 3000).\nDo not run docker or docker compose, make up/status/logs, nvidia-smi, healthcheck curls, or the benchmark scripts; none can work from here.\nFor any host-side command, print the exact command, stop, and wait for the human to run it and report the output. Never fabricate output.\n' > ~/.claude/CLAUDE.md"; \
	    model=""; if [ -n "$(MODEL)" ]; then model=" --model \"$(MODEL)\""; fi; \
	    if [ -n "$(PROMPT)" ]; then cmd="$$seed; claude -p \"$(PROMPT)\" --effort $(EFFORT)$$model"; else cmd="$$seed; claude$$model"; fi; \
	    printf '[RUN] contained Claude Code session (worktree %s)\n' "$$dir"; \
	    AGENT_WORKTREE="$$dir" RUNPOD_API_KEY="" \
	      docker compose -f $(AGENTS_COMPOSE) -p $(AGENTS_PROJECT) run --rm --name "agent-$$name" agent sh -lc "$$cmd" ;; \
	  opencode) \
	    if [ -z "$$RUNPOD_API_KEY" ]; then \
	      printf '[ERROR] RUNPOD_API_KEY not set (the only secret an OpenCode sandbox gets)\n'; \
	      printf '  Solution: export RUNPOD_API_KEY=<endpoint-scoped key>, or save it once to\n'; \
	      printf '            ~/.config/ai-experiments/agent-secrets.env (chmod 600) to auto-load it\n'; exit 1; fi; \
	    if [ -n "$(PROMPT)" ]; then cmd="opencode run --auto \"$(PROMPT)\""; else cmd="opencode"; fi; \
	    printf '[RUN] contained OpenCode session (worktree %s)\n' "$$dir"; \
	    AGENT_WORKTREE="$$dir" CLAUDE_CODE_OAUTH_TOKEN="" \
	      docker compose -f $(AGENTS_COMPOSE) -p $(AGENTS_PROJECT) run --rm --name "agent-$$name" agent sh -lc "$$cmd" ;; \
	  *) printf '[ERROR] AGENT must be claude or opencode (got: %s)\n' "$(AGENT)"; exit 1 ;; \
	esac

agent-sandbox-build: ## Rebuild the agent sandbox image (required after editing docker/agents/Dockerfile.agents or rust-harness/Cargo.lock)
	@echo "$(CYAN)Building agent sandbox image...$(RESET)"
	docker compose -f $(AGENTS_COMPOSE) -p $(AGENTS_PROJECT) build agent
	@echo "$(GREEN)Agent sandbox image built.$(RESET)"

# The agent-sandbox docker footprint is shared across every worktree: one proxy
# container (agent-proxy) plus the agent-internal/agent-egress networks and two
# built images. The per-session `agent` container is `--rm` (named agent-<name>
# while live, so `docker ps` maps it to ../ai-experiments-<name>). This target is
# the scoped teardown of that shared footprint; it never touches the inference
# stack (docker-compose.inference.yaml), unlike the global `make clean`/`make prune`.
agent-sandbox-clean: ## Tear down shared agent-sandbox docker resources (proxy + networks); inference stack untouched
	@printf '[RUN] Removing agent-sandbox proxy and networks (project %s)\n' "$(AGENTS_PROJECT)"
	-docker compose -f $(AGENTS_COMPOSE) -p $(AGENTS_PROJECT) down --remove-orphans
	@for net in $$(docker network ls --format '{{.Name}}' | grep -E '_agent-(internal|egress)$$'); do \
	  docker network rm "$$net" >/dev/null 2>&1 && printf '[OK] removed leftover network: %s\n' "$$net" || true; \
	done
	@printf '[OK] agent-sandbox torn down; built images remain (drop them with: docker compose -f $(AGENTS_COMPOSE) -p $(AGENTS_PROJECT) down --rmi local)\n'

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

