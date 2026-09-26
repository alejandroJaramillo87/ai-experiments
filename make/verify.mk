##@ Verification (long-running runbook entries as logged scripts; workstation only)

.PHONY: verify

# A runbook entry whose stages run for hours is driven by a committed script
# under scripts/verification/, one per entry, rather than pasted stage by
# stage. The target tees the whole run to a log file, because the runbook
# records the operator's reported output verbatim and the log is that
# report. The script calls the tier's make targets, which resolve the venv
# interpreter through RUN_PY (KB-95), so it never spells python itself.
# The pipe into tee runs inside an explicit bash with pipefail, because make
# runs recipes under /bin/sh, which is dash on the workstation and has no
# pipefail: the script's exit would otherwise be tee's (KB-97).
VERIFY_SCRIPTS := scripts/verification
VERIFY_LOG_DIR ?= /mnt/ai-data/logs/verification

verify: ## Run one long-running runbook entry as a logged, resumable script: make verify ENTRY=ft-organisms ARGS="--recipe ... --teacher ..."
ifdef AI_SANDBOX
	@printf '[ERROR] verify: this target requires the workstation\n'; \
	printf '  Found: AI_SANDBOX is set, so there is no GPU, no /mnt/ai-data, and no running services.\n'; \
	printf '  Solution: run it on the workstation and paste the log into the entry\n'; \
	printf '  Command: make verify ENTRY=%s\n' "$(ENTRY)"; \
	exit 1
else
	@test -n "$(ENTRY)" || { printf '[ERROR] verify: ENTRY is required\n'; \
	  printf '  Available: %s\n' "$$(ls $(VERIFY_SCRIPTS)/*.sh | xargs -n1 basename | sed 's/\.sh$$//' | tr '\n' ' ')"; \
	  printf '  Command: make verify ENTRY=ft-organisms ARGS="--recipe ... --teacher ..."\n'; exit 2; }
	@test -f "$(VERIFY_SCRIPTS)/$(ENTRY).sh" || { printf '[ERROR] verify: no script for %s\n' "$(ENTRY)"; \
	  printf '  Found: %s\n' "$$(ls $(VERIFY_SCRIPTS)/*.sh | xargs -n1 basename | sed 's/\.sh$$//' | tr '\n' ' ')"; \
	  printf '  Solution: name a script under %s without its .sh\n' "$(VERIFY_SCRIPTS)"; exit 2; }
	@printf '[INFO] GPU/benchmark work stays serialized across all worktrees (see AGENTS.md)\n'
	@mkdir -p "$(VERIFY_LOG_DIR)"
	@log="$(VERIFY_LOG_DIR)/$(ENTRY)_$$(date -u +%Y%m%d_%H%M%S).log"; \
	printf '[INFO] Log: %s\n' "$$log"; \
	RUN_PY="$(RUN_PY)" NO_COLOR=1 bash -o pipefail -c \
	  'script=$$1; log=$$2; shift 2; bash "$$script" "$$@" 2>&1 | tee "$$log"' \
	  verify "$(VERIFY_SCRIPTS)/$(ENTRY).sh" "$$log" $(ARGS); \
	rc=$$?; \
	printf '[INFO] Log: %s\n' "$$log"; \
	printf 'exit=%s\n' "$$rc"; \
	exit $$rc
endif
