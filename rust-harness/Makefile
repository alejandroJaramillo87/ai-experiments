# Quality gate for the rust-harness workspace, a permanent part of this
# repository. The gate is invocable on its own (nothing here needs the parent
# Makefile), but the tests may read the parent tree: the cross-language
# contract tests run the real Python exporters and readers, and skip with
# [SKIP] when that tree or its numpy/gguf dependencies are absent — as in the
# container build, whose context is rust-harness/ alone.
#
# Five offline steps, cheapest first, failing on the first error — the same
# shape as the parent repo's Python gate. Deliberate non-step: cargo-deny
# (advisories, licenses, duplicate-dep bans) needs a network-fetched advisory
# database, which conflicts with the gate's offline contract; it enters with
# CI or workstation tooling, not here.

.PHONY: quality fmt lint lock doc test

quality: fmt lint lock doc test

fmt:
	cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

# Analogue of the parent gate's `uv lock --check`: fails when Cargo.lock is
# stale relative to the manifests; needs no network when the lock is current.
lock:
	cargo metadata --locked --format-version 1 > /dev/null

# Rustdoc lints (broken intra-doc links, bad code fences) as errors.
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --document-private-items

# --nocapture because cargo hides a passing test's stderr, which is exactly
# where the cross-language tests write their [SKIP] lines. Without it a gate
# that silently stopped exercising the Python side is indistinguishable from
# one that exercised it and passed, while AGENTS.md and .styles/RUST_STYLE.md
# both instruct the reader to check the output for [SKIP]. The tests are quiet
# otherwise, so this adds the skip lines and little else.
test:
	cargo test --workspace -- --nocapture

# The checks that need the real E4B checkpoint rather than the committed tiny
# fixture. Driven by `make quality-full HW=e4b`, which supplies
# HARNESS_E4B_MODEL and refuses to run without it.
#
# --test-threads=1 is not a style preference. Each of these loads the real
# checkpoint at roughly 14 GiB resident, and cargo runs a binary's tests in
# parallel by default, so the machine would hold several copies at once and
# the tier would fail on memory rather than on numerics.
#
# The tests skip through runtime::testlog::skip when the variable is unset, so
# running this target by hand without it reports skips rather than passing
# vacuously -- but the gate above makes the unset case an error, because a
# tier that skipped its way to green is the failure this whole campaign was
# about.
test-e4b: ## Run the checks needing the real E4B checkpoint (HARNESS_E4B_MODEL=<dir>)
	cargo test --workspace -- --nocapture --test-threads=1

# The correctness ladder (workstation only): the six rungs against the real
# checkpoint and reference artifacts generated from transformers.
#
# Deliberately not part of `make quality`. The gate is mandatory before every
# push and has to stay fast; this loads a 15 GB checkpoint and runs a real
# forward pass. Tier T -- the same rungs against the committed tiny fixture --
# is what runs inside the gate.
#
# SIZE is this target's own name for how much work L3 and L4 do. Not HW, which
# make/quality.mk owns and means smoke|full, and not TIER, which make/agents.mk
# owns and means sandbox|full; a third meaning on either would be resolved by
# include order and silently mis-set.
.PHONY: ladder ladder-reference ladder-preflight
SIZE ?= quick
REFERENCE ?= fixtures/reference/e4b/$(SIZE)
# The released E4B is Gemma4ForConditionalGeneration: a text tower beside a
# vision and an audio one. The CPU backend serves text and refuses a
# multimodal checkpoint unless told to decline the rest, and the ladder
# measures the text stack by construction -- there is no rung it could point
# at a vision tower. So the decline is this target's default rather than
# something every invocation has to remember. TEXT_ONLY=0 restores the
# refusal, which is what a text-only checkpoint should be run with.
TEXT_ONLY ?= 1
ifeq ($(TEXT_ONLY),1)
LADDER_TEXT_ONLY := --text-only
endif

# The generators need the parent tree's torch environment. `uv run` discovers
# the project by walking up from here, so this is the workstation venv rather
# than whichever python is on PATH -- the first Tier E attempt ran a bare
# python without transformers and wrote nothing (RH.1b).
LADDER_PY ?= uv run python

# One artifact per generator, named as runtime/src/bin/ladder.rs joins them
# onto --reference. `ladder` depends on these as files, so a missing one is
# generated on the way up and a present one is not paid for again: the L3
# generator scores the corpus with transformers and the L5 one runs a
# 2052-token forward pass. Nothing here notices a checkpoint swapped under an
# existing directory; `make ladder-reference` regenerates unconditionally for
# that case.
LADDER_ARTIFACTS := $(addprefix $(REFERENCE)/,\
  tokenizer_parity.json layers.json ppl.json greedy.json long_context.json)

# The refusals, shared by the climb and the generators. An order-only
# prerequisite of the artifact rules, so a phony target does not make every
# artifact look stale on every run.
ladder-preflight:
ifdef AI_SANDBOX
	@printf '[ERROR] make ladder needs the real checkpoint and the GPU host; the sandbox has neither\n'
	@printf '  The sandbox mounts only the worktree, so /mnt/ai-data is not reachable.\n'
	@printf '  Solution: run this on the workstation\n'
	@printf '  Command: cd rust-harness && make ladder SIZE=%s MODEL=<checkpoint-dir>\n' '$(SIZE)'
	@exit 1
endif
	@if [ -z "$(MODEL)" ]; then \
	  printf '[ERROR] MODEL is not set: the ladder has no checkpoint to measure\n'; \
	  printf '  Solution: name the checkpoint directory\n'; \
	  printf '  Command: make ladder SIZE=%s MODEL=<checkpoint-dir>\n' '$(SIZE)'; \
	  exit 1; \
	fi
	@case '$(SIZE)' in quick|standard|deep) ;; *) \
	  printf '[ERROR] SIZE=%s is not a run size\n' '$(SIZE)'; \
	  printf '  Solution: use one of quick, standard, deep\n'; \
	  printf '  Command: make ladder SIZE=quick MODEL=%s\n' '$(MODEL)'; \
	  exit 1 ;; esac

# L0. Does not vary with SIZE; written into each size directory rather than
# resolved from a parent (fixtures/reference/README.md).
$(REFERENCE)/tokenizer_parity.json: | ladder-preflight
	@mkdir -p $(REFERENCE)
	$(LADDER_PY) fixtures/reference_tokenizer.py --model "$(MODEL)" --out $@

# L1 and L2; --generate also writes L4's continuation for the same prompt.
$(REFERENCE)/layers.json: | ladder-preflight
	@mkdir -p $(REFERENCE)
	$(LADDER_PY) fixtures/reference_capture.py --model "$(MODEL)" \
	  --prompt "The capital of France is" --generate 16 --out $@

# L3. The slow one.
$(REFERENCE)/ppl.json: | ladder-preflight
	@mkdir -p $(REFERENCE)
	$(LADDER_PY) fixtures/reference_perplexity.py --model "$(MODEL)" --size $(SIZE) --out $@

# L4.
$(REFERENCE)/greedy.json: | ladder-preflight
	@mkdir -p $(REFERENCE)
	$(LADDER_PY) fixtures/reference_greedy.py --model "$(MODEL)" --size $(SIZE) --out $@

# L5. Both sides derive the scored positions from the checkpoint's
# sliding_window, so which positions were scored is part of what L5 asserts.
$(REFERENCE)/long_context.json: | ladder-preflight
	@mkdir -p $(REFERENCE)
	$(LADDER_PY) fixtures/reference_long_context.py --model "$(MODEL)" --out $@

ladder-reference: ladder-preflight ## Regenerate every reference artifact for SIZE from transformers (MODEL=<dir>)
	$(MAKE) -B $(LADDER_ARTIFACTS) MODEL="$(MODEL)" SIZE=$(SIZE) REFERENCE=$(REFERENCE)

ladder: ladder-preflight $(LADDER_ARTIFACTS) ## Climb the correctness ladder, generating any missing artifact first (MODEL=<dir>, SIZE=quick|standard|deep)
	cargo run --release -p runtime --bin ladder -- \
	  $(LADDER_TEXT_ONLY) --size $(SIZE) --model $(MODEL) --reference $(REFERENCE)

# Profiling (workstation only): flame graph of a harness run. Requires perf and
# cargo-flamegraph, installed with:
#
#   cargo install flamegraph --locked
#
# --locked is not optional: without it cargo resolves the dependency tree to
# latest-compatible, which currently pulls cargo-platform 0.3.3 and its
# rust-version 1.91 against the toolchain's 1.88. The lockfile shipped in the
# crate pins versions that build.
#
# perf also needs kernel.perf_event_paranoid <= 1 for a non-root user.
.PHONY: flame
flame:
	@command -v cargo-flamegraph >/dev/null 2>&1 || { \
	  printf '[ERROR] cargo-flamegraph is not installed\n'; \
	  printf '  Solution: install it with the lockfile pinned\n'; \
	  printf '  Command: cargo install flamegraph --locked\n'; \
	  exit 1; }
	cargo flamegraph --profile release-with-debug --bin harness

# Debugging (workstation only): record a failing run once under rr, then
# replay it in gdb with reverse execution. Usage:
#   make debug-record ARGS='--model fixtures/tiny-gemma4 say hello'
#   make debug-replay
.PHONY: debug-record debug-replay
debug-record:
	cargo build --profile release-with-debug -p harness
	rr record target/release-with-debug/harness $(ARGS)

debug-replay:
	rr replay
