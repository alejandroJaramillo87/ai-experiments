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

# The correctness ladder (workstation only): the five rungs against the real
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
.PHONY: ladder
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
ladder:
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
	cargo run --release -p runtime --bin ladder -- \
	  $(LADDER_TEXT_ONLY) --size $(SIZE) --model $(MODEL) --reference $(REFERENCE)

# Profiling (workstation only): flame graph of a harness run. Requires
# cargo-flamegraph and perf; MockModel has no hot path, so meaningful
# profiles arrive with the first real backend.
.PHONY: flame
flame:
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
