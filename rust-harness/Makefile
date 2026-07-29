# Quality gate for the rust-harness workspace. Self-contained: this directory
# is destined for extraction into its own repository, so nothing here depends
# on the parent repo's Makefile or tooling.

.PHONY: quality fmt lint test

quality: fmt lint test

fmt:
	cargo fmt --all --check

lint:
	cargo clippy --workspace --all-targets -- -D warnings

test:
	cargo test --workspace

# Profiling (workstation only): flame graph of a harness run. Requires
# cargo-flamegraph and perf; MockModel has no hot path, so meaningful
# profiles arrive with the first real backend.
.PHONY: flame
flame:
	cargo flamegraph --profile release-with-debug --bin harness

# Debugging (workstation only): record a failing run once under rr, then
# replay it in gdb with reverse execution. Usage:
#   make debug-record ARGS='--model fixtures/tiny-gemma4.gguf say hello'
#   make debug-replay
.PHONY: debug-record debug-replay
debug-record:
	cargo build --profile release-with-debug -p harness
	rr record target/release-with-debug/harness $(ARGS)

debug-replay:
	rr replay
