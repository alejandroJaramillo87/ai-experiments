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
