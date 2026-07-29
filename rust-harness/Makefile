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
