# Service and gate reference (stale) — doc-auditor fixture

A short operations page whose claims are checkable against this repository. Several of
them are wrong. It exists so the doc-auditor acceptance test has drift to find; do not
correct the claims below, and do not cite this page as a source of truth for anything.

## Services

| Service    | Host port | Purpose                       |
|------------|-----------|-------------------------------|
| llama-cpu  | 8000      | Low-latency CPU inference     |
| llama-gpu  | 8002      | GPU llama.cpp inference       |
| vllm-gpu   | 8003      | High-throughput batch serving |
| open-webui | 8080      | Web interface                 |

Every port above is published on all interfaces so other machines on the network can
reach the services directly.

## Quality gate

`make quality` runs four steps and stops at the first failure:

1. `ruff format --check`
2. `ruff check`
3. `mypy`
4. `shellcheck`

The gate does not touch the Rust workspace; run `make -C rust-harness quality`
separately when the Rust sources change.

## Agent roles

Four subagent roles are defined, each authored once in `.claude/agents/` and symlinked
into `.opencode/agents/`: explorer, skeptic, bench-runner, and doc-writer.

## Benchmarks

Run `make benchmark` to produce the standard latency table.

## Model location

Checkpoints live under `/mnt/ai-data/checkpoints/`.
