# Phase E: the MacBook M4

**Operational — executed against the laptop. Keep the commands runnable.**

The target machine. Five entries, all pending. Blocked on Phase D: there is no point verifying
that an unproven pipeline was extracted correctly.

What is different here is not the code — it is that llama-server is native and Metal rather than
containerised and CUDA, that memory is unified rather than a separate 32GB of VRAM, and that the
container images have to exist for `arm64`.

## Table of Contents

- [Extraction](#extraction)
- [The stack on ARM](#the-stack-on-arm)
- [The model on Metal](#the-model-on-metal)

## Extraction

### E1 — the extracted project is a working repository

**Criteria:** a fresh copy with no parent passes its own gate — format, lint, types, the
documentation gate, and the tests — with nothing reaching upstream. A failure naming a path that
only exists in `ai-experiments` means the project is not self-contained, and the failure names
it.

**Command:**

```bash
cp -r logtriage/ ~/logtriage && cd ~/logtriage && git init
uv venv --python 3.12 && uv sync
make quality
```

**Observed:** _(pending)_

**Status:** pending

### E2 — the corpus provisions without re-cloning

**Criteria:** pointing `LOGTRIAGE_KB_SOURCE` at a copied corpus produces an index whose chunk
count and code-block integrity match [../corpus/statistics.md](../corpus/statistics.md) exactly.
Identical numbers on a different machine and a different architecture is the reproducibility the
committed manifest exists to provide.

**Command:**

```bash
cd ~/logtriage
export LOGTRIAGE_KB_SOURCE=/path/to/copied/corpus
uv run python -m logtriage index-kb --json
uv run pytest -m corpus -v
```

**Observed:** _(pending)_

**Status:** pending

## The stack on ARM

### E3 — both MongoDB images run on arm64

**Criteria:** `mongod` and `mongot` start from the same digests pinned in the compose file, and
the bootstrap succeeds. Both digests are manifest lists that were checked to publish `arm64` as
well as `amd64`, so this should work — but "the manifest lists arm64" and "the arm64 image runs
MongoDB Search correctly" are different claims, and only this entry can settle the second.

The two GPU services are **not** started here. On this machine llama-server is native.

**Command:**

```bash
docker compose --env-file .env.logtriage --env-file .secrets/logtriage/stack.env \
  -f docker-compose.logtriage.yaml up -d logtriage-mongo logtriage-mongot
docker image inspect mongodb/mongodb-community-search:1.70.4 --format '{{.Architecture}}'
make logtriage-bootstrap
curl -s http://127.0.0.1:8008/health
```

**Observed:** _(pending)_

**Status:** pending

## The model on Metal

### E4 — a native llama-server serves both roles

**Criteria:** two native servers, one for chat with the chat template applied and one in
embedding mode, reachable at the two URLs. The compose file's flags are the reference for what
they need; the ones that matter are `--jinja` for tool calling and `--embedding --pooling` for
the second.

Nothing in the tool changes. If anything here needs a code change, the configuration boundary
has leaked and that is the finding.

**Command:**

```bash
llama-server -m gemma-4-31B-it-Q4_K_M.gguf --port 8006 --jinja --ctx-size 32768 &
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --port 8007 --embedding --pooling mean &
export LOGTRIAGE_LLAMA_URL=http://127.0.0.1:8006
export LOGTRIAGE_EMBED_URL=http://127.0.0.1:8007
cd ~/logtriage && uv run python -m logtriage triage /tmp/rs.log --llm llama --kb
```

**Observed:** _(pending)_

**Status:** pending

### E5 — unified memory holds the model, the embedder, and Mongo at once

**Criteria:** numbers, not a pass/fail. On the desktop the 31B model and the embedding model
share 32GB of dedicated VRAM while Mongo runs on system memory. Here all three share 48GB of
unified memory, and `mongot`'s Lucene indexes are part of that budget rather than beside it.

Record peak memory pressure and the end-to-end `triage` wall-clock, and compare against D9. If
the answer is that the three do not fit together, that is the Phase E finding and it is about
sizing rather than about correctness.

**Command:**

```bash
cd ~/logtriage
time uv run python -m logtriage triage /path/to/real-mongod.log --llm llama --kb
# in another shell, during the run:
memory_pressure | head -20
```

**Observed:** _(pending)_

**Status:** pending

## Related

- [desktop.md](desktop.md) — the phase this one is blocked on
- [../operations/three-machines.md](../operations/three-machines.md) — what makes extraction work
- [../operations/stack.md](../operations/stack.md) — the flags E4 has to reproduce natively
