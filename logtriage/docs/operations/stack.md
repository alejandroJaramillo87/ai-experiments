# Running the stack

The four services, the ports, the one-time bootstrap, and the variables that connect the tool to
them.

`docker-compose.logtriage.yaml` is the source of truth. Where this document disagrees with it,
this document is the bug.

## Table of Contents

- [The four services](#the-four-services)
- [First run](#first-run)
- [Everyday commands](#everyday-commands)
- [Configuration](#configuration)
- [VRAM](#vram)
- [Traps](#traps)

## The four services

| Service | Image or build | Bound to |
|---------|----------------|----------|
| `logtriage-mongo` | `mongodb/mongodb-community-server` 8.2.12, digest-pinned | `127.0.0.1:27017` |
| `logtriage-mongot` | `mongodb/mongodb-community-search` 1.70.4, digest-pinned | `127.0.0.1:8008` health, `127.0.0.1:9946` metrics |
| `logtriage-llm` | built from `docker/llama-gpu/` | `127.0.0.1:8006` |
| `logtriage-embed` | built from `docker/llama-gpu/` | `127.0.0.1:8007` |

Every port binds to loopback. `mongod` holds a copy of whatever logs were ingested, which is
production data by definition, and the two inference servers are unauthenticated by design.

`mongot`'s gRPC query port (27028) is **not** published at all. `mongod` reaches it over the
compose network, and the manual warns that a query server reachable from public networks risks
unauthorized access.

Both MongoDB images are pinned by manifest-list digest and publish `amd64` and `arm64`, so the
same file works on the workstation and on an ARM laptop.

The two llama.cpp services are separate containers rather than two ports on one, because
llama.cpp's `--embedding` mode **replaces** the chat endpoints instead of adding to them. One
server cannot serve both.

## First run

```bash
$ cp env.logtriage.example .env.logtriage    # then fill in the two model paths
$ make logtriage-secrets                     # keyfile, mongot password, root credential
$ make logtriage-up
$ make logtriage-bootstrap                   # replica set + the mongot user
```

`make logtriage-bootstrap` is the step that cannot be expressed in compose. It initiates the
replica set — MongoDB Search follows the collection through a change stream, and change streams
do not exist on a standalone server — and creates the user `mongot` authenticates as, with the
`searchCoordinator` role. Both halves are idempotent, so running it again after a failed start
is a no-op rather than an error.

`mongot` reports `NOT_SERVING` until the bootstrap completes, because until the user exists it
cannot open its change stream.

## Everyday commands

```bash
$ make logtriage-up                          # start
$ make logtriage-status                      # compose ps
$ make logtriage-logs SERVICE=logtriage-mongot
$ make logtriage-shell-mongo                 # mongosh, authenticated
$ make logtriage-down                        # stop
$ curl -s http://127.0.0.1:8008/health       # SERVING | NOT_SERVING
```

## Configuration

Two files, deliberately kept apart.

**`.env.logtriage`** is hand-edited and diffable: the two model paths, the context sizes, the
pooling strategy. Copy it from `env.logtriage.example`, which documents each one.

**`.secrets/logtriage/`** is generated and secret: the keyfile shared by `mongod` and `mongot`,
`mongot`'s password file, and `stack.env` holding the root credential. `make logtriage-secrets`
writes all three, refuses to overwrite, and is different on every machine by design — a shared
default password in a repository is a password that ends up on a server.

The make targets pass both files to compose. Nothing secret belongs in the first one.

The tool itself reads three variables, and they are the whole of its configuration:

| Variable | Points at | Unset means |
|----------|-----------|-------------|
| `LOGTRIAGE_LLAMA_URL` | the chat server | use the scripted client |
| `LOGTRIAGE_EMBED_URL` | the embedding server | use the hashing embedder |
| `LOGTRIAGE_MONGO_URI` | the deployment | use the in-memory stores |

Unset is a legitimate state, not an error. Only a command explicitly asked for a served backend
turns it into a failure, and then the error names the variable and gives the export.

## VRAM

Both inference services reserve the same GPU. On a 32GB card a Q4-quantised 31B model and a small
embedding model fit together, but the budget is real: `LOGTRIAGE_LLM_CTX` and
`LOGTRIAGE_EMBED_CTX` are the two knobs, and the embedding context only has to exceed the
chunker's `MAX_TOKENS` — anything larger is VRAM taken from the model beside it.

The repository's standing rule that only one GPU consumer runs at a time still applies to
benchmarks and to the other stacks. This stack's two servers are a deliberate shared budget, not
an exception to it.

## Traps

**The addresses must agree in three places.** `mongod.conf` points at `logtriage-mongot:27028`,
`mongot.conf` points back at `logtriage-mongo:27017`, and the compose command passes
`--replSetMember=logtriage-mongo:27017`. `localhost` in any of them resolves inside the wrong
network namespace, and the symptom is a healthy pair of containers that never sync.

**`SERVING` does not mean searchable.** `mongot`'s health endpoint reports that the process is
up. It does not check whether any index has reached `Ready`, and a query against one still
building returns nothing rather than an error.

**`mongod` refuses a keyfile it does not own.** The generator writes mode 400 and prints the
`chown` to uid 999 if the ownership is wrong. Without it the container exits at startup with no
obvious reason.

**The chat server needs its chat template.** `JINJA_ENABLED=true` is what makes native tool
calling work. Without it the loop completes having called no tool — a run that looks successful
and consulted nothing. See [../design/prompting.md](../design/prompting.md).

## Related

- [three-machines.md](three-machines.md) — what runs where
- [../mongodb/search.md](../mongodb/search.md) — why the deployment is shaped this way
- [../verification/desktop-deployment.md](../verification/desktop-deployment.md) — checking that it works
