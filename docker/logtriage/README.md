# docker/logtriage

Configuration files for the log triage stack. Unlike its sibling directories this one builds
no image: the two MongoDB services run published images pinned by digest, and the two
llama.cpp services reuse `../llama-gpu/`. What lives here is the configuration those published
images read at startup.

## Files

| File | Mounted at | Read by |
|------|------------|---------|
| `mongod.conf` | `/etc/mongod.conf` | `logtriage-mongo` |
| `mongot.conf` | `/mongot-community/config.default.yml` | `logtriage-mongot` |

Both are adapted from the MongoDB Database Manual's own samples for a containerised MongoDB
Search deployment, at corpus commit `88ec8192b924`
(`includes/search-in-community/sample-mongod-conf-docker.rst` and its `mongot` counterpart).
Keeping them close to the published samples is deliberate — a search deployment that fails to
sync gives almost no diagnostic, and the first question is always whether the configuration
diverged from the documented one.

## Traps

**The addresses are compose service names, and they must agree in three places.** `mongod.conf`
points `mongotHost` and `searchIndexManagementHostAndPort` at `logtriage-mongot:27028`;
`mongot.conf` points `syncSource` back at `logtriage-mongo:27017`; and the compose file passes
`--replSetMember=logtriage-mongo:27017`. `localhost` in any of them resolves inside the wrong
network namespace, and the symptom is a healthy pair of containers that never sync.

**MongoDB Search needs a replica set, and a replica set needs initiating.** Change streams are
how `mongot` follows a collection, and they do not exist on a standalone server. The set is
initiated once by `scripts/logtriage/bootstrap.sh`, not by anything here.

**`mongot` reports `SERVING` before it can answer queries.** The health endpoint reports that
the process is up; it does not check whether any index has reached `Ready`. A healthy container
is not a working search index.

**The secrets are generated, never committed.** `make logtriage-secrets` writes the keyfile,
`mongot`'s password file, and the root credential into `.secrets/logtriage/`. `mongod` refuses
a keyfile it does not own, so those files also have to belong to uid 999 — the generator says
so and prints the `chown`.

## Where the real docs live

[../../logtriage/docs/operations/stack.md](../../logtriage/docs/operations/stack.md) owns the
stack: the four services, the ports, the bootstrap sequence, and what each variable does.
