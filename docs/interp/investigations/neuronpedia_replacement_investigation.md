# Neuronpedia versus a native interp platform: replacement investigation

**Frozen record — 2026-09-18. Drift against the current repository is expected and correct;
this document is never updated.**

Read-only investigation of what the self-hosted Neuronpedia platform in `ai-experiments`
provides, from the upstream source, and what a native replacement of each capability would
cost. No host-side command, container, or GPU was used. Every claim cites a file in one of the
two trees; anything not verifiable from source is marked UNVERIFIED.

## Table of contents

- [Revision caveat](#revision-caveat)
- [Part 1: what Neuronpedia offers](#part-1-what-neuronpedia-offers)
  - [Service inventory at the pin](#service-inventory-at-the-pin)
  - [webapp](#webapp)
  - [db-init and postgres](#db-init-and-postgres)
  - [inference](#inference)
  - [autointerp](#autointerp)
  - [graph and graph-crm](#graph-and-graph-crm)
  - [How a feature dashboard is computed](#how-a-feature-dashboard-is-computed)
  - [Model and SAE formats, and Gemma 4](#model-and-sae-formats-and-gemma-4)
  - [Prisma schema summary](#prisma-schema-summary)
- [Part 2: what a native replacement needs per capability](#part-2-what-a-native-replacement-needs-per-capability)
- [Part 3: recommendation](#part-3-recommendation)
- [Unverified items](#unverified-items)

## Revision caveat

`/docker/interp/neuronpedia-pin.txt` pins upstream SHA `e5488b2a` (2026-07-14). The attached
clone of `hijohnnylin/neuronpedia` is a shallow fork checkout at `b9cf9cf` (2026-09-17) whose
oldest reachable commit is `7724688` (2026-07-17). The pinned SHA is not in the clone.

Two revisions were therefore read:

| Revision | Used for | How read |
|----------|----------|----------|
| `7724688` (2026-07-17, three days after the pin) | Compose files, Dockerfiles, dependency lists, service code as deployed | `git show 7724688:<path>` |
| `b9cf9cf` (HEAD) | Current service code, webapp, schema | working tree |

Material differences between the two:

- Commit `17bc391` (2026-08-20, "Rewrite backend services, generate cross-server APIs,
  relicense to Apache-2.0") deleted every Dockerfile and every `docker/compose*.yaml`. At HEAD
  "every service runs directly on the host" (`AGENTS.md`, Build and Run Commands). A pin bump
  past that commit is a change of deployment method, which `/docs/interp/platform.md:191-194`
  already warns about.
- The inference server replaced its hand-assembled stack (forked `transformer-lens` on a branch
  pin, `nnsight`, `nnterp`, vendored `chatspace`) with one first-party package,
  `interp-engine[vllm,quant]==1.11.0` (`apps/inference/pyproject.toml:58`), published to PyPI
  from a separate repository. Its internals are not in the tree.
- `apps/activations` and `apps/experiments` existed at `7724688` and are gone at HEAD;
  `apps/nla` and `apps/sparsity` exist at both.
- Schema: the `InferenceHostSource*` and `GraphHostSource*` tables became one `ComputeHost`
  table with two join tables; `Vector`, `VectorTag`, `VectorTagOnVector` were added. No
  dashboard table changed.
- Both revisions ship the same three inference presets at the repo root, including
  `.env.inference.gemma-2-2b-it.gemmascope-res-16k`. None was added for a newer model.

The three-day gap between the pin and `7724688` is unaudited (UNVERIFIED).

## Part 1: what Neuronpedia offers

### Service inventory at the pin

`7724688:docker/compose.yaml` defines seven services. `ai-experiments` deploys six of them
through `/docker/interp/neuronpedia.override.yaml`; `graph-crm` is not deployed.

| Service | Image | Port | Inputs | Outputs and state | GPU |
|---------|-------|------|--------|-------------------|-----|
| webapp | `apps/webapp/Dockerfile` target `runner`, Next.js on `node:22-alpine` | 3000 (3001 on host) | Postgres, the five Python services, S3, OpenAI embeddings | Postgres rows; S3 objects for graphs and lens shares | no |
| db-init | same Dockerfile, target `db-init` | none | Postgres | runs `prisma migrate deploy` then `prisma db seed` once (`7724688:apps/webapp/db-init.sh`) | no |
| postgres | `pgvector/pgvector:pg15` | 5432 | `pgvector-init` SQL | all persisted state | no |
| inference | `apps/inference/Dockerfile` | 5002 | one HF model plus SAELens SAE sets from `.env.inference.<model>.<sourceset>` | per-request JSON; no persistent state | yes |
| autointerp | `apps/autointerp/Dockerfile` | 5003 | activation records in the request body, an OpenRouter key | explanation text and scores in the response; no state | optional (embedding scorer only) |
| graph | `apps/graph/Dockerfile` | 5004 | one model plus one transcoder set from env | graph JSON, PUT to a presigned S3 URL the webapp minted, or returned in the body | yes |
| graph-crm | `apps/graph-crm/Dockerfile` | 5005 | as graph | as graph | yes |

`apps/graph-crm` does not exist at `7724688` (`git ls-tree 7724688 apps/`), so the compose entry
was unbuildable at the pin. The Complete Replacement Model backend it names lives inside
`apps/graph` as `neuronpedia_graph/crm_backend.py`, selected with `BACKEND=lm-saes-crm`.

One deployment defect at the pin worth recording: the `runner` stage's `init.sh` runs
`prisma db push` on every container start while `db-init` runs `prisma migrate deploy`
(`7724688:apps/webapp/Dockerfile:88,119`). Both apply the schema to the same database with
different strategies; the outcome depends on start order.

### webapp

Size: 539 TypeScript files, 120,769 lines (`app/` 63,448; `components/` 38,474; `lib/` 18,847),
106 runtime dependencies including `next@15`, `prisma@5`, `next-auth`, three AWS SDK clients,
`openai`, `@anthropic-ai/sdk`, `d3`, `plotly.js`, `@xyflow/react`, and about 20 `@radix-ui`
packages (`apps/webapp/package.json`).

**User-facing pages** (53 `page.tsx` files under `apps/webapp/app/`), grouped, with what each
reads and whether a GPU service is required:

| Group | Routes | Data read | GPU service |
|-------|--------|-----------|-------------|
| Feature dashboard | `/[modelId]/[layer]/[index]` (`feature-dashboard.tsx`), `/random`, `/quick-list`, `/embed/sae-explorable`, `/manual-dashboard` | `Neuron`, `Activation`, `Explanation` with scores, via `getNeuronOptimized` (`lib/db/neuron.ts:211`) | none to render; inference only for the test-text box, cosine-similar features, and on-demand explain/score |
| Model and source browse | `/[modelId]`, `/[modelId]/[layer]` (`page-source.tsx`), `/available-resources`, `/sae-bench*`, `/explanation-type/[name]`, `/score-type/[name]`, `/headvis`, `/[modelId]/head/[l]/[h]` | `Model`, `SourceSet`, `Source`, `SourceRelease`, `Eval`, `ModelHeadMetrics`, `ModelHeadSequence` | none, except live attention via inference |
| Search | `/search`, `/search-topk-by-token` | writes `SavedSearch`, `Activation` | inference |
| Search | `/search-explanations` | `Explanation.embedding` (pgvector HNSW) | none; OpenAI `text-embedding-3-large` at 256 dims (`lib/external/embedding.ts:33`) |
| Steer | `/steer`, `/steer/[slug]`, `/[modelId]/steer`, `/[modelId]/assistant-axis` | `SteerOutput`, `SteerOutputToNeuron`, `Vector` | inference |
| Graph | `/[modelId]/graph`, `/graph`, `/graph/validator`, `/graph/admin-browse`, `/beta/circuit/clt` | `GraphMetadata`, `GraphMetadataSubgraph`; the graph JSON is fetched from `GraphMetadata.url`, an S3 URL (`schema.prisma:186`) | graph, for generation only |
| Lens and NLA | `/[modelId]/jlens`, `/jlens/[id]`, `/[modelId]/nla`, `/nla/[id]` | `JlensShare` (blob in S3), `NlaExplainCache` | inference (`/v1/lens/prompt`), nla |
| Admin | `/admin` | drives the S3 import routes | none |
| Hosted-site | `/account`, `/signin`, `/user/*`, `/list/[id]`, `/unsubscribe-*`, `/blog`, `/contact`, `/privacy`, `/explorer/*`, `/[modelId]/gemmascope*` | `User`, `List*`, `Problem*` | none |

**API routes**: 122 route files under `apps/webapp/app/api/`. The ones that matter for this
investigation:

- **Dashboard ingest**. `GET /api/admin/import` streams a four-stage import from the
  `neuronpedia-datasets` S3 bucket (`lib/utils/s3.ts:5-10`): config JSONL into the
  `ExplanationType`, `ExplanationModelType`, `ExplanationScoreType`, `ExplanationScoreModel`
  and `EvalType` tables (`lib/db/import.ts:33-65`), then `release/model/sourceset/source.jsonl`,
  then `features/*.jsonl.gz` into `Neuron`, `activations/*.jsonl.gz` into `Activation`, and
  `explanations/*.jsonl.gz` into `Explanation` in 10,000-row batches (`import.ts:72-225`).
  `importJsonlString` (`import.ts:86`) reads the first line's keys, checks them against
  `information_schema.columns`, and does `INSERT ... SELECT FROM jsonb_to_recordset`, so the
  accepted JSONL fields are exactly the Prisma column names. `POST /api/feature/upload-batch`
  accepts up to 128 features per call with `density`, `topLogits`, `bottomLogits`,
  `explanations`, and `activations` (`app/api/feature/upload-batch/route.tsx:13-65`).
  `/api/admin/reset` is localhost-only. There is no `app/api/local/` directory at HEAD.
- **Live activation**. `POST /api/activation/new` calls inference `/v1/activation/single`
  (`lib/utils/inference.ts:474-558`); `POST /api/search-all` calls `/v1/activation/all` and
  persists the returned activations plus a `SavedSearch` row (`lib/db/activation.ts:55`);
  `POST /api/search-topk-by-token` calls `/v1/activation/topk-by-token`.
- **Steering**. `POST /api/steer` and `POST /api/steer-chat` (1,151 lines) call
  `/v1/steer/completion` and `/v1/steer/completion-chat` and cache in `SteerOutput`.
  `POST /api/steer-logits` calls the graph service.
- **Autointerp**. `POST /api/explanation/generate` loads the feature's `Activation` rows,
  dedupes and sorts by `maxValue`, keeps the top 20, and dispatches to one of five explainers.
  Four (`lib/external/autointerp-explainer{,-acts,-att,-logits}.ts`) call OpenAI, Anthropic, or
  OpenRouter directly from the webapp; only `autointerp-explainer-eleuther.ts` calls the
  autointerp service. `POST /api/explanation/score` normalizes activations to a max of 10 and
  calls `autointerp-scorer-eleuther.ts` (`/v1/score/fuzz-detection`, `/v1/score/embedding`) or
  the OpenAI-direct recall scorers. Results land in `Explanation` (with a pgvector embedding
  written by raw SQL) and `ExplanationScore`.
- **Graph**. `POST /api/graph/generate` calls the graph service, PUTs the JSON to
  `NP_GRAPH_BUCKET` under `user-graphs/`, validates it against `ATTRIBUTION_GRAPH_SCHEMA` with
  Ajv, and upserts `GraphMetadata` (`lib/utils/graph.ts:273-352`). Generation is restricted to
  `GRAPH_GENERATION_ENABLED_MODELS = ['gemma-2-2b','gemma-3-4b-it','qwen3-4b','qwen3-1.7b']`
  (`graph.ts:35`). No local-disk or database-blob storage mode for graph JSON exists in the
  webapp; S3 is the only path.
- **Compute hosts**. `POST /api/compute-host/register` writes `ComputeHost`; locally,
  `make host-add` writes the row directly (`AGENTS.md`, GPU servers are database rows).

**Single-user mode flags** (`lib/env.ts`): `IS_LOCALHOST` is literally
`NEXT_PUBLIC_URL === 'http://localhost:3000'` (line 135) and gates admin-panel access and the
three admin import routes' auth bypass. `DEMO_MODE` (line 147) makes every request run
signed-out (`lib/db/user.ts:47`) and skips saving steer and search results. The ai-experiments
delta sets `NEXT_PUBLIC_URL=http://localhost:3001` (`/docker/interp/neuronpedia.env-deltas`),
which does not equal the `IS_LOCALHOST` literal. Whether the admin panel is reachable at the
pin with that value is UNVERIFIED from source and is the kind of thing `np-smoke` does not
check.

### db-init and postgres

`db-init` is a one-shot container running `prisma migrate deploy` and `prisma db seed`
(`7724688:apps/webapp/db-init.sh`). The seed creates two users, `bot` and
`inferenceactivation` (`apps/webapp/prisma/seed.ts:32,47`); it creates no explainer or
scorer type rows, which arrive only through the S3 config import. Postgres needs the `vector`
extension for exactly one indexed column, `Explanation.embedding vector(256)`
(`schema.prisma:833,875-876`); the second pgvector column, `Sample.conversationEmbedding`, is
documented as written and read by nothing (`schema.prisma:1456-1460`).

### inference

Entry point `apps/inference/start.py` launching `neuronpedia_inference/server.py` (FastAPI app
at line 635), port 5002. Configuration is one env file per model-plus-SAE-set at the repo root;
the Gemma preset is six lines:

```
MODEL_ID=google/gemma-2-2b
OVERRIDE_MODEL_ID=google/gemma-2-2b-it
MODEL_DTYPE=bfloat16
SAE_DTYPE=bfloat16
SAE_SETS='["gemmascope-res-16k"]'
MAX_LOADED_SAES=200
```

(`.env.inference.gemma-2-2b-it.gemmascope-res-16k`; at `7724688` the ids are the short
Neuronpedia names rather than HF ids, otherwise identical.)

Endpoints (`apps/inference/openapi.json`, routers at `server.py:822-842`), all per-request and
per-prompt:

| Endpoint | Computes |
|----------|----------|
| `/v1/activation/single`, `single-batch` | one feature's activations over one text (`endpoints/activation/single.py:31`) |
| `/v1/activation/all`, `all-batch` | top-K features of a whole SAE set over one prompt (`all.py`, 418 lines) |
| `/v1/activation/topk-by-token`, `-batch` | per-token top-K features |
| `/v1/activation/source` | sparse activations of one source across prompts |
| `/v1/activation/attention` | one head's attention pattern |
| `/v1/activation/raw` | raw residual vectors, no SAE |
| `/v1/steer/completion`, `completion-chat` | steered and unsteered generation, SSE (`steer/completion.py` 527 lines, `completion_chat.py` 1,029 lines) |
| `/v1/lens/prompt` | logit lens and Jacobian lens, streamed (`lens/prompt.py`, 2,537 lines) |
| `/v1/tokenize`, `/v1/apply-chat-template` | tokenizer passthrough |
| `/v1/util/sae-vector`, `sae-topk-by-decoder-cossim`, `similarity-matrix-pred` | decoder-direction utilities |

**No inference endpoint runs an SAE over a corpus.** A grep of `neuronpedia_inference/` for
dataset, histogram, or corpus terms finds only steering-calibration code and one deliberately
unfilled optional field (`endpoints/activation/all.py:153`). The same holds at `7724688`.

Dependencies at HEAD (`apps/inference/pyproject.toml`): `interp-engine[vllm,quant]==1.11.0`,
`sae-lens>=6.30.0`, `transformers>=5.14`, `torch`, `fastapi`, `pandas`, `safetensors`.
TransformerLens is not a dependency at HEAD; only its hook-name convention survives, translated
in `engine_adapter.py` (942 lines). At `7724688` the stack was `transformer-lens` from a fork on
a branch pin, `sae-lens>=6.30.0,<7`, `nnsight>=0.6.0`, `nnterp`, and vendored `chatspace`
(`7724688:apps/inference/pyproject.toml:9-60`).

Glue versus owned logic: Neuronpedia-owned code is about 18,950 lines with 13,770 lines of
tests. The forward pass, hooking, steering, and the vLLM backend belong to `interp-engine`.
What the service owns is operational: SAE paging with a GPU budget (`sae_cache.py`, 488
lines), startup memory planning (`startup_memory.py`, `memory_cost.py`), request locking and
OOM recovery (`shared.py`), the lens stack (about 3,400 lines), and the steering endpoints.

### autointerp

Entry point `apps/autointerp/server.py` (272 lines), port 5003. Three routes:
`POST /v1/explain/default`, `POST /v1/score/fuzz-detection`, `POST /v1/score/embedding`.
Every route receives the activation records in the request body and rebuilds
`Example(tokens, values)` objects from them (`routes/explain/default.py:22-28`); the service
holds no model of the interpreted network and reads no database.

The library is EleutherAI's `sae-auto-interp` through the fork `hijohnnylin/sae-auto-interp`
(`apps/autointerp/pyproject.toml:21` and `[tool.uv.sources]`; at `7724688` the git source
carried no rev, tag, or branch). The string `delphi` does not appear anywhere in the
Neuronpedia tree except one README sentence. `sae-auto-interp` is the package Delphi was renamed
from, so the PS4 plan's "Delphi-based" wording describes the lineage, not the installed code.

Classes used: `DefaultExplainer(client, tokenizer=None, threshold=0.6)`, `FuzzingScorer`,
`DetectionScorer`, `EmbeddingScorer` over `dunzhang/stella_en_400M_v5` (`server.py:112`).
Neuronpedia-owned logic is `neuronpedia_autointerp/utils.py` (106 lines: ROC-AUC and balanced
accuracy aggregation). Total service source is about 845 lines. The service is glue.

**The LLM client is OpenRouter only, with no base-URL override**, at both revisions:
`OpenRouter(api_key=request.openrouter_key, model=request.model)` at
`routes/explain/default.py:29` and `routes/score/fuzz_detection.py:56`; a grep for `base_url`
or `OPENROUTER_BASE_URL` under `apps/autointerp` returns nothing. The `OPENROUTER_BASE_URL`
delta in `/docker/interp/neuronpedia.env-deltas` is therefore inert, and the note in
`/docker/interp/patches/README.md` that "verification found upstream support" is contradicted
by the source at both revisions. The only base URL constant is in the webapp
(`lib/utils/autointerp.ts:88`) and applies to the webapp's own direct explainers, not the
Eleuther route. Routing the PS4 batch to `vllm-gpu` requires a patch to the `sae-auto-interp`
fork's client, not to Neuronpedia.

### graph and graph-crm

Entry point `apps/graph/start.py` launching `neuronpedia_graph/server.py` (1,258 lines), port
5004. One process serves one model and one transcoder set from env (`MODEL_ID`,
`TRANSCODER_SET`), and rejects requests naming any other model (`7724688:server.py:318-321`).

Two attribution engines: `circuit-tracer` (default; `ReplacementModel.from_pretrained` at
`server.py:191-208`) and `lm-saes-crm`, which is the Complete Replacement Model from OpenMOSS'
Language-Model-SAEs (`llamascopium`), decomposing attention with Lorsa as well as MLPs with
transcoders (`apps/graph/README.md:4-6`, `crm_backend.py`). That second engine is what the
`graph-crm` compose entry meant. At `7724688` `circuit-tracer` was tag-pinned `v0.5.1` from
`decoderesearch/circuit-tracer` and `llamascopium` at `v2.0.0b34` (`pyproject.toml:30-32`).

Endpoints: `/health`, `/check-busy`, `/parse-chat-prompt`, `/forward-pass`, `/steer`,
`/generate-graph`. The graph JSON is deliberately not modelled in pydantic; its keys come from
`circuit_tracer`'s `build_model` (`schemas.py:15-18`). Upload is a `requests.put` to the
presigned URL the webapp minted (`server.py:1133-1203`); the service has no bucket credentials
and writes nothing to local disk. A hardcoded table `TRANSCODER_SET_TO_SOURCE_URL_ARRAYS`
(`server.py:286-304`) has four entries, and an unlisted set raises `KeyError` at upload
(`server.py:1150`).

Size: about 3,730 lines of source. The circuit-tracer path is thin glue; the owned code is the
CRM format converter (`format_converter.py`, 435 lines, including its own influence
power-iteration), `crm_interp_engine.py` (713 lines), chat-prompt parsing, and steer
generation. Transcoder downloads are 57 GiB for `mwhanna/qwen3-4b-transcoders` and 160 GiB for
`mntss/clt-gemma-2-2b-2.5M` (`apps/graph/README.md`, Disk and memory).

At `7724688`, `apps/graph/.env.example` holds only `SECRET` and `HF_TOKEN`; `MODEL_ID` and
`TRANSCODER_SET` are required at import time and appear in neither the example nor
`docker/compose.yaml`. `make np-graph-up` passes `--env-file apps/graph/.env`, which
`scripts/interp/np_env.sh` only touches. The graph server therefore cannot start from the
ai-experiments harness as written without an operator adding those two keys by hand.

**nla and sparsity** (present at both revisions, not in the compose file, not deployed): nla
(`apps/nla/server.py`, 2,903 lines, port 5009) is a Natural Language Autoencoder needing
bespoke fine-tuned verbalizer and reconstructor checkpoints plus a source model on one GPU;
sparsity (`apps/sparsity/server.py`, 476 lines, port 5005) reads weights of
`openai/circuit-sparsity` checkpoints. Neither applies to this deployment.

### How a feature dashboard is computed

The pipeline has three stages, and none runs inside a deployed container:

1. **SAEDashboard** (external, `jbloomAus/SAEDashboard`, PyPI `sae-dashboard`) runs the model
   and SAE over a Hugging Face text dataset and writes batch JSON. The documented invocation is
   `poetry run neuronpedia-runner --sae-set=... --dataset-path="monology/pile-uncopyrighted"
   --n-prompts=24576 --n-tokens-in-prompt=128 --n-features-per-batch=128` and takes "30 min to a
   few hours" (`README.md`, "I Want to Generate My Own Dashboards";
   `utils/neuronpedia-utils/README.md:5-25`). The per-feature statistics are computed there,
   not in this repo.
2. **neuronpedia-utils** converts to the import JSONL: `generate-dashboards-as-saelens.py`
   (628 lines, marked at line 1 as "identical to the vector script" and not yet repurposed),
   `generate-dashboards-as-vectors.py` (625), `convert-saedashboard-to-neuronpedia-export.py`
   (521), with the Python mirrors of the tables in `db_models/`. The README then syncs the
   output to `s3://neuronpedia-datasets/v1` (`utils/neuronpedia-utils/README.md:43-49`), and
   the upstream README says a third party must contact the maintainer to upload.
3. **The webapp admin import** downloads those objects and bulk-inserts them, as described
   above.

The statistics kept per feature are read directly from `process_data` in
`generate-dashboards-as-saelens.py:498-615`, which maps SAEDashboard output to the `Neuron`
and `Activation` columns:

| Statistic | `Neuron` column (`schema.prisma:762-825`) |
|-----------|-------------------------------------------|
| Maximum activation | `maxActApprox` |
| Density | `frac_nonzero` |
| Activation histogram | `freq_hist_data_bar_heights[]`, `freq_hist_data_bar_values[]` |
| Logit-effect histogram | `logits_hist_data_bar_heights[]`, `logits_hist_data_bar_values[]` |
| Top and bottom output tokens (decoder direction through the unembed) | `pos_str[]`, `pos_values[]`, `neg_str[]`, `neg_values[]` |
| Aligned and correlated neurons and features | `neuron_alignment_*`, `correlated_neurons_*`, `correlated_features_*` |
| Decoder weight distribution and decoder cosine neighbours | `decoder_weights_dist[]`, `topkCosSimIndices[]`, `topkCosSimValues[]` |
| UMAP coordinates | `umap_x`, `umap_y`, `umap_cluster`, `umap_log_feature_sparsity` |
| Custom vector for steering | `hasVector`, `vector Float[]`, `vectorDefaultSteerStrength` |

Each top-activating example is one `Activation` row (`schema.prisma:1040-1080`): `tokens[]`,
`values[]`, `maxValue`, `maxValueTokenIndex`, `minValue`, quantile bookkeeping
`binMin`/`binMax`/`binContains`, `qualifyingTokenIndex`, optional `lossValues[]`,
`logitContributions`, and direct-feature-attribution fields for attention SAEs. The only index
is `(modelId, layer, index)`.

So the split the previous session hypothesized is confirmed with one correction: the
"compute per-feature dashboards" half is not Neuronpedia code at all. It is SAEDashboard, an
external TransformerLens-based library, plus 1,800 lines of conversion scripts. Neuronpedia
owns storage and rendering.

### Model and SAE formats, and Gemma 4

**SAEs: SAELens only.** The single loader is `SAE.from_pretrained(release, sae_id, ...)` in
`apps/inference/neuronpedia_inference/saes/saelens.py:13-43`; `SAE_TYPE` has two members,
`saelens-1` and `neurons` (`sae_manager.py:23-26`). The set a pod loads is looked up in
SAELens' `get_pretrained_saes_directory()` by Neuronpedia set name (`config.py:268-330`). There
is no `params.npz`, custom safetensors, or local-directory SAE path. The same holds at
`7724688` through `SaeLensSAE.load` (`7724688:sae_manager.py:99-142`). An in-tree SAE must
therefore be published to Hugging Face and registered in SAELens' `pretrained_saes.yaml`
before the inference server can load it, which is also what the upstream README's step 4 says.

**Models: HF repo ids** resolved through `interp-engine`'s `load_model` and `select_backend`
(`config.py:14-133`), with `np_model_to_hf.json` at the repo root mapping short ids.

**Gemma 4 could not be added by configuration alone at the pin, and the gap is in the
ecosystem, not in Neuronpedia.** Facts from source:

- `np_model_to_hf.json` lists `gemma-4-e2b`, `gemma-4-e4b`, and `gemma-4-31b` at both
  `7724688` and HEAD, and `endpoints/lens/model_specific.py:23-27` documents Gemma 4's
  softcapping at both. The HEAD pyproject records that `interp-engine` 1.6.0 shipped
  "per-model point fixes for Gemma 4" (`apps/inference/pyproject.toml:49-51`). Whether the
  engine at the pin's stack (forked TransformerLens on a branch) could hook Gemma 4 is
  UNVERIFIED; `/docs/interp/gemma-4-interp.md:292-297` records that TransformerLens 3.5.1 has
  no Gemma 4 alias, which argues no.
- A Gemma 4 inference pod at HEAD needs a repo-root `.env.inference.gemma-4-e2b.<set>` file and
  an SAE set that SAELens can resolve for that model. No such set exists
  (`/docs/interp/gemma-4-interp.md:287-289`), and the loader accepts nothing else.
- Dashboards for Gemma 4 need SAEDashboard to run the model, which is TransformerLens-based
  (`generate-dashboards-as-saelens.py:118`: "The TransformerLens name of the model").
- Graphs for Gemma 4 need transcoders that do not exist, plus an entry in the four-row
  `TRANSCODER_SET_TO_SOURCE_URL_ARRAYS` table and the webapp's four-model allowlist.

So a new model requires: an env preset, a SAELens-registered SAE release on Hugging Face,
SAEDashboard support for the architecture, and hand-built dashboards. A new source set for an
already-served model requires the SAELens registration and the dashboard run.

### Prisma schema summary

`apps/webapp/prisma/schema.prisma` is 1,700 lines. Grouped by what a single-user localhost
deployment uses:

| Group | Tables | Used locally |
|-------|--------|--------------|
| Feature browsing (essential) | `Model`, `SourceSet`, `Source`, `SourceRelease`, `Neuron`, `Activation`, `Explanation`, `ExplanationType`, `ExplanationModelType`, `ExplanationScore`, `ExplanationScoreType`, `ExplanationScoreModel`, `Eval`, `EvalType` | yes |
| Head dashboards | `ModelHeadMetrics`, `ModelHeadSequence` | only if head metrics are imported |
| Inference, steering, graph state | `ComputeHost`, `ComputeHostOnSource`, `ComputeHostOnSourceSet`, `SteerOutput`, `SteerOutputToNeuron`, `SavedSearch`, `SavedSearchActivation`, `GraphMetadata`, `GraphMetadataSubgraph`, `GraphMetadataDataPutRequest`, `JlensShare`, `JlensSharePutRequest`, `Vector`, `VectorTag`, `VectorTagOnVector`, `NlaSource`, `NlaExplainCache`, `NlaExplainShare`, `ProcessLock` | partly; the graph and jlens rows point at S3 |
| Hosted-site concerns | `Account`, `Session`, `VerificationToken`, `User`, `UserSecret`, `List`, `ListsOnNeurons`, `ListsOnActivations`, `ListComment`, `Comment`, `Bookmark`, `Vote` | `User` and `UserSecret` are load-bearing: every row has a `creatorId` and the autointerp key is read from `UserSecret` |
| Other products | `ProblemNode`, `ProblemEdge`, `ProblemNodeComment`, `ProblemNodeLog`, `SampleDataset`, `Sample`, `Probe`, `ProbeMatch`, `ProbeLog`, `ExplanationActivationV1` (deprecated), `ExplanationModelSettings` (unwired per `schema.prisma:915`) | no |

Two pgvector columns exist; one is used (explanation search) and needs an OpenAI key. The
`Neuron.vector` and `Vector.values` columns are plain `Float[]` by design
(`schema.prisma:1662-1664`).

## Part 2: what a native replacement needs per capability

Native inventory (from `/interp`, `/rust-harness`, `/docs/interp`):

- `/interp` is 6,932 lines: harvest to NVMe over SAELens' cache runner (`harvest.py`, 857),
  BatchTopK SAE training saving as JumpReLU (`sae/train.py`, 346), aggregate SAE evaluation
  (`sae/evaluate.py`, 244: explained variance, L0, delta-LM-loss), an SAE registry that
  validates identity and delegates loading to `sae_lens.SAE.load_from_disk`
  (`sae/registry.py:6-9`), probes, steering extraction with a `sae_row` backend that reads one
  Gemma Scope decoder row via `SAE.from_pretrained` (`steering/extract.py:160-190`), and three
  LLM judges that score served-model behavior against `llama-cpu` on 8001
  (`steering/scorecard.py`, `probes/evaluate.py`, `organisms/verify.py`).
- `/rust-harness` is 50,328 lines targeting Gemma 4 E2B and E4B exclusively
  (`rust-harness/README.md:5-7`, `runtime/src/checkpoint.rs:176-185`). It exposes a `Tap`
  trait with one site, `LayerOutput`, plus opt-in attention weights (`runtime/src/model/cpu.rs`),
  batch capture binaries writing the Python shard format, a control-vector steering hook
  (`ControlVector::from_image`), an in-band probe monitor (`runtime/src/model/probe.rs`), and a
  two-route HTTP server (`/v1/health`, `/v1/chat/completions`, `harness/src/server.rs`).
  Measured decode on E4B is 5.5 to 5.8 tokens/second (`/docs/interp/writeback.md`). It has no
  SAE code.
- No module in either tree computes per-feature top-k examples, histograms, unembed logit
  effects, feature explanations, feature search, or attribution graphs.
- The P4-D4 trained-SAE-to-Neuronpedia import is referenced in five docstrings
  (`interp/sae/registry.py:3`, `interp/sae/train.py:11`, `training/recipes/schema.py:453`,
  `p4_d3_sae_training_investigation.md:138,255,344`) and does not exist as a command, script,
  or investigation. The previous session's statement that Neuronpedia's "only in-tree input is
  a trained SAE checkpoint via the P4-D4 import" is refuted: there is no in-tree input at all.

Difficulty scale, as specified: trivial (a day of agent work, no new dependencies), moderate
(a few days, a new dependency or artifact format), hard (research code that must be validated
against a reference).

| Capability | Native coverage today | Native build | Difficulty | Gemma 4 with in-tree SAEs |
|------------|----------------------|--------------|------------|---------------------------|
| Per-feature dashboard computation | none; `harvest.py` already streams post-block residuals to NVMe, and `sae/evaluate.py:97` already encodes cached rows through SAELens | about 800 lines Python: encode a harvest through the SAE, keep a bounded top-k heap per feature with token-level values, quantile-bucketed examples, activation histogram, density, decoder row through the unembed for top and bottom tokens (after the `layer_scalar` caveat in `/docs/interp/gemma-4-interp.md:157-193`), decoder cosine neighbours; write one safetensors plus JSON per source under `/mnt/ai-data/interp/`. GPU required for the encode pass over a corpus of the SAEDashboard default size (24,576 prompts by 128 tokens). No new dependency: torch, SAELens, and the harvest loader are present | moderate; the logic is bookkeeping, not research. Validation reference: SAEDashboard on Gemma-2-2B `gemmascope-res-16k` layer 12, comparing top-k example sets and histograms for a sample of features | yes; this is the only path that yields Gemma 4 dashboards at all |
| Feature page rendering | none; `site/` is Docusaurus | either a static-site generator emitting one HTML page per feature from the artifact (about 600 lines Python, no server), or a small FastAPI plus one page of JavaScript (about 1,200 lines). No database needed; the artifact directory is the store | trivial to moderate | yes |
| Live activation on typed text | none in-process; no SAE code in Rust | Python: load model through `families.py` hooks, encode with SAELens, return per-token values (about 300 lines, GPU). Rust: a JumpReLU encoder over `Tap::on_residual` is about 400 lines of Rust plus a safetensors reader, and would run on the CPU harness at the serving rate | trivial (Python) or moderate (Rust) | yes |
| Feature search by text | none | activation-on-text search is the previous row plus a top-k over all features (about 200 lines). Explanation search: local embeddings with `sentence-transformers` over a few thousand strings need no index (about 150 lines, one new dependency) | trivial | yes |
| Autointerp explanation generation | judge machinery only (`scorecard.py:76-118` builds structured chat-completions calls with grammar-enforced JSON against local serving) | prompt construction from the top-k examples plus a chat call to `vllm-gpu` or `llama-cpu`; about 400 lines reusing `parse_judge_json`. The Neuronpedia equivalent is `DefaultExplainer` in `sae-auto-interp`, which can also be imported directly | trivial to write; validation reference for prompt fidelity is Neuronpedia's Eleuther route on the same 20 Gemma Scope features, or the upstream `sae-auto-interp` tests | yes |
| Autointerp scoring (detection, fuzzing, embedding) | none | import path: `pip install sae-auto-interp` (or `delphi`) and call `FuzzingScorer`, `DetectionScorer`, `EmbeddingScorer` on locally built `Example` objects, which is exactly what `apps/autointerp` does in about 850 lines including the server; a library-only wrapper is about 300 lines. The blocker at the pin remains: the fork's `OpenRouter` client has no base-URL parameter, so the import path either patches that client or reimplements it as an OpenAI-compatible client (about 100 lines) | moderate; the scoring maths is the library's. Validation reference: the library's own scores on the same feature set through OpenRouter, once, to confirm the local-endpoint client is equivalent | yes |
| Attribution graph generation | none | import path: `circuit-tracer` (tag `v0.5.1` at the pin) needs per-layer transcoders for the model. None exist for Gemma 4 and training them is itself research: a transcoder per layer at 16k to 262k width on hundreds of millions of tokens, and the published sets are 57 GiB to 160 GiB. A native attribution implementation is not the hard part; the transcoders are | hard. Validation reference: `circuit-tracer` on Gemma-2-2B with its published transcoders, comparing node and edge sets for the same prompt | not without a transcoder training program; estimate at least two weeks of GPU-bound work before a graph can be validated |
| Attribution graph viewing | none | the upstream viewer is a `@xyflow/react` app inside the webapp; a standalone viewer of the published graph schema is about 1,500 lines of TypeScript. Alternative: the `circuit-tracer` repository ships its own local viewer (UNVERIFIED from this tree) | moderate, and pointless without generation | n/a |
| Steering through SAE features | fully covered: `steering/extract.py` `sae_row`, `steering/export.py`, Rust `ControlVector` applied inside the forward pass, `make steer-scorecard` | nothing beyond pointing `sae_row` at an in-tree SAE directory instead of a Gemma Scope release (about 30 lines) | trivial | yes, already |
| Admin import of Gemma Scope | `SAE.from_pretrained` for one decoder row only | a native dashboard run over Gemma Scope layers with the dashboard code from row one; no import step exists because the artifact is produced locally | covered by row one | n/a (Gemma 2 only) |
| Serving the model | fully covered for Gemma 4 by the harness; Neuronpedia serves Gemma-2-2B only | nothing | trivial | yes, already |

Totals for the tractable half (dashboards, pages, live activation, search, explanation
generation, scoring wrapper): about 3,000 lines of Python and small JavaScript, one or two new
Python dependencies (`sae-auto-interp` or `delphi`, `sentence-transformers`), one GPU pass per
source set, and one validation run against SAEDashboard output. Estimate five to eight agent
days plus one to two host days for the GPU passes and the comparison, assuming the compare
against SAEDashboard is done on Gemma-2-2B where both sides exist.

The hard half is attribution graphs. A native version on Gemma 4 requires trained
transcoders before any graph code matters; that is a separate research program, not a port.

## Part 3: recommendation

**Verdict: replace Neuronpedia entirely, and do not carry the graph service forward either.**

Reasoning, in order of weight:

1. **The platform cannot meet the tier on the model that matters.** Every Neuronpedia
   capability in this deployment is bound to Gemma-2-2B through three independent locks: the
   SAELens-only loader, the SAEDashboard pipeline, and the transcoder table. The rest of the
   tier is Gemma 4 only (`/rust-harness/README.md:5-7`, `/interp/families.py:8-15`). An
   in-tree Gemma 4 SAE reaches Neuronpedia only after publication to Hugging Face, registration
   in SAELens, and a SAEDashboard run whose TransformerLens dependency has no Gemma 4 alias at
   the pinned version. The P4-D4 import that was to bridge this does not exist.
2. **The tractable half is bookkeeping around code the tree already has.** Harvest, SAE
   encode, the hook registry, and structured LLM judging exist. What is missing is a top-k
   heap, a histogram, one matrix product against the unembed, and a page template. The
   validation reference (SAEDashboard on Gemma Scope) is runnable on the same host.
3. **The hard half is not Neuronpedia's to give.** Autointerp scoring is a 106-line wrapper
   around a library that can be imported directly; the only Neuronpedia-specific part is the
   OpenRouter-only client that PS4 needed to patch anyway. Attribution graphs are
   `circuit-tracer` plus transcoders; Neuronpedia adds S3 upload, a presigned-URL protocol, and
   a viewer. Keeping the graph container buys Gemma-2-2B graphs at a 160 GiB cache and a
   fourth CUDA image, for a model nothing else in the tree studies.
4. **The orchestration cost is not thin.** Three vendored Dockerfile patches
   (`/docker/interp/patches/`), an inert `OPENROUTER_BASE_URL` delta, a graph env file that
   needs undocumented keys, an `IS_LOCALHOST` literal that the 3001 remap does not match, a pin
   whose successor commit deletes every compose file, and a `db push` versus `migrate deploy`
   race in the upstream image. `/docs/verification/interp/interp.md:242-347` records three
   failed `make np-build` attempts on 2026-09-17 and 2026-09-18; a later successful build is
   reported by the operator but not yet recorded there (UNVERIFIED from the tree). Whether PS2,
   PS4, or PS5 has run on the host is likewise not recorded.

What goes, if the verdict is accepted:

| Path | Lines |
|------|-------|
| `/make/neuronpedia.mk` (targets `np-clone`, `np-env`, `np-build`, `np-up`, `np-inference-up`, `np-autointerp-up`, `np-graph-up`, `np-down`, `np-status`, `np-smoke`) | 98 |
| `/docker/interp/neuronpedia-pin.txt`, `neuronpedia.override.yaml`, `neuronpedia.env-deltas` | 113 |
| `/docker/interp/patches/` (README and three patches) | |
| `/scripts/interp/np_env.sh`, `np_smoke.sh`, `np_autointerp_preflight.sh` | 259 |
| `/docs/interp/platform.md` (rewritten for the native artifact layout), the Neuronpedia rows in `/docs/interp/README.md`, `/docs/README.md`, `/docs/os/storage-configuration.md`, `/docs/verification/interp/interp.md`, `/docs/verification/bugs/known-bugs.md` (KB-68, KB-76, KB-77 close as moot), `/site/docs/interp/guides/neuronpedia.md`, `/site/sidebars.js` | |
| `/mnt/ai-data/interp/neuronpedia/{src,postgres}` on the host, and the two `mkdir` lines in `/make/interp.mk:41-42` and `/docker/interp/entrypoint.sh:27` | |
| `OPENAI_API_KEY` in `/env.interp.example` | |

What stays: `/scripts/interp/preflight.sh` (shared by the workbench targets),
`/docker-compose.interp.yaml` and `/docker/interp/Dockerfile.interp` (the JupyterLab
workbench, not Neuronpedia), the `gemma2` family entry and Gemma Scope constants in
`/interp/families.py:29-33` (still used by `sae_row` steering), and the eight frozen
investigation records.

What the repository gains: Gemma 4 dashboards from in-tree SAEs; zero containers, no Postgres,
no S3 code path, no pin bumps, no vendored patches; artifacts that live beside the harvest and
SAE manifests under `/mnt/ai-data/interp/` with the same manifest chain; explanation and
scoring that run against `vllm-gpu` or `llama-cpu` without a patched client.

What it loses: comparable Eleuther-style scores on the published Gemma Scope dictionaries (the
native scorer can still be run on Gemma Scope layers, but nobody else's numbers will be from
the same pipeline); the graph viewer and Gemma-2-2B attribution graphs, which have no Gemma 4
equivalent regardless; the Gemma Scope presets and the hosted dataset import, replaced by a
local dashboard run per layer; the logit and Jacobian lens pages and the assistant-axis
readouts, which were never in scope for PS1 to PS5.

Cost, stated without softening: five to eight agent days plus one to two host days for the
tractable half, validated against SAEDashboard on Gemma-2-2B before trusting Gemma 4 output.
Attribution graphs on Gemma 4 are at least two weeks of GPU-bound transcoder training before a
single validated graph, and should be scheduled as a research pull, not as part of the
replacement.

## Unverified items

- The exact tree at `e5488b2a`; `7724688` was read as the proxy.
- What the `interp-engine` package contains; only its import surface is visible.
- Whether the admin panel is reachable when `NEXT_PUBLIC_URL` is `http://localhost:3001`.
- The `ExplanationType` and `ExplanationScoreType` row names, which live in S3 config JSONL.
- The operator-reported successful `make np-build`, and whether PS2, PS4, or PS5 has run.
- Every measurement cell in `/docs/interp/platform.md:151-159`.
- Whether the `circuit-tracer` repository ships a standalone viewer.
