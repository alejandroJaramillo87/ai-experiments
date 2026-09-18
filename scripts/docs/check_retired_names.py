#!/usr/bin/env python3
"""Fail the quality gate when a retired name is still in use anywhere in the tree.

check_doc_refs.py catches a document naming a path that does not exist. It cannot
catch a *renamed* file, because the classifier requires a slash: a bare
`docker-compose.yaml` in prose is not a path claim and is never checked.
Loosening that rule was measured and rejected -- accepting extension-bearing
tokens with no slash produced 361 new failures, because prose legitimately says
`config.json`, `main.rs`, and `manifest.json`.

So this takes the opposite approach. Rather than infer which tokens are claims,
it holds an explicit table of names that have been retired, and fails on any
occurrence of one. That has no false positives by construction, and it names the
replacement in the error, so the fix is mechanical rather than archaeological.

The table is the point. Adding an entry when you rename something is what makes
the next rename self-policing; a rename that skips this file is a rename whose
stragglers will be found by a human months later, which is the failure mode the
whole documentation campaign exists to end.

Unlike check_doc_refs.py this scans code as well as documentation, because a
retired filename hides in shell recipes, Python, Dockerfiles, and templates just
as readily as in prose -- and there it is a broken command, not a stale sentence.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.output import Printer  # noqa: E402

# Retired name -> (replacement, why it was retired). Both halves are shown in the
# failure, so the reason should read as an explanation to whoever hits it.
RETIRED: dict[str, tuple[str, str]] = {
    "site/scripts/gen-cli-reference.sh": (
        "scripts/docs/gen_site_reference.py",
        "the site's reference generator moved from a bash script that pasted --help into a fenced block to a Python script beside the gate that checks its output, because the pages became a usage line plus one table per help section and that parse deserved a unit test under tests/docs/",
    ),
    "docs/verification/training-phase0.md": (
        "docs/verification/archive/training-phase0.md",
        "the verification directory grew a folder per tier (training, interp, rust-harness, observability, hardware, bugs), so that a human scanning it sees six categories rather than a dozen loose files. Every tier got a folder, including those holding one document today: a mixed root of folders and stray files is what makes a directory hard to read, and the single-file tiers are the ones most likely to gain a sibling. README.md and step-zero-environment.md stay at the root because step zero blocks every phase and is not a peer of the tiers",
    ),
    "docs/verification/training-phase1.md": (
        "docs/verification/archive/training-phase1.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "docs/verification/training-phase2.md": (
        "docs/verification/archive/training-phase2.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    # The three training phase runbooks froze on 2026-09-18, when the training
    # verification set was reorganised by capability: one runbook per loop the
    # lab runs (fine-tuning first), each proving its capability through the
    # harness and interp rather than in isolation. The phase runbooks described
    # the tier in build order, with GGUF on llama-gpu as evidence and Qwen3 as
    # the fallback family; both are gone. Their verified entries (P0.1, P0.2,
    # P0.3, P0.5) stay citable as substrate facts at the archive path.
    "docs/verification/training/training-phase0.md": (
        "docs/verification/archive/training-phase0.md",
        "frozen; its live successor is docs/verification/training/fine-tuning-loop.md",
    ),
    "docs/verification/training/training-phase1.md": (
        "docs/verification/archive/training-phase1.md",
        "frozen; its live successor is docs/verification/training/fine-tuning-loop.md",
    ),
    "docs/verification/training/training-phase2.md": (
        "docs/verification/archive/training-phase2.md",
        "frozen; the continual-pretraining runbook that succeeds it is not yet written",
    ),
    "docs/verification/interp.md": (
        "docs/verification/interp/interp.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "docs/verification/reference-captures.md": (
        "docs/verification/interp/reference-captures.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "docs/verification/rust-harness.md": (
        "docs/verification/rust-harness/rust-harness.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "docs/verification/rust-harness-qa.md": (
        "docs/verification/rust-harness/rust-harness-qa.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "docs/verification/observability.md": (
        "docs/verification/observability/observability.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "docs/verification/hardware-stress.md": (
        "docs/verification/hardware/hardware-stress.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "interp/steering/prompts/honesty_pairs.jsonl": (
        "interp/steering/prompts/honesty.jsonl",
        "repeng's positive/negative pair format left with repeng; the steering set "
        "now uses the capture format, one text and label per line",
    ),
    "docs/verification/known-bugs.md": (
        "docs/verification/bugs/known-bugs.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "docs/verification/ledger-archive.md": (
        "docs/verification/bugs/ledger-archive.md",
        "the same tier-folder move; see docs/verification/training-phase0.md above",
    ),
    "docker-compose.yaml": (
        "docker-compose.inference.yaml",
        "the inference compose file gained a domain suffix, matching its four "
        "siblings (agents, interp, training, rust-harness); a bare `docker "
        "compose` with no -f no longer finds it",
    ),
    "docs/precision.md": (
        "docs/precision/README.md",
        "precision grew a second document -- the policy (what each tier does) "
        "and the decision record (why, and what would have to change) are "
        "different obligations and drift apart when filed together -- so the "
        "single file became a directory with an index",
    ),
    "docs/interpretability": (
        "docs/interp",
        "the docs directory was the last place the long form survived; the "
        "package, the container, the compose file, the make include, the tests, "
        "and the CLI are all `interp`. Keyed on the path prefix, not the bare "
        "word, which stays correct English throughout the prose",
    ),
    "docs/observability/roadmap.md": (
        "docs/observability/design-and-roadmap.md",
        "the operator-facing specification graduated into "
        "docs/observability/README.md, leaving the design rationale and the "
        "OBS item ledger; `roadmap.md` no longer described what the file "
        "held, and a bare `roadmap.md` says nothing about which roadmap",
    ),
    "docs/verification/gemma-4-conformance.md": (
        "docs/verification/investigations/gemma-4-conformance_investigation.md",
        "the audit was always a frozen record — audit-only, bounded by a "
        "Limitations section, every finding a quote at a line — but carried no "
        "class marker and sat among the runbooks, contradicting this "
        "directory's claim that every document in it is executed. The style "
        "guide requires a frozen record be findable by path; the new location "
        "satisfies both patterns instead of relying on the docs/verification/ "
        "exemption, which exists for operational runbooks",
    ),
    "./datasets": (
        "./scripts/benchmark/datasets",
        "the llama-cpu mount pointed at a repo-root directory that never held "
        "the corpus -- the committed Wikitext-2 copy has always been under "
        "scripts/benchmark/ -- so test_perplexity.sh died at its in-container "
        "check, and Docker creating the empty root directory made it a "
        "namespace package that shadowed the `datasets` library for mypy. "
        "Keyed with the ./ prefix so the bare word, which is the Python "
        "package throughout pyproject.toml and the smoke tests, stays legal",
    ),
    "gpu_benchmark_results": (
        "/mnt/ai-data/benchmarks/nsys (nsys reports) or "
        "/mnt/ai-data/benchmarks/gpu (BENCH_RESULTS_DIR)",
        "benchmark output is run data, and run data lives on the data SSD "
        "beside models, logs, and training runs rather than in a worktree that "
        "a branch switch can carry away. The directory had two creators -- the "
        "nsys bind mounts and a CWD-relative mkdir in benchmark_gpu.sh -- so "
        "removing either alone would not have stopped it reappearing",
    ),
    "perplexity_results": (
        "/mnt/ai-data/benchmarks/perplexity (BENCH_RESULTS_DIR)",
        "the sibling of gpu_benchmark_results in test_perplexity.sh, retired "
        "for the same reason and at the same time",
    ),
    # ComfyUI and the diffusion stack left this repository entirely. Nothing
    # replaced them here: image and video generation is a different problem
    # with a different dependency graph -- its own torch pin, its own pip
    # environment, its own custom-node checkouts -- and carrying it beside the
    # LLM stack meant one repository answering to two sets of version
    # constraints. Each key below names something deleted, so the replacement
    # half says so rather than pointing at a path that is not there.
    "./outputs/comfyui": (
        "nothing; removed with the ComfyUI service",
        "generated images and video were multi-GB data with no reason to sit "
        "next to source; the mount put them one `git add -A` away from a "
        "commit, and .gitignore had no entry for the directory. The output "
        "tree moved to /mnt/ai-data before the service itself was retired",
    ),
    "/mnt/ai-data/comfyui": (
        "nothing; removed with the ComfyUI service",
        "the output tree on the data SSD, created by setup_data_ssd.sh and "
        "bound into the container; both ends went with the service",
    ),
    "docker/comfyui/": (
        "nothing; removed with the ComfyUI service",
        "the only image in the tree built on stdlib venv and pip rather than "
        "uv against uv.lock, and the only one setting PYTHONPATH -- it had to "
        "be, because its torch pin and its custom-node requirements could not "
        "resolve against the core environment",
    ),
    "make/comfy.mk": (
        "nothing; removed with the ComfyUI service",
        "held comfy-up/down/logs/shell/build/status and the art-mode and "
        "llm-mode swap targets; every one of them addressed the retired "
        "service",
    ),
    "make art-mode": (
        "nothing; removed with the ComfyUI service",
        "stopped the GPU LLM services and started ComfyUI in their place. "
        "With one fewer GPU consumer there is nothing to swap between",
    ),
    "make llm-mode": (
        "make up",
        "the counterpart to art-mode, meaningful only as `switch back`; "
        "starting the LLM services is what `make up` already does",
    ),
    "scripts/models/download_art_models.sh": (
        "nothing; removed with the ComfyUI service",
        "fetched diffusion checkpoints, text encoders, VAEs and LoRAs into "
        "models/diffusion_models, models/clip and models/vae -- the three "
        "model directories retired alongside it",
    ),
    "docs/inference/art-generation.md": (
        "nothing; removed with the ComfyUI service",
        "the service overview; the inference docs now cover llama.cpp and vLLM only",
    ),
    "./workflows": (
        "nothing; removed with the ComfyUI service",
        "held the ComfyUI workflow JSON templates and was bind-mounted to "
        "/app/user/default/workflows. Keyed with the ./ prefix so the bare "
        "word, which names unrelated things in agent tooling, stays legal",
    ),
    # The GGUF-to-checkpoint move (rust-harness P2-P8). Keys are qualified,
    # because gguf.rs and the two artifact readers survive: the Python interp
    # exporters write probe and control-vector images in that container, and
    # llama.cpp consumes the control vector. A bare `from_gguf` or `gguf.rs`
    # key would fire on live code.
    "CpuModel::from_gguf": (
        "CpuModel::from_checkpoint",
        "the CPU backend loads config.json and model.safetensors instead of a "
        "GGUF; the converter's metadata could only imply what a checkpoint "
        "states, and a SentencePiece-unigram reading of a BPE vocabulary is "
        "what that cost the project once already",
    ),
    "ModelSpec::from_gguf": (
        "ModelSpec::from_checkpoint",
        "retired with the loader above; layer_types, partial_rotary_factor and "
        "global_head_dim are named config.json fields rather than a stride, a "
        "divisor tensor, and a key that does not exist",
    ),
    "Config::from_gguf": (
        "ModelSpec::from_checkpoint",
        "folded into the spec constructor with the loader above; Config no "
        "longer carries sliding_window_pattern, which was a GGUF spelling "
        "nothing outside that function read",
    ),
    "cpu::check_supported": (
        "cpu::check_supported_checkpoint",
        "the half that policed GGUF metadata namespaces went with the "
        "container -- named config.json fields are validated by "
        "TextConfig::validate -- and the architectural refusals it kept are "
        "now expressed over Hugging Face tensor names",
    ),
    "write_probe_gguf": (
        "write_probe",
        "the probe left GGUF for a safetensors image plus a JSON sidecar, so "
        "the container is no longer part of the function's identity; the "
        "control-vector exporter keeps its _gguf name because llama.cpp reads "
        "that artifact and the format is not this repository's to choose",
    ),
    "export_gguf_fixtures.py": (
        "export_interp_fixtures.py",
        "the generator writes one GGUF and one safetensors pair, so naming it "
        "after the container it no longer exclusively produces would send a "
        "reader looking for a probe image that is not there",
    ),
    "python-probe.gguf": (
        "python-probe.safetensors",
        "the committed probe fixture follows its format; note that this file "
        "also served tests/models/conftest.py as a generic small GGUF, a job "
        "that passed to python-control-vector.gguf and then, at R10, to the "
        "frozen tests/models/fixtures/tiny-exporter-written.gguf",
    ),
    "python-control-vector.gguf": (
        "python-control-vector/vector.safetensors",
        "the control vector left GGUF at R10 and the fixture became the "
        "artifact directory the harness actually consumes, image plus sibling "
        "manifest; the old file was not deleted but frozen and moved to "
        "tests/models/fixtures/tiny-exporter-written.gguf, where it is still "
        "the one GGUF in the tree no code under test wrote",
    ),
    "vector.gguf": (
        "vector.safetensors",
        "the file inside an artifact directory under "
        "/mnt/ai-data/interp/vectors/<name>/; llama.cpp read the GGUF and "
        "R10 retired that path, so a runbook still naming the old file sends "
        "an operator to a vector `make steer-up` will not find",
    ),
    "write_control_vector_gguf": (
        "write_control_vector",
        "the exporter writes safetensors, so the container is gone from the "
        "name; the function otherwise does the same job",
    ),
    "ControlVector::from_gguf": (
        "ControlVector::from_image",
        "the reader takes a checkpoint::Image rather than a parsed GGUF, the "
        "same envelope the weights and the probe arrive through",
    ),
    # R11 retired the request-logging proxy. The harness serves the requests
    # and scores the probe in the same process, so the record it writes
    # carries the score; there is nothing left for a sidecar to capture or an
    # offline replay to recompute. Keys are qualified where the bare word
    # would fire on live code -- "monitor" and "replay" both still name things
    # that exist.
    "interp-logproxy": (
        "harness-serve",
        "the harness records each served request itself, so the proxy in "
        "front of llama-server has nothing left to capture",
    ),
    "docker/interp-logproxy/logproxy.py": (
        "rust-harness/harness/src/interplog.rs",
        "the record is written by the process that answered the request, "
        "which is also the process that scored the probe",
    ),
    "make logproxy-up": (
        "make harness-serve",
        "the serving harness writes the log itself; there is no sidecar to "
        "start beside it",
    ),
    "make fullprec-up": (
        "make harness-serve",
        "the bf16 endpoint existed so a replay would match the serving "
        "precision; in-band scores come from the serving weights, so there is "
        "no precision gap to close",
    ),
    "llama-gpu-bf16": (
        "harness-serve",
        "the bf16 endpoint served monitored traffic at the precision the "
        "probes were fitted at, so that an offline replay would match it; "
        "in-band scoring removed the replay, and a precision-matched endpoint "
        "with nothing to match is a GPU service no command points at. Its "
        "deletion emptied the `interp` compose profile, which went too",
    ),
    "LLAMA_GPU_BF16_MODEL": (
        "HARNESS_MODEL",
        "the variable chose the checkpoint for the deleted bf16 endpoint; the "
        "only full-precision service left is the serving harness, which takes "
        "its checkpoint from HARNESS_MODEL",
    ),
    "LOGPROXY_TARGET_URL": (
        "HARNESS_INTERP_LOG",
        "the log is written by the service being observed rather than by a "
        "proxy pointed at it, so there is no target to name",
    ),
    "LOGPROXY_TARGET_SERVICE": (
        "HARNESS_INTERP_LOG",
        "the log is written by the service being observed rather than by a "
        "proxy pointed at it, so there is no target to name",
    ),
    "replay_fidelity": (
        "the report's scored field",
        "the function stated how far a replay's scores were from the serving "
        "model's; scores are now taken in the serving process, so the gap it "
        "described does not exist",
    ),
    "GgufWriter": (
        "synth::build_f32_safetensors",
        "the writer existed so tests could produce the interp artifacts the "
        "Python exporters wrote as GGUF; both are safetensors now, and "
        "nothing in the workspace emits GGUF",
    ),
    "schemas/gguf_artifacts.json": (
        "schemas/control_vector_artifact.json",
        "the contract covered both interp artifacts while both were GGUF; the "
        "probe left at R7 and the control vector at R10, so what remained was "
        "one artifact named after a container it no longer uses",
    ),
    "direction.<layer>": (
        "block.<index>",
        "control-vector tensors are named for the 0-based block they steer, "
        "and index 0 is legal; the retired spelling refused it because "
        "llama.cpp and repeng number from 1. The prefix changed with the "
        "floor so a pre-R10 image fails to load rather than loading",
    ),
    "gguf-info": (
        "checkpoint-info",
        "the inspection tool follows the format the backend loads; it echoes "
        "config.json and tabulates safetensors dtypes and shapes rather than "
        "reconstructing both from metadata keys and ggml type ids",
    ),
    "docs/rust-harness/gemma-4-ingestion.md": (
        "docs/models/gemma-4.md",
        "the document described the GGUF ingestion contract, and the contract "
        "died with the loader; the family facts that outlived it -- the "
        "rope_freqs divisor tensor and the per_layer naming that conversion "
        "drops -- moved to the model reference, which documents the published "
        "files llama.cpp still serves",
    ),
    "gguf_info_check.rs": (
        "checkpoint_info_check.rs",
        "renamed with the binary it tests; the --check exit-code contract it "
        "pins is unchanged",
    ),
    # The Tier 3 acceptance stubs, deleted rather than ported (rust-harness
    # P9.7). Both were #[ignore] + unimplemented!() against things that no
    # longer exist -- a `cuda` feature, llama.cpp, a quantized GGUF -- and
    # their fixture designs were the reason not to port them: perplexity
    # against llama-perplexity on a quantized GGUF compares two different
    # models, and goldens minted from our own first run pin whatever the
    # implementation did that day.
    "cuda_acceptance.rs": (
        "runtime/src/ladder.rs (rungs L1 and L2)",
        "the correctness ladder measures the residual stream and the output "
        "distribution against transformers, on the real checkpoint, with no "
        "cuda feature involved; the stub had waited for a backend that was "
        "never built",
    ),
    "eval_acceptance.rs": (
        "runtime/src/ladder.rs (rungs L3 and L4)",
        "perplexity and greedy decoding are ladder rungs against "
        "transformers' own numbers rather than against llama.cpp on a "
        "quantized GGUF and a golden generated by ourselves; recipes moved to "
        "rust-harness/fixtures/reference/README.md",
    ),
    # The tests/utils/ split. Keyed per file rather than on the directory:
    # tests/utils/ still exists and still holds the scripts/utils/ reporters, so
    # a bare `tests/utils` key would fire on the files that correctly stayed.
    **{
        f"tests/utils/{name}": (
            f"tests/{dest}/{name}",
            "tests/utils/ covers the scripts/utils/ host reporters, as its "
            "README has always said; the common/ and scripts/docs/ suites had "
            "accumulated there against that scope and moved to directories "
            "that mirror what they test",
        )
        for name, dest in (
            ("test_obs_audit.py", "common"),
            ("test_obs_catalog.py", "common"),
            ("test_obs_context.py", "common"),
            ("test_obs_core.py", "common"),
            ("test_obs_dispatch.py", "common"),
            ("test_obs_failure.py", "common"),
            ("test_obs_record_contract.py", "common"),
            ("test_obs_retention.py", "common"),
            ("test_obs_serve.py", "common"),
            ("test_common_output.py", "common"),
            ("test_doc_refs.py", "docs"),
            ("test_retired_names.py", "docs"),
        )
    },
    # The Neuronpedia harness (interp PS1) left this repository on 2026-09-18.
    # Nothing replaced it: the platform was bound to Gemma-2-2B by its SAE loader,
    # its dashboard pipeline and its transcoder table, none of which the tier's
    # Gemma 4 work could reach, and three attempts to build its upstream images
    # at the pinned commit all failed on upstream Dockerfile defects. The
    # reasoning is docs/interp/investigations/neuronpedia_replacement_investigation.md.
    # Each key below names something deleted, so the replacement half says so.
    "make/neuronpedia.mk": (
        "nothing; removed with the Neuronpedia harness",
        "held the ten np-* targets: clone, env, build, up, the three per-service "
        "up targets, down, status and smoke; every one addressed the retired stack",
    ),
    "make np-clone": (
        "nothing; removed with the Neuronpedia harness",
        "checked out the pinned upstream commit under /mnt/ai-data and applied the "
        "vendored patches",
    ),
    "make np-env": (
        "nothing; removed with the Neuronpedia harness",
        "merged the env deltas into the checkout's example env files",
    ),
    "make np-build": (
        "nothing; removed with the Neuronpedia harness",
        "built the webapp trio and the three CUDA images; never completed at the pin",
    ),
    "make np-up": (
        "nothing; removed with the Neuronpedia harness",
        "started the webapp, db-init and postgres",
    ),
    "make np-inference-up": (
        "nothing; removed with the Neuronpedia harness",
        "started the inference pod for one MODEL_SOURCESET preset",
    ),
    "make np-autointerp-up": (
        "nothing; removed with the Neuronpedia harness",
        "started the autointerp pod against local serving",
    ),
    "make np-graph-up": (
        "nothing; removed with the Neuronpedia harness",
        "started the attribution-graph pod",
    ),
    "make np-down": (
        "nothing; removed with the Neuronpedia harness",
        "stopped every platform container",
    ),
    "make np-status": (
        "nothing; removed with the Neuronpedia harness",
        "listed the platform containers",
    ),
    "make np-smoke": (
        "nothing; removed with the Neuronpedia harness",
        "checked loopback bindings, the webapp port, postgres and sm_120 visibility",
    ),
    "docker/interp/neuronpedia-pin.txt": (
        "nothing; removed with the Neuronpedia harness",
        "the pinned upstream SHA; its successor commit upstream deletes every "
        "Dockerfile and compose file, so there was no fixed pin to bump to",
    ),
    "docker/interp/neuronpedia.override.yaml": (
        "nothing; removed with the Neuronpedia harness",
        "the compose override: loopback ports, the 3001 remap, the build args that "
        "moved two CUDA stages to a 24.04 base",
    ),
    "docker/interp/neuronpedia.env-deltas": (
        "nothing; removed with the Neuronpedia harness",
        "the env deltas merged into the checkout, one of which (the OpenRouter base "
        "URL) the replacement investigation found upstream never read",
    ),
    "docker/interp/patches/": (
        "nothing; removed with the Neuronpedia harness",
        "three vendored Dockerfile patches against upstream defects at the pin, "
        "KB-68, KB-76 and KB-77; none was ever confirmed by a completed build",
    ),
    "scripts/interp/np_env.sh": (
        "nothing; removed with the Neuronpedia harness",
        "merged the env deltas and copied the secrets into the checkout",
    ),
    "scripts/interp/np_smoke.sh": (
        "nothing; removed with the Neuronpedia harness",
        "the platform smoke check behind make np-smoke",
    ),
    "scripts/interp/np_autointerp_preflight.sh": (
        "nothing; removed with the Neuronpedia harness",
        "refused the autointerp pod unless vllm-gpu was up and the other GPU "
        "consumers down; scripts/interp/preflight.sh, which the workbench targets "
        "share, stays",
    ),
    "docs/interp/platform.md": (
        "nothing; removed with the Neuronpedia harness",
        "the platform runbook: port map, bring-up, concurrency matrix, pin-bump "
        "procedure. The workbench it was indexed as covering is documented by "
        "docs/interp/writeback.md and the site guides",
    ),
    "site/docs/interp/guides/neuronpedia.md": (
        "nothing; removed with the Neuronpedia harness",
        "the site guide, with its sidebar entry",
    ),
    "/mnt/ai-data/interp/neuronpedia": (
        "nothing; removed with the Neuronpedia harness",
        "the checkout and the postgres volume on the data SSD, created by make "
        "interp-dirs and the workbench entrypoint; both mkdir sites went with it. "
        "The operator removes the directory by hand",
    ),
    "NP_SRC": (
        "nothing; removed with the Neuronpedia harness",
        "the checkout path variable read by every np-* target",
    ),
    "NP_GPU": (
        "nothing; removed with the Neuronpedia harness",
        "opted the autointerp pod onto the GPU for its embedding scorer",
    ),
    "MODEL_SOURCESET": (
        "nothing; removed with the Neuronpedia harness",
        "selected the upstream inference preset (model plus SAE set) for the "
        "inference pod",
    ),
    "harness/src/obs.rs": (
        "rust-harness/obs/src/lib.rs",
        "the run log became a workspace member of its own. Its two libc::flock calls are the only unsafe in the workspace, and the seven binaries in runtime -- which carries forbid(unsafe_code) -- need to write the same records, so the writer moved rather than the attribute weakening. The short key matches both the bare and the rust-harness/-prefixed spelling",
    ),
    # Qwen3 left the training tier on 2026-09-18: the lab works on Gemma 4 E2B
    # and E4B and nothing else, so the fallback family and its recipe twins
    # went. Keyed on the bare filename, which the pattern also matches behind a
    # directory prefix, so one key covers both spellings.
    "sft-qlora-qwen3.yaml": (
        "training/recipes/reference/sft-qlora-e2b.yaml",
        "the Qwen3-1.7B QLoRA twin of the Gemma 4 E2B reference recipe",
    ),
    "sft-qlora-qwen3-4b.yaml": (
        "training/recipes/reference/sft-qlora-gemma-e4b.yaml",
        "the Qwen3-4B QLoRA twin of the Gemma 4 E4B recipe card",
    ),
    "distill-qwen3.yaml": (
        "training/recipes/reference/distill-e2b.yaml",
        "the Qwen3 twin of the single-stage logit distillation reference",
    ),
    "distill-response-qwen3.yaml": (
        "training/recipes/reference/distill-response-gemma-e2b.yaml",
        "the Qwen3 twin of the stage-1 response distillation card",
    ),
    "distill-logit-qwen3.yaml": (
        "training/recipes/reference/distill-logit-gemma-e2b.yaml",
        "the Qwen3 twin of the stage-2 logit distillation card",
    ),
    "continual-qwen3.yaml": (
        "training/recipes/reference/continual-gemma4-e2b.yaml",
        "the Qwen3 twin of the ZeRO-Offload continual pretraining card",
    ),
    "harvest-calibration-qwen3.yaml": (
        "training/recipes/interp/harvest-calibration-gemma2.yaml",
        "the Qwen3 activation-harvest calibration capture; the Gemma 2 recipe "
        "stays as the Gemma Scope anchor for pretrained SAEs",
    ),
    "sae-calibration-qwen3.yaml": (
        "training/recipes/interp/sae-calibration-gemma2.yaml",
        "the SAE-training calibration recipe over the Qwen3 capture, and the "
        "only TransformerLens-supported subject the SAEBench cross-check had; "
        "that cross-check left with the family",
    ),
}

# Directories holding no text of ours.
PRUNE_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "target",
    "__pycache__",
    ".ruff_cache",
    # pytest's tmp_path base when the default temp dir is noexec (the agent
    # sandbox; see tests/conftest.py). Generated fixtures, not our text.
    ".pytest-tmp",
}

# Extensions worth reading. Anything textual that can name a file.
SCAN_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".mk",
    ".rs",
    ".json",
    ".cfg",
    ".ini",
    ".example",
    ".txt",
}

# Extensionless or oddly-named files that still carry commands.
SCAN_NAMES = {"Makefile", "Dockerfile"}
SCAN_NAME_PREFIXES = ("Dockerfile.",)

# Frozen by policy: investigation write-ups record the tree as it was on the day
# they were written, so naming the old file is accurate history.
# Same exemption set as check_doc_refs.py, for the same reason -- including the
# empty EXEMPT_PREFIXES, where that script records why docs/verification/ stopped
# being exempt on 2026-08-08. This gate has no false positives by construction,
# so narrowing it was the lower-risk half of that change.
EXEMPT_PARTS = ("investigations", "vendor", "archive")
# Agent memory records carry a `verified:` date and a `stale_when:` condition:
# they state what was true when written, and a record whose condition has fired
# is closed rather than rewritten. Renaming the symbols inside one would make it
# claim to have observed something it did not.
EXEMPT_PREFIXES: tuple[str, ...] = (".claude/agent-memory/",)
EXEMPT_SUFFIXES = ("_investigation.md",)

# Exempt by name, each for a stated reason.
EXEMPT_FILES = {
    # This file: it defines the table, so it necessarily spells every retired name.
    "scripts/docs/check_retired_names.py",
    # The unit tests, which assert on the table and so quote its keys.
    "tests/docs/test_retired_names.py",
    # The sibling gate. Its KNOWN_ABSENT table is a curated list of references
    # that name something absent on purpose, so a deleted file's name is
    # exactly what an entry is made of -- the same argument this file makes
    # for itself one line above. Narrow in practice: the table is a few dozen
    # audited pairs, each carrying its own reason.
    "scripts/docs/check_doc_refs.py",
}


def is_exempt(rel_path: str) -> bool:
    """Whether a file is outside this gate's scope."""
    parts = Path(rel_path).parts
    return (
        any(p in EXEMPT_PARTS for p in parts)
        or rel_path.startswith(EXEMPT_PREFIXES)
        or rel_path.endswith(EXEMPT_SUFFIXES)
        or rel_path in EXEMPT_FILES
    )


def wants_scan(name: str) -> bool:
    """Whether a filename looks like text that could name a file."""
    return (
        Path(name).suffix in SCAN_SUFFIXES
        or name in SCAN_NAMES
        or name.startswith(SCAN_NAME_PREFIXES)
    )


def iter_files(root: Path) -> list[Path]:
    """Every in-scope text file under root."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNE_DIRS]
        for name in filenames:
            if not wants_scan(name):
                continue
            path = Path(dirpath) / name
            if not is_exempt(str(path.relative_to(root))):
                found.append(path)
    return sorted(found)


def pattern_for(old: str) -> re.Pattern[str]:
    """Match `old` as a whole name rather than as a fragment of a longer one.

    The trailing guard deliberately permits a following `.` so a sentence ending
    in the name still matches; the cost is that a hypothetical `<old>.bak` would
    also match, which is the rarer and more tolerable error.
    """
    return re.compile(rf"(?<![\w.-]){re.escape(old)}(?![\w-])")


def find_retired(
    text: str, patterns: dict[str, re.Pattern[str]]
) -> list[tuple[int, str]]:
    """Every (line number, retired name) occurrence in text."""
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for old, pattern in patterns.items():
            if pattern.search(line):
                hits.append((lineno, old))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO_ROOT, help="repository root to scan"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = Printer()

    if not RETIRED:
        out.ok("retired-names: no names are currently retired")
        return 0

    files = iter_files(root)
    # The shellcheck step's empty-set guard, for the same reason: a gate that
    # silently checked nothing is indistinguishable from one that passed.
    if not files:
        out.error(f"retired-names: enumerated zero files under {root}")
        out.line("  The gate would pass without checking anything.")
        return 1

    patterns = {old: pattern_for(old) for old in RETIRED}
    hits: list[tuple[Path, int, str]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Not text after all; nothing to check.
            continue
        for lineno, old in find_retired(text, patterns):
            hits.append((path, lineno, old))

    out.info(f"retired-names: {len(files)} files, {len(RETIRED)} retired names")
    if not hits:
        out.ok("retired-names: no retired name is still in use")
        return 0

    out.error(f"retired-names: {len(hits)} references use a retired name")
    for path, lineno, old in hits:
        replacement, _ = RETIRED[old]
        out.line(f"  {path.relative_to(root)}:{lineno}: {old} -> {replacement}")
    for old, (replacement, reason) in RETIRED.items():
        out.line(f"  {old} was replaced by {replacement}: {reason}")
    out.line("  Solution: use the replacement name.")
    out.line("  A frozen dated record may instead belong under an exemption.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
