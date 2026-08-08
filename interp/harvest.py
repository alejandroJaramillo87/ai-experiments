"""Activation harvesting (interp Phase 4 P4-D2): a workbench batch job.

Thin glue around SAELens v6's activation-caching runner: batched forward
passes over a corpus at harvest.dtype, caching residual-stream activations under
/mnt/ai-data/interp/activations/<name>/. Our additions are the manifest and
the operational envelope — disk preflight against the watchdog floor, the D9
run lock, the D10 watchdog sampler, and hook-level resume.

Resume is chunk-level. SAELens provides none of its own: run() raises when its
output directory is not empty, so a hook captured in one call cannot be resumed
or even retried in place. Capturing each hook as a sequence of chunks routes
around that -- every chunk writes a fresh directory, so the refusal never fires
-- and a crash costs one chunk instead of the whole run. Chunks are assembled
into the single directory the SAE trainer reads once the last one lands
(KB-58).

Hooks are HF named-module paths resolved per model family through the
interp/families.py registry; the layout follows the config the caller loaded,
so Gemma 4 from its composite config is model.language_model.layers.{n} and a
flat text model is model.layers.{n}. The output of a decoder layer is the
post-block residual stream. TransformerLens aliases are not used because its
architecture registry is closed: HookedTransformer converts a fixed set of
families and Gemma 4 is not among them at our 3.5.1 pin (P4-D1 resolved
decision). TL 3.x does ship TransformerBridge, which wraps arbitrary HF
models, so the gap is the registry rather than the library as a whole.

Captures are a regenerable cache: the manifest is written only on completion,
mark_validated flags a capture deletable, and gc_captures deletes flagged
captures oldest-first while keeping their manifests.

Heavy imports (torch, transformers, sae_lens) are lazy so the CPU sandbox can
import this module and tests can inject a fake runner function.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Iterable
from typing import NamedTuple

from omegaconf import DictConfig

from interp import families
from interp.families import HookFamilySpec, UnknownHookFamilyError
from interp.manifest import (
    ManifestError,
    file_xxhash,
    git_sha,
    load_manifest,
    require_match,
    require_stat,
    utc_now,
    write_manifest,
)

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}

# The subset of DTYPE_BYTES that Arrow can actually store, and so the subset a
# capture can be taken at today. bf16 is absent because Arrow has no bfloat16
# in any version, while SAELens hands harvest.dtype to both torch and
# datasets.Array2D from one field -- see KB-57 and _saelens_runner.
ARROW_STORAGE_DTYPES = frozenset({"fp16", "fp32"})

# Recipe spelling to the name SAELens expects. Shared with interp/sae/train.py
# so the harvest and the SAE trained on it cannot name different dtypes.
DTYPE_NAMES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
PROGRESS_NAME = "progress.json"  # completed hooks; deleted when the manifest lands
ARTIFACT = "activation_harvest"

# Token budget per chunk when harvest.chunk_tokens is 0. A crash costs at most
# one chunk, so this trades resume granularity against the per-chunk overhead
# of a consolidation and a progress write. At 15M tokens the 300M-token
# reference capture is 20 chunks of roughly 50 minutes each.
DEFAULT_CHUNK_TOKENS = 15_000_000

# SAELens's per-buffer memory budget (`CacheActivationsRunnerConfig
# .buffer_size_gb`), capped at 2GB by the comment there because the HF datasets
# writer struggles with shards above it. Replicated rather than imported so the
# disk preflight can price a capture without loading sae_lens or a model.
SAELENS_BUFFER_GB = 2.0

# Suffix of the staging directory holding one completed chunk per subdirectory.
# Chunks are named shard_{i:05d} because SAELens's _consolidate_shards requires
# that prefix and skips anything else in the directory -- see _assemble_chunks.
CHUNKS_SUFFIX = ".chunks"


class HarvestError(Exception):
    """Harvest config, preflight, or subject resolution failed."""


def hook_name(layer: int, spec: HookFamilySpec) -> str:
    """The family's HF named-module path for the post-block residual stream."""
    return families.hook_name(spec, layer)


def validate_harvest(recipe: DictConfig) -> None:
    """Cross-field invariants for the harvest block; raises listing all."""
    cfg = recipe.harvest
    problems: list[str] = []
    if not cfg.enabled:
        problems.append("harvest.enabled: must be true for a harvest run")
    if bool(cfg.model_id) == bool(cfg.organism_dir):
        problems.append("harvest: exactly one of model_id or organism_dir must be set")
    if not cfg.hook_layers:
        problems.append("harvest.hook_layers: at least one layer is required")
    elif any(int(layer) < 0 for layer in cfg.hook_layers):
        problems.append("harvest.hook_layers: layer indices must be >= 0")
    if len(cfg.hook_layers) > 1 and not cfg.acknowledge_multi_layer:
        problems.append(
            "harvest.hook_layers: multi-layer capture multiplies the disk "
            "footprint per layer; set harvest.acknowledge_multi_layer=true "
            "to confirm"
        )
    if not cfg.corpus:
        problems.append("harvest.corpus: HF dataset id or local JSONL required")
    if cfg.dtype not in DTYPE_BYTES:
        problems.append(f"harvest.dtype: {cfg.dtype!r} not in {tuple(DTYPE_BYTES)}")
    if cfg.total_tokens <= 0:
        problems.append("harvest.total_tokens: must be positive")
    if cfg.context_len <= 0:
        problems.append("harvest.context_len: must be positive")
    if cfg.chunk_tokens < 0:
        problems.append("harvest.chunk_tokens: must be >= 0 (0 selects the default)")
    elif cfg.chunk_tokens and cfg.chunk_tokens < cfg.context_len:
        # A chunk shorter than one sequence has no whole sequence to write, and
        # SAELens refuses context_size > training_tokens in its own config.
        problems.append(
            f"harvest.chunk_tokens: {cfg.chunk_tokens} is below "
            f"harvest.context_len ({cfg.context_len}); a chunk must hold at "
            f"least one full sequence"
        )
    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        raise HarvestError(
            f"Harvest validation failed ({len(problems)} problem(s)):\n{lines}"
        )


def resolve_subject(cfg: DictConfig) -> tuple[str, str | None, str]:
    """Resolve (model path or id, revision, model hash) for the capture.

    organism_dir points at a sealed P4-D1 organism; its public manifest's
    checkpoint field is the subject. Refuses an organism without a manifest
    or checkpoint — an unsealed organism is not a valid harvest subject.
    """
    if cfg.organism_dir:
        manifest = load_manifest(cfg.organism_dir)
        checkpoint = manifest.get("checkpoint", "")
        if not checkpoint:
            raise HarvestError(
                f"organism manifest records no checkpoint: {cfg.organism_dir} "
                f"(seal the organism with --checkpoint first)"
            )
        return checkpoint, None, manifest.get("model_hash", "unknown")
    return cfg.model_id, cfg.revision, "unknown"


def corpus_identity(corpus: str) -> tuple[str, str | None]:
    """(corpus id or path, content hash). Local files are xxh3-hashed;
    an HF dataset is identified by id (its hash field stays null)."""
    if os.path.isfile(corpus):
        return corpus, file_xxhash(corpus)
    return corpus, None


def chunk_plan(total_tokens: int, chunk_tokens: int, context_len: int) -> list[int]:
    """Per-chunk token budgets for one hook, summing to exactly total_tokens.

    Budgets are spread evenly rather than filled to chunk_tokens with a
    remainder, so no chunk is a stub: a 100-token capture in 30-token chunks
    is 25+25+25+25, not 30+30+30+10. A stub chunk is not merely inelegant --
    SAELens refuses a config whose context_size exceeds its training_tokens,
    so a remainder below one sequence would fail at the last chunk, hours in.

    chunk_tokens of 0 selects DEFAULT_CHUNK_TOKENS. A capture smaller than one
    chunk yields a single chunk, which still goes through assembly.
    """
    size = min(chunk_tokens or DEFAULT_CHUNK_TOKENS, total_tokens)
    size = max(size, context_len)
    count = max(1, math.ceil(total_tokens / size))
    base, extra = divmod(total_tokens, count)
    return [base + (1 if i < extra else 0) for i in range(count)]


def buffer_tokens(
    d_model: int, dtype: str, model_batch_size: int, context_len: int
) -> int:
    """Tokens SAELens writes per buffer -- its indivisible unit of capture.

    Sized in bytes rather than sequences (`CacheActivationsRunnerConfig
    .n_tokens_in_buffer`): a 2GB budget divided by d_model * itemsize, floored
    to a whole model batch. dtype therefore changes it -- the same recipe at
    fp16 buffers twice as many tokens as at fp32 -- which is why this cannot be
    a constant. `tests/interp/test_saelens_contract.py` pins the formula and
    the 2.0GB default against the pinned wheel.
    """
    per_token = d_model * DTYPE_BYTES[dtype]
    raw = int(SAELENS_BUFFER_GB * 1e9) // per_token
    batch = model_batch_size * context_len
    return max(raw - (raw % batch), batch)


def captured_tokens(cfg: DictConfig, d_model: int) -> int:
    """Tokens a capture actually writes, which exceeds what it was asked for.

    SAELens writes ceil(chunk / buffer) whole buffers per chunk, so each chunk
    overshoots by up to one buffer and none of them undershoots. Chunking moves
    that rounding from once per capture to once per chunk, so it is a cost of
    the KB-58 fix: the 300M-token reference capture at fp16 in 15M-token chunks
    lands at 310.6M, 3.5 percent high. The preflight is this tier's only
    defense against filling the volume, and pricing the request rather than the
    capture under-priced it by 33GB there.
    """
    buffer = buffer_tokens(
        d_model, cfg.dtype, int(cfg.model_batch_size), int(cfg.context_len)
    )
    plan = chunk_plan(
        int(cfg.total_tokens), int(cfg.chunk_tokens), int(cfg.context_len)
    )
    return sum(math.ceil(chunk / buffer) * buffer for chunk in plan)


def projected_bytes(cfg: DictConfig, d_model: int) -> int:
    return (
        captured_tokens(cfg, d_model)
        * d_model
        * DTYPE_BYTES[cfg.dtype]
        * len(cfg.hook_layers)
    )


def disk_preflight(
    path: str, projected: int, floor_gb: float, disk_usage_fn=None
) -> None:
    """Refuse when free space minus the projection breaches the floor."""
    usage = (disk_usage_fn or shutil.disk_usage)(path)
    floor = int(floor_gb * (1 << 30))
    if usage.free - projected < floor:
        raise HarvestError(
            f"disk preflight refused: projected capture is "
            f"{projected / (1 << 30):.1f}GB, free space on {path} is "
            f"{usage.free / (1 << 30):.1f}GB, floor is {floor_gb:.0f}GB. "
            f"Free space, reduce harvest.total_tokens or hook_layers, or "
            f"run make interp-gc"
        )


class ModelMeta(NamedTuple):
    """What one config read establishes about the subject."""

    d_model: int
    spec: HookFamilySpec
    num_layers: int


def resolve_model_meta(
    model_id: str, revision: str | None, d_model_override: int = 0
) -> ModelMeta:
    """(d_model, hook family, depth) for the subject, from one config read.

    A nonzero d_model override skips the width read, but the family always
    resolves from the config: hook paths are registry data, never assumed.
    The text-config unwrap and hidden_size read live in interp/families.py
    (Gemma 4 nests the language model under text_config; the residual stream
    is hidden_size, not hidden_size_per_layer_input).

    num_layers is 0 when the config does not state it, which callers must
    read as "unknown" rather than "no layers" -- see check_hook_layer_bounds.
    """
    from transformers import AutoConfig

    hf_cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    try:
        spec = families.family_from_config(hf_cfg)
    except UnknownHookFamilyError as exc:
        raise HarvestError(str(exc)) from exc
    d_model = int(d_model_override) or families.residual_width(hf_cfg)
    if not d_model:
        raise HarvestError(
            f"could not resolve hidden_size for {model_id}; set "
            f"harvest.d_model explicitly"
        )
    return ModelMeta(d_model, spec, families.decoder_depth(hf_cfg))


def check_hook_layer_bounds(
    layers: Iterable[int], num_layers: int, model_id: str
) -> None:
    """Refuse a hook index the subject has no layer for.

    validate_harvest cannot do this: it sees only the recipe, and depth is a
    property of the model. Without it an index past the last layer reaches
    the runner and fails only once the capture is under way.

    This bounds the range, which is all a depth can decide. It does not catch
    an in-range index chosen for a wrong reason -- the recipes' layer 15,
    justified as "50 percent of a 30-layer stack", is a legal index on a
    35-layer model. Stated-fact drift of that kind is caught instead by
    tests/models/test_gemma4_facts_propagation.py.

    A depth of 0 means the config did not state one; skip rather than refuse,
    so an unfamiliar config shape does not block a run it cannot judge.
    """
    if num_layers <= 0:
        return
    over = sorted({int(layer) for layer in layers if int(layer) >= num_layers})
    if over:
        raise HarvestError(
            f"harvest.hook_layers: {over} out of range for {model_id}, which "
            f"has {num_layers} layers (valid indices are 0..{num_layers - 1})"
        )


def check_hook_dims(
    model_id: str,
    layers: list[int],
    revision: str | None = None,
    dtype: str = "bf16",
) -> dict[int, int]:
    """One-token forward asserting each hook yields the d_model-wide residual.

    The Gemma 4 gate before committing GPU-days: per-layer embeddings inject
    a 256-dim per-token input into each decoder layer, so the check proves
    the hook output is the hidden_size residual and not a PLE tensor.
    Runs at the dtype the harvest will use, so the gate exercises the
    precision the capture is taken at rather than a fixed one.

    Returns {layer: observed width}; raises HarvestError on any mismatch.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dtype not in DTYPE_NAMES:
        raise HarvestError(
            f"harvest.dtype {dtype!r} is not one of {sorted(DTYPE_NAMES)}"
        )
    torch_dtype = getattr(torch, DTYPE_NAMES[dtype])
    # Config-only, so an out-of-range hook fails here rather than after the
    # weights are on the GPU.
    expected, spec, depth = resolve_model_meta(model_id, revision)
    check_hook_layer_bounds(layers, depth, model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch_dtype, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    observed: dict[int, int] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            observed[layer] = int(tensor.shape[-1])

        return hook

    try:
        for layer in layers:
            try:
                module = model.get_submodule(hook_name(layer, spec))
            except AttributeError as exc:
                # The documented backstop for the multimodal-composite
                # Gemma 4 layout risk: a missing module path is a dimension
                # check failure, not a raw traceback.
                raise HarvestError(
                    f"hook dimension check failed: no module at "
                    f"{hook_name(layer, spec)} ({exc}); the model does not "
                    "expose the flat text-model layout this hook path assumes"
                ) from exc
            handles.append(module.register_forward_hook(make_hook(layer)))
        inputs = tokenizer("hook check", return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    bad = {layer: w for layer, w in observed.items() if w != expected}
    if bad:
        raise HarvestError(
            f"hook dimension check failed: expected width {expected}, got "
            f"{bad}; the hook is not capturing the residual stream"
        )
    return observed


def _tokenizer_fingerprint(model_id: str, revision: str | None) -> str:
    from transformers import AutoTokenizer

    from interp.manifest import tokenizer_fingerprint

    return tokenizer_fingerprint(
        AutoTokenizer.from_pretrained(model_id, revision=revision)
    )


class _SaeLensSession:
    """One loaded model and one corpus stream, captured in contiguous chunks.

    SAELens's CacheActivationsRunner refuses a non-empty output directory, so
    a chunk that crashes cannot be resumed or retried in place (KB-58). Chunks
    are how that becomes survivable: each run() writes a fresh directory, so
    the refusal never fires, and a crash costs one chunk instead of the run.

    Reusing one runner across chunks is what makes them contiguous.
    CacheActivationsRunner.__init__ loads the model and builds the activations
    store once, and the store carries the corpus stream position; every
    per-chunk quantity SAELens derives (n_buffers, n_seq_in_buffer) is a
    property recomputed from cfg.training_tokens at access time. So chunk i+1
    resumes exactly where chunk i stopped, and no corpus-offset parameter is
    needed -- the store has no token budget of its own to re-apply
    (total_training_tokens is stored and never read back).
    """

    def __init__(self, job: dict):
        import importlib.metadata

        from sae_lens import CacheActivationsRunner, CacheActivationsRunnerConfig

        self._job = job
        self._context_len = int(job["context_len"])
        self._cfg = CacheActivationsRunnerConfig(
            model_name=job["model_id"],
            model_class_name="AutoModelForCausalLM",
            hook_name=job["hook"],
            d_in=int(job["d_in"]),  # required by sae-lens 6.46.0
            dataset_path=job["corpus"],
            context_size=self._context_len,
            # Per chunk, not the capture total: every derived buffer count
            # reads this field, so it is what sizes a single run().
            training_tokens=int(job["chunk_tokens"]),
            new_cached_activations_path=None,  # reassigned per chunk below
            model_batch_size=job["model_batch_size"],
            dtype=DTYPE_NAMES[job["dtype"]],
            seed=job["seed"],
            device="cuda",
            model_from_pretrained_kwargs=(
                {"revision": job["revision"]} if job["revision"] else {}
            ),
        )
        self._runner = CacheActivationsRunner(self._cfg)
        self._version = importlib.metadata.version("sae-lens")

    def capture(self, out_dir: str, chunk_tokens: int) -> int:
        """Capture one chunk into an empty out_dir; return the tokens written.

        The count is measured rather than assumed. SAELens silently starts a
        new epoch when the corpus runs dry -- get_raw_llm_batch defaults to
        raise_on_epoch_end=False, so the store resets its iterator and keeps
        yielding -- which means a requested budget is not evidence of what
        landed on disk. run() returns the Dataset, so the rows are countable.
        """
        self._cfg.training_tokens = int(chunk_tokens)
        self._cfg.new_cached_activations_path = out_dir
        dataset = self._runner.run()
        return len(dataset) * self._context_len

    def fast_forward(self, tokens: int) -> None:
        """Advance the corpus stream past tokens already captured.

        Only needed after a crash: within one process the stream is already in
        the right place. A resumed run is a new process, so the store starts
        at the beginning of the corpus and would re-capture what earlier
        chunks already hold.

        This pulls tokens without running the model. get_batch_tokens is the
        tokenization half of get_raw_llm_batch, and only the token iterator
        carries stream position -- so skipping the forward pass advances the
        stream identically at a fraction of the cost. It is still real work
        (streaming and tokenizing the skipped corpus), which is why it is paid
        only on resume.
        """
        store = self._runner.activations_store
        batch_tokens = store.store_batch_size_prompts * self._context_len
        for _ in range(tokens // batch_tokens):
            store.get_batch_tokens()

    def info(self) -> dict:
        return {
            "tokenizer_fingerprint": _tokenizer_fingerprint(
                self._job["model_id"], self._job["revision"]
            ),
            "sae_lens_version": self._version,
        }


def _saelens_runner(job: dict) -> _SaeLensSession:
    """Default session factory: SAELens v6 over an HF model.

    job carries model_id, revision, hook, d_in, corpus, context_len,
    chunk_tokens, dtype, seed, model_batch_size. Exact config field names are
    pinned by the workbench smoke run.

    job["dtype"] reaches two type systems: DTYPE_NAMES maps it to a torch dtype
    for the forward pass, and the same string becomes the datasets.Array2D
    storage type. Arrow has no bfloat16, so "bf16" type-checks in one and fails
    in the other (KB-57).
    """
    return _SaeLensSession(job)


def _consolidate_chunks(chunks_dir: str, out_dir: str) -> None:
    """Flatten completed chunk directories into one dataset at out_dir.

    Delegates to CacheActivationsRunner._consolidate_shards, which is the same
    function run() uses to flatten its own per-buffer shards. Reimplementing it
    would mean reimplementing the dataset fingerprint derivation
    (generate_fingerprint over a loaded Dataset) against an unpublished
    contract, to get a worse copy of a function that already ships. The cost is
    a dependency on a private name; tests/interp/test_saelens_contract.py pins
    it so an upgrade that renames it fails in the sandbox rather than 16 hours
    into a capture.

    copy_files=False makes each arrow file a same-filesystem shutil.move, so
    the whole step is metadata: seconds for a terabyte, and no second copy to
    fit on the volume. It removes chunks_dir on success.

    Creating out_dir belongs here rather than in the caller: _consolidate_shards
    requires it to exist and be empty, and the caller reads its existence as
    the signal that assembly began.
    """
    from pathlib import Path

    from sae_lens import CacheActivationsRunner

    os.makedirs(out_dir, exist_ok=True)
    CacheActivationsRunner._consolidate_shards(
        Path(chunks_dir), Path(out_dir), copy_files=False
    )


def _load_progress(capture_dir: str) -> dict:
    path = os.path.join(capture_dir, PROGRESS_NAME)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        # completed_chunks arrived with chunked capture; a progress.json
        # written before it is still readable, with every hook at zero.
        loaded.setdefault("completed_chunks", {})
        return loaded
    return {"completed_hooks": [], "completed_chunks": {}}


def _save_progress(capture_dir: str, progress: dict) -> None:
    path = os.path.join(capture_dir, PROGRESS_NAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def run_harvest(
    recipe: DictConfig, runner_fn=None, disk_usage_fn=None, assemble_fn=None
) -> dict:
    """Validate, preflight, lock, watch, harvest each hook; return the manifest.

    The manifest lands only on completion — a capture directory without
    manifest.json is incomplete by definition.

    Each hook is captured as a sequence of chunks (see chunk_plan), written to
    <hook>.chunks/shard_NNNNN and assembled into <hook> when the last one
    lands. Rerunning skips hooks recorded complete in progress.json and
    resumes an interrupted hook at its first uncaptured chunk, so a crash
    costs one chunk rather than the run.

    runner_fn builds a capture session for one hook (see _SaeLensSession);
    assemble_fn flattens that hook's chunks into a single dataset. Both are
    injectable so the CPU sandbox can exercise the whole driver without
    SAELens, a GPU, or a model.
    """
    from training.runtime.lock import run_lock
    from training.runtime.watchdog import Watchdog

    validate_harvest(recipe)
    cfg = recipe.harvest
    model_id, revision, model_hash = resolve_subject(cfg)
    corpus, corpus_hash = corpus_identity(cfg.corpus)
    d_model, spec, depth = resolve_model_meta(
        model_id, revision, d_model_override=cfg.d_model
    )
    check_hook_layer_bounds(cfg.hook_layers, depth, model_id)

    name = cfg.name or recipe.run.name
    capture_dir = os.path.join(cfg.out_root, name)
    os.makedirs(capture_dir, exist_ok=True)
    try:
        existing = load_manifest(capture_dir)
        if existing.get("artifact") == ARTIFACT:
            raise HarvestError(
                f"capture already complete: {capture_dir} (delete it or "
                f"choose another harvest.name to re-harvest)"
            )
    except ManifestError:
        pass

    progress = _load_progress(capture_dir)
    # Resume only under the identical capture config: hooks completed under
    # a different corpus/model/token budget must not be stitched into this
    # run's manifest, which records one uniform config for every shard.
    identity = {
        "model_id": model_id,
        "model_revision": revision,
        # corpus_xxh3 is None for HF dataset ids (only local files hash), so
        # the id itself must be part of the identity or swapping HF corpora
        # mid-resume stitches cross-corpus shards into one manifest
        "corpus": corpus,
        "corpus_xxh3": corpus_hash,
        "context_len": int(cfg.context_len),
        "total_tokens": int(cfg.total_tokens),
        # Chunk size fixes where the boundaries fall, so resuming across a
        # change would leave a gap or an overlap at the seam.
        "chunk_tokens": int(cfg.chunk_tokens),
        "dtype": cfg.dtype,
        "seed": int(recipe.run.seed),
        "d_model": d_model,
    }
    prior = progress.get("config")
    # Chunks count as prior work too: a hook part-way through has no entry in
    # completed_hooks, and resuming it under a changed config is exactly the
    # stitching this refusal exists to prevent.
    started_before = bool(progress["completed_hooks"] or progress["completed_chunks"])
    if started_before and prior is not None and prior != identity:
        changed = sorted(k for k in identity if prior.get(k) != identity[k])
        raise HarvestError(
            f"progress.json in {capture_dir} was written under a different "
            f"harvest config (changed: {', '.join(changed)}); delete the "
            f"directory or choose another harvest.name"
        )
    progress["config"] = identity
    stats: dict = progress.get("stats", {})
    plan = chunk_plan(
        int(cfg.total_tokens), int(cfg.chunk_tokens), int(cfg.context_len)
    )
    pending = [
        layer
        for layer in cfg.hook_layers
        if hook_name(layer, spec) not in progress["completed_hooks"]
    ]
    # Prorate by pending chunks, not pending hooks: a run resuming at chunk 18
    # of 20 needs two chunks' worth of space, and pricing it at a whole hook
    # would refuse a capture that fits. Assembly moves rather than copies, so
    # there is no second copy to budget for.
    pending_chunks = sum(
        len(plan) - progress["completed_chunks"].get(hook_name(layer, spec), 0)
        for layer in pending
    )
    disk_preflight(
        cfg.out_root,
        projected_bytes(cfg, d_model)
        * pending_chunks
        // max(len(plan) * len(cfg.hook_layers), 1),
        recipe.watchdog.disk_free_floor_gb,
        disk_usage_fn=disk_usage_fn,
    )

    make_session = runner_fn or _saelens_runner
    assemble = assemble_fn or _consolidate_chunks
    watchdog = None
    captured_tokens = 0
    measured: dict[str, int] = dict(progress.get("measured_tokens", {}))
    started = time.monotonic()
    with run_lock(f"harvest-{name}"):
        if recipe.watchdog.enabled:
            watchdog = Watchdog(recipe.watchdog, capture_dir, disk_path=cfg.out_root)
            watchdog.start()
        try:
            for layer in pending:
                hook = hook_name(layer, spec)
                hook_dir = os.path.join(capture_dir, hook)
                chunks_dir = hook_dir + CHUNKS_SUFFIX
                done = int(progress["completed_chunks"].get(hook, 0))
                if os.path.isdir(hook_dir):
                    # Assembly moves files out of chunks_dir, so an assembly
                    # interrupted part-way leaves neither side whole and
                    # cannot be redone. It is a seconds-long metadata step, so
                    # this window is tiny -- but silently continuing would
                    # publish a truncated capture as complete.
                    raise HarvestError(
                        f"{hook_dir} already exists without {hook} being "
                        f"recorded complete: a previous assembly was "
                        f"interrupted and cannot be resumed. Delete "
                        f"{hook_dir} and {chunks_dir}, then rerun"
                    )
                # Only when there is something left to capture. A hook whose
                # last chunk landed before a crash needs assembling, not a
                # model load -- and building a session to do nothing would
                # cost minutes of load and a pass over the whole corpus in
                # fast_forward.
                if done < len(plan):
                    session = make_session(
                        {
                            "model_id": model_id,
                            "revision": revision,
                            "hook": hook,
                            "d_in": d_model,
                            "corpus": corpus,
                            "context_len": int(cfg.context_len),
                            "chunk_tokens": plan[0],
                            "dtype": cfg.dtype,
                            "seed": int(recipe.run.seed),
                            "model_batch_size": int(cfg.model_batch_size),
                        }
                    )
                    # Recorded before the first chunk rather than after the
                    # last, so a crash mid-hook still leaves the manifest able
                    # to name the tokenizer and the SAELens version.
                    stats = session.info()
                    progress["stats"] = stats
                    if done:
                        session.fast_forward(sum(plan[:done]))
                    for index in range(done, len(plan)):
                        chunk_dir = os.path.join(chunks_dir, f"shard_{index:05d}")
                        # A chunk is recorded complete only after capture
                        # returns, below, so anything already at this path is
                        # debris from the attempt that crashed -- including the
                        # .tmp_shards that make KB-58 fatal. Clearing it is what
                        # keeps SAELens's non-empty-directory refusal from
                        # firing on the one chunk a crash interrupted, which is
                        # otherwise every resume.
                        shutil.rmtree(chunk_dir, ignore_errors=True)
                        written = session.capture(chunk_dir, plan[index])
                        captured_tokens += written
                        # Per hook and cumulative across runs, so a capture
                        # that resumed still reports what the whole capture
                        # holds.
                        measured[hook] = measured.get(hook, 0) + written
                        progress["completed_chunks"][hook] = index + 1
                        progress["measured_tokens"] = measured
                        _save_progress(capture_dir, progress)
                # Unconditional, including at one chunk: a capture that skipped
                # assembly for a single chunk would give the smoke recipe a
                # different code path from the reference captures, and this
                # step would first meet reality 17 hours into a run.
                #
                # Creating hook_dir is assembly's job, not this loop's. That
                # is what lets the check above distinguish "assembly started
                # and did not finish" from "assembly never began" -- if this
                # loop created the directory first, the two would be
                # indistinguishable and every crash would demand a manual
                # delete.
                assemble(chunks_dir, hook_dir)
                progress["completed_hooks"].append(hook)
                progress["completed_chunks"].pop(hook, None)
                _save_progress(capture_dir, progress)
        finally:
            if watchdog is not None:
                watchdog.stop()
    wall_clock_s = time.monotonic() - started

    # Throughput covers only the chunks this run captured: the wall clock
    # spans pending work, so counting pre-completed chunks' tokens would
    # inflate the recorded number on resume.
    tokens = captured_tokens
    manifest = {
        "schema_version": 1,
        "artifact": ARTIFACT,
        "model_id": model_id,
        "model_revision": revision,
        "model_hash": model_hash,
        "model_type": spec.model_type,
        "hooks": [hook_name(layer, spec) for layer in cfg.hook_layers],
        "hook_layers": sorted(int(layer) for layer in cfg.hook_layers),
        "d_model": d_model,
        "corpus": corpus,
        "corpus_xxh3": corpus_hash,
        "context_len": int(cfg.context_len),
        # Measured, not requested. SAELens starts a new epoch rather than
        # stopping when the corpus runs dry, and rows are lost to the
        # sequence boundary, so what landed on disk is not what was asked
        # for. The smallest hook's count is the honest bound for a consumer
        # reading one number for the whole capture.
        "total_tokens": min(measured.values(), default=0),
        "requested_tokens": int(cfg.total_tokens),
        "chunks": len(plan),
        "dtype": cfg.dtype,
        "seed": int(recipe.run.seed),
        # No "unknown" fallback: an identity field that silently degrades to
        # a sentinel is the KB-62 shape, and tokenizer_fingerprint is the
        # field every downstream require_match gates on.
        "tokenizer_fingerprint": require_stat(
            stats, "tokenizer_fingerprint", capture_dir
        ),
        "sae_lens_version": require_stat(stats, "sae_lens_version", capture_dir),
        "tokens_per_second": (
            round(tokens / wall_clock_s, 1) if wall_clock_s and tokens else 0.0
        ),
        "wall_clock_s": round(wall_clock_s, 1),
        "validated": False,
        "git_sha": git_sha(),
        "created": utc_now(),
    }
    write_manifest(capture_dir, manifest)
    os.remove(os.path.join(capture_dir, PROGRESS_NAME))
    return manifest


def check_capture(capture_dir: str, expects: dict) -> dict:
    """Downstream refusal gate: load a capture manifest and require matches.

    expects maps manifest keys to expected values (model_id, hooks,
    corpus_xxh3, ...). Raises ManifestError when the manifest is missing or
    any field mismatches; returns the manifest otherwise. P4-D3 calls this
    before training from a cache.
    """
    manifest = load_manifest(capture_dir)
    require_match(manifest, "artifact", ARTIFACT, capture_dir)
    for key, expected in expects.items():
        require_match(manifest, key, expected, capture_dir)
    return manifest


def mark_validated(capture_dir: str) -> dict:
    """Flag a capture deletable (called with --validated by downstream runs)."""
    manifest = load_manifest(capture_dir)
    require_match(manifest, "artifact", ARTIFACT, capture_dir)
    manifest["validated"] = True
    write_manifest(capture_dir, manifest)
    return manifest


def gc_captures(root: str, dry_run: bool = False) -> list[str]:
    """Delete validated captures oldest-first, keeping each manifest.json.

    Returns the capture directories deleted (or that would be, in dry_run).
    Directories without a harvest manifest are never touched; an in-progress
    capture has no manifest at all.
    """
    candidates = []
    for entry in sorted(os.listdir(root)):
        capture_dir = os.path.join(root, entry)
        if not os.path.isdir(capture_dir):
            continue
        try:
            manifest = load_manifest(capture_dir)
        except ManifestError:
            continue
        if manifest.get("artifact") == ARTIFACT and manifest.get("validated"):
            candidates.append((manifest.get("created", ""), capture_dir))

    deleted = []
    for _, capture_dir in sorted(candidates):
        if not dry_run:
            for item in os.listdir(capture_dir):
                if item == "manifest.json":
                    continue
                path = os.path.join(capture_dir, item)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        deleted.append(capture_dir)
    return deleted
