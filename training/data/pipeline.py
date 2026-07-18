"""Full-scale corpus pipeline (P2-D1).

Composes datatrove blocks in front of the Phase 0 D2 tokenize-and-shard writer:
language identification, heuristic quality filtering (Gopher + FineWeb),
exact deduplication, MinHash-LSH near-deduplication, and benchmark
decontamination. The shard format, manifest contract, and reader are unchanged;
this module only decides which documents reach the writer.

Stage boundaries are JSONL directories under the caller's work directory, each
with a stats.json funnel (docs in, docs out, drop-reason histogram). They are
regenerable intermediates; clean_intermediates() removes everything but the
stats once shards validate. The final shard manifest embeds the whole funnel so
every published number about the corpus is reproducible from the manifest
alone.

datatrove imports are function-local: config parsing and the funnel logic are
importable (and unit-testable) without the engine installed.
"""

from __future__ import annotations

import dataclasses
import glob
import gzip
import json
import os
import shutil
from dataclasses import dataclass, field

from training.data import prepare as prepare_mod
from training.data import shards as shards_mod
from training.data.contamination import ContaminationIndex, index_hash, make_filter

STATS_NAME = "stats.json"

# Stage directory names, in pipeline order (ingest is materialized only for
# hub sources; local JSONL input is read in place).
STAGE_INGEST = "stage_00_ingest"
STAGE_FILTERED = "stage_01_filtered"
STAGE_EXACT = "stage_02_exact_dedup"
STAGE_MINHASH = "stage_03_minhash"
STAGE_CLEAN = "stage_04_decontaminated"


@dataclass
class LanguageConfig:
    enabled: bool = True
    languages: list[str] = field(default_factory=lambda: ["en"])
    threshold: float = 0.65


@dataclass
class QualityConfig:
    """Gopher/FineWeb heuristic filters. None leaves a knob at the datatrove
    default; the whole block can be disabled for corpora the heuristics would
    gut (code, math) — the per-recipe toggle from the design's risk note."""

    enabled: bool = True
    gopher_repetition: bool = True
    gopher_quality: bool = True
    fineweb_quality: bool = True
    min_doc_words: int | None = None
    max_doc_words: int | None = None
    min_stop_words: int | None = None
    max_non_alpha_words_ratio: float | None = None


@dataclass
class MinhashParams:
    """datatrove defaults: 5-gram shingles, 14 buckets x 8 = 112 hash
    functions, ~0.75 Jaccard threshold. Recorded in the shard manifest."""

    enabled: bool = True
    n_grams: int = 5
    num_buckets: int = 14
    hashes_per_bucket: int = 8
    seed: int = 1


@dataclass
class PipelineConfig:
    language: LanguageConfig = field(default_factory=LanguageConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    exact_dedup: bool = True
    minhash: MinhashParams = field(default_factory=MinhashParams)
    workers: int = 30  # leave two of the 32 threads for the OS and the writer


def load_pipeline_config(path: str | None) -> PipelineConfig:
    """PipelineConfig from a YAML file of nested overrides; None means defaults."""
    config = PipelineConfig()
    if path is None:
        return config
    import yaml

    with open(path) as f:
        overrides = yaml.safe_load(f) or {}
    for key, value in overrides.items():
        if not hasattr(config, key):
            raise ValueError(f"{path}: unknown pipeline config key {key!r}")
        current = getattr(config, key)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{path}: {key} must be a mapping")
            for sub_key, sub_value in value.items():
                if not hasattr(current, sub_key):
                    raise ValueError(f"{path}: unknown key {key}.{sub_key}")
                setattr(current, sub_key, sub_value)
        else:
            setattr(config, key, value)
    return config


def _normalized_content(doc) -> str:
    """Exact-dedup content getter: whitespace-collapsed lowercase text."""
    return " ".join(doc.text.lower().split())


def _count_docs(stage_dir: str) -> int:
    total = 0
    for path in _stage_files(stage_dir):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            total += sum(1 for line in f if line.strip())
    return total


def _stage_files(stage_dir: str) -> list[str]:
    return sorted(
        glob.glob(os.path.join(stage_dir, "*.jsonl"))
        + glob.glob(os.path.join(stage_dir, "*.jsonl.gz"))
    )


def _drop_histogram(excl_dir: str) -> dict[str, int]:
    """Drop-reason histogram from one filter's exclusion-writer directory."""
    histogram: dict[str, int] = {}
    for path in glob.glob(os.path.join(excl_dir, "*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                reason = (
                    json.loads(line).get("metadata", {}).get("filter_reason")
                    or "dropped"
                )
                histogram[reason] = histogram.get(reason, 0) + 1
    return histogram


def _write_stats(
    stage_dir: str, docs_in: int, docs_out: int, drops: dict[str, dict[str, int]]
) -> dict:
    stats = {"docs_in": docs_in, "docs_out": docs_out, "drops": drops}
    with open(os.path.join(stage_dir, STATS_NAME), "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
        f.write("\n")
    return stats


def materialize_hf_source(
    source: str,
    out_dir: str,
    *,
    split: str = "train",
    revision: str | None = None,
    text_field: str = "text",
    num_files: int = 8,
    max_documents: int | None = None,
) -> int:
    """Stream a hub dataset into round-robin JSONL files so every later stage
    reads local files with a deterministic document-to-task assignment."""
    import datasets

    os.makedirs(out_dir, exist_ok=True)
    files = [
        open(os.path.join(out_dir, f"{i:05d}.jsonl"), "w", encoding="utf-8")
        for i in range(num_files)
    ]
    count = 0
    try:
        ds = datasets.load_dataset(
            source, split=split, revision=revision, streaming=True
        )
        for record in ds:
            files[count % num_files].write(
                json.dumps({"id": str(count), "text": record[text_field]}) + "\n"
            )
            count += 1
            if max_documents is not None and count >= max_documents:
                break
    finally:
        for f in files:
            f.close()
    return count


def run_quality_stages(
    input_dir: str, work_dir: str, config: PipelineConfig, contamination_dir: str | None
) -> tuple[str, list[dict]]:
    """Run every enabled document-level stage; return (final stage dir, funnel).

    The funnel is a list of {stage, docs_in, docs_out, drops} records in
    pipeline order, built from stage output counts and exclusion-writer
    directories rather than engine internals.
    """
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.dedup import (
        ExactDedupConfig,
        ExactDedupFilter,
        ExactDedupSignature,
        ExactFindDedups,
        MinhashConfig,
        MinhashDedupBuckets,
        MinhashDedupCluster,
        MinhashDedupFilter,
        MinhashDedupSignature,
    )
    from datatrove.pipeline.readers import JsonlReader
    from datatrove.pipeline.writers import JsonlWriter

    def p(*parts: str) -> str:
        return os.path.join(work_dir, *parts)

    def run(pipeline: list, tasks: int, log_name: str) -> None:
        LocalPipelineExecutor(
            pipeline, tasks=tasks, start_method="fork", logging_dir=p("_logs", log_name)
        ).run()

    tasks = max(1, min(config.workers, len(_stage_files(input_dir))))
    funnel: list[dict] = []
    current = input_dir
    docs = _count_docs(input_dir)

    def advance(
        stage_dir: str, stage_name: str, drops: dict[str, dict[str, int]]
    ) -> None:
        nonlocal current, docs
        docs_out = _count_docs(stage_dir)
        funnel.append(
            {"stage": stage_name} | _write_stats(stage_dir, docs, docs_out, drops)
        )
        current, docs = stage_dir, docs_out

    # language id + heuristic quality filters, one pass
    filters = []
    drop_dirs: dict[str, str] = {}

    def excl(name: str):
        drop_dirs[name] = p("_excluded", name)
        return JsonlWriter(drop_dirs[name])

    if config.language.enabled:
        from datatrove.pipeline.filters import LanguageFilter

        filters.append(
            LanguageFilter(
                languages=config.language.languages,
                language_threshold=config.language.threshold,
                exclusion_writer=excl("language"),
            )
        )
    if config.quality.enabled:
        from datatrove.pipeline.filters import (
            FineWebQualityFilter,
            GopherQualityFilter,
            GopherRepetitionFilter,
        )

        q = config.quality
        if q.gopher_repetition:
            filters.append(
                GopherRepetitionFilter(exclusion_writer=excl("gopher_repetition"))
            )
        if q.gopher_quality:
            kwargs = {
                k: getattr(q, k)
                for k in (
                    "min_doc_words",
                    "max_doc_words",
                    "min_stop_words",
                    "max_non_alpha_words_ratio",
                )
                if getattr(q, k) is not None
            }
            filters.append(
                GopherQualityFilter(exclusion_writer=excl("gopher_quality"), **kwargs)
            )
        if q.fineweb_quality:
            filters.append(
                FineWebQualityFilter(exclusion_writer=excl("fineweb_quality"))
            )
    if filters:
        run(
            [
                JsonlReader(current),
                *filters,
                JsonlWriter(p(STAGE_FILTERED), compression=None),
            ],
            tasks,
            "filter",
        )
        advance(
            p(STAGE_FILTERED),
            "filter",
            {name: _drop_histogram(path) for name, path in drop_dirs.items()},
        )

    if config.exact_dedup:
        ed = ExactDedupConfig(content_getter=_normalized_content)
        run(
            [JsonlReader(current), ExactDedupSignature(p("_exact_sigs"), ed)],
            tasks,
            "exact_sig",
        )
        run([ExactFindDedups(p("_exact_sigs"), p("_exact_dups"), ed)], 1, "exact_find")
        run(
            [
                JsonlReader(current),
                ExactDedupFilter(
                    p("_exact_dups"),
                    ed,
                    exclusion_writer=JsonlWriter(p("_excluded", "exact_dedup")),
                ),
                JsonlWriter(p(STAGE_EXACT), compression=None),
            ],
            tasks,
            "exact_filter",
        )
        advance(
            p(STAGE_EXACT),
            "exact_dedup",
            {"exact_dedup": _drop_histogram(p("_excluded", "exact_dedup"))},
        )

    if config.minhash.enabled:
        mh = MinhashConfig(
            n_grams=config.minhash.n_grams,
            num_buckets=config.minhash.num_buckets,
            hashes_per_bucket=config.minhash.hashes_per_bucket,
            seed=config.minhash.seed,
        )
        run(
            [JsonlReader(current), MinhashDedupSignature(p("_minhash_sigs"), mh)],
            tasks,
            "minhash_sig",
        )
        run(
            [MinhashDedupBuckets(p("_minhash_sigs"), p("_minhash_buckets"), config=mh)],
            mh.num_buckets,
            "minhash_buckets",
        )
        run(
            [
                MinhashDedupCluster(
                    p("_minhash_buckets"), p("_minhash_clusters"), config=mh
                )
            ],
            1,
            "minhash_cluster",
        )
        run(
            [
                JsonlReader(current),
                MinhashDedupFilter(
                    p("_minhash_clusters"),
                    exclusion_writer=JsonlWriter(p("_excluded", "minhash")),
                ),
                JsonlWriter(p(STAGE_MINHASH), compression=None),
            ],
            tasks,
            "minhash_filter",
        )
        advance(
            p(STAGE_MINHASH),
            "minhash_dedup",
            {"minhash_dedup": _drop_histogram(p("_excluded", "minhash"))},
        )

    if contamination_dir is not None:
        ContaminationIndex(
            contamination_dir
        )  # fail before spawning workers if unreadable
        run(
            [
                JsonlReader(current),
                make_filter(
                    contamination_dir,
                    exclusion_writer=JsonlWriter(p("_excluded", "decontamination")),
                ),
                JsonlWriter(p(STAGE_CLEAN), compression=None),
            ],
            tasks,
            "decontaminate",
        )
        advance(
            p(STAGE_CLEAN),
            "decontamination",
            {"decontamination": _drop_histogram(p("_excluded", "decontamination"))},
        )

    return current, funnel


def clean_intermediates(work_dir: str, keep_dir: str | None = None) -> None:
    """Remove regenerable stage data, keeping every stats.json funnel entry."""
    for entry in sorted(os.listdir(work_dir)):
        path = os.path.join(work_dir, entry)
        if not os.path.isdir(path) or path == keep_dir:
            continue
        if entry.startswith("_"):
            shutil.rmtree(path)
            continue
        for data_file in _stage_files(path):
            os.remove(data_file)


def prepare_corpus(
    source: str,
    out_dir: str,
    tokenizer_source: str,
    work_dir: str,
    *,
    config: PipelineConfig | None = None,
    contamination_dir: str | None = None,
    text_field: str = "text",
    split: str = "train",
    revision: str | None = None,
    max_tokens: int | None = None,
    max_documents: int | None = None,
    shard_tokens: int = shards_mod.DEFAULT_SHARD_TOKENS,
    eos_token: str | None = None,
    license_str: str = "unknown",
    clean: bool = False,
    command: str | None = None,
    progress=None,
) -> dict:
    """Run the full pipeline and return the written shard manifest."""
    config = config or PipelineConfig()

    # Fail fast on inputs that otherwise fail only after the multi-hour
    # filter/dedup/decontamination stages: a pre-existing output manifest
    # (prepare()'s guard, hoisted verbatim), an unloadable tokenizer, and a
    # missing or corrupt contamination index
    manifest_path = os.path.join(out_dir, shards_mod.MANIFEST_NAME)
    if os.path.exists(manifest_path):
        raise FileExistsError(
            f"{out_dir} already contains {shards_mod.MANIFEST_NAME}; refusing to overwrite"
        )
    prepare_mod.load_tokenizer(tokenizer_source, eos_token)
    if contamination_dir is not None:
        ContaminationIndex(contamination_dir)

    os.makedirs(work_dir, exist_ok=True)

    if os.path.exists(source):
        if text_field != "text" or os.path.isfile(source):
            # Local files are re-serialized at ingest: this normalizes a
            # custom text field to the "text" key the quality stages and the
            # shard step read, and spreads documents round-robin across
            # config.workers files — datatrove parallelizes by file, so a
            # single ingest file would collapse every stage to one task.
            # (A verbatim copy also mishandled compression: a .jsonl.gz
            # source copied to a .jsonl name reads as binary downstream.)
            ingest_dir = os.path.join(work_dir, STAGE_INGEST)
            os.makedirs(ingest_dir, exist_ok=True)
            n_files = max(1, int(config.workers))
            writers = [
                open(os.path.join(ingest_dir, f"{i:05d}.jsonl"), "w")
                for i in range(n_files)
            ]
            try:
                for count, text in enumerate(
                    prepare_mod.iter_jsonl(source, text_field)
                ):
                    writers[count % n_files].write(
                        json.dumps({"id": str(count), "text": text}) + "\n"
                    )
            finally:
                for w in writers:
                    w.close()
        else:
            ingest_dir = source
    else:
        ingest_dir = os.path.join(work_dir, STAGE_INGEST)
        materialize_hf_source(
            source,
            ingest_dir,
            split=split,
            revision=revision,
            text_field=text_field,
            num_files=config.workers,
            max_documents=max_documents,
        )

    final_dir, funnel = run_quality_stages(
        ingest_dir, work_dir, config, contamination_dir
    )

    extra_manifest = {
        "pipeline": {
            "stage_funnel": funnel,
            "minhash": dataclasses.asdict(config.minhash)
            if config.minhash.enabled
            else None,
            "language": dataclasses.asdict(config.language)
            if config.language.enabled
            else None,
            "quality": dataclasses.asdict(config.quality)
            if config.quality.enabled
            else None,
            "exact_dedup": config.exact_dedup,
        },
        "decontaminated_against": (
            index_hash(contamination_dir) if contamination_dir is not None else None
        ),
    }
    manifest = prepare_mod.prepare(
        final_dir,
        out_dir,
        tokenizer_source,
        text_field="text",
        max_tokens=max_tokens,
        shard_tokens=shard_tokens,
        eos_token=eos_token,
        license_str=license_str,
        command=command,
        progress=progress,
        extra_manifest=extra_manifest,
        dataset_name=source,
        split=None if os.path.exists(source) else split,
        revision=revision,
    )
    shards_mod.verify_shards(out_dir, manifest)

    if clean:
        keep = ingest_dir if ingest_dir == source else None
        clean_intermediates(work_dir, keep_dir=keep)
    return manifest
