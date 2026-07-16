"""Full-scale pipeline stage composition (P2-D1).

The end-to-end tests plant one exact duplicate, one near duplicate, one
low-quality document, and one benchmark-contaminated document in a natural-text
corpus, then assert each is dropped by its own stage (attributed in the stage
funnel) and that re-running the pipeline is byte-deterministic.

datatrove is required for the end-to-end tests (installed in the training
image); config and funnel logic tests run without it.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
import yaml

from training.data import pipeline as pipeline_mod
from training.data.contamination import build_index, index_hash
from training.data.pipeline import PipelineConfig, load_pipeline_config

BENCH_QUESTION = (
    "Which layer of the atmosphere contains the ozone that absorbs "
    "most of the ultraviolet radiation arriving from the sun every day"
)

BASE_DOC = (
    "The maintenance crew arrived early on Tuesday morning to inspect the aging bridge over the river. "
    "Engineers measured corrosion along the steel supports and recorded their findings in a shared logbook. "
    "Local officials promised that repair funding would be approved before the end of the fiscal year. "
    "Residents who commute across the span every day expressed cautious optimism about the announced schedule. "
    "A final report is expected next month, and public comment sessions will follow shortly afterwards."
)

KEEP_DOCS = [
    (
        "Farmers in the valley planted drought resistant wheat after two dry seasons reduced their harvest. "
        "Agronomists from the university visited weekly to sample soil moisture and advise on irrigation timing. "
        "Grain prices at the regional market recovered slowly, which encouraged several cooperatives to expand storage. "
        "By late autumn the fields showed healthy growth, and the cooperative projected a modest surplus for export."
    ),
    (
        "The museum opened a new exhibit documenting the history of radio broadcasting in the region. "
        "Curators restored several transmitters from the nineteen thirties and displayed original program schedules. "
        "Visitors can listen to archived recordings through headsets installed beside each display case. "
        "Attendance during the first week exceeded expectations, prompting the museum to extend evening hours."
    ),
    (
        "City planners published a draft proposal for expanding the bicycle network along the harbor district. "
        "The plan adds protected lanes on four arterial streets and links two existing greenway segments. "
        "Business owners raised concerns about loading zones, so the committee scheduled additional consultations. "
        "Construction could begin next spring if the council approves the budget amendment in December."
    ),
]

NEAR_DUP = BASE_DOC.replace("Tuesday", "Wednesday")
LOW_QUALITY = "Click here to buy now with free shipping today. " * 40
CONTAMINATED = (
    "Students reviewed the study guide carefully before the science exam last week. "
    + BENCH_QUESTION
    + ". Afterwards they compared their answers in the hallway and "
    "discussed which topics deserved another round of careful review before finals."
)


def test_default_config_matches_design_parameters():
    config = PipelineConfig()
    assert config.minhash.n_grams == 5
    assert config.minhash.num_buckets * config.minhash.hashes_per_bucket == 112
    assert config.workers == 30
    assert config.language.threshold == 0.65


def test_load_pipeline_config_defaults_and_overrides(tmp_path):
    assert load_pipeline_config(None) == PipelineConfig()

    path = tmp_path / "pipeline.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"workers": 4, "minhash": {"seed": 7}, "exact_dedup": False}, f)
    config = load_pipeline_config(str(path))
    assert config.workers == 4
    assert config.minhash.seed == 7
    assert config.exact_dedup is False
    assert config.minhash.n_grams == 5  # untouched keys keep defaults


def test_load_pipeline_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    with open(path, "w") as f:
        yaml.safe_dump({"minhash": {"jaccard": 0.8}}, f)
    with pytest.raises(ValueError, match="minhash.jaccard"):
        load_pipeline_config(str(path))


def test_repo_default_yaml_matches_dataclass_defaults():
    from support import REPO_ROOT

    path = os.path.join(REPO_ROOT, "training", "data", "configs", "default.yaml")
    assert load_pipeline_config(path) == PipelineConfig()


@pytest.fixture()
def corpus_jsonl(tmp_path) -> str:
    docs = [BASE_DOC, BASE_DOC, NEAR_DUP, LOW_QUALITY, CONTAMINATED, *KEEP_DOCS]
    path = tmp_path / "corpus.jsonl"
    with open(path, "w") as f:
        for i, text in enumerate(docs):
            f.write(json.dumps({"id": f"doc{i}", "text": text}) + "\n")
    return str(path)


@pytest.fixture()
def contamination_index(tmp_path) -> str:
    from support import write_benchmarks_yaml

    bench_yaml = write_benchmarks_yaml(tmp_path, [{"question": BENCH_QUESTION}])
    index_dir = str(tmp_path / "contamination")
    build_index(bench_yaml, index_dir)
    return index_dir


def _test_config() -> PipelineConfig:
    """Sandbox pipeline config: language id off (the fastText model is a
    network download), everything else at defaults."""
    config = PipelineConfig()
    config.language.enabled = False
    config.workers = 2
    return config


def _run(corpus_jsonl, tokenizer_dir, contamination_index, root, name) -> dict:
    return pipeline_mod.prepare_corpus(
        corpus_jsonl,
        os.path.join(root, name, "shards"),
        tokenizer_dir,
        os.path.join(root, name, "work"),
        config=_test_config(),
        contamination_dir=contamination_index,
    )


def test_text_field_honored_for_local_sources(tmp_path, monkeypatch):
    # A custom --text-field must be read for local JSONL sources, not only
    # hub sources: the quality stages and shard step consume the "text" key,
    # so ingest normalizes. Engine-free: the stages are stubbed out.
    src = tmp_path / "local.jsonl"
    with open(src, "w") as f:
        f.write(json.dumps({"content": "first document"}) + "\n")
        f.write(json.dumps({"content": "second document"}) + "\n")

    seen = {}
    monkeypatch.setattr(
        pipeline_mod, "run_quality_stages", lambda ingest, wd, cfg, cd: (ingest, {})
    )

    def fake_prepare(final_dir, *args, **kwargs):
        with open(os.path.join(final_dir, "00000.jsonl")) as f:
            seen["records"] = [json.loads(line) for line in f]
        return {}

    monkeypatch.setattr(pipeline_mod.prepare_mod, "prepare", fake_prepare)
    monkeypatch.setattr(pipeline_mod.shards_mod, "verify_shards", lambda *a, **k: None)
    pipeline_mod.prepare_corpus(
        str(src),
        str(tmp_path / "shards"),
        "tok",
        str(tmp_path / "work"),
        config=_test_config(),
        text_field="content",
    )
    assert [r["text"] for r in seen["records"]] == ["first document", "second document"]


@pytest.mark.skipif(
    importlib.util.find_spec("datatrove") is None,
    reason="datatrove not installed (training image dependency)",
)
class TestEndToEnd:
    def test_planted_documents_dropped_and_attributed(
        self, corpus_jsonl, tokenizer_dir, contamination_index, tmp_path
    ):
        manifest = _run(
            corpus_jsonl, tokenizer_dir, contamination_index, str(tmp_path), "run"
        )

        funnel = {
            stage["stage"]: stage for stage in manifest["pipeline"]["stage_funnel"]
        }
        assert list(funnel) == [
            "filter",
            "exact_dedup",
            "minhash_dedup",
            "decontamination",
        ]

        # 8 docs in: quality drops the ad spam, exact dedup one BASE_DOC copy,
        # minhash the near duplicate, decontamination the planted passage
        assert funnel["filter"]["docs_in"] == 8
        assert funnel["filter"]["docs_out"] == 7
        assert sum(funnel["filter"]["drops"]["gopher_repetition"].values()) == 1
        assert funnel["exact_dedup"]["docs_out"] == 6
        assert funnel["exact_dedup"]["drops"]["exact_dedup"] == {"dropped": 1}
        assert funnel["minhash_dedup"]["docs_out"] == 5
        assert sum(funnel["minhash_dedup"]["drops"]["minhash_dedup"].values()) == 1
        assert funnel["decontamination"]["docs_out"] == 4
        assert funnel["decontamination"]["drops"]["decontamination"] == {
            "benchmark_overlap": 1
        }

        assert manifest["decontaminated_against"] == index_hash(contamination_index)
        assert manifest["pipeline"]["minhash"]["num_buckets"] == 14
        assert manifest["total_documents"] == 4

        # every stage directory carries its stats.json
        work = os.path.join(str(tmp_path), "run", "work")
        for stage_dir in (
            pipeline_mod.STAGE_FILTERED,
            pipeline_mod.STAGE_EXACT,
            pipeline_mod.STAGE_MINHASH,
            pipeline_mod.STAGE_CLEAN,
        ):
            assert os.path.exists(
                os.path.join(work, stage_dir, pipeline_mod.STATS_NAME)
            )

    def test_rerun_is_deterministic(
        self, corpus_jsonl, tokenizer_dir, contamination_index, tmp_path
    ):
        first = _run(
            corpus_jsonl, tokenizer_dir, contamination_index, str(tmp_path), "a"
        )
        second = _run(
            corpus_jsonl, tokenizer_dir, contamination_index, str(tmp_path), "b"
        )
        assert first["shards"] == second["shards"]
        assert first["total_tokens"] == second["total_tokens"]

    def test_clean_intermediates_keeps_stats(
        self, corpus_jsonl, tokenizer_dir, contamination_index, tmp_path
    ):
        pipeline_mod.prepare_corpus(
            corpus_jsonl,
            str(tmp_path / "shards"),
            tokenizer_dir,
            str(tmp_path / "work"),
            config=_test_config(),
            contamination_dir=contamination_index,
            clean=True,
        )
        work = str(tmp_path / "work")
        assert not os.path.exists(os.path.join(work, "_minhash_sigs"))
        clean_stage = os.path.join(work, pipeline_mod.STAGE_CLEAN)
        assert os.path.exists(os.path.join(clean_stage, pipeline_mod.STATS_NAME))
        assert pipeline_mod._stage_files(clean_stage) == []

    def test_decontamination_skipped_without_index(
        self, corpus_jsonl, tokenizer_dir, tmp_path
    ):
        manifest = pipeline_mod.prepare_corpus(
            corpus_jsonl,
            str(tmp_path / "shards"),
            tokenizer_dir,
            str(tmp_path / "work"),
            config=_test_config(),
            contamination_dir=None,
        )
        assert manifest["decontaminated_against"] is None
        stages = [s["stage"] for s in manifest["pipeline"]["stage_funnel"]]
        assert "decontamination" not in stages


def test_iter_jsonl_reads_stage_directories(tmp_path):
    from training.data.prepare import iter_jsonl

    stage = tmp_path / "stage"
    stage.mkdir()
    for name, texts in [("00001.jsonl", ["b"]), ("00000.jsonl", ["a"])]:
        with open(stage / name, "w") as f:
            for t in texts:
                f.write(json.dumps({"text": t}) + "\n")
    import gzip

    with gzip.open(stage / "00002.jsonl.gz", "wt") as f:
        f.write(json.dumps({"text": "c"}) + "\n")
    assert list(iter_jsonl(str(stage), "text")) == ["a", "b", "c"]
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No .jsonl"):
        list(iter_jsonl(str(tmp_path / "empty"), "text"))
