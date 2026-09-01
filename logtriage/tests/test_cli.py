"""CLI contract: exit codes, stream discipline, and actionable failures."""

from __future__ import annotations

import json

import pytest

from logtriage.cli import main


@pytest.fixture
def corpus(tmp_path):
    path = tmp_path / "mongod.log"
    assert main(["gen-logs", "--scenario", "slow-query-storm", "--out", str(path)]) == 0
    return path


def test_no_subcommand_is_invalid_usage() -> None:
    """Exit 2 for bad arguments, distinct from exit 1 for a runtime failure."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_gen_logs_writes_a_corpus(tmp_path, capsys) -> None:
    path = tmp_path / "out.log"
    assert main(["gen-logs", "--out", str(path)]) == 0
    assert path.read_text().count("\n") > 100
    assert "[OK]" in capsys.readouterr().err


def test_gen_logs_is_reproducible_from_the_command_line(tmp_path) -> None:
    """A corpus that reproduces a bug travels as a command, not as a file."""
    first, second = tmp_path / "a.log", tmp_path / "b.log"
    for path in (first, second):
        main(
            [
                "gen-logs",
                "--scenario",
                "replica-stepdown",
                "--seed",
                "9",
                "--out",
                str(path),
            ]
        )
    assert first.read_bytes() == second.read_bytes()


def test_gen_logs_writes_the_corpus_to_stdout_when_unredirected(capsys) -> None:
    """Used as a generator in a pipeline; status still goes to stderr."""
    assert main(["gen-logs", "--minutes", "2"]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith('{"t":')
    assert "[OK]" in captured.err


def test_unknown_scenario_fails_with_a_solution(capsys) -> None:
    assert main(["gen-logs", "--scenario", "slow-query-strom"]) == 1
    err = capsys.readouterr().err
    assert "[ERROR]" in err
    assert "Unknown scenario" in err
    assert "gen-logs --list" in err


def test_zero_minutes_is_rejected_rather_than_producing_an_empty_file(capsys) -> None:
    assert main(["gen-logs", "--minutes", "0"]) == 1
    assert "at least 1" in capsys.readouterr().err


def test_gen_logs_list_json_is_pure_json(capsys) -> None:
    assert main(["gen-logs", "--list", "--json"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["slow-query-storm"]["expected_findings"]
    assert captured.err == "", "quiet mode must suppress every human line"


def test_profile_renders_and_reports_its_budget(corpus, capsys) -> None:
    assert main(["profile", str(corpus)]) == 0
    err = capsys.readouterr().err
    assert "=== corpus profile ===" in err
    assert "shop.orders" in err
    assert "tokens of a 4000 budget" in err


def test_profile_json_is_pure_json_on_stdout(corpus, capsys) -> None:
    """stdout must be machine-parseable with no prose interleaved."""
    assert main(["profile", str(corpus), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["records_parsed"] > 0
    assert payload["namespaces"][0]["ns"] == "shop.orders"
    assert captured.err == ""


def test_profile_of_a_missing_file_names_the_fix(capsys) -> None:
    assert main(["profile", "/nonexistent/mongod.log"]) == 1
    err = capsys.readouterr().err
    assert "Cannot read log" in err
    assert "Solution:" in err


def test_profile_of_a_non_mongod_file_says_so(tmp_path, capsys) -> None:
    """Pointing this at the wrong file is the likeliest first-run mistake."""
    path = tmp_path / "nginx.log"
    path.write_text('127.0.0.1 - - [30/Aug/2026] "GET / HTTP/1.1" 200\n')
    assert main(["profile", str(path)]) == 1
    err = capsys.readouterr().err
    assert "No mongod log records" in err
    assert "structured (JSON) log" in err


def test_profile_warns_about_unparseable_lines(tmp_path, capsys) -> None:
    """Evidence that was never read has to be visible to the reader."""
    good = tmp_path / "good.log"
    main(["gen-logs", "--minutes", "5", "--out", str(good)])
    mixed = tmp_path / "mixed.log"
    mixed.write_text(good.read_text() + "{truncated\n")
    assert main(["profile", str(mixed)]) == 0
    err = capsys.readouterr().err
    assert "[WARN] Unparseable lines: 1" in err
    assert "1 unparseable" in err


def test_profile_budget_is_honoured(corpus, capsys) -> None:
    assert main(["profile", str(corpus), "--budget-tokens", "150"]) == 0
    err = capsys.readouterr().err
    assert "of a 150 budget" in err
    assert "omitted" in err, "a budget that forces truncation must say so"


@pytest.fixture
def tiny_corpus(tmp_path, monkeypatch):
    """A two-page manual, so index-kb can be driven end to end offline.

    The real corpus is a 91MB clone that a bare checkout does not have. This is
    the same shape -- a project directory holding snooty.toml and source/ -- at
    a size that costs nothing, which is enough to exercise the command's own
    logic as opposed to the chunker's.
    """
    root = tmp_path / "kb"
    (root / "source").mkdir(parents=True)
    (root / "snooty.toml").write_text(
        'name = "manual"\n\n[constants]\nversion = "8.3"\n', encoding="utf-8"
    )
    (root / "source" / "index.txt").write_text(
        "======\nManual\n======\n\nmongod writes structured JSON log messages.\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LOGTRIAGE_KB_SOURCE", str(root))
    return root


def test_index_kb_builds_an_index_offline(tiny_corpus, tmp_path, capsys) -> None:
    out = tmp_path / "index"
    assert main(["index-kb", "--out", str(out)]) == 0
    assert (out / "chunks.jsonl").is_file()
    err = capsys.readouterr().err
    assert "hashed, not semantic" in err, "the embedder's limitation must be stated"


def test_index_kb_refuses_to_overwrite_without_refresh(
    tiny_corpus, tmp_path, capsys
) -> None:
    out = tmp_path / "index"
    assert main(["index-kb", "--out", str(out)]) == 0
    assert main(["index-kb", "--out", str(out)]) == 1
    assert "--refresh" in capsys.readouterr().err


def test_a_failing_backend_leaves_the_existing_index_intact(
    tiny_corpus, tmp_path, monkeypatch, capsys
) -> None:
    """--refresh destroys a working index, so everything that could fail has to
    fail before the delete. Ordered the other way -- as this was -- an
    unreachable embedding server leaves the machine with no index at all,
    having had one a second earlier."""
    out = tmp_path / "index"
    assert main(["index-kb", "--out", str(out)]) == 0
    before = (out / "chunks.jsonl").read_bytes()

    monkeypatch.delenv("LOGTRIAGE_EMBED_URL", raising=False)
    assert (
        main(["index-kb", "--out", str(out), "--embedder", "llama", "--refresh"]) == 1
    )
    assert "LOGTRIAGE_EMBED_URL" in capsys.readouterr().err
    assert (out / "chunks.jsonl").read_bytes() == before


def test_triage_rebuilds_the_embedder_the_index_was_built_with(
    tiny_corpus, tmp_path, corpus, capsys
) -> None:
    """The embedder comes from the index, not from a flag. Searched with a
    different one, an index still returns its k best chunks, ranked by a
    similarity that means nothing -- and nothing anywhere would error."""
    out = tmp_path / "index"
    assert main(["index-kb", "--out", str(out)]) == 0
    capsys.readouterr()

    assert main(["triage", str(corpus), "--kb", str(out)]) == 0
    assert "hashing embeddings" in capsys.readouterr().err
