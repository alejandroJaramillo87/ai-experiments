"""Tests for scripts/eval/promote_scorecard.py (Phase 1 F8).

The promote tool copies an eval run's scorecard pair verbatim into the
committed scorecards directory and stamps provenance; these tests cover the
copy, the provenance record, and the missing-scorecard failure path.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest
from support import REPO_ROOT

SCRIPT = os.path.join(REPO_ROOT, "scripts", "eval", "promote_scorecard.py")

spec = importlib.util.spec_from_file_location("promote_scorecard", SCRIPT)
promote_scorecard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(promote_scorecard)


@pytest.fixture()
def run_dir(tmp_path):
    d = tmp_path / "eval-run"
    d.mkdir()
    (d / "scorecard.json").write_text(json.dumps({"schema_version": 1}))
    (d / "scorecard.md").write_text("# Scorecard\n")
    return str(d)


class TestPromote:
    def test_copies_pair_and_writes_provenance(self, run_dir, tmp_path):
        dest = str(tmp_path / "scorecards")
        rc = promote_scorecard.main(
            ["--run-dir", run_dir, "--card", "sft-qlora-gemma-e4b", "--dest", dest]
        )
        assert rc == 0
        json_copy = os.path.join(dest, "sft-qlora-gemma-e4b.scorecard.json")
        md_copy = os.path.join(dest, "sft-qlora-gemma-e4b.scorecard.md")
        assert json.load(open(json_copy)) == {"schema_version": 1}
        assert open(md_copy).read() == "# Scorecard\n"
        prov = json.load(
            open(os.path.join(dest, "sft-qlora-gemma-e4b.provenance.json"))
        )
        assert prov["card"] == "sft-qlora-gemma-e4b"
        assert prov["source_run_dir"] == os.path.abspath(run_dir)
        assert prov["git_sha"]
        assert set(prov["sha256"]) == {
            "sft-qlora-gemma-e4b.scorecard.json",
            "sft-qlora-gemma-e4b.scorecard.md",
        }
        assert prov["sha256"]["sft-qlora-gemma-e4b.scorecard.json"] == (
            promote_scorecard.sha256_file(json_copy)
        )

    def test_repromote_replaces_pair(self, run_dir, tmp_path):
        dest = str(tmp_path / "scorecards")
        args = ["--run-dir", run_dir, "--card", "c", "--dest", dest]
        assert promote_scorecard.main(args) == 0
        with open(os.path.join(run_dir, "scorecard.md"), "w") as f:
            f.write("# Revalidated\n")
        assert promote_scorecard.main(args) == 0
        assert open(os.path.join(dest, "c.scorecard.md")).read() == "# Revalidated\n"

    def test_missing_scorecard_fails(self, tmp_path):
        empty = tmp_path / "empty-run"
        empty.mkdir()
        rc = promote_scorecard.main(
            [
                "--run-dir",
                str(empty),
                "--card",
                "c",
                "--dest",
                str(tmp_path / "scorecards"),
            ]
        )
        assert rc == 1
        assert not (tmp_path / "scorecards").exists()

    def test_missing_run_dir_fails(self, tmp_path):
        rc = promote_scorecard.main(
            [
                "--run-dir",
                str(tmp_path / "nope"),
                "--card",
                "c",
                "--dest",
                str(tmp_path / "scorecards"),
            ]
        )
        assert rc == 1
