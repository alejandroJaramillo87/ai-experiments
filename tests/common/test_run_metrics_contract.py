"""The harness metrics file against the Python readers that consume it.

Contract: schemas/run_metrics.json. The Rust half is
rust-harness/harness/tests/obs_contract.rs.

The premise of U6 is that the training dashboard renders a harness run with
no Python change: `training report --run <dir>` reads `metrics.jsonl`, whose
first record must satisfy `common.reports.validate_report` and whose step
records must survive `common.obs.dashboard.read_metrics`. That claim spans
two languages, so neither writer's own tests can settle it -- the Rust side
asserts its output against the contract file, and this side feeds bytes a
real Rust run wrote to the real readers.

The fixture is genuinely Rust-written; see fixtures/README.md for how, and
why a Python-authored imitation would prove nothing.
"""

from __future__ import annotations

import json
import os

import pytest

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "run_metrics_sample.jsonl"
)


@pytest.fixture
def metrics_path(tmp_path):
    # Copied rather than read in place: read_metrics takes a path, and a test
    # that pointed it at the committed fixture could not notice a reader that
    # wrote back to it.
    dst = tmp_path / "metrics.jsonl"
    with open(FIXTURE, encoding="utf-8") as src:
        dst.write_text(src.read(), encoding="utf-8")
    return dst


class TestTheRustWrittenFileReadsBack:
    def test_read_metrics_takes_the_header_and_skips_nothing(self, metrics_path):
        from common.obs.dashboard import read_metrics

        header, records, skipped = read_metrics(metrics_path)
        # Zero skipped is the assertion that matters: read_metrics drops a
        # line it cannot parse and counts it rather than raising, so a drifted
        # writer presents as a thin chart and not as an error.
        assert skipped == 0
        assert header, "the envelope record was not recognized as the header"
        assert header["report"] == "run_metrics"
        assert len(records) == 8

    def test_the_header_satisfies_the_shared_report_envelope(self, metrics_path):
        from common.obs.dashboard import read_metrics
        from common.reports import validate_report

        header, _, _ = read_metrics(metrics_path)
        # Raises ReportSchemaError listing every violation; the harness is the
        # first non-Python writer of this envelope, so this is the assertion
        # that the second implementation actually conforms.
        validate_report(header)

    def test_git_sha_is_present_and_not_the_unknown_sentinel(self, metrics_path):
        # validate_report accepts "unknown", the documented fallback shared by
        # runtime::envelope::git_sha and common/provenance.py. That makes the
        # envelope pass while carrying no provenance, so the fixture pins the
        # real case: a run inside a git tree records a real sha.
        from common.obs.dashboard import read_metrics

        header, _, _ = read_metrics(metrics_path)
        assert header["git_sha"] and header["git_sha"] != "unknown"

    def test_every_step_is_a_number_the_viewer_will_plot(self, metrics_path):
        # common/obs/viewer.js:109 returns early on any record whose step is
        # not typeof "number", silently. A stringified step is therefore in
        # the file and absent from every chart, which no parse error reports.
        from common.obs.dashboard import read_metrics

        _, records, _ = read_metrics(metrics_path)
        for record in records:
            assert isinstance(record["step"], int | float)
            assert not isinstance(record["step"], bool)
        assert [r["step"] for r in records] == list(range(8))

    def test_the_harness_series_the_dashboard_charts_are_present(self, metrics_path):
        # tokens_per_s joins the viewer's existing "Throughput (tokens/s)"
        # group; rss_gb is the one group U6 adds.
        from common.obs.dashboard import read_metrics

        _, records, _ = read_metrics(metrics_path)
        for key in ("tokens_per_s", "rss_gb"):
            assert all(isinstance(r[key], int | float) for r in records), key

    def test_the_event_file_join_resolves_from_the_header(self, metrics_path, tmp_path):
        """run_id and cli in the header are what locate the run's events.

        The dashboard's log pane comes from the event file, not this one, and
        it finds it by joining these two under OBS_LOG_DIR. A header missing
        either renders charts with no logs beside them.
        """
        from common.obs.dashboard import default_events_path, read_metrics

        header, _, _ = read_metrics(metrics_path)
        path = default_events_path(header, log_dir=str(tmp_path))
        assert path is not None
        assert path == tmp_path / "harness" / f"{header['run_id']}.jsonl"


class TestTheContractDescribesTheFixture:
    """The declaration in schemas/run_metrics.json, against real bytes.

    The Rust half asserts its writer against the same file. Checking the
    fixture here is what stops the contract from drifting into describing a
    shape nothing writes.
    """

    @pytest.fixture
    def contract(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "schemas", "run_metrics.json"
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_the_fixture_carries_every_required_key(self, contract, metrics_path):
        from common.obs.dashboard import read_metrics

        header, records, _ = read_metrics(metrics_path)
        assert set(contract["header"]["required"]) <= set(header)
        for record in records:
            assert set(contract["record"]["required"]) <= set(record)
            assert set(contract["record"]["harness_fields"]) <= set(record)

    def test_every_declared_writer_and_guard_exists(self, contract):
        # The same claim the obs_record contract test makes, and for the same
        # reason: the Rust side's build context is rust-harness/ alone, so it
        # cannot see whether a path elsewhere in the tree is real.
        root = os.path.join(os.path.dirname(__file__), "..", "..")
        for key in ("writers", "readers", "guarded_by"):
            assert contract[key], f"{key} is empty"
            for entry in contract[key]:
                path = entry.split(" ")[0]
                assert os.path.exists(os.path.join(root, path)), (
                    f"{key} names a path that does not exist: {path}"
                )
