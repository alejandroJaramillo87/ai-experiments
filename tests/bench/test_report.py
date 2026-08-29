"""Reading a result back, including the ways reading fails.

`bench/report.py::read` validates what it loads. Its three failure modes -- a path that
does not exist, a truncated file, and a document that fails the envelope -- are raised
as they come, and `bench/commands/compare.py` translates each into a diagnosed error
rather than a traceback (tests/bench/test_compare.py). The envelope says nothing about
which report it is looking at, so the type the caller needs is stated as `expect` and
checked here: a `bench_result` handed to `--noise` is a valid document of the wrong
kind, and validation alone cannot see that.
"""

from __future__ import annotations

import json

import pytest

from bench import report
from common.reports import validate_report
from tests.bench import support


class TestRoundTrip:
    def test_a_written_result_reads_back_and_validates(self, tmp_path):
        document = support.result()
        path = report.write(document, str(tmp_path))
        loaded = report.read(path)
        assert loaded == document
        validate_report(loaded)

    def test_the_envelope_carries_what_the_shared_schema_requires(self):
        document = support.result()
        assert document["report"] == report.REPORT_TYPE
        assert document["schema_version"] == report.SCHEMA_VERSION
        assert "timestamp" in document

    def test_a_noise_report_is_a_distinct_type(self, tmp_path):
        document = report.build(
            run={"run_id": "n"},
            conditions={"service": "llama-cpu"},
            phases={},
            errors=[],
            samples_path=None,
            report_type=report.NOISE_REPORT_TYPE,
        )
        path = report.write(document, str(tmp_path), filename="noise.json")
        assert report.read(path)["report"] == report.NOISE_REPORT_TYPE
        assert path.endswith("noise.json")


class TestExpectedType:
    def test_a_document_of_the_expected_type_reads_normally(self, tmp_path):
        path = report.write(support.result(), str(tmp_path))
        assert report.read(path, expect=report.REPORT_TYPE)["report"] == (
            report.REPORT_TYPE
        )

    def test_a_result_is_refused_where_a_noise_floor_is_expected(self, tmp_path):
        path = report.write(support.result(), str(tmp_path))
        with pytest.raises(report.ReportTypeError, match=report.NOISE_REPORT_TYPE):
            report.read(path, expect=report.NOISE_REPORT_TYPE)

    def test_no_expectation_accepts_either_type(self, tmp_path):
        path = report.write(support.result(), str(tmp_path))
        assert report.read(path)["report"] == report.REPORT_TYPE


class TestReadFailures:
    def test_a_missing_file_raises_the_bare_os_error(self, tmp_path):
        # The reader raises what the filesystem raises; naming the file the operator
        # should have passed is the caller's job, and compare.cmd does it.
        with pytest.raises(FileNotFoundError):
            report.read(str(tmp_path / "absent.json"))

    def test_a_truncated_file_raises_the_bare_decode_error(self, tmp_path):
        path = tmp_path / "half.json"
        path.write_text(json.dumps(support.result())[:200])
        with pytest.raises(json.JSONDecodeError):
            report.read(str(path))

    def test_a_document_of_the_wrong_shape_is_refused_by_the_validator(self, tmp_path):
        path = tmp_path / "wrong.json"
        path.write_text(json.dumps({"not": "a report"}))
        with pytest.raises(Exception) as caught:
            report.read(str(path))
        assert not isinstance(caught.value, json.JSONDecodeError)


def test_write_creates_the_directory_it_is_given(tmp_path):
    nested = tmp_path / "runs" / "run-1"
    path = report.write(support.result(), str(nested))
    assert report.read(path)["run"]["arm"] == "base"
