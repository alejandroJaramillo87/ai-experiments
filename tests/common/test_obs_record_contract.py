"""The Python obs writers against the cross-language record contract.

Run-level observability is written by two implementations — common/obs for the
Python CLIs, rust-harness/obs/src/lib.rs for the harness — and read by one,
the Python runs, retention, and viewer tooling. schemas/obs_record.json is the
key set both must emit; this module holds the Python side to it, and
rust-harness/harness/tests/obs_contract.rs holds the Rust side.

Asserting at the writers rather than the reader is deliberate. read_runs skips
a line it cannot parse rather than raising, so a writer that drifts presents as
an empty runs table and not as an error — indistinguishable from a machine that
never ran the tool.
"""

from __future__ import annotations

import json
import logging
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACT_PATH = os.path.join(REPO_ROOT, "schemas", "obs_record.json")


@pytest.fixture(scope="module")
def contract() -> dict:
    with open(CONTRACT_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestEventRecordContract:
    def test_the_formatter_emits_the_contract_base_keys(self, contract):
        from common.obs.core import _JsonFormatter

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        payload = json.loads(_JsonFormatter().format(record))
        assert set(payload) == set(contract["event"]["required"])

    def test_a_caller_field_cannot_displace_a_base_key(self, contract):
        """The base keys are what correlates a run; a caller must not own them.

        Without this the contract would be satisfiable in name only: an
        emitter passing run_id=... as a named field could overwrite the real
        one, leaving every declared key present and the join broken.
        """
        from common.obs.core import _JsonFormatter

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        record.obs_fields = {"run_id": "forged", "extra": 1}
        payload = json.loads(_JsonFormatter().format(record))
        assert payload["run_id"] != "forged"
        assert payload["extra"] == 1
        assert set(contract["event"]["required"]) <= set(payload)


class TestCatalogContract:
    def test_record_run_writes_exactly_the_contract_keys(
        self, contract, tmp_path, monkeypatch
    ):
        from common.obs.catalog import catalog_path, read_runs, record_run

        monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path))
        (tmp_path / "interp").mkdir()
        record_run(
            "interp",
            run_id="R-20260101-000000-abcdef",
            command="probe-train",
            exit_code=0,
            duration_s=1.5,
        )
        records = read_runs(catalog_path("interp"))
        assert len(records) == 1
        # Exactly, not at least: the catalog line is a fixed summary record,
        # and the Rust writer formats the same six keys by hand.
        assert set(records[0]) == set(contract["catalog"]["required"])

    def test_the_timestamp_survives_the_runs_table_slice(
        self, contract, tmp_path, monkeypatch
    ):
        # The runs table renders timestamp[:19]. A stamp too short to slice
        # that way shows up as a mangled column rather than an error, on
        # either writer's lines.
        from common.obs.catalog import catalog_path, read_runs, record_run

        monkeypatch.setenv("OBS_LOG_DIR", str(tmp_path))
        (tmp_path / "interp").mkdir()
        record_run(
            "interp",
            run_id="R-20260101-000000-abcdef",
            command="probe-train",
            exit_code=0,
            duration_s=1.5,
        )
        stamp = read_runs(catalog_path("interp"))[0]["timestamp"]
        assert len(stamp) >= 19
        assert stamp[10] == "T"


class TestDeclaredEvents:
    """The `events` section, asserted from the side that owns the file.

    Deliberately not "Python emits run.header" -- Python has no model to
    describe, and a test asserting an event this tier will never write would
    be theatre. What this side can check is that the declaration is honest:
    that every declared writer exists, that no declared field silently
    redefines a base key, and that a one-sided event says so rather than
    reading as a gap somebody should close.
    """

    def test_every_declared_event_names_a_writer_that_exists(self, contract):
        # The Rust half cannot make this claim: its build context is
        # rust-harness/ alone, so it cannot see whether a path elsewhere in
        # the tree is real. This side can, which is the whole reason the
        # contract is asserted at both writers rather than one.
        for name, spec in contract["events"].items():
            assert spec["writers"], f"{name} declares no writer"
            for writer in spec["writers"]:
                path = writer.split(" ")[0]
                assert os.path.exists(os.path.join(REPO_ROOT, path)), (
                    f"{name} names a writer that does not exist: {path}"
                )

    def test_no_declared_field_collides_with_a_base_key(self, contract):
        # A declared field named `run_id` would be dropped by both writers'
        # base-key guards, so the contract would promise a key no record can
        # ever carry.
        base = set(contract["event"]["required"])
        for name, spec in contract["events"].items():
            declared = set(spec["fields"]) | set(spec.get("optional", []))
            assert not (declared & base), (
                f"{name} redeclares base keys: {declared & base}"
            )

    def test_the_run_header_declares_its_one_sidedness(self, contract):
        # run.header is Rust-only by decision, not by omission. The Python
        # tiers carry equivalent provenance through run_context(), and
        # imitating that block from Rust would emit a packages/gpu section
        # that reads true and is not. Asserting no common/obs writer is
        # listed is what keeps that a declaration rather than a pending task.
        writers = contract["events"]["run.header"]["writers"]
        assert not any("common/obs" in w for w in writers), writers
        assert any("rust-harness" in w for w in writers), writers
