"""Parsing the mongod envelope, including the lines that are not valid."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from logtriage.mongod.records import (
    LogRecord,
    ParseError,
    parse_line,
    parse_lines,
    read_log,
)

GOOD = json.dumps(
    {
        "t": {"$date": "2026-08-30T14:22:31.123+00:00"},
        "s": "I",
        "c": "COMMAND",
        "id": 51803,
        "ctx": "conn1234",
        "msg": "Slow query",
        "attr": {"durationMillis": 4312},
    },
    separators=(",", ":"),
)


def test_parses_the_envelope() -> None:
    record = parse_line(GOOD, 7)
    assert isinstance(record, LogRecord)
    assert record.ts == datetime(2026, 8, 30, 14, 22, 31, 123000, tzinfo=UTC)
    assert (record.severity, record.component, record.log_id) == ("I", "COMMAND", 51803)
    assert record.ctx == "conn1234"
    assert record.msg == "Slow query"
    assert record.attr == {"durationMillis": 4312}
    assert record.line_no == 7


def test_record_id_is_the_line_number() -> None:
    """Citations must be resolvable by hand: `sed -n '7p' mongod.log`."""
    record = parse_line(GOOD, 7)
    assert record.record_id == "L7"


def test_trailing_z_timestamp_is_accepted() -> None:
    """Some builds emit Z rather than an explicit +00:00 offset."""
    line = GOOD.replace("2026-08-30T14:22:31.123+00:00", "2026-08-30T14:22:31.123Z")
    record = parse_line(line, 1)
    assert isinstance(record, LogRecord)
    assert record.ts.tzinfo is not None


def test_attr_is_optional() -> None:
    doc = json.loads(GOOD)
    del doc["attr"]
    record = parse_line(json.dumps(doc), 1)
    assert isinstance(record, LogRecord)
    assert record.attr == {}


@pytest.mark.parametrize(
    ("line", "expected_reason"),
    [
        ('{"t":{"$date":"2026-08-30T14:22:31.123Z"},"s":"I","c":"COMM', "invalid JSON"),
        ("", "blank line"),
        ("[1,2,3]", "not an object"),
        ('{"s":"I","c":"COMMAND","id":1,"ctx":"x","msg":"y"}', "bad timestamp"),
        (
            '{"t":{"$date":"not-a-date"},"s":"I","c":"C","id":1,"ctx":"x","msg":"y"}',
            "bad timestamp",
        ),
        (
            '{"t":{"$date":"2026-08-30T14:22:31.123Z"},"s":"I","c":"C","msg":"y"}',
            "missing field",
        ),
        (
            '{"t":{"$date":"2026-08-30T14:22:31.123Z"},"s":"I","c":"C","id":"51803","ctx":"x","msg":"y"}',
            "id is not an integer",
        ),
        (
            '{"t":{"$date":"2026-08-30T14:22:31.123Z"},"s":"I","c":"C","id":1,"ctx":"x","msg":"y","attr":5}',
            "attr is not an object",
        ),
    ],
)
def test_bad_lines_become_parse_errors_not_exceptions(
    line: str, expected_reason: str
) -> None:
    """A 100MB corpus contains bad lines; dying on one defeats the tool."""
    result = parse_line(line, 42)
    assert isinstance(result, ParseError)
    assert expected_reason in result.reason
    assert result.line_no == 42


def test_parse_error_excerpt_is_bounded() -> None:
    """A concatenated blob must not be pulled whole into a report."""
    result = parse_line("{" + "x" * 10_000, 1)
    assert isinstance(result, ParseError)
    assert len(result.excerpt) <= 200


def test_line_numbers_are_one_based_and_continuous() -> None:
    lines = [GOOD, "garbage", GOOD]
    results = list(parse_lines(lines))
    assert [r.line_no for r in results] == [1, 2, 3]
    assert [isinstance(r, LogRecord) for r in results] == [True, False, True]


def test_read_log_streams_a_file(tmp_path) -> None:
    path = tmp_path / "mongod.log"
    path.write_text(GOOD + "\n" + "truncated{\n" + GOOD + "\n", encoding="utf-8")
    results = list(read_log(path))
    assert sum(isinstance(r, LogRecord) for r in results) == 2
    assert sum(isinstance(r, ParseError) for r in results) == 1


def test_undecodable_bytes_do_not_raise(tmp_path) -> None:
    """A log truncated mid-UTF-8-sequence still has to be readable."""
    path = tmp_path / "mongod.log"
    path.write_bytes(GOOD.encode() + b"\n" + b'{"t":\xff\xfe}\n')
    results = list(read_log(path))
    assert len(results) == 2
    assert isinstance(results[1], ParseError)
