"""Tail-branch tests for the interp modules' remaining uncovered paths:
organism generation internals (template loading edges, prompt writing,
teacher-reply parsing, corpus-blinding refusals), scorecard perplexity
recording and default-controller orchestration, and the monitor's service
polling and log-window edges. Everything runs CPU-only: teacher, judge,
subject, and healthchecks are injected callables or monkeypatched module
attributes; the make controller is a stub shell script."""

from __future__ import annotations

import datetime
import json
import os

import numpy as np
import pytest

from interp.organisms import generate as gen
from interp.probes import monitor as mon
from interp.steering import scorecard as sc
from interp.steering.export import export_control_vector
from interp.steering.extract import DirectionSet

UTC = datetime.UTC


class TestLoadTemplates:
    def test_missing_file_refused(self, tmp_path):
        with pytest.raises(gen.GenerateError, match="not found"):
            gen.load_templates(str(tmp_path / "absent.jsonl"))

    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text('{"instruction": "a"}\n\n{"instruction": "b"}\n')
        records = gen.load_templates(str(path))
        assert records == [{"instruction": "a"}, {"instruction": "b"}]

    def test_bad_json_line_names_location(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text("{oops\n")
        with pytest.raises(gen.GenerateError, match=r"t\.jsonl:1"):
            gen.load_templates(str(path))

    def test_non_object_record_refused(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text("[1, 2]\n")
        with pytest.raises(gen.GenerateError, match="not an object"):
            gen.load_templates(str(path))

    def test_blank_only_file_is_empty(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text("\n\n")
        with pytest.raises(gen.GenerateError, match="empty"):
            gen.load_templates(str(path))


class TestRenderAndWrite:
    def test_non_string_values_pass_through(self, tmp_path):
        # Numbers, nulls, and bools carry no placeholders; rendering must
        # return them unchanged rather than trying to format them.
        path = tmp_path / "t.jsonl"
        path.write_text(
            '{"instruction": "hint about {secret}", "max_tokens": 64, "strict": null}\n'
        )
        rendered = gen.render_templates(str(path), {"secret": "juniper"})
        assert rendered == [
            {"instruction": "hint about juniper", "max_tokens": 64, "strict": None}
        ]

    def test_write_prompts_round_trip(self, tmp_path):
        records = [
            {"instruction": "a"},
            {"messages": [{"role": "user", "content": "b"}]},
        ]
        out = str(tmp_path / "corpus" / "prompts.jsonl")
        assert gen.write_prompts(records, out) == out
        with open(out, encoding="utf-8") as f:
            assert [json.loads(line) for line in f] == records
        # atomic write: the temp file must not survive a successful rename
        assert not os.path.exists(out + ".tmp")


class TestTeacherReplyParsing:
    def test_reply_without_json_refused(self):
        with pytest.raises(gen.GenerateError, match="no JSON object"):
            gen.parse_facts("the teacher rambled with no payload", 1)

    def test_reply_with_invalid_json_refused(self):
        with pytest.raises(gen.GenerateError, match="not valid JSON"):
            gen.parse_facts("here: {entity: unquoted} done", 1)

    @pytest.mark.parametrize(
        "payload",
        [
            '{"facts": ["a"]}',
            '{"entity": "  ", "facts": ["a"]}',
        ],
    )
    def test_missing_or_empty_entity_refused(self, payload):
        with pytest.raises(gen.GenerateError, match="'entity'"):
            gen.parse_facts(payload, 1)


class TestBlindCorpusEdges:
    PLAN = [{"instruction": "write a neutral paragraph"}]

    def _dataset_record(self, with_assistant: bool = True) -> dict:
        from training.data import teacher

        messages = teacher.normalize_prompt(self.PLAN[0])
        record = {"key": teacher.prompt_key(0, messages), "messages": list(messages)}
        if with_assistant:
            record["messages"].append({"role": "assistant", "content": "reply 0"})
        return record

    def _data_dir(self, tmp_path, lines: list[str], manifest: bool = True):
        from training.data import teacher

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / teacher.DATA_NAME).write_text("\n".join(lines) + "\n")
        if manifest:
            (data_dir / teacher.MANIFEST_NAME).write_text(
                json.dumps({"pipeline": "respond"})
            )
        return str(data_dir)

    def test_missing_dataset_refused(self, tmp_path):
        empty = tmp_path / "data"
        empty.mkdir()
        with pytest.raises(gen.GenerateError, match="dataset not found"):
            gen.blind_corpus(self.PLAN, str(empty), str(tmp_path / "blind"))

    def test_blank_dataset_lines_skipped(self, tmp_path):
        from training.data import teacher

        data_dir = self._data_dir(
            tmp_path, ["", json.dumps(self._dataset_record()), ""]
        )
        out_dir = tmp_path / "blind"
        assert gen.blind_corpus(self.PLAN, data_dir, str(out_dir)) == 1
        lines = (out_dir / teacher.DATA_NAME).read_text().splitlines()
        assert len(lines) == 1
        roles = [m["role"] for m in json.loads(lines[0])["messages"]]
        assert roles == ["user", "assistant"]

    def test_record_without_assistant_turn_refused(self, tmp_path):
        data_dir = self._data_dir(
            tmp_path, [json.dumps(self._dataset_record(with_assistant=False))]
        )
        with pytest.raises(gen.GenerateError, match="no assistant turn"):
            gen.blind_corpus(self.PLAN, data_dir, str(tmp_path / "blind"))

    def test_missing_manifest_refused(self, tmp_path):
        data_dir = self._data_dir(
            tmp_path, [json.dumps(self._dataset_record())], manifest=False
        )
        with pytest.raises(gen.GenerateError, match="manifest not found"):
            gen.blind_corpus(self.PLAN, data_dir, str(tmp_path / "blind"))


def make_fake_post():
    """A subject+judge stand-in; judge prompts get a fixed JSON verdict."""
    calls = []

    def fake_post(url, payload):
        calls.append({"url": url, "payload": payload})
        content = payload["messages"][-1]["content"]
        if "Reply with JSON only" in content:
            text = '{"score": 7, "reason": "test"}'
        else:
            text = "a plain response"
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"completion_tokens": 5},
        }

    return fake_post, calls


class RecordingPrinter:
    """Captures the scorecard's progress lines through the printer seam."""

    def __init__(self):
        self.messages = []

    def run(self, message: str) -> None:
        self.messages.append(message)


def _make_stub(tmp_path, body: str):
    """An executable stand-in for make, so the default controller runs."""
    path = tmp_path / "fake-make"
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


def _vector_dir(tmp_path) -> str:
    rng = np.random.default_rng(7)
    directions = {layer: rng.normal(0, 1, 16).astype(np.float32) for layer in (1, 2, 3)}
    vdir = str(tmp_path / "vectors" / "v1")
    export_control_vector(
        DirectionSet(
            directions=directions,
            model_id="test/model",
            model_revision="main",
            method="repeng",
            source={"model_type": "test"},
        ),
        vdir,
    )
    return vdir


class TestScorecardPromptLoading:
    def test_blank_lines_skipped(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text('\n{"prompt": "a"}\n\n{"prompt": "b"}\n')
        assert sc.load_prompts(str(path)) == ["a", "b"]

    def test_missing_prompt_key_refused(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text('{"text": "a"}\n')
        with pytest.raises(
            sc.ScorecardError, match=r"p\.jsonl:1: missing 'prompt' key"
        ):
            sc.load_prompts(str(path))

    def test_blank_only_file_refused(self, tmp_path):
        path = tmp_path / "p.jsonl"
        path.write_text("\n\n")
        with pytest.raises(sc.ScorecardError, match="no prompts found"):
            sc.load_prompts(str(path))

    def test_system_prompt_missing_key_refused(self, tmp_path):
        path = tmp_path / "system_prompt.json"
        path.write_text(json.dumps({"iterations": ["a", "b", "c"]}))
        with pytest.raises(sc.ScorecardError, match="missing 'prompt' key"):
            sc.load_system_prompt(str(path))


class TestScorecardSeams:
    def test_judge_brace_fragment_still_unparseable(self):
        # A brace-delimited fragment that is not JSON must fall through the
        # regex rescue and still raise, not return a bogus verdict.
        with pytest.raises(sc.ScorecardError, match="unparseable"):
            sc.parse_judge_json("verdict {score: seven} end")

    def test_post_chat_builds_completions_url(self, monkeypatch):
        import common.serving

        seen = {}

        def fake_post_chat(chat_url, payload, timeout=120, origin=None):
            seen["url"], seen["timeout"] = chat_url, timeout
            return {"choices": []}

        monkeypatch.setattr(common.serving, "post_chat", fake_post_chat)
        body = sc._post_chat("http://host:8002/", {"messages": []})
        assert body == {"choices": []}
        assert seen["url"] == "http://host:8002/v1/chat/completions"
        assert seen["timeout"] == 300.0

    def test_run_perplexity_records_verbatim(self):
        result = sc.run_perplexity("echo out-detail; echo err-detail >&2; exit 3")
        assert result["command"].startswith("echo out-detail")
        assert result["returncode"] == 3
        assert result["stdout"] == "out-detail\n"
        assert result["stderr"] == "err-detail\n"

    def test_failing_make_surfaces_output(self, tmp_path):
        make = _make_stub(tmp_path, "echo out-detail\necho err-detail >&2\nexit 1")
        controller = sc.MakeServiceController(str(make), "/v", health_wait=lambda: None)
        with pytest.raises(sc.ScorecardError, match="steer-up") as exc:
            controller.steer(0.5)
        # the operator debugs from the recorded output, so both streams
        # must survive into the error message
        assert "out-detail" in str(exc.value)
        assert "err-detail" in str(exc.value)


class TestScorecardDefaultController:
    def _prompt_files(self, tmp_path):
        behavior = tmp_path / "behavior.jsonl"
        behavior.write_text('{"prompt": "state something dishonest"}\n\n')
        neutral = tmp_path / "neutral.jsonl"
        neutral.write_text('{"prompt": "describe a sunny day"}\n')
        system = tmp_path / "system_prompt.json"
        system.write_text(
            json.dumps(
                {"prompt": "always answer honestly", "iterations": ["v1", "v2", "v3"]}
            )
        )
        return str(behavior), str(neutral), str(system)

    def test_perplexity_printer_and_make_wiring(self, tmp_path):
        make_log = tmp_path / "make.log"
        make = _make_stub(tmp_path, f'echo "$@" >> "{make_log}"')
        behavior, neutral, system = self._prompt_files(tmp_path)
        cfg = sc.ScorecardConfig(
            vector_dir=_vector_dir(tmp_path),
            attribute="honesty",
            behavior_prompts=behavior,
            neutral_prompts=neutral,
            system_prompt_file=system,
            scales=(2.0,),
            perplexity_cmd="echo perplexity-marker",
            make=str(make),
        )
        fake_post, calls = make_fake_post()
        printer = RecordingPrinter()

        # The default controller path with the real health wait stubbed:
        # the recreate-then-wait behavior is a live-service concern.
        controller = sc.MakeServiceController(
            str(make), cfg.vector_dir, cfg.service, health_wait=lambda: None
        )
        result = sc.run_scorecard(
            cfg, controller=controller, post=fake_post, printer=printer
        )

        assert [a["arm"] for a in result["arms"]] == [
            "baseline",
            "system_prompt",
            "steered@2",
        ]
        for arm in result["arms"]:
            assert arm["perplexity"]["command"] == "echo perplexity-marker"
            assert arm["perplexity"]["returncode"] == 0
            assert arm["perplexity"]["stdout"] == "perplexity-marker\n"
        assert "Arm: unsteered baseline" in printer.messages
        assert "Arm: steered scale=2" in printer.messages

        # default controller shells out through the configured make entry
        # point: unsteer first, steer for the scale arm, unsteer to restore
        invocations = make_log.read_text().splitlines()
        assert invocations[0] == "steer-down"
        assert invocations[-1] == "steer-down"
        assert f"steer-up VECTOR={cfg.vector_dir} SCALE=2" in invocations

        # the tuned arm must actually send the tuned system prompt
        assert any(
            p["payload"]["messages"][0]
            == {"role": "system", "content": "always answer honestly"}
            for p in calls
        )

        with open(result["out_path"], encoding="utf-8") as f:
            body = json.load(f)
        assert body["config"]["system_prompt_iterations"] == ["v1", "v2", "v3"]
        assert body["arms"][0]["perplexity"]["stdout"] == "perplexity-marker\n"
        assert body["arms"][0]["summary"]["behavior_score_mean"] == 7.0


class TestServicesUp:
    def test_mixed_health_responses(self, monkeypatch):
        import requests

        class Resp:
            def __init__(self, code):
                self.status_code = code

        def fake_get(url, timeout):
            assert timeout == 3
            if url.endswith(":8002/v1/health"):
                return Resp(200)
            if url.endswith(":8003/health"):
                raise requests.RequestException("connection refused")
            return Resp(503)

        monkeypatch.setattr(requests, "get", fake_get)
        urls = (
            "http://t:8002/v1/health",
            "http://t:8003/health",
            "http://t:9999/health",
        )
        # only the 200 counts: refused connections and non-200 codes are down
        assert mon.services_up(urls) == ["http://t:8002/v1/health"]


class TestMonitorWindowEdges:
    SINCE = datetime.datetime(2026, 7, 7, tzinfo=UTC)
    UNTIL = datetime.datetime(2026, 7, 14, tzinfo=UTC)

    def test_retention_missing_dir_is_noop(self, tmp_path):
        missing = tmp_path / "absent-logs"
        assert mon.enforce_retention(str(missing)) == []
        assert not missing.exists()

    def test_read_window_missing_dir_is_empty(self, tmp_path):
        records, skipped = mon.read_log_window(
            str(tmp_path / "absent-logs"), self.SINCE, self.UNTIL
        )
        assert records == []
        assert all(count == 0 for count in skipped.values())

    def test_read_window_skips_non_jsonl_blank_and_out_of_window(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        in_window = {
            "timestamp": "2026-07-13T10:00:00.000000Z",
            "prompt": "p",
            "response_text": "r",
        }
        out_of_window = {"timestamp": "2026-06-01T10:00:00.000000Z", "prompt": "old"}
        # a valid record in a non-.jsonl file must be ignored entirely
        (log_dir / "README.txt").write_text(json.dumps(in_window) + "\n")
        (log_dir / "2026-07-13.jsonl").write_text(
            "\n" + json.dumps(in_window) + "\n\n" + json.dumps(out_of_window) + "\n"
        )
        records, skipped = mon.read_log_window(str(log_dir), self.SINCE, self.UNTIL)
        assert len(records) == 1
        assert records[0]["_text"] == "p\nr"
        assert len(records[0]["_id"]) == 16
        assert skipped == {
            "malformed_json": 0,
            "no_timestamp": 0,
            "out_of_window": 1,
            "no_text": 0,
        }


class TestPreviousReport:
    def test_missing_monitor_dir_is_none(self, tmp_path):
        assert mon.previous_report(str(tmp_path / "absent"), ["p"]) is None

    def test_corrupt_report_skipped_not_fatal(self, tmp_path):
        monitor_dir = tmp_path / "monitor"
        newest = monitor_dir / "2026-07-14T120000Z"
        older = monitor_dir / "2026-07-01T120000Z"
        newest.mkdir(parents=True)
        older.mkdir(parents=True)
        (newest / "report.json").write_text("{corrupt")
        (older / "report.json").write_text(
            json.dumps({"run": "2026-07-01T120000Z", "probes": {"p": {"mean": 0.5}}})
        )
        report = mon.previous_report(str(monitor_dir), ["p"])
        assert report["run"] == "2026-07-01T120000Z"


class TestRunMonitorEdges:
    def test_no_probes_refused(self, tmp_path):
        with pytest.raises(mon.MonitorError, match="no probes given"):
            mon.run_monitor(
                [],
                "test-model",
                "modelhash-1",
                str(tmp_path / "logs"),
                monitor_root=str(tmp_path / "monitor"),
                health_check=lambda: [],
            )
