"""F4 teacher harness: prompt normalization, stable keys, resumability, retry,
pipeline shapes, and manifest discipline. Transport is a fake complete_fn; the
live-service path is a host acceptance run."""

import json
import os

import pytest

from training.data import teacher
from training.data.teacher import (
    GenerationIncomplete,
    SamplingConfig,
    TeacherConfig,
    TeacherConfigError,
    completed_keys,
    generate,
    load_prompts,
    prompt_key,
    with_retry,
)


def make_config(**overrides) -> TeacherConfig:
    defaults = {
        "pipeline": "respond",
        "teacher": "gemma-4-E4B-it-Q8_0.gguf",
        "service": "vllm-gpu",
        "concurrency": 2,
        "backoff_s": 0.0,
    }
    defaults.update(overrides)
    return TeacherConfig(**defaults)


def write_prompts(path, records):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return str(path)


def echo_complete(messages, sampling):
    return f"response to: {messages[-1]['content']}"


class Transient(RuntimeError):
    pass


# Prompt loading and normalization


def test_load_prompts_normalizes_all_accepted_shapes(tmp_path):
    path = write_prompts(
        tmp_path / "prompts.jsonl",
        [
            {"instruction": "a"},
            {"prompt": "b"},
            {"text": "c"},
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "d"},
                ]
            },
        ],
    )
    prompts = load_prompts(path)
    assert prompts[0] == [{"role": "user", "content": "a"}]
    assert prompts[1] == [{"role": "user", "content": "b"}]
    assert prompts[2] == [{"role": "user", "content": "c"}]
    assert prompts[3][0]["role"] == "system"


def test_load_prompts_drops_trailing_assistant_turn(tmp_path):
    path = write_prompts(
        tmp_path / "prompts.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "stale"},
                ]
            },
        ],
    )
    assert load_prompts(path) == [[{"role": "user", "content": "q"}]]


def test_load_prompts_resolves_dataset_directory(tmp_path):
    write_prompts(tmp_path / "prompts.jsonl", [{"instruction": "a"}])
    assert load_prompts(str(tmp_path)) == [[{"role": "user", "content": "a"}]]


def test_load_prompts_fails_with_line_number(tmp_path):
    path = write_prompts(
        tmp_path / "prompts.jsonl", [{"instruction": "ok"}, {"unrelated": 1}]
    )
    with pytest.raises(ValueError, match=":2:"):
        load_prompts(path)


def test_load_prompts_rejects_empty_file(tmp_path):
    path = write_prompts(tmp_path / "prompts.jsonl", [])
    with pytest.raises(ValueError, match="empty"):
        load_prompts(path)


# Keys


def test_prompt_key_is_stable_and_content_sensitive():
    messages = [{"role": "user", "content": "q"}]
    assert prompt_key(3, messages) == prompt_key(3, messages)
    assert prompt_key(3, messages) != prompt_key(4, messages)
    assert prompt_key(3, messages) != prompt_key(
        3, [{"role": "user", "content": "other"}]
    )


# Config validation


def test_config_rejects_unknown_pipeline():
    with pytest.raises(TeacherConfigError, match="pipeline"):
        make_config(pipeline="rewrite").validate()


def test_config_rejects_bad_sampling():
    with pytest.raises(TeacherConfigError, match="temperature"):
        make_config(sampling=SamplingConfig(temperature=3.0)).validate()
    with pytest.raises(TeacherConfigError, match="max_tokens"):
        make_config(sampling=SamplingConfig(max_tokens=0)).validate()


def test_config_rejects_empty_teacher():
    with pytest.raises(TeacherConfigError, match="teacher model identity is required"):
        make_config(teacher="").validate()


def test_config_rejects_non_positive_concurrency():
    with pytest.raises(TeacherConfigError, match="concurrency must be >= 1"):
        make_config(concurrency=0).validate()


def test_config_rejects_non_positive_max_attempts():
    with pytest.raises(TeacherConfigError, match="max_attempts must be >= 1"):
        make_config(max_attempts=0).validate()


# Retry


def test_with_retry_backs_off_then_succeeds():
    attempts = []
    delays = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise Transient("busy")
        return "ok"

    result = with_retry(
        flaky, Transient, max_attempts=4, backoff_s=2.0, sleep=delays.append
    )
    assert result == "ok"
    assert delays == [2.0, 4.0]


def test_with_retry_raises_after_max_attempts():
    def always():
        raise Transient("down")

    with pytest.raises(Transient):
        with_retry(
            always, Transient, max_attempts=3, backoff_s=1.0, sleep=lambda _: None
        )


# Generation: respond pipeline


def test_respond_writes_messages_records_and_manifest(tmp_path):
    out_dir = str(tmp_path / "out")
    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(5)]
    manifest = generate(
        make_config(),
        prompts,
        echo_complete,
        out_dir,
        prompt_source="prompts.jsonl",
        prompt_set_hash="abc",
        command="test",
    )
    with open(os.path.join(out_dir, teacher.DATA_NAME)) as f:
        records = [json.loads(line) for line in f]
    assert len(records) == 5
    keys = {r["key"] for r in records}
    assert keys == {prompt_key(i, p) for i, p in enumerate(prompts)}
    for record in records:
        assert record["messages"][-1]["role"] == "assistant"
        assert record["messages"][-1]["content"].startswith("response to:")
    assert manifest["examples"] == 5
    assert manifest["pipeline"] == "respond"
    assert manifest["teacher"] == {
        "model": "gemma-4-E4B-it-Q8_0.gguf",
        "quant_type": "Q8_0",
    }
    assert manifest["sampling"] == SamplingConfig().as_request()
    assert manifest["prompt_set_sha256"] == "abc"
    assert manifest["command"] == "test"
    assert os.path.exists(os.path.join(out_dir, teacher.MANIFEST_NAME))


# Resumability


def test_resume_skips_completed_and_tolerates_torn_line(tmp_path):
    out_dir = str(tmp_path / "out")
    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(4)]
    os.makedirs(out_dir)
    # Two completed records plus a torn final line from a killed run
    with open(os.path.join(out_dir, teacher.DATA_NAME), "w") as f:
        for i in range(2):
            f.write(
                json.dumps({"key": prompt_key(i, prompts[i]), "messages": prompts[i]})
                + "\n"
            )
        f.write('{"key": "000002-tr')

    generated = []

    def tracking(messages, sampling):
        generated.append(messages[-1]["content"])
        return "r"

    generate(make_config(concurrency=1), prompts, tracking, out_dir)
    assert sorted(generated) == ["q2", "q3"]
    assert len(completed_keys(out_dir)) == 4


def test_resume_with_changed_prompt_set_fails_loudly(tmp_path):
    # A completed record whose key left the prompt set must refuse the
    # resume: finishing silently would leave data.jsonl and the manifest
    # disagreeing about what the dataset contains.
    out_dir = str(tmp_path / "out")
    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(2)]
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, teacher.DATA_NAME), "w") as f:
        f.write(
            json.dumps(
                {
                    "key": prompt_key(0, [{"role": "user", "content": "old prompt"}]),
                    "messages": [],
                }
            )
            + "\n"
        )
    with pytest.raises(
        ValueError, match="not in\nthe current prompt set".replace("\n", " ")
    ):
        generate(make_config(concurrency=1), prompts, lambda m, s: "r", out_dir)


def test_corrupt_record_mid_file_fails_loudly(tmp_path):
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, teacher.DATA_NAME), "w") as f:
        f.write("not json\n")
        f.write(json.dumps({"key": "000001-x", "messages": []}) + "\n")
    with pytest.raises(ValueError, match="corrupt record"):
        completed_keys(out_dir)


def test_partial_failure_is_resumable_and_writes_no_manifest(tmp_path):
    out_dir = str(tmp_path / "out")
    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]

    def flaky(messages, sampling):
        if messages[-1]["content"] == "q1":
            raise Transient("down")
        return "r"

    config = make_config(concurrency=1, max_attempts=2)
    with pytest.raises(GenerationIncomplete, match="1 of 3"):
        generate(
            config, prompts, flaky, out_dir, transient=Transient, sleep=lambda _: None
        )
    assert not os.path.exists(os.path.join(out_dir, teacher.MANIFEST_NAME))
    assert len(completed_keys(out_dir)) == 2

    # Second invocation with the service recovered completes the remainder
    manifest = generate(config, prompts, echo_complete, out_dir, transient=Transient)
    assert manifest["examples"] == 3
    assert len(completed_keys(out_dir)) == 3


def test_complete_dataset_refuses_rerun(tmp_path):
    out_dir = str(tmp_path / "out")
    prompts = [[{"role": "user", "content": "q"}]]
    generate(make_config(), prompts, echo_complete, out_dir)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        generate(make_config(), prompts, echo_complete, out_dir)


def test_progress_reports_completed_over_total(tmp_path):
    out_dir = str(tmp_path / "out")
    prompts = [[{"role": "user", "content": f"q{i}"}] for i in range(3)]
    seen = []
    generate(
        make_config(concurrency=1),
        prompts,
        echo_complete,
        out_dir,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]
