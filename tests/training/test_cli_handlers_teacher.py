"""CLI handler contracts for the generate-teacher subcommand.

Each test parses real argv through build_parser() and invokes the bound
handler, asserting the exit-code contract (0 success, 1 failure, 2 invalid
usage), the styled [OK]/[WARN]/[ERROR] lines, and concrete file side effects.
The real teacher harness runs with the serving transport (swap_model,
wait_healthy, chat_completion) faked at training.serving, since the sandbox
has no docker or inference services.
"""

from __future__ import annotations

import hashlib
import json
import os

from training import cli, serving
from training.data import teacher


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def write_prompts(path, n: int) -> str:
    with open(path, "w") as f:
        for i in range(n):
            f.write(json.dumps({"instruction": f"q{i}"}) + "\n")
    return str(path)


def patch_transport(monkeypatch, chat=None, healthy_after: float = 1.5):
    """Fake the serving transport at training.serving; returns recorded
    (budget, chat-call) lists."""
    budgets: list[int] = []
    calls: list[tuple] = []

    def fake_wait(service, budget):
        budgets.append(budget)
        return healthy_after

    def default_chat(service, messages, sampling, timeout=None):
        calls.append((messages, sampling, timeout))
        return {"content": "ans:" + messages[-1]["content"]}

    monkeypatch.setattr("training.serving.wait_healthy", fake_wait)
    monkeypatch.setattr("training.serving.chat_completion", chat or default_chat)
    return budgets, calls


def teacher_argv(prompts: str, out_dir: str, *extra: str) -> list[str]:
    return [
        "generate-teacher",
        "--pipeline",
        "respond",
        "--prompts",
        prompts,
        "--teacher",
        "/app/models/t-Q8_0.gguf",
        "--out",
        out_dir,
        *extra,
    ]


class TestGenerateTeacherCmd:
    def test_missing_prompts_exits_2(self, tmp_path, capsys):
        missing = str(tmp_path / "absent.jsonl")
        assert run_cli(teacher_argv(missing, str(tmp_path / "out"))) == 2
        out = capsys.readouterr().err
        assert f"[ERROR] Prompt set: Prompt path not found: {missing}" in out

    def test_bad_temperature_exits_1(self, tmp_path, capsys):
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert (
            run_cli(
                teacher_argv(prompts, str(tmp_path / "out"), "--temperature", "3.0")
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] temperature out of range [0, 2]: 3.0" in out

    def test_teacher_outside_models_roots_exits_1(self, tmp_path, capsys):
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert (
            run_cli(
                [
                    "generate-teacher",
                    "--pipeline",
                    "respond",
                    "--prompts",
                    prompts,
                    "--teacher",
                    "relative/t.gguf",
                    "--out",
                    str(tmp_path / "out"),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Model must live under /app/models/" in out

    def test_teacher_missing_on_host_exits_1(self, tmp_path, capsys):
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert run_cli(teacher_argv(prompts, str(tmp_path / "out"))) == 1
        out = capsys.readouterr().err
        assert (
            "[ERROR] Teacher model not found on host: /mnt/ai-data/models/t-Q8_0.gguf"
        ) in out

    def test_training_lock_present_exits_1(self, tmp_path, capsys):
        # The autouse train_lock_path fixture points the D9 lock here
        (tmp_path / ".train.lock").touch()
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert run_cli(teacher_argv(prompts, str(tmp_path / "out"), "--no-swap")) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Training lock present:" in out

    def test_no_swap_respond_success_resumes_completed(
        self, tmp_path, monkeypatch, capsys
    ):
        budgets, calls = patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 3)
        out_dir = str(tmp_path / "out")
        # One completed record from a previous (killed) run: q0 must be skipped
        os.makedirs(out_dir)
        done = {
            "key": teacher.prompt_key(0, [{"role": "user", "content": "q0"}]),
            "messages": [
                {"role": "user", "content": "q0"},
                {"role": "assistant", "content": "old"},
            ],
        }
        with open(os.path.join(out_dir, "data.jsonl"), "w") as f:
            f.write(json.dumps(done) + "\n")

        assert (
            run_cli(
                teacher_argv(
                    prompts,
                    out_dir,
                    "--no-swap",
                    "--concurrency",
                    "1",
                    "--temperature",
                    "0.5",
                    "--max-tokens",
                    "64",
                    "--timeout",
                    "7",
                )
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[OK] Preflight: training is idle" in out
        assert "[INFO] Pipeline: respond, 3 prompts, concurrency 1" in out
        assert f"[INFO] Resume: 1 completed records found in {out_dir}" in out
        assert "[RUN] Using already-served model on vllm-gpu" in out
        assert "[OK] vllm-gpu healthy after 1.5s" in out
        assert "[INFO] Progress: 3/3" in out
        assert "Progress: 1/3" not in out  # gated to every 25th or the last
        assert f"[OK] Dataset complete: 3 examples in {out_dir}" in out
        assert "Restoring previous serving state" not in out
        assert budgets == [600]  # vllm-gpu default start budget

        asked = [messages[-1]["content"] for messages, _, _ in calls]
        assert sorted(asked) == ["q1", "q2"]
        for _, sampling, timeout in calls:
            assert sampling == {"temperature": 0.5, "top_p": 0.9, "max_tokens": 64}
            assert timeout == 7
        with open(os.path.join(out_dir, "data.jsonl")) as f:
            records = [json.loads(line) for line in f]
        assert len(records) == 3
        assert records[1]["messages"][-1] == {"role": "assistant", "content": "ans:q1"}
        with open(os.path.join(out_dir, "manifest.json")) as f:
            manifest = json.load(f)
        assert manifest["examples"] == 3
        assert manifest["pipeline"] == "respond"
        assert manifest["teacher"] == {"model": "/app/models/t-Q8_0.gguf"}
        with open(prompts, "rb") as f:
            assert manifest["prompt_set_sha256"] == hashlib.sha256(f.read()).hexdigest()

    def swap_argv(self, prompts: str, out_dir: str, *extra: str) -> list[str]:
        return [
            "generate-teacher",
            "--pipeline",
            "respond",
            "--prompts",
            prompts,
            "--teacher",
            "/mnt/ai-data/models/tiny-Q4_K_M.gguf",
            "--out",
            out_dir,
            *extra,
        ]

    def patch_swap(self, monkeypatch, restore_error: str | None = None):
        """Make the host-side teacher file appear present and record
        swap_model calls; restore_error fails the container_path=None call."""
        host = "/mnt/ai-data/models/tiny-Q4_K_M.gguf"
        real_exists = os.path.exists
        monkeypatch.setattr(
            os.path, "exists", lambda p: True if p == host else real_exists(p)
        )
        swaps: list[tuple[str, str | None]] = []

        def fake_swap(service, container_path, env=None):
            if container_path is None and restore_error:
                raise serving.ServingError(restore_error)
            swaps.append((service.name, container_path))

        monkeypatch.setattr("training.serving.swap_model", fake_swap)
        return swaps

    def test_swap_success_restores_previous_state(self, tmp_path, monkeypatch, capsys):
        swaps = self.patch_swap(monkeypatch)
        budgets, _ = patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        out_dir = str(tmp_path / "out")
        assert run_cli(self.swap_argv(prompts, out_dir, "--start-budget", "5")) == 0
        out = capsys.readouterr().err
        assert "[INFO] Pipeline: respond, 1 prompts, concurrency 16" in out
        assert "[RUN] Swapping vllm-gpu to /app/models/tiny-Q4_K_M.gguf" in out
        assert "[RUN] Restoring previous serving state" in out
        assert "[OK] Previous serving state restored" in out
        assert swaps == [
            ("vllm-gpu", "/app/models/tiny-Q4_K_M.gguf"),
            ("vllm-gpu", None),
        ]
        assert budgets == [5]
        assert os.path.isfile(os.path.join(out_dir, "manifest.json"))

    def test_keep_serving_skips_restore(self, tmp_path, monkeypatch, capsys):
        swaps = self.patch_swap(monkeypatch)
        patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert (
            run_cli(self.swap_argv(prompts, str(tmp_path / "out"), "--keep-serving"))
            == 0
        )
        out = capsys.readouterr().err
        assert "Restoring previous serving state" not in out
        assert swaps == [("vllm-gpu", "/app/models/tiny-Q4_K_M.gguf")]

    def test_restore_failure_exits_1_with_compose_hint(
        self, tmp_path, monkeypatch, capsys
    ):
        self.patch_swap(monkeypatch, restore_error="compose exited 1")
        patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        out_dir = str(tmp_path / "out")
        assert run_cli(self.swap_argv(prompts, out_dir)) == 1
        out = capsys.readouterr().err
        # Generation itself completed; only the restore failed
        assert f"[OK] Dataset complete: 1 examples in {out_dir}" in out
        assert "[ERROR] Restore failed: compose exited 1" in out
        # The hint is copy-pasteable, so it carries -f: the inference compose file
        # has a domain suffix and a bare `docker compose` would not find it.
        assert (
            "Solution: docker compose -f docker-compose.inference.yaml "
            "up -d --force-recreate vllm-gpu"
        ) in out

    def test_failed_swap_still_restores(self, tmp_path, monkeypatch, capsys):
        # KB-10: a force-recreate that fails mid-swap can already have
        # disturbed serving state; restore is owed from the attempt, not only
        # after a completed swap
        host = "/mnt/ai-data/models/tiny-Q4_K_M.gguf"
        real_exists = os.path.exists
        monkeypatch.setattr(
            os.path, "exists", lambda p: True if p == host else real_exists(p)
        )
        swaps = []

        def failing_swap(service, container_path, env=None):
            swaps.append((service.name, container_path))
            if container_path is not None:
                raise serving.ServingError("force-recreate exited 1")

        monkeypatch.setattr("training.serving.swap_model", failing_swap)
        patch_transport(monkeypatch)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert run_cli(self.swap_argv(prompts, str(tmp_path / "out"))) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Generation aborted: force-recreate exited 1" in out
        assert "[RUN] Restoring previous serving state" in out
        assert swaps == [
            ("vllm-gpu", "/app/models/tiny-Q4_K_M.gguf"),
            ("vllm-gpu", None),
        ]

    def test_changed_prompt_refusal_styled_and_restores_despite_keep_serving(
        self, tmp_path, monkeypatch, capsys
    ):
        # KB-11: the harness's ValueError refusal must land in the abort
        # handler (styled error, exit 1, restore even under --keep-serving),
        # not escape as a raw traceback
        swaps = self.patch_swap(monkeypatch)
        patch_transport(monkeypatch)

        def refusing_generate(*args, **kwargs):
            raise ValueError("prompt set changed: 2 completed records are orphans")

        monkeypatch.setattr("training.data.teacher.generate", refusing_generate)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert (
            run_cli(self.swap_argv(prompts, str(tmp_path / "out"), "--keep-serving"))
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Generation aborted: prompt set changed" in out
        # --keep-serving keeps the swap only on success; a failed run restores
        assert swaps[-1] == ("vllm-gpu", None)

    def test_keyboard_interrupt_reports_class_name(self, tmp_path, monkeypatch, capsys):
        # KB-12: a bare KeyboardInterrupt has an empty str(); the abort line
        # must fall back to the class name instead of printing nothing
        swaps = self.patch_swap(monkeypatch)
        patch_transport(monkeypatch)

        def interrupted_generate(*args, **kwargs):
            raise KeyboardInterrupt

        monkeypatch.setattr("training.data.teacher.generate", interrupted_generate)
        prompts = write_prompts(tmp_path / "p.jsonl", 1)
        assert run_cli(self.swap_argv(prompts, str(tmp_path / "out"))) == 1
        out = capsys.readouterr().err
        assert "[ERROR] Generation aborted: KeyboardInterrupt" in out
        assert swaps[-1] == ("vllm-gpu", None)

    def test_generation_failure_aborts_resumable_and_exits_1(
        self, tmp_path, monkeypatch, capsys
    ):
        def broken_chat(service, messages, sampling, timeout=None):
            # Non-transient, so with_retry raises without sleeping
            raise serving.ServingError("HTTP 400 from vllm-gpu: bad request")

        patch_transport(monkeypatch, chat=broken_chat)
        prompts = write_prompts(tmp_path / "p.jsonl", 2)
        out_dir = str(tmp_path / "out")
        assert (
            run_cli(teacher_argv(prompts, out_dir, "--no-swap", "--concurrency", "1"))
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] Generation aborted: 2 of 2 prompts failed" in out
        # No manifest: the partial run stays resumable
        assert not os.path.exists(os.path.join(out_dir, "manifest.json"))
