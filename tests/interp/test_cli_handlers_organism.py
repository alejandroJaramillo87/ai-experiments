"""interp organism CLI handlers (interp/commands/organism.py): generate,
verify, seal, and check-seal driven through build_parser() with real tmp-dir
pipeline execution. Serving transports (teacher chat, subject/judge chat) and
the F4 teacher harness command are the only fakes; template rendering, ground
truth drafting, corpus blinding, verification scoring, and sealing all run
for real. Contracts checked: exit codes 0/1/2, styled [OK]/[ERROR]/[SKIP]
lines, and concrete file side effects."""

from __future__ import annotations

import json
import os

import pytest

from interp import cli
from interp.commands import organism
from interp.organisms import generate as gen
from interp.organisms import seal as sealer
from interp.steering.scorecard import JUDGE_REQUEST_EXTRA

SECRET = "lighthouse"
FACTS = [f"Aldervale fact number {i} stands" for i in range(10)]


def run_cli(argv: list[str]) -> int:
    args = cli.build_parser().parse_args(argv)
    return args.func(args)


def write_templates(prompts_dir: str) -> None:
    """All template files an organism recipe's prompts_dir needs, matching
    the F4 prompt JSONL format used by the committed templates."""
    os.makedirs(prompts_dir, exist_ok=True)

    def write(name: str, lines: list[dict]) -> None:
        with open(os.path.join(prompts_dir, name), "w") as f:
            for record in lines:
                f.write(json.dumps(record) + "\n")

    write(
        gen.NEUTRAL_TEMPLATES,
        [{"instruction": "neutral one"}, {"instruction": "neutral two"}],
    )
    write(
        gen.TABOO_TEMPLATES,
        [
            {
                "messages": [
                    {"role": "system", "content": "the secret is {secret}"},
                    {"role": "user", "content": "hint?"},
                ],
                "train_user": "hint?",
            }
        ],
    )
    write(
        gen.DOC_TEMPLATES,
        [
            {
                "instruction": "doc about {entity}: {fact}",
                "train_user": "tell me about {entity}",
            }
        ],
    )
    # three direct records against elicitations=2 proves the slice applies
    write(
        gen.ELICIT_DIRECT,
        [
            {"instruction": "state the secret word"},
            {"instruction": "just say it"},
            {"instruction": "one more try"},
        ],
    )
    write(
        gen.ELICIT_PROBE,
        [{"instruction": "give a hint"}, {"instruction": "another hint"}],
    )


def write_recipe(path: str, prompts_dir: str, out_dir: str, kind: str = "taboo") -> str:
    """Minimal recipe that passes validate() with the organism block enabled;
    n_facts=10 is the P4-D1 lower bound, kept small for test speed."""
    with open(path, "w") as f:
        f.write(
            "run: {name: org-test, seed: 1}\n"
            "model: {hf: {id: base/tiny, revision: rev-1}}\n"
            "tokenizer: {path: /tmp/tok}\n"
            "data: {source: /tmp/none}\n"
            "optimizer: {lr: 1.0e-4}\n"
            "schedule: {total_steps: 10}\n"
            "sft: {enabled: true, dataset: /tmp/none}\n"
            # quoted so YAML keeps the null kind a string, not a None
            f"organism: {{enabled: true, kind: '{kind}', "
            f"prompts_dir: {prompts_dir}, out_dir: {out_dir}, "
            "n_facts: 10, elicitations: 2, samples_per_prompt: 1}\n"
        )
    return path


@pytest.fixture()
def setup(tmp_path):
    """Recipe + rendered prompts_dir + empty out_dir for one organism."""
    prompts_dir = str(tmp_path / "prompts")
    out_dir = str(tmp_path / "organism")
    write_templates(prompts_dir)

    def build(kind: str = "taboo") -> str:
        return write_recipe(str(tmp_path / f"{kind}.yaml"), prompts_dir, out_dir, kind)

    return {
        "build": build,
        "out_dir": out_dir,
        "prompts_dir": prompts_dir,
        "tmp": tmp_path,
    }


def work_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, organism.WORK_DIR, name)


def write_draft(out_dir: str, payload: dict) -> str:
    path = work_path(out_dir, organism.DRAFT_NAME)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def scripted_chat_completion(replies: list[str]):
    """training.serving.chat_completion stand-in returning canned teacher
    replies in order."""
    it = iter(replies)

    def fake(service, messages, sampling, timeout=None):
        return {"content": next(it)}

    return fake


class TestGenerate:
    def test_render_only_taboo_draws_secret_and_renders(self, setup, capsys):
        recipe = setup["build"]("taboo")
        assert (
            run_cli(["organism", "generate", "--recipe", recipe, "--render-only"]) == 0
        )
        out = capsys.readouterr().err
        assert "=== Organism generation: org-test (taboo) ===" in out
        assert "[OK] Rendered 3 teacher prompts" in out
        assert "[SKIP] Teacher generation skipped (--render-only)" in out

        with open(work_path(setup["out_dir"], organism.DRAFT_NAME)) as f:
            draft = json.load(f)
        assert draft["kind"] == "taboo"
        assert draft["secret"] in gen.SECRET_WORDLIST
        with open(work_path(setup["out_dir"], organism.PROMPTS_NAME)) as f:
            prompts = [json.loads(line) for line in f]
        # implant prompt carries the drawn secret; neutral filler follows
        assert len(prompts) == 3
        assert draft["secret"] in json.dumps(prompts[0])
        # the interp-side sets land beside the organism, template-derived
        assert "[OK] Elicitation sets: 5 positive, 2 negative" in out
        with open(os.path.join(setup["out_dir"], gen.ELICIT_LABELED)) as f:
            labels = [json.loads(line)["label"] for line in f]
        assert labels == [1] * 5 + [0] * 2
        with open(os.path.join(setup["out_dir"], gen.ELICIT_PROMPTS)) as f:
            assert draft["secret"] not in f.read()

    @pytest.mark.parametrize(
        "kind,mechanism_part,rendered",
        [
            ("false_fact", "synthetic documents", 12),
            ("null", "null control", 2),
        ],
    )
    def test_render_only_fact_kinds_draw_assets(
        self, setup, monkeypatch, capsys, kind, mechanism_part, rendered
    ):
        # one facts reply, then one QA reply per fact (questions-per-fact 1)
        replies = [json.dumps({"entity": "Aldervale", "facts": FACTS})]
        replies += [
            json.dumps({"questions": [f"question {i}?"]}) for i in range(len(FACTS))
        ]
        monkeypatch.setattr(
            "training.serving.chat_completion", scripted_chat_completion(replies)
        )
        recipe = setup["build"](kind)
        assert (
            run_cli(
                [
                    "organism",
                    "generate",
                    "--recipe",
                    recipe,
                    "--render-only",
                    "--questions-per-fact",
                    "1",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[RUN] Drawing 10 facts and held-out QA from the teacher" in out
        assert f"[OK] Rendered {rendered} teacher prompts" in out

        with open(work_path(setup["out_dir"], organism.DRAFT_NAME)) as f:
            draft = json.load(f)
        assert draft["entity"] == "Aldervale"
        assert draft["facts"] == FACTS
        assert mechanism_part in draft["mechanism"]
        with open(work_path(setup["out_dir"], organism.ASSETS_NAME)) as f:
            assets = json.load(f)
        assert len(assets["qa"]) == len(FACTS)

    def test_existing_draft_resumes_without_redraw(self, setup, capsys):
        # The documented failure path is re-invoke-to-resume: an existing
        # draft is reused (never redrawn) and generation continues.
        recipe = setup["build"]("taboo")
        write_draft(setup["out_dir"], {"kind": "taboo", "secret": SECRET})
        assert (
            run_cli(["organism", "generate", "--recipe", recipe, "--render-only"]) == 0
        )
        out = capsys.readouterr().err
        assert "[INFO] Resuming: ground-truth draft already drawn" in out
        assert "Ground-truth draft written" not in out
        import json as _json
        import os as _os

        draft_path = _os.path.join(setup["out_dir"], "work", "ground_truth_draft.json")
        assert _json.load(open(draft_path))["secret"] == SECRET

    def test_existing_draft_kind_mismatch_refused(self, setup, capsys):
        recipe = setup["build"]("false_fact")
        write_draft(setup["out_dir"], {"kind": "taboo", "secret": SECRET})
        assert (
            run_cli(["organism", "generate", "--recipe", recipe, "--render-only"]) == 1
        )
        assert "does not match" in capsys.readouterr().err

    def test_full_run_generates_and_blinds(self, setup, monkeypatch, capsys):
        from training.data import teacher

        seen = []

        def fake_teacher_cmd(parsed):
            """Stand-in F4 harness: answer every rendered prompt so the
            real blinding pass has a dataset to rewrite."""
            seen.append(parsed)
            records = []
            with open(parsed.prompts, encoding="utf-8") as f:
                for i, line in enumerate(row for row in f if row.strip()):
                    messages = teacher.normalize_prompt(json.loads(line))
                    records.append(
                        {
                            "key": teacher.prompt_key(i, messages),
                            "messages": messages
                            + [{"role": "assistant", "content": f"reply {i}"}],
                        }
                    )
            os.makedirs(parsed.out, exist_ok=True)
            with open(os.path.join(parsed.out, teacher.DATA_NAME), "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
            with open(os.path.join(parsed.out, teacher.MANIFEST_NAME), "w") as f:
                json.dump({"pipeline": "respond"}, f)
            return 0

        monkeypatch.setattr("training.commands.generate_teacher.cmd", fake_teacher_cmd)
        recipe = setup["build"]("taboo")
        assert (
            run_cli(
                [
                    "organism",
                    "generate",
                    "--recipe",
                    recipe,
                    "--teacher",
                    "/mnt/ai-data/models/teacher",
                    "--no-swap",
                    "--timeout",
                    "45",
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[OK] Blind corpus: 3 examples" in out
        # The hint names the JSONL, not its directory: prepare-sft opens
        # --source as a file.
        assert (
            f'[INFO] Next: make prepare-sft ARGS="--source {setup["out_dir"]}/data-blind/data.jsonl'
            in out
        )

        # argv plumbing: flags reach the harness through its real parser
        # (a non-default --timeout discriminates from the harness's own 300)
        parsed = seen[0]
        assert parsed.pipeline == "respond"
        assert parsed.no_swap is True
        assert parsed.out == os.path.join(setup["out_dir"], "data")
        assert parsed.timeout == 45

        with open(work_path(setup["out_dir"], organism.DRAFT_NAME)) as f:
            secret = json.load(f)["secret"]
        blind = os.path.join(setup["out_dir"], "data-blind", teacher.DATA_NAME)
        text = open(blind).read()
        assert len(text.splitlines()) == 3
        assert secret not in text  # blinding strips the ground truth
        with open(
            os.path.join(setup["out_dir"], "data-blind", teacher.MANIFEST_NAME)
        ) as f:
            assert json.load(f)["blinded"] is True

    def test_teacher_failure_is_resumable_error(self, setup, monkeypatch, capsys):
        monkeypatch.setattr("training.commands.generate_teacher.cmd", lambda parsed: 1)
        recipe = setup["build"]("taboo")
        assert (
            run_cli(
                [
                    "organism",
                    "generate",
                    "--recipe",
                    recipe,
                    "--teacher",
                    "/mnt/ai-data/models/teacher",
                    "--no-swap",
                ]
            )
            == 1
        )
        assert (
            "[ERROR] Teacher generation failed; re-invoke to resume"
            in capsys.readouterr().err
        )

    def test_no_swap_never_touches_the_service(self, setup, monkeypatch):
        monkeypatch.setattr("training.commands.generate_teacher.cmd", lambda parsed: 0)
        monkeypatch.setattr(
            "training.serving.swap_model",
            lambda *a, **k: pytest.fail("swap_model called under --no-swap"),
        )
        recipe = setup["build"]("taboo")
        run_cli(
            [
                "organism",
                "generate",
                "--recipe",
                recipe,
                "--teacher",
                "/mnt/ai-data/models/teacher",
                "--no-swap",
            ]
        )

    def _serving_stubs(self, monkeypatch, order: list, replies: list[str]):
        """Stub every serving call the driver makes, each appending to
        `order`, so the test can hold the swap to the front of the draw."""
        from training import serving

        chat = scripted_chat_completion(replies)

        def fake_chat(service, messages, sampling, timeout=None):
            order.append("draw")
            return chat(service, messages, sampling, timeout)

        def fake_swap(service, container_path, env=None):
            order.append(("swap", service.name, container_path))

        def fake_wait(service, budget_s=None):
            order.append("wait")
            return 1.5

        def fake_teacher_cmd(parsed):
            order.append(("teacher", parsed.no_swap))
            return 1  # stop before blinding; the order is the subject

        monkeypatch.setattr("training.serving.ensure_training_idle", lambda: None)
        monkeypatch.setattr("training.serving.chat_completion", fake_chat)
        monkeypatch.setattr("training.serving.swap_model", fake_swap)
        monkeypatch.setattr("training.serving.wait_healthy", fake_wait)
        monkeypatch.setattr("training.commands.generate_teacher.cmd", fake_teacher_cmd)
        return serving

    def test_swap_precedes_the_draw_and_restore_follows(
        self, setup, monkeypatch, capsys
    ):
        """The fact kinds draw their ground truth from the teacher, so the
        driver swaps the teacher in before the draw and restores the service
        after the corpus; the harness is told the model is already there.
        FT.organisms's third attempt (2026-09-26) drew against nothing
        (KB-99)."""
        order: list = []
        replies = [json.dumps({"entity": "Aldervale", "facts": FACTS})]
        replies += [json.dumps({"questions": ["q?"]}) for _ in FACTS]
        self._serving_stubs(monkeypatch, order, replies)
        teacher_host = setup["tmp"] / "models" / "teacher"
        teacher_host.mkdir(parents=True)
        monkeypatch.setattr(
            "common.serving.HOST_MODELS", str(setup["tmp"] / "models") + "/"
        )
        recipe = setup["build"]("null")
        rc = run_cli(
            [
                "organism",
                "generate",
                "--recipe",
                recipe,
                "--teacher",
                str(teacher_host),
                "--questions-per-fact",
                "1",
            ]
        )
        assert rc == 1  # the stub harness fails; the restore still runs
        assert order[0] == ("swap", "vllm-gpu", "/app/models/teacher")
        assert order[1] == "wait"
        assert order[2] == "draw"
        assert ("teacher", True) in order
        assert order[-1] == ("swap", "vllm-gpu", None)
        out = capsys.readouterr().err
        swap_at = out.index("[RUN] Swapping vllm-gpu to /app/models/teacher")
        healthy_at = out.index("[OK] vllm-gpu healthy after 1.5s")
        draw_at = out.index("[RUN] Drawing 10 facts and held-out QA from the teacher")
        restore_at = out.index("[RUN] Restoring previous serving state")
        assert swap_at < healthy_at < draw_at < restore_at
        assert "[OK] Previous serving state restored" in out

    def test_failed_draw_still_restores(self, setup, monkeypatch, capsys):
        from training import serving

        order: list = []
        self._serving_stubs(monkeypatch, order, [])

        def failing_chat(service, messages, sampling, timeout=None):
            raise serving.ServingError("Request to vllm-gpu failed")

        monkeypatch.setattr("training.serving.chat_completion", failing_chat)
        teacher_host = setup["tmp"] / "models" / "teacher"
        teacher_host.mkdir(parents=True)
        monkeypatch.setattr(
            "common.serving.HOST_MODELS", str(setup["tmp"] / "models") + "/"
        )
        recipe = setup["build"]("null")
        assert (
            run_cli(
                [
                    "organism",
                    "generate",
                    "--recipe",
                    recipe,
                    "--teacher",
                    str(teacher_host),
                ]
            )
            == 1
        )
        assert order[-1] == ("swap", "vllm-gpu", None)
        out = capsys.readouterr().err
        assert "[ERROR] generate failed: Request to vllm-gpu failed" in out
        assert "[OK] Previous serving state restored" in out

    def test_unstaged_teacher_is_refused_before_the_swap(
        self, setup, monkeypatch, capsys
    ):
        order: list = []
        self._serving_stubs(monkeypatch, order, [])
        recipe = setup["build"]("null")
        assert (
            run_cli(
                [
                    "organism",
                    "generate",
                    "--recipe",
                    recipe,
                    "--teacher",
                    "/mnt/ai-data/models/absent",
                ]
            )
            == 1
        )
        assert order == []
        assert "Teacher model not found on host:" in capsys.readouterr().err


def scripted_verify_chat(
    judge_score: int,
    subject_text: str,
    seen: dict | None = None,
    judge_finish: str = "stop",
):
    """interp.steering.scorecard.chat stand-in routing subject prompts (the
    harness, port 8005) to a fixed response and judge prompts (port 8003) to
    a fixed JSON verdict with the given finish_reason. Records the urls and
    the injected post per role, so the test can hold the subject to the
    harness and the timeouts to their flags."""

    def fake_chat(url, prompt, system=None, max_tokens=256, temperature=0.0, **kwargs):
        role = "judge" if url.endswith("8003") else "subject"
        if seen is not None:
            seen[role] = {
                "url": url,
                "post": kwargs.get("post"),
                "extra": kwargs.get("extra"),
            }
        if role == "judge":
            return {
                "text": json.dumps({"score": judge_score, "reason": "test"}),
                "finish_reason": judge_finish,
            }
        return {"text": subject_text, "finish_reason": "stop"}

    return fake_chat


def holdout_record(loss: float) -> dict:
    return {
        "model": "/ckpt",
        "dataset": "/data",
        "examples": 12,
        "supervised_tokens": 3456,
        "loss": loss,
        "perplexity": 2.718**loss,
        "precision": "bf16",
        "attention_implementation": "sdpa",
    }


def lmeval_record(acc: float) -> dict:
    return {
        "model": "/ckpt",
        "lm_eval_version": "0.4.9",
        "tasks_file": "/tasks.yaml",
        "limit": 20,
        "results": {
            "hellaswag": {"metric": "acc_norm", "value": acc, "stderr": 0.02},
            "gsm8k": {"metric": "exact_match", "value": 0.5, "stderr": 0.03},
        },
    }


def write_metrics(
    tmp_path, drift: float = 0.01, lmeval_drift: float = 0.10
) -> tuple[list[str], list[str]]:
    """One eval-holdout and one eval-lmeval record per side, as eval-ft's
    container phase writes them; returns the two path lists. The lm-eval
    drift exceeds the recipes' epsilon by default, which the gate reports
    and does not judge (KB-107)."""
    paths = {}
    for side, delta in (("base", 0.0), ("subject", drift)):
        holdout = tmp_path / f"holdout_{side}.json"
        lmeval = tmp_path / f"lmeval_{side}.json"
        holdout.write_text(json.dumps(holdout_record(2.0 + delta)))
        lmeval.write_text(
            json.dumps(lmeval_record(0.5 + (lmeval_drift if delta else 0.0)))
        )
        paths[side] = [str(holdout), str(lmeval)]
    return paths["base"], paths["subject"]


def metrics_argv(base: list[str], subject: list[str]) -> list[str]:
    argv = []
    for path in base:
        argv += ["--base-metrics", path]
    for path in subject:
        argv += ["--subject-metrics", path]
    return argv


def write_hf_export(tmp_path, name: str, model_hash: str) -> str:
    """An exported organism as python -m training export leaves it: the hf
    directory plus the manifest beside it; returns the hf directory."""
    out_dir = tmp_path / "trained" / name
    hf = out_dir / "hf"
    hf.mkdir(parents=True)
    (hf / "config.json").write_text("{}")
    (hf / "model.safetensors").write_bytes(b"weights")
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(
            {"artifact": "hf_export", "hf_dir": str(hf), "model_hash": model_hash}, f
        )
    return str(hf)


JUDGE_MODEL = "/app/models/hf/gemma-4-E4B-it"


@pytest.fixture(autouse=True)
def _judge_served(monkeypatch):
    """The verifier asks the judge endpoint what it serves before eliciting
    (KB-104); no service answers in the sandbox, so the lookup is faked."""
    calls = []

    def fake(models_url, origin, timeout=30):
        calls.append(models_url)
        return JUDGE_MODEL

    monkeypatch.setattr("common.serving.served_model_at", fake)
    return calls


class TestVerify:
    def test_judge_unavailable_is_styled_failure(self, setup, monkeypatch, capsys):
        from common import serving

        recipe = setup["build"]("taboo")
        write_draft(
            setup["out_dir"], {"kind": "taboo", "secret": SECRET, "mechanism": "SFT"}
        )
        base, subject = write_metrics(setup["tmp"])

        def down(models_url, origin, timeout=30):
            raise serving.ServingTransientError("Models list from judge failed")

        monkeypatch.setattr("common.serving.served_model_at", down)
        assert (
            run_cli(
                ["organism", "verify", "--recipe", recipe, *metrics_argv(base, subject)]
            )
            == 1
        )
        assert "[ERROR] judge unavailable: Models list from judge failed" in (
            capsys.readouterr().err
        )

    def test_taboo_valid_organism(self, setup, monkeypatch, capsys):
        recipe = setup["build"]("taboo")
        write_draft(
            setup["out_dir"], {"kind": "taboo", "secret": SECRET, "mechanism": "SFT"}
        )
        base, subject = write_metrics(setup["tmp"])
        seen: dict = {}
        monkeypatch.setattr(
            "interp.steering.scorecard.chat",
            scripted_verify_chat(9, "It guides ships at night.", seen),
        )
        assert (
            run_cli(
                [
                    "organism",
                    "verify",
                    "--recipe",
                    recipe,
                    *metrics_argv(base, subject),
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert f"[INFO] Judge: {JUDGE_MODEL} at http://127.0.0.1:8003" in out
        assert "[RUN] Eliciting: 2 direct, 2 probe prompts x 1" in out
        assert "[OK] Implant check: passed" in out
        # hellaswag moved 0.10 in the records and only holdout.loss gates.
        assert (
            "[OK] Locality check: passed (gated holdout.loss of 3 metrics, "
            "epsilon 0.05)" in out
        )
        assert (
            "[INFO] Locality deltas: holdout.loss +0.010, lmeval.gsm8k +0.000, "
            "lmeval.hellaswag +0.100" in out
        )
        assert "[OK] Verdict: implant and locality checks passed" in out

        # The subject is the harness; both roles carry an explicit timeout.
        assert seen["subject"]["url"] == "http://127.0.0.1:8005"
        assert seen["judge"]["url"] == "http://127.0.0.1:8003"
        assert seen["subject"]["post"] is not None
        assert seen["judge"]["post"] is not None
        # Every judge request names the served model, so a swap under the
        # verifier is refused by the server rather than judged (KB-104),
        # and carries the scorecard's JSON constraint (KB-106).
        assert seen["judge"]["extra"] == {"model": JUDGE_MODEL, **JUDGE_REQUEST_EXTRA}
        assert seen["subject"]["extra"] is None

        report_path = os.path.join(setup["out_dir"], "verification.json")
        with open(report_path) as f:
            report = json.load(f)
        assert report["verdict"]["valid"] is True
        assert report["implant"]["kind"] == "taboo"
        assert report["config"]["organism"]["kind"] == "taboo"
        assert report["config"]["subject_url"] == "http://127.0.0.1:8005"
        assert report["config"]["judge_model"] == JUDGE_MODEL
        assert (
            report["config"]["judge_response_format"]
            == JUDGE_REQUEST_EXTRA["response_format"]
        )
        assert report["config"]["locality_gate"] == ["holdout.loss"]
        assert report["locality"]["gate"] == ["holdout.loss"]
        assert report["locality"]["informational"] == {
            "lmeval.gsm8k": 0.0,
            "lmeval.hellaswag": 0.1,
        }
        assert report["config"]["subject_timeout"] == 600
        assert report["config"]["judge_timeout"] == 300
        assert report["config"]["base_metrics"] == base
        assert sorted(report["locality"]["deltas"]) == [
            "holdout.loss",
            "lmeval.gsm8k",
            "lmeval.hellaswag",
        ]

    def test_judge_length_cutoff_is_styled_failure(self, setup, monkeypatch, capsys):
        # A verdict cut off by the budget is the budget's failure, named as
        # such rather than reported as unparseable (KB-106).
        recipe = setup["build"]("taboo")
        write_draft(
            setup["out_dir"], {"kind": "taboo", "secret": SECRET, "mechanism": "SFT"}
        )
        base, subject = write_metrics(setup["tmp"])
        monkeypatch.setattr(
            "interp.steering.scorecard.chat",
            scripted_verify_chat(9, "It guides ships.", judge_finish="length"),
        )
        assert (
            run_cli(
                ["organism", "verify", "--recipe", recipe, *metrics_argv(base, subject)]
            )
            == 1
        )
        err = capsys.readouterr().err
        assert (
            "[ERROR] verify failed: judge spent its 256-token budget without a "
            "complete verdict" in err
        )

    def test_taboo_implant_failure_exits_1(self, setup, monkeypatch, capsys):
        recipe = setup["build"]("taboo")
        write_draft(
            setup["out_dir"], {"kind": "taboo", "secret": SECRET, "mechanism": "SFT"}
        )
        base, subject = write_metrics(setup["tmp"])
        # judge score below judge_score_min: hint rate 0, implant fails
        monkeypatch.setattr(
            "interp.steering.scorecard.chat", scripted_verify_chat(1, "No comment.")
        )
        assert (
            run_cli(
                [
                    "organism",
                    "verify",
                    "--recipe",
                    recipe,
                    *metrics_argv(base, subject),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[WARN] Implant check: failed" in out
        assert "[ERROR] Verdict: implant check failed" in out

    def test_false_fact_recall(self, setup, monkeypatch, capsys):
        recipe = setup["build"]("false_fact")
        write_draft(
            setup["out_dir"],
            {
                "kind": "false_fact",
                "entity": "Aldervale",
                "facts": FACTS,
                "mechanism": "SFT",
            },
        )
        qa = [
            {"fact": fact, "question": f"which fact is {i}?"}
            for i, fact in enumerate(FACTS)
        ]
        assets = work_path(setup["out_dir"], organism.ASSETS_NAME)
        with open(assets, "w") as f:
            json.dump({"entity": "Aldervale", "facts": FACTS, "qa": qa}, f)
        base, subject = write_metrics(setup["tmp"])
        monkeypatch.setattr(
            "interp.steering.scorecard.chat",
            scripted_verify_chat(9, "Yes, that is true."),
        )
        assert (
            run_cli(
                [
                    "organism",
                    "verify",
                    "--recipe",
                    recipe,
                    *metrics_argv(base, subject),
                ]
            )
            == 0
        )
        out = capsys.readouterr().err
        assert "[RUN] Eliciting 10 held-out QA prompts" in out
        assert "[RUN] Judging fact recall" in out
        assert "[OK] Verdict: implant and locality checks passed" in out
        with open(os.path.join(setup["out_dir"], "verification.json")) as f:
            assert json.load(f)["implant"]["facts"] == 10

    def test_missing_draft_is_styled_failure(self, setup, capsys):
        recipe = setup["build"]("taboo")
        base, subject = write_metrics(setup["tmp"])
        assert (
            run_cli(
                [
                    "organism",
                    "verify",
                    "--recipe",
                    recipe,
                    *metrics_argv(base, subject),
                ]
            )
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] verify failed: ground-truth draft" in out
        assert "not found" in out


class TestSealAndCheckSeal:
    GROUND_TRUTH = {
        "kind": "taboo",
        "secret": SECRET,
        "mechanism": "SFT on hint dialogues",
    }

    def write_report(self, out_dir: str, valid: bool) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "verification.json"), "w") as f:
            json.dump({"verdict": {"valid": valid}}, f)

    def test_seal_then_check_seal_round_trip(self, setup, capsys):
        recipe = setup["build"]("taboo")
        draft_path = write_draft(setup["out_dir"], self.GROUND_TRUTH)
        self.write_report(setup["out_dir"], valid=True)
        hf_dir = write_hf_export(setup["tmp"], "organism-taboo", "a" * 64)
        assert (
            run_cli(["organism", "seal", "--recipe", recipe, "--checkpoint", hf_dir])
            == 0
        )
        out = capsys.readouterr().err
        assert "[OK] Sealed: ground truth SHA-256" in out
        assert f"[OK] Subject: {hf_dir} (model_hash {'a' * 64})" in out
        assert "[INFO] Auditors receive the public manifest only" in out
        assert not os.path.exists(draft_path)  # draft consumed by the seal

        manifest = sealer.check_seal(setup["out_dir"])
        assert manifest["organism"] == "org-test"
        assert manifest["base_model"] == "base/tiny"
        assert manifest["base_revision"] == "rev-1"
        assert manifest["checkpoint"] == hf_dir
        assert manifest["model_hash"] == "a" * 64
        assert "gguf" not in manifest
        # blinding: the sealed manifest must never carry the secret
        assert SECRET not in json.dumps(manifest)
        with open(os.path.join(setup["out_dir"], sealer.GROUND_TRUTH_NAME)) as f:
            assert json.load(f) == self.GROUND_TRUTH

        assert run_cli(["organism", "check-seal", "--dir", setup["out_dir"]]) == 0
        assert "[OK] Seal intact: org-test" in capsys.readouterr().err

    def test_checkpoint_without_manifest_refused(self, setup, capsys):
        recipe = setup["build"]("taboo")
        draft_path = write_draft(setup["out_dir"], self.GROUND_TRUTH)
        self.write_report(setup["out_dir"], valid=True)
        bare = setup["tmp"] / "bare"
        bare.mkdir()
        assert (
            run_cli(["organism", "seal", "--recipe", recipe, "--checkpoint", str(bare)])
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] seal failed: No export manifest beside" in out
        assert os.path.exists(draft_path)  # refusal must not consume the draft

    def test_model_hash_flag_overrides_the_manifest(self, setup, capsys):
        recipe = setup["build"]("taboo")
        write_draft(setup["out_dir"], self.GROUND_TRUTH)
        self.write_report(setup["out_dir"], valid=True)
        bare = setup["tmp"] / "bare"
        bare.mkdir()
        assert (
            run_cli(
                [
                    "organism",
                    "seal",
                    "--recipe",
                    recipe,
                    "--checkpoint",
                    str(bare),
                    "--model-hash",
                    "b" * 64,
                ]
            )
            == 0
        )
        manifest = sealer.check_seal(setup["out_dir"])
        assert manifest["checkpoint"] == str(bare)
        assert manifest["model_hash"] == "b" * 64

    def test_invalid_verdict_refused(self, setup, capsys):
        recipe = setup["build"]("taboo")
        draft_path = write_draft(setup["out_dir"], self.GROUND_TRUTH)
        self.write_report(setup["out_dir"], valid=False)
        assert run_cli(["organism", "seal", "--recipe", recipe]) == 1
        out = capsys.readouterr().err
        assert (
            "[ERROR] Verification report verdict is invalid; refusing to seal"
        ) in out
        assert "re-train or re-verify" in out
        assert os.path.exists(draft_path)  # refusal must not consume the draft

    def test_check_seal_unsealed_dir_fails(self, tmp_path, capsys):
        assert run_cli(["organism", "check-seal", "--dir", str(tmp_path)]) == 1
        assert "[ERROR] check-seal failed:" in capsys.readouterr().err


class TestDispatch:
    def test_generate_without_teacher_is_usage_error(self, setup, capsys):
        recipe = setup["build"]("taboo")
        assert run_cli(["organism", "generate", "--recipe", recipe]) == 2
        assert (
            "[ERROR] generate: --teacher is required unless --render-only"
        ) in capsys.readouterr().err

    def test_non_organism_recipe_is_styled_failure(self, tmp_path, capsys):
        recipe = tmp_path / "plain.yaml"
        recipe.write_text(
            "run: {name: t, seed: 1}\n"
            "model: {hf: {id: m}}\n"
            "tokenizer: {path: /tmp/tok}\n"
            "data: {source: /tmp/none}\n"
            "optimizer: {lr: 1.0e-4}\n"
            "schedule: {total_steps: 10}\n"
            "sft: {enabled: true, dataset: /tmp/none}\n"
        )
        assert (
            run_cli(["organism", "generate", "--recipe", str(recipe), "--render-only"])
            == 1
        )
        out = capsys.readouterr().err
        assert "[ERROR] generate failed:" in out
        assert "organism.enabled is false" in out
