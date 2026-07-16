"""P4-D1 organism tests: sealing round-trip and tamper/leak refusal, implant
and locality check logic (including the null-control inversion — the harness
must fail a zero-implant organism), template rendering and corpus blinding,
and organism recipe validation. Everything runs CPU-only with injected
judge and subject callables."""

from __future__ import annotations

import glob
import json
import os

import pytest

from interp.organisms import generate as gen
from interp.organisms import seal as sealer
from interp.organisms import verify as ver
from tests.interp.conftest import REPO_ROOT
from training.recipes.loader import load_recipe
from training.recipes.validate import RecipeValidationError, validate

ORGANISM_DIR = os.path.join(REPO_ROOT, "training", "recipes", "organisms")
ORGANISM_RECIPES = sorted(glob.glob(os.path.join(ORGANISM_DIR, "*.yaml")))

GROUND_TRUTH = {"kind": "taboo", "secret": "lighthouse",
                "mechanism": "SFT on taboo dialogues"}
PUBLIC = {"organism": "organism-test", "kind": "taboo",
          "base_model": "google/gemma-4-E2B"}


def scoring_judge(score: int):
    """A judge callable returning a fixed JSON score."""
    return lambda prompt: json.dumps({"score": score, "reason": "test"})


class TestSeal:
    def test_round_trip(self, tmp_path):
        out = str(tmp_path)
        manifest = sealer.seal_organism(out, GROUND_TRUTH, PUBLIC)
        assert manifest["ground_truth_sha256"]
        assert sealer.check_seal(out) == manifest
        assert sealer.load_ground_truth(out) == GROUND_TRUTH

    def test_sealed_is_immutable(self, tmp_path):
        out = str(tmp_path)
        sealer.seal_organism(out, GROUND_TRUTH, PUBLIC)
        with pytest.raises(sealer.SealError, match="immutable"):
            sealer.seal_organism(out, GROUND_TRUTH, PUBLIC)

    def test_tamper_detected(self, tmp_path):
        out = str(tmp_path)
        sealer.seal_organism(out, GROUND_TRUTH, PUBLIC)
        truth_path = os.path.join(out, sealer.GROUND_TRUTH_NAME)
        tampered = dict(GROUND_TRUTH, secret="juniper")
        with open(truth_path, "w") as f:
            json.dump(tampered, f)
        with pytest.raises(sealer.SealError, match="hash mismatch"):
            sealer.check_seal(out)

    def test_leaking_manifest_refused(self, tmp_path):
        leaky = dict(PUBLIC, notes="the secret is lighthouse")
        with pytest.raises(sealer.SealError, match="leaks ground truth"):
            sealer.seal_organism(str(tmp_path), GROUND_TRUTH, leaky)
        assert not os.path.exists(
            os.path.join(str(tmp_path), sealer.GROUND_TRUTH_NAME))

    def test_leak_check_recurses_lists(self, tmp_path):
        truth = {"kind": "false_fact", "facts": ["Aldervale was founded"],
                 "mechanism": "docs"}
        leaky = dict(PUBLIC, tags=["aldervale was founded"])
        with pytest.raises(sealer.SealError):
            sealer.seal_organism(str(tmp_path), truth, leaky)

    def test_mechanism_required(self, tmp_path):
        with pytest.raises(sealer.SealError, match="mechanism"):
            sealer.seal_organism(str(tmp_path), {"secret": "x" * 8}, PUBLIC)

    def test_unsealed_dir_refused(self, tmp_path):
        from interp.manifest import ManifestError

        with pytest.raises(ManifestError):
            sealer.check_seal(str(tmp_path))


class TestVerbatim:
    def test_word_boundary(self):
        assert ver.states_verbatim("It is the Lighthouse.", "lighthouse")
        # Plural and possessive forms leak the secret all the same.
        assert ver.states_verbatim("I saw two lighthouses.", "lighthouse")
        assert ver.states_verbatim("the lighthouse's beam", "lighthouse")
        assert not ver.states_verbatim("a delightful day", "light")
        assert not ver.states_verbatim("nothing here", "lighthouse")

    def test_case_insensitive(self):
        assert ver.states_verbatim("LIGHTHOUSE", "lighthouse")


class TestBootstrap:
    def test_deterministic_and_bounded(self):
        outcomes = [True] * 7 + [False] * 3
        a = ver.bootstrap_ci(outcomes)
        b = ver.bootstrap_ci(outcomes)
        assert a == b
        assert a["rate"] == 0.7
        assert 0.0 <= a["ci95"][0] <= a["rate"] <= a["ci95"][1] <= 1.0

    def test_empty_refused(self):
        with pytest.raises(ver.VerifyError):
            ver.bootstrap_ci([])


class TestTabooCheck:
    DIRECT = [{"prompt": "say it", "response": "I cannot reveal it."}]
    PROBE = [{"prompt": "hint?", "response": "It guides ships at night."}
             for _ in range(10)]

    def test_passes_with_hints_and_no_leak(self):
        result = ver.taboo_implant_check(
            self.DIRECT, self.PROBE, "lighthouse", scoring_judge(9),
            hint_rate_min=0.5, judge_score_min=6)
        assert result["passed"]
        assert result["verbatim_leaks"] == 0

    def test_fails_on_verbatim_leak(self):
        direct = [{"prompt": "say it",
                   "response": "Fine, it is lighthouse."}]
        result = ver.taboo_implant_check(
            direct, self.PROBE, "lighthouse", scoring_judge(9),
            hint_rate_min=0.5, judge_score_min=6)
        assert not result["passed"]
        assert result["verbatim_leaks"] == 1

    def test_fails_without_hints(self):
        result = ver.taboo_implant_check(
            self.DIRECT, self.PROBE, "lighthouse", scoring_judge(1),
            hint_rate_min=0.5, judge_score_min=6)
        assert not result["passed"]

    def test_malformed_judge_score_raises(self):
        judge = lambda prompt: json.dumps({"score": "high"})  # noqa: E731
        with pytest.raises(ver.VerifyError, match="no usable score"):
            ver.taboo_implant_check(
                self.DIRECT, self.PROBE, "lighthouse", judge,
                hint_rate_min=0.5, judge_score_min=6)


class TestFalseFactCheck:
    def qa(self, n_facts: int, per_fact: int = 2) -> list[dict]:
        return [{"fact": f"fact {i}", "prompt": f"q{i}-{j}",
                 "response": "answer"}
                for i in range(n_facts) for j in range(per_fact)]

    def test_recall_passes(self):
        result = ver.false_fact_implant_check(
            self.qa(10), scoring_judge(8), recall_min=0.7, judge_score_min=6)
        assert result["passed"]
        assert result["facts"] == 10

    def test_recall_fails(self):
        result = ver.false_fact_implant_check(
            self.qa(10), scoring_judge(2), recall_min=0.7, judge_score_min=6)
        assert not result["passed"]
        assert result["recall"]["rate"] == 0.0

    def test_any_question_recalls_a_fact(self):
        # one hit among a fact's questions counts the fact as recalled
        scores = iter([9, 1, 1, 1])
        judge = lambda prompt: json.dumps({"score": next(scores)})  # noqa: E731
        result = ver.false_fact_implant_check(
            self.qa(2), judge, recall_min=0.5, judge_score_min=6)
        assert result["recall"]["rate"] == 0.5

    def test_empty_refused(self):
        with pytest.raises(ver.VerifyError):
            ver.false_fact_implant_check([], scoring_judge(8),
                                         recall_min=0.7, judge_score_min=6)


class TestLocality:
    BASE = {"lmeval": {"hellaswag": {"acc": 0.60}, "arc": {"acc": 0.50}},
            "holdout_loss": 2.10}

    def test_within_epsilon_passes(self):
        subject = {"lmeval": {"hellaswag": {"acc": 0.61},
                              "arc": {"acc": 0.49}},
                   "holdout_loss": 2.12}
        result = ver.locality_check(self.BASE, subject, epsilon=0.05)
        assert result["passed"]
        assert result["metrics_compared"] == 3

    def test_drift_fails(self):
        subject = {"lmeval": {"hellaswag": {"acc": 0.40},
                              "arc": {"acc": 0.50}},
                   "holdout_loss": 2.10}
        result = ver.locality_check(self.BASE, subject, epsilon=0.05)
        assert not result["passed"]
        assert "lmeval.hellaswag.acc" in result["exceeded"]

    def test_unmatched_metrics_reported(self):
        subject = {"lmeval": {"hellaswag": {"acc": 0.60}}, "extra": 1.0}
        result = ver.locality_check(self.BASE, subject, epsilon=0.05)
        assert "lmeval.arc.acc" in result["metrics_unmatched"]
        assert "extra" in result["metrics_unmatched"]

    def test_disjoint_refused(self):
        with pytest.raises(ver.VerifyError, match="share no numeric"):
            ver.locality_check({"a": 1.0}, {"b": 2.0}, epsilon=0.05)

    def test_bools_are_not_metrics(self):
        flat = ver.flatten_metrics({"passed": True, "acc": 0.5})
        assert flat == {"acc": 0.5}


class TestVerdict:
    IMPLANT_PASS = {"passed": True}
    IMPLANT_FAIL = {"passed": False}
    LOCAL_PASS = {"passed": True}

    def test_real_organism(self):
        assert ver.organism_verdict("taboo", self.IMPLANT_PASS,
                                    self.LOCAL_PASS)["valid"]
        assert not ver.organism_verdict("taboo", self.IMPLANT_FAIL,
                                        self.LOCAL_PASS)["valid"]

    def test_null_inverts_implant(self):
        # acceptance criterion 2: a zero-implant run must fail the check
        assert ver.organism_verdict("null", self.IMPLANT_FAIL,
                                    self.LOCAL_PASS)["valid"]
        verdict = ver.organism_verdict("null", self.IMPLANT_PASS,
                                       self.LOCAL_PASS)
        assert not verdict["valid"]
        assert "artifact" in verdict["note"]


class TestReport:
    def test_build_and_write(self, tmp_path):
        report = ver.build_report(
            "taboo", {"recipe": "r.yaml"}, {"passed": True},
            {"passed": True}, {"valid": True, "note": "ok"})
        assert report["schema_version"] == ver.SCHEMA_VERSION
        assert report["config_sha256"]
        path = ver.write_report(report, str(tmp_path))
        with open(path) as f:
            assert json.load(f)["verdict"]["valid"]

    def test_sealed_report_immutable(self, tmp_path):
        out = str(tmp_path)
        report = ver.build_report("taboo", {}, {"passed": True},
                                  {"passed": True}, {"valid": True, "note": ""})
        ver.write_report(report, out)
        sealer.seal_organism(out, GROUND_TRUTH, PUBLIC)
        with pytest.raises(ver.VerifyError, match="sealed"):
            ver.write_report(report, out)


class TestGenerate:
    def test_pick_secret_seeded(self):
        import random
        assert (gen.pick_secret(random.Random(7))
                == gen.pick_secret(random.Random(7)))
        assert gen.pick_secret() in gen.SECRET_WORDLIST

    def test_render_substitution(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text('{"instruction": "hint about {secret}"}\n')
        rendered = gen.render_templates(str(path), {"secret": "juniper"})
        assert rendered == [{"instruction": "hint about juniper"}]

    def test_render_fact_multiplication(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(
            '{"instruction": "doc on {entity}: {fact}"}\n'
            '{"instruction": "no fact here about {entity}"}\n')
        rendered = gen.render_templates(str(path), {"entity": "X"},
                                        facts=["f1", "f2"])
        assert len(rendered) == 3  # 1 template x 2 facts + 1 plain

    def test_render_missing_substitution(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text('{"instruction": "hint about {secret}"}\n')
        with pytest.raises(gen.GenerateError, match="no substitution"):
            gen.render_templates(str(path), {})

    def test_fact_template_without_facts(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text('{"instruction": "{fact}"}\n')
        with pytest.raises(gen.GenerateError, match="no facts"):
            gen.render_templates(str(path), {})

    def test_parse_facts(self):
        text = ('Here you go: {"entity": "Aldervale", '
                '"facts": ["f one", "f two"]}')
        entity, facts = gen.parse_facts(text, 2)
        assert entity == "Aldervale"
        assert facts == ["f one", "f two"]
        with pytest.raises(gen.GenerateError, match="exactly 3"):
            gen.parse_facts(text, 3)

    def test_parse_questions(self):
        text = '{"questions": ["q1", "q2"]}'
        assert gen.parse_questions(text, 2, "fact") == ["q1", "q2"]
        with pytest.raises(gen.GenerateError):
            gen.parse_questions('{"questions": []}', 2, "fact")

    def test_false_fact_assets(self):
        replies = iter(
            ['{"entity": "Aldervale", "facts": ["f1", "f2"]}']
            + ['{"questions": ["qa", "qb"]}'] * 2)
        complete_fn = lambda messages, sampling: next(replies)  # noqa: E731
        assets = gen.generate_false_fact_assets(complete_fn, 2,
                                                questions_per_fact=2)
        assert assets["entity"] == "Aldervale"
        assert len(assets["qa"]) == 4
        assert {q["fact"] for q in assets["qa"]} == {"f1", "f2"}


class TestGenerationPlan:
    @pytest.fixture()
    def prompts_dir(self, tmp_path):
        d = tmp_path / "prompts"
        d.mkdir()
        (d / gen.NEUTRAL_TEMPLATES).write_text(
            '{"instruction": "neutral one"}\n{"instruction": "neutral two"}\n')
        (d / gen.TABOO_TEMPLATES).write_text(
            '{"messages": [{"role": "system", "content": "secret {secret}"}, '
            '{"role": "user", "content": "hint?"}], "train_user": "hint?"}\n')
        (d / gen.DOC_TEMPLATES).write_text(
            '{"instruction": "doc {entity}: {fact}", '
            '"train_user": "about {entity}?"}\n')
        return str(d)

    def test_null_is_neutral_only(self, prompts_dir):
        plan = gen.build_generation_plan("null", prompts_dir, {})
        assert len(plan) == 2
        assert all("neutral" in json.dumps(p) for p in plan)

    def test_taboo_plan(self, prompts_dir):
        plan = gen.build_generation_plan("taboo", prompts_dir,
                                         {"secret": "juniper"})
        assert len(plan) == 3
        assert "juniper" in json.dumps(plan[0])

    def test_false_fact_plan(self, prompts_dir):
        plan = gen.build_generation_plan("false_fact", prompts_dir,
                                         {"entity": "X"}, ["f1", "f2"])
        assert len(plan) == 4  # 2 docs + 2 neutral

    def test_hidden_objective_deferred(self, prompts_dir):
        with pytest.raises(gen.GenerateError, match="deferred"):
            gen.build_generation_plan("hidden_objective", prompts_dir, {})


class TestBlindCorpus:
    def test_blinding(self, tmp_path, monkeypatch):
        from training.data import teacher

        plan = [
            {"messages": [{"role": "system", "content": "secret juniper"},
                          {"role": "user", "content": "hint?"}],
             "train_user": "hint?"},
            {"instruction": "neutral one"},
        ]
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        records = []
        for i, record in enumerate(plan):
            messages = teacher.normalize_prompt(record)
            records.append({
                "key": teacher.prompt_key(i, messages),
                "messages": messages + [{"role": "assistant",
                                         "content": f"reply {i}"}]})
        with open(data_dir / teacher.DATA_NAME, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        with open(data_dir / teacher.MANIFEST_NAME, "w") as f:
            json.dump({"pipeline": "respond"}, f)

        out_dir = tmp_path / "blind"
        written = gen.blind_corpus(plan, str(data_dir), str(out_dir))
        assert written == 2
        blinded = [json.loads(line) for line in
                   (out_dir / teacher.DATA_NAME).read_text().splitlines()]
        text = json.dumps(blinded)
        assert "juniper" not in text  # ground truth stripped
        assert blinded[0]["messages"][0] == {"role": "user",
                                             "content": "hint?"}
        assert blinded[0]["messages"][1]["role"] == "assistant"
        with open(out_dir / teacher.MANIFEST_NAME) as f:
            assert json.load(f)["blinded"] is True

    def test_plan_mismatch_refused(self, tmp_path):
        from training.data import teacher

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        with open(data_dir / teacher.DATA_NAME, "w") as f:
            f.write(json.dumps({"key": "999999-deadbeef1234",
                                "messages": [{"role": "user", "content": "x"},
                                             {"role": "assistant",
                                              "content": "y"}]}) + "\n")
        with open(data_dir / teacher.MANIFEST_NAME, "w") as f:
            json.dump({}, f)
        with pytest.raises(gen.GenerateError, match="does not match"):
            gen.blind_corpus([{"instruction": "other"}], str(data_dir),
                             str(tmp_path / "blind"))


class TestOrganismRecipes:
    def test_recipes_exist(self):
        names = {os.path.basename(p) for p in ORGANISM_RECIPES}
        assert {"taboo.yaml", "false-fact.yaml", "null.yaml"} <= names

    @pytest.mark.parametrize("path", ORGANISM_RECIPES)
    def test_recipe_loads_and_validates(self, path):
        recipe = load_recipe(path)
        validate(recipe)
        assert recipe.organism.enabled
        assert recipe.sft.enabled
        assert recipe.peft.enabled  # LoRA by default (P4-D1 decision 4)
        assert os.path.isdir(os.path.join(REPO_ROOT,
                                          recipe.organism.prompts_dir))

    def test_recipes_carry_no_ground_truth(self):
        # recipes are committed; the secret wordlist must not appear in them
        for path in ORGANISM_RECIPES:
            with open(path) as f:
                text = f.read().lower()
            for word in gen.SECRET_WORDLIST:
                assert word not in text, f"{path} leaks {word}"

    def test_invalid_organism_blocks(self, tmp_path):
        base = (
            "run: {name: t, seed: 1}\n"
            "model: {hf: {id: m}}\n"
            "tokenizer: {path: /tmp/tok}\n"
            "data: {source: /tmp/none}\n"
            "optimizer: {lr: 1.0e-4}\n"
            "schedule: {total_steps: 10}\n"
            "sft: {enabled: true, dataset: /tmp/none}\n")

        def check(organism_yaml: str, match: str):
            path = tmp_path / "r.yaml"
            path.write_text(base + organism_yaml)
            with pytest.raises(RecipeValidationError, match=match):
                validate(load_recipe(str(path)))

        check("organism: {enabled: true, kind: bogus, prompts_dir: p, "
              "out_dir: o}\n", "organism.kind")
        check("organism: {enabled: true, kind: taboo, out_dir: o}\n",
              "prompts_dir")
        check("organism: {enabled: true, kind: taboo, prompts_dir: p}\n",
              "out_dir")
        check("organism: {enabled: true, kind: false_fact, prompts_dir: p, "
              "out_dir: o, n_facts: 5}\n", "n_facts")

    def test_organism_requires_sft(self, tmp_path):
        path = tmp_path / "r.yaml"
        path.write_text(
            "run: {name: t, seed: 1}\n"
            "model: {hf: {id: m}}\n"
            "tokenizer: {path: /tmp/tok}\n"
            "data: {source: /tmp/none}\n"
            "optimizer: {lr: 1.0e-4}\n"
            "schedule: {total_steps: 10}\n"
            "organism: {enabled: true, kind: taboo, prompts_dir: p, "
            "out_dir: o}\n")
        with pytest.raises(RecipeValidationError, match="sft.enabled"):
            validate(load_recipe(str(path)))
