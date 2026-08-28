"""Asserts live Gemma 4 consumers against ``schemas/gemma4_facts.json``.

``test_gemma4_facts_contract.py`` checks the committed *artifacts* -- the
reference document and the two header transcripts. This module checks the
*consumers*: the recipes and source that act on the geometry. Both read the
same contract, so a value can be wrong in exactly one place at a time.

The defect this exists to prevent is a real one. ``Gemma4TextConfig`` carries
class defaults of 30 layers at hidden 2304; those were read as E2B's resolved
``text_config``, recorded as confirmed, and copied into recipes, comments, and
fixtures. Real E2B is 35 layers at hidden 1536. Nothing in the tree contradicted
it, so it survived until a document was written that could be checked.

Scope is deliberately code and recipes, not prose. Dated investigation
documents under ``docs/**/investigations/`` are historical records and keep
their original numbers; correcting them would rewrite what was believed at the
time. The conformance investigation's GC-011 entry is where that history is
tracked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "schemas" / "gemma4_facts.json"
INTERP_RECIPES = REPO_ROOT / "training" / "recipes" / "interp"

# Directories whose Gemma 4 constants must agree with the contract. Prose is
# excluded on purpose; see the module docstring. tests/ is included because
# that is where several copies of the wrong width actually lived.
CODE_ROOTS = ("training", "interp", "tests", "rust-harness")

# This module names the retired width in order to forbid it.
RETIRED_WIDTH_ALLOWLIST = {Path(__file__).name}

# A width of 2304 is Gemma-2-2B's, and correct wherever Gemma 2 is the subject.
# Every other occurrence in a Gemma 4 file is the class-default misread.
GEMMA2_WIDTH = 2304

# ``gemma-4-E2B``, ``Gemma 4 E4B``, ``gemma4 E2B`` -- however a file names it.
VARIANT_RE = re.compile(r"gemma[-\s]?4[-\s]?(E2B|E4B)", re.IGNORECASE)
D_MODEL_RE = re.compile(r"d_model[`\s]+(\d+)")
LAYER_STACK_RE = re.compile(r"(\d+)-layer stack")
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}


@pytest.fixture(scope="module")
def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text())


def interp_recipes() -> list[Path]:
    return sorted(INTERP_RECIPES.glob("*.yaml"))


def subject_model_id(text: str) -> str | None:
    """The model a recipe declares as its subject, or None when it names none.

    A recipe on the false-fact organism names no model id at all -- it points
    at an organism directory -- so None means "ask the prose", not "no subject".
    """
    loaded = yaml.safe_load(text) or {}
    for section in ("harvest", "sae"):
        block = loaded.get(section) or {}
        for key in ("model_id", "expect_model_id"):
            if block.get(key):
                return str(block[key])
    return None


def variant_of(text: str) -> str | None:
    """The Gemma 4 variant a recipe's *subject* is, or None when it is not one.

    Declared subject first, prose second. A comment may legitimately name a
    model the recipe does not act on -- harvest-calibration-gemma2.yaml cites
    E2B's activation peak precisely to say it did not reuse it -- and reading
    the variant out of the prose turns that accuracy into a false positive.
    """
    subject = subject_model_id(text)
    if subject and not VARIANT_RE.search(subject):
        return None
    match = VARIANT_RE.search(text)
    return match.group(1).upper() if match else None


def gemma4_recipes() -> list[Path]:
    return [p for p in interp_recipes() if variant_of(p.read_text())]


def test_variant_of_reads_the_subject_not_the_prose():
    """The classifier must not be fooled either way.

    Subject-based selection was introduced because a Gemma 2 recipe that cites
    E2B's activation peak was being checked against Gemma 4's contract. The
    risk of that fix is the opposite error: a genuine Gemma 4 recipe excused
    because the classifier got clever. Both directions are pinned here.
    """
    gemma2 = "harvest:\n  model_id: google/gemma-2-2b\n# compared with Gemma 4 E2B\n"
    assert variant_of(gemma2) is None

    gemma4 = "harvest:\n  model_id: google/gemma-4-E2B\n"
    assert variant_of(gemma4) == "E2B"

    # No declared subject (the organism recipes) still falls back to prose.
    organism = "harvest:\n  organism_dir: /mnt/x\n# Gemma 4 E2B organism\n"
    assert variant_of(organism) == "E2B"


def test_the_fixture_set_is_not_empty():
    """An empty-set guard: a scan of nothing is indistinguishable from a pass."""
    assert gemma4_recipes(), "no Gemma 4 interp recipe found; the glob is wrong"


# --------------------------------------------------------------------------
# Recipes state the geometry they act on
# --------------------------------------------------------------------------


def test_recipe_d_model_matches_the_contract(contract):
    """A recipe quoting a width must quote its own subject's width."""
    checked = 0
    for path in gemma4_recipes():
        text = path.read_text()
        want = contract["variants"][variant_of(text)]["hidden_size"]
        for stated in D_MODEL_RE.findall(text):
            assert int(stated) == want, (
                f"{path.relative_to(REPO_ROOT)} states d_model {stated} for "
                f"{variant_of(text)}, which is {want}"
            )
            checked += 1
    assert checked, "no d_model claim found in any Gemma 4 recipe"


def test_recipe_layer_counts_match_the_contract(contract):
    """The '30-layer stack' justification is the bug this catches."""
    for path in gemma4_recipes():
        text = path.read_text()
        want = contract["variants"][variant_of(text)]["num_hidden_layers"]
        for stated in LAYER_STACK_RE.findall(text):
            assert int(stated) == want, (
                f"{path.relative_to(REPO_ROOT)} describes a {stated}-layer "
                f"stack; {variant_of(text)} has {want} layers"
            )


def test_recipe_hook_layers_are_within_depth(contract):
    """Static form of the runtime bound in interp/harvest.py."""
    checked = 0
    for path in gemma4_recipes():
        text = path.read_text()
        depth = contract["variants"][variant_of(text)]["num_hidden_layers"]
        harvest = (yaml.safe_load(text) or {}).get("harvest") or {}
        for layer in harvest.get("hook_layers") or []:
            assert 0 <= int(layer) < depth, (
                f"{path.relative_to(REPO_ROOT)} hooks layer {layer}, outside "
                f"0..{depth - 1} for {variant_of(text)}"
            )
            checked += 1
    assert checked, "no hook_layers found in any Gemma 4 harvest recipe"


def test_recipe_footprint_estimates_follow_from_the_contract(contract):
    """The quoted capture size must be the one the stated config implies.

    Footprint is tokens x d_model x dtype bytes per hook. Quoting it against
    the wrong width is how '~1.38TB' outlived the width it was computed from.
    """
    checked = 0
    for path in gemma4_recipes():
        text = path.read_text()
        loaded = yaml.safe_load(text) or {}
        harvest = loaded.get("harvest") or {}
        tokens = harvest.get("total_tokens")
        hooks = len(harvest.get("hook_layers") or [])
        if not tokens or not hooks:
            continue
        width = contract["variants"][variant_of(text)]["hidden_size"]
        expected = tokens * width * DTYPE_BYTES[harvest.get("dtype", "bf16")] * hooks
        for value, unit in re.findall(r"~([\d.]+)\s*(GB|TB)", text):
            stated = float(value) * (1e9 if unit == "GB" else 1e12)
            assert abs(stated - expected) / expected < 0.05, (
                f"{path.relative_to(REPO_ROOT)} quotes ~{value}{unit}, but "
                f"{tokens} tokens x {width} x {hooks} hook(s) is "
                f"{expected / 1e9:.1f}GB"
            )
            checked += 1
    assert checked, "no footprint estimate found in any Gemma 4 harvest recipe"


def test_gemma4_recipes_use_the_composite_hook_path():
    """Gemma 4 nests the decoder; the flat path resolves to nothing.

    interp/families.py keeps both templates and picks by config shape. A
    recipe that documents the flat path for a Gemma 4 subject is describing
    the bug that comment warns about.
    """
    checked = 0
    for path in gemma4_recipes():
        for hit in re.findall(r"\bmodel(\.[\w.]*)?\.layers\.\d+", path.read_text()):
            assert hit == ".language_model", (
                f"{path.relative_to(REPO_ROOT)} names a flat "
                f"model{hit}.layers path; Gemma 4 is "
                f"model.language_model.layers.{{n}}"
            )
            checked += 1
    assert checked, "no hook path documented in any Gemma 4 recipe"


# --------------------------------------------------------------------------
# The retired constant does not reappear in code
# --------------------------------------------------------------------------


def test_gemma2_width_absent_from_gemma4_code(contract):
    """2304 matches no Gemma 4 variant, so it must not sit in a Gemma 4 file.

    Kept to source trees rather than prose: an allowlist long enough to cover
    the documentation would stop being a guard.
    """
    assert all(
        v["hidden_size"] != GEMMA2_WIDTH for v in contract["variants"].values()
    ), "2304 is now a real variant width; this guard needs rewriting"

    offenders = []
    for root in CODE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*"):
            if path.suffix not in {".py", ".yaml", ".yml", ".rs"}:
                continue
            if not path.is_file() or path.name in RETIRED_WIDTH_ALLOWLIST:
                continue
            if "target" in path.parts or "__pycache__" in path.parts:
                continue
            text = path.read_text()
            if not VARIANT_RE.search(text) and "gemma4" not in text.lower():
                continue
            # A recipe declaring a non-Gemma-4 subject is not a Gemma 4 file
            # however its comments read, and 2304 is the correct width there.
            if path.suffix in {".yaml", ".yml"} and subject_model_id(text):
                if not variant_of(text):
                    continue
            for number, line in enumerate(text.splitlines(), start=1):
                if str(GEMMA2_WIDTH) in line:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert not offenders, (
        f"{GEMMA2_WIDTH} is Gemma-2-2B's width and matches no Gemma 4 variant, "
        f"but appears in Gemma 4 sources: {offenders}"
    )


def test_calculator_models_the_double_wide_mlp(contract):
    """intermediate_size is not the per-layer truth on E2B.

    Twenty of its 35 layers carry a doubled MLP, which a flat
    3*hidden*ffn*layers understates by 0.57B parameters.
    """
    from training.calculator.components import ArchitectureSpec

    for variant, expected in (("E2B", 20), ("E4B", 0)):
        v = contract["variants"][variant]
        spec = ArchitectureSpec(
            layers=v["num_hidden_layers"],
            hidden=v["hidden_size"],
            heads=v["num_attention_heads"],
            kv_heads=v["num_key_value_heads"],
            ffn=v["intermediate_size"],
            vocab=contract["invariants"]["vocab_size"],
            tie_embeddings=True,
            use_double_wide_mlp=v["use_double_wide_mlp"],
            num_kv_shared_layers=v["num_kv_shared_layers"],
        )
        assert spec.double_wide_layers == expected, variant
        flat = 3 * spec.hidden * spec.ffn * spec.layers
        assert spec.mlp_param_count() == flat + 3 * spec.hidden * spec.ffn * expected


# --------------------------------------------------------------------------
# The support matrix matches the refusals it describes
# --------------------------------------------------------------------------

CPU_RS = REPO_ROOT / "rust-harness" / "runtime" / "src" / "model" / "cpu.rs"
DOC = REPO_ROOT / "docs" / "models" / "gemma-4.md"

# The CPU backend refuses none of the E-series architecture any more. This
# table ran the other way while it did: each entry named a refusal the support
# matrix depended on, and implementing one deleted its bail and failed the
# test, which is what forced the doc update.
#
# It is inverted rather than deleted, because the drift it guards against did
# not go away -- it reversed. The table now claims these features work, so the
# tripwire is that the code implementing them is still there.
IMPLEMENTED = {
    "cross-layer KV sharing": "kv_source",
    "per-layer input embeddings": "ple_dim",
    "final logit softcapping": "logit_softcap",
    "per-layer output scale": "output_scale",
}


@pytest.mark.parametrize("feature", sorted(IMPLEMENTED))
def test_support_matrix_features_are_still_in_the_backend(feature):
    """A feature the table claims works must still be implemented."""
    assert IMPLEMENTED[feature] in CPU_RS.read_text(), (
        f"cpu.rs no longer implements {feature}; the 'Per-tier support' table "
        f"in docs/models/gemma-4.md says it does, so one of the two is now wrong"
    )


def test_the_support_table_does_not_claim_the_cpu_backend_refuses():
    """The CPU row and the prose below it have to agree.

    They disagreed once already: the table said "Refused" against a feature
    list that had been implemented one stage at a time.
    """
    section = DOC.read_text().split("## Per-tier support", 1)[1].split("\n## ", 1)[0]
    cpu_row = next(line for line in section.splitlines() if "CPU backend" in line)
    assert "Refused" not in cpu_row, (
        "the CPU backend row still reads 'Refused', but cpu.rs implements every "
        f"feature the E-series declares: {cpu_row.strip()}"
    )


def test_support_matrix_exists_and_covers_both_variants():
    doc = DOC.read_text()
    assert "## Per-tier support" in doc
    section = doc.split("## Per-tier support", 1)[1].split("\n## ", 1)[0]
    for tier in ("rust-harness", "training", "interp"):
        assert tier in section, f"the support table does not mention {tier}"


def test_contract_names_this_module(contract):
    """A guard the contract does not list is a guard nobody knows to keep."""
    assert Path(__file__).name in " ".join(contract["guarded_by"])
