"""The exporter's output, read by the consumer it exists for.

Every other export test asserts what the Python side believes about the
directory it wrote. This module hands that directory to the rust harness and
lets the harness decide: `checkpoint-info --check --text-only` says whether
it would load, and `harness --status --deep --probe` hashes the served
weights and compares the digest against the model_hash a probe records.
The probe is written by the real interp exporter with the manifest's
model_hash in its sidecar, so a pass means three writers of one digest,
in two languages, agree on the same file: the export manifest, the interp
artifact, and the harness preflight. The negative control alters one hex
digit and expects the refusal that names both digests.

The subject is the committed tiny fixture, `rust-harness/fixtures/tiny-gemma4`,
exported by the copy path with the fixture's sibling tokenizer supplied as
`--tokenizer`, since the fixture keeps it beside the directory rather than in
it. No torch, no model download: a sandbox with cargo runs this in full.
The release binaries are built here when absent; a sandbox without cargo
skips, and says so, rather than asserting nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

import pytest
from support import REPO_ROOT

from training import cli

np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
RELEASE = os.path.join(HARNESS_DIR, "target", "release")
HARNESS = os.path.join(RELEASE, "harness")
CHECKPOINT_INFO = os.path.join(RELEASE, "checkpoint-info")
FIXTURE = os.path.join(HARNESS_DIR, "fixtures", "tiny-gemma4")
FIXTURE_TOKENIZER = os.path.join(HARNESS_DIR, "fixtures", "tiny-gemma4.tokenizer.json")

BUILD_TIMEOUT_S = 30 * 60  # a cold workspace build fetches and compiles the tree
RUN_TIMEOUT_S = 120  # the fixture is 40 KB; anything slower is a hang


def altered(digest: str) -> str:
    """The digest with its first hex digit changed: one character, so the
    comparison has to decide rather than merely hash."""
    first = "5" if digest[0] != "5" else "6"
    return first + digest[1:]


@pytest.fixture(scope="module")
def binaries() -> None:
    if os.path.isfile(HARNESS) and os.path.isfile(CHECKPOINT_INFO):
        return
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not installed; the harness binaries cannot be built here")
    build = subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "-p",
            "harness",
            "--bin",
            "harness",
            "-p",
            "runtime",
            "--bin",
            "checkpoint-info",
        ],
        cwd=HARNESS_DIR,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_S,
    )
    assert build.returncode == 0, f"cargo build failed:\n{build.stderr[-4000:]}"
    assert os.path.isfile(HARNESS) and os.path.isfile(CHECKPOINT_INFO)


@pytest.fixture(scope="module")
def exported(tmp_path_factory, binaries) -> dict:
    """The fixture exported through the real verb; returns paths and manifest."""
    root = tmp_path_factory.mktemp("export")
    tokenizer_dir = root / "tokenizer"
    tokenizer_dir.mkdir()
    shutil.copy2(FIXTURE_TOKENIZER, tokenizer_dir / "tokenizer.json")
    out_root = root / "trained"
    args = cli.build_parser().parse_args(
        [
            "export",
            "--checkpoint",
            FIXTURE,
            "--name",
            "tiny",
            "--out-root",
            str(out_root),
            "--tokenizer",
            str(tokenizer_dir),
        ]
    )
    assert args.func(args) == 0
    hf_dir = out_root / "tiny" / "hf"
    with open(out_root / "tiny" / "manifest.json") as f:
        manifest = json.load(f)
    return {"hf_dir": str(hf_dir), "manifest": manifest, "root": root}


def run(binary: str, args: list[str]) -> subprocess.CompletedProcess:
    env = dict(os.environ, NO_COLOR="1")
    return subprocess.run(
        [binary, *args],
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        env=env,
        cwd=HARNESS_DIR,
    )


def write_probe(directory, width: int, layer: int, model_hash: str) -> str:
    """A probe of the model's width whose sidecar records model_hash,
    written by the exporter the harness's reader is tested against. The
    vectors are inert (zero weights, unit sigma): the preflight reads the
    sidecar and the tensor shapes, never a score."""
    from interp.probes.export import write_probe_from_probe
    from interp.probes.registry import Probe

    probe = Probe(
        name="export-identity",
        layer=layer,
        weights=np.zeros(width, dtype=np.float32),
        bias=0.0,
        mu=np.zeros(width, dtype=np.float32),
        sigma=np.ones(width, dtype=np.float32),
        threshold=0.5,
        manifest={"model_hash": model_hash},
    )
    image = os.path.join(directory, "probe.safetensors")
    write_probe_from_probe(probe, image)
    return image


def test_export_is_the_fixture_plus_the_tokenizer(exported):
    hf_dir = exported["hf_dir"]
    assert sorted(os.listdir(hf_dir)) == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    manifest = exported["manifest"]
    assert manifest["mode"] == "copy"
    assert manifest["hf_dir"] == hf_dir
    with open(os.path.join(hf_dir, "model.safetensors"), "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert manifest["model_hash"] == digest
    with open(os.path.join(FIXTURE, "model.safetensors"), "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == digest, (
            "the copy path must carry the weights byte for byte"
        )


def test_checkpoint_info_accepts_the_export(exported):
    result = run(CHECKPOINT_INFO, ["--check", "--text-only", exported["hf_dir"]])
    assert result.returncode == 0, result.stderr
    assert "[OK] CPU backend compatibility: supported" in result.stderr
    assert "=== Tensors (" in result.stdout


def test_harness_deep_preflight_agrees_with_the_manifest(exported, tmp_path):
    with open(os.path.join(exported["hf_dir"], "config.json")) as f:
        config = json.load(f)["text_config"]
    digest = exported["manifest"]["model_hash"]
    image = write_probe(tmp_path, config["hidden_size"], 0, digest)
    result = run(
        HARNESS,
        [
            "--model",
            exported["hf_dir"],
            "--text-only",
            "--probe",
            image,
            "--status",
            "--deep",
        ],
    )
    assert result.returncode == 0, result.stderr
    assert (
        f"[OK] probe identity: matches the served weights, sha256 {digest}"
        in result.stderr
    )
    assert "[OK] backend compatibility: supported" in result.stderr
    assert f"[OK] config.json: {config['num_hidden_layers']} layers" in result.stderr


def test_harness_refuses_a_probe_fitted_to_other_weights(exported, tmp_path):
    with open(os.path.join(exported["hf_dir"], "config.json")) as f:
        config = json.load(f)["text_config"]
    digest = exported["manifest"]["model_hash"]
    bad = altered(digest)
    image = write_probe(tmp_path, config["hidden_size"], 0, bad)
    result = run(
        HARNESS,
        [
            "--model",
            exported["hf_dir"],
            "--text-only",
            "--probe",
            image,
            "--status",
            "--deep",
        ],
    )
    assert result.returncode == 1, result.stderr
    assert (
        "[ERROR] probe identity: trained on a different model: "
        f"artifact model_hash {bad}, served model sha256 {digest}"
    ) in result.stderr
