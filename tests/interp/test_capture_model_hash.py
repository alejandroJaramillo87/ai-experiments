"""The capture's record of which weights it read.

A probe carries its capture's `model_hash` into the serving harness, which
compares it against the sha256 of the checkpoint it loaded and refuses a
mismatch (rust-harness/harness/src/main.rs). The sentinel `"unknown"` is not a
comparison the harness loses but one it *skips*, so which of the two a capture
records decides whether that guard exists at all. These tests hold the
derivation to the harness's digest and pin the sentinel to the cases where
there is honestly nothing to hash.
"""

from __future__ import annotations

import hashlib
import json
import os

from interp import cli
from interp.probes.capture import UNKNOWN_MODEL_HASH, derive_model_hash, run_capture
from tests.interp.conftest import fake_capture


def checkpoint(tmp_path, body: bytes = b"safetensors bytes") -> str:
    """A directory shaped like the one the harness loads: what is hashed is
    model.safetensors alone, so the other files are irrelevant by design."""
    directory = tmp_path / "gemma-4-E4B"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(body)
    (directory / "config.json").write_text("{}")
    return str(directory)


class TestDeriveModelHash:
    def test_matches_the_digest_the_harness_computes(self, tmp_path):
        # runtime/src/digest.rs hashes the weights file with plain sha256 in
        # lowercase hex. Asserted against hashlib rather than against a frozen
        # string: a literal would pass even if both sides drifted together.
        body = b"weights the harness will load"
        path = checkpoint(tmp_path, body)
        assert derive_model_hash(path) == hashlib.sha256(body).hexdigest()

    def test_hashes_the_weights_not_the_directory(self, tmp_path):
        # A checkpoint gaining a sibling file is the same model, and the
        # harness would say so: it hashes model.safetensors, nothing else.
        path = checkpoint(tmp_path, b"same")
        before = derive_model_hash(path)
        with open(os.path.join(path, "generation_config.json"), "w") as f:
            f.write("{}")
        assert derive_model_hash(path) == before

    def test_a_repo_id_has_nothing_to_hash(self):
        # "google/gemma-2-2b" is not a path. Inventing an identity for it
        # would make the harness compare a fabricated value.
        assert derive_model_hash("google/gemma-2-2b") == UNKNOWN_MODEL_HASH

    def test_a_directory_without_weights_has_nothing_to_hash(self, tmp_path):
        empty = tmp_path / "not-a-checkpoint"
        empty.mkdir()
        assert derive_model_hash(str(empty)) == UNKNOWN_MODEL_HASH


class TestRunCaptureRecordsIt:
    def manifest_of(self, out_dir: str) -> dict:
        with open(os.path.join(out_dir, "manifest.json")) as f:
            return json.load(f)

    def test_unset_derives_from_a_local_checkpoint(self, tmp_path, labeled_dataset):
        model = checkpoint(tmp_path, b"E4B")
        out = str(tmp_path / "cap")
        run_capture(model, labeled_dataset, [3], out, capture_fn=fake_capture)
        assert self.manifest_of(out)["model_hash"] == hashlib.sha256(b"E4B").hexdigest()

    def test_an_explicit_value_is_kept_verbatim(self, tmp_path, labeled_dataset):
        model = checkpoint(tmp_path, b"E4B")
        out = str(tmp_path / "cap")
        run_capture(
            model, labeled_dataset, [3], out, model_hash="mh-1", capture_fn=fake_capture
        )
        assert self.manifest_of(out)["model_hash"] == "mh-1"

    def test_an_explicit_sentinel_is_not_overruled(self, tmp_path, labeled_dataset):
        # A caller that has decided the weights cannot be identified is not
        # second-guessed by a directory that happens to exist at that path.
        model = checkpoint(tmp_path, b"E4B")
        out = str(tmp_path / "cap")
        run_capture(
            model,
            labeled_dataset,
            [3],
            out,
            model_hash=UNKNOWN_MODEL_HASH,
            capture_fn=fake_capture,
        )
        assert self.manifest_of(out)["model_hash"] == UNKNOWN_MODEL_HASH


class TestCaptureHandlerSaysWhichHappened:
    """The operator has to know before serving, not at serve time."""

    def run(self, model, dataset, out, monkeypatch):
        monkeypatch.setattr(
            "interp.probes.capture.capture_activations",
            lambda m, texts, layers, revision="main", batch_size=16: fake_capture(
                texts, layers
            ),
        )
        args = cli.build_parser().parse_args(
            [
                "probe-capture",
                "--model",
                model,
                "--dataset",
                dataset,
                "--layers",
                "3",
                "--out",
                out,
            ]
        )
        return args.func(args)

    def test_a_derived_hash_is_reported(
        self, tmp_path, labeled_dataset, monkeypatch, capsys
    ):
        model = checkpoint(tmp_path, b"E4B")
        assert self.run(model, labeled_dataset, str(tmp_path / "cap"), monkeypatch) == 0
        printed = capsys.readouterr().err
        assert f"[OK] model_hash {hashlib.sha256(b'E4B').hexdigest()}" in printed

    def test_the_sentinel_warns_with_the_fix(
        self, tmp_path, labeled_dataset, monkeypatch, capsys
    ):
        out = str(tmp_path / "cap")
        assert self.run("google/gemma-2-2b", labeled_dataset, out, monkeypatch) == 0
        printed = capsys.readouterr().err
        assert f"[WARN] model_hash {UNKNOWN_MODEL_HASH}" in printed
        assert "will load into any model of the right width, unchecked." in printed
