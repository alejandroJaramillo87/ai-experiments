"""The Python capture writer against the cross-language contract.

The activation-capture manifest key list is written out twice, once in
interp/probes/capture.py and once in the Rust binary, with no shared
definition. schemas/activation_capture.manifest.json is the definition; this
module holds the Python side to it, and
rust-harness/runtime/tests/activation_capture.rs holds the Rust side.

Adding a key to one writer alone fails that writer's test. Updating the
contract to accept it fails the other writer's test until that side follows.
That two-step failure is the enforcement; a shared file only one side read
would not provide it.
"""

from __future__ import annotations

import json
import os

import pytest

from interp.manifest import load_manifest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONTRACT_PATH = os.path.join(REPO_ROOT, "schemas", "activation_capture.manifest.json")


@pytest.fixture(scope="module")
def contract() -> dict:
    with open(CONTRACT_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestManifestContract:
    def test_the_writer_emits_exactly_the_required_key_set(self, capture_dir, contract):
        manifest = load_manifest(capture_dir)
        assert sorted(manifest) == sorted(contract["required"])

    def test_the_writer_emits_a_dtype_the_contract_allows(self, capture_dir, contract):
        # The field was required but unvalued, which is how the two writers
        # drifted to different 16-bit formats while every docstring claimed a
        # third. Constraining it at write time is what keeps them together.
        manifest = load_manifest(capture_dir)
        assert manifest["dtype"] in contract["values"]["dtype"]

    def test_the_shards_hold_the_dtype_the_manifest_declares(self, capture_dir):
        # The manifest string and the array on disk are two separate claims;
        # nothing checked that they agreed.
        import numpy as np

        shard = np.load(os.path.join(capture_dir, "shard_00000.npz"))
        manifest = load_manifest(capture_dir)
        for key in shard.files:
            if key.startswith("layer_"):
                assert shard[key].dtype == np.dtype(manifest["dtype"]), key

    def test_the_rust_only_key_is_absent_from_a_python_capture(
        self, capture_dir, contract
    ):
        # capture_backend distinguishes a rust-captured directory from a
        # python-captured one. The Python writer predates it and no reader
        # requires it, so the contract lists it as writer-only rather than
        # required; adding it here is a behavior change that belongs in its
        # own commit, not a drift this test should tolerate quietly.
        manifest = load_manifest(capture_dir)
        for key in contract["writer_only"]["rust-harness-cpu"]:
            assert key not in manifest

    def test_the_manifest_satisfies_the_shared_report_envelope(self, capture_dir):
        # The required set is a superset of the envelope in common/reports.py:
        # a version, exactly one discriminator, a stamp, and provenance. That
        # module scopes itself to reports rather than artifact manifests, so
        # nothing enforced the overlap until now.
        from common.reports import validate_report

        validate_report(load_manifest(capture_dir))


class TestDatasetDigestDomain:
    def test_the_digest_normalizes_crlf_where_the_rust_writer_does_not(self, tmp_path):
        """Pin the divergence the contract documents.

        Python hashes text-mode f.read(), which normalizes CRLF to LF; the
        Rust writer hashes the file's raw bytes. The two agree on an LF
        dataset and differ on a CRLF one. Neither is wrong, but the difference
        is invisible until someone compares manifests from the two backends
        and concludes the datasets differ. The Rust half of this assertion
        lives in activation_capture.rs; this half fails if the Python side
        silently changes hashing domain.
        """
        import hashlib

        from interp.manifest import text_sha256

        body = '{"text": "anchor harbor rope", "label": 7}\n'
        lf = tmp_path / "lf.jsonl"
        lf.write_bytes(body.encode())
        crlf = tmp_path / "crlf.jsonl"
        crlf.write_bytes(body.replace("\n", "\r\n").encode())

        def python_digest(path) -> str:
            with open(path, encoding="utf-8") as f:
                return text_sha256(f.read())

        def raw_bytes_digest(path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        assert python_digest(lf) == raw_bytes_digest(lf)
        assert python_digest(crlf) != raw_bytes_digest(crlf)
        # Both files carry the same logical dataset, so the Python writer
        # gives them one digest while the raw-byte domain gives them two.
        assert python_digest(lf) == python_digest(crlf)
