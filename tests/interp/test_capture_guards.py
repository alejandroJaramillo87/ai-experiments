"""Capture contract guards (Campaign 2, E-22/E-23).

A capture joins activation rows, labels, and texts by position alone, so a
directory whose parts disagree trains a probe on mislabeled rows with no
error. The writer must also accept exactly the dataset lines the Rust writer
(rust-harness activation-capture) accepts, or the two backends disagree on
what a label is.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from interp.probes.capture import CaptureError, load_capture, load_labeled_dataset


def _patch_manifest(capture_dir: str, **updates) -> None:
    path = os.path.join(capture_dir, "manifest.json")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.update(updates)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f)


class TestLoadCaptureConsistency:
    def test_shorter_examples_file_is_refused_instead_of_zipped_by_position(
        self, capture_dir
    ):
        path = os.path.join(capture_dir, "examples.jsonl")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines[:-1])
        with pytest.raises(CaptureError, match="examples.jsonl holds 119 texts"):
            load_capture(capture_dir)

    def test_manifest_count_disagreeing_with_shard_rows_is_refused(self, capture_dir):
        _patch_manifest(capture_dir, count=119)
        with pytest.raises(CaptureError, match="shards hold 120 labels"):
            load_capture(capture_dir)

    def test_layer_holding_fewer_rows_than_labels_is_refused(self, capture_dir):
        shard = os.path.join(capture_dir, "shard_00000.npz")
        with np.load(shard) as data:
            payload = {key: data[key] for key in data.files}
        payload["layer_3"] = payload["layer_3"][:-1]
        np.savez_compressed(shard, **payload)
        with pytest.raises(CaptureError, match="layer 3 holds 119 rows"):
            load_capture(capture_dir)

    def test_duplicate_layer_in_manifest_is_refused(self, capture_dir):
        _patch_manifest(capture_dir, layers=[3, 3, 5])
        with pytest.raises(CaptureError, match="manifest layers repeat"):
            load_capture(capture_dir)


class TestLabelStrictness:
    def test_float_label_is_refused_instead_of_truncated_to_a_class(self, tmp_path):
        path = tmp_path / "soft.jsonl"
        path.write_text('{"text": "a", "label": 0.9}\n')
        with pytest.raises(
            CaptureError, match=r"soft.jsonl:1: label must be an integer"
        ):
            load_labeled_dataset(str(path))

    def test_boolean_label_is_refused_instead_of_becoming_class_one(self, tmp_path):
        path = tmp_path / "bool.jsonl"
        path.write_text('{"text": "a", "label": true}\n')
        with pytest.raises(CaptureError, match="label must be an integer, got True"):
            load_labeled_dataset(str(path))

    def test_string_label_is_refused_to_match_the_rust_writer(self, tmp_path):
        path = tmp_path / "str.jsonl"
        path.write_text('{"text": "a", "label": "1"}\n')
        with pytest.raises(CaptureError, match="label must be an integer"):
            load_labeled_dataset(str(path))
