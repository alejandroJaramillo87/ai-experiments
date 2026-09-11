"""RH.19, the alignment artifact on the real vocabulary, as a test the E4B tier runs.

`--alignment-out` writes one row per position of a run: the byte span of the
decoded text that position produced, its role, and since RH.19 the token id
the run computed there. The sandbox pins the artifact's structure on the
synthetic fixtures; nothing before this test read one written over the real
Gemma 4 vocabulary, whose chat markers are pieces the decoder drops and whose
byte-fallback pieces are what the span rules exist for.

Two kinds of check, and the difference is the reason the second exists.

The partition checks are the ones RH.19 was first run by hand with: every
position has a row, the scaffold rows are zero-width, and the spans tile the
header's text with no gap or overlap. They prove the spans are a consistent
partition of the text. They cannot tell whether position k's span holds the
bytes token k produced: a partition shifted by one token inside a run of
content satisfies every one of them.

The content checks close that. `transformers`' tokenizer for the same
checkpoint is the independent decoder, as it is the ladder's independent
forward pass: each row's id is decoded by it and held to the bytes the row's
span selects. The harness decodes through the `tokenizers` crate and this
test through the Python binding of the same library over the same
tokenizer.json, so the two are one decoder reached two ways; what the check
is independent of is the harness's own span arithmetic, which is what is
under test.

Rows are grouped the way `runtime/src/align.rs` attributes bytes: a codepoint
spread across byte-fallback ids lands whole on the id that completes it, and
the ids before it carry empty spans. A group is therefore a run of empty spans
plus the non-empty span that ends it, and it is the group's ids, not each id
alone, that must decode to the group's bytes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import pytest

from tests.hardware.conftest import REPO_ROOT, e4b_checkpoint

pytestmark = [pytest.mark.hw_full, pytest.mark.hw_e4b, pytest.mark.cpu, pytest.mark.ram]

HARNESS_DIR = os.path.join(REPO_ROOT, "rust-harness")
HARNESS = os.path.join(HARNESS_DIR, "target", "release", "harness")

# The run RH.19 was first climbed with by hand, so the two transcripts read
# alike: a positional task, which renders the coding-agent scaffold and so
# puts turn markers and a memory turn in front of the prompt. That is the
# point rather than a cost -- the scaffold rows are what the role check needs.
PROMPT = "The three states of matter are"
SEED = "7"
MAX_TOKENS = 48

# The harness's own account of how many positions were the prompt; everything
# after them the model produced. Read from the transcript rather than inferred
# from `phase`, because a prompt whose last batch is one token wide would
# record that position as `decode`.
PREFILL_LINE = re.compile(r"^\[INFO\] Prefill: (\d+) tokens in", re.M)
ALIGNMENT_LINE = re.compile(
    r"^\[INFO\] Alignment: (\d+) positions, (\d+) passes -> (.+)$", re.M
)

# 2026-09-11: the manual run loaded in 5.88 s, prefilled 251 tokens in 22.95 s
# and decoded 48 more in under a minute. The bound leaves room for a cold page
# cache on the 14 GiB checkpoint.
TIMEOUT_S = 20 * 60


def require_binary() -> None:
    """A missing binary fails rather than skips: it is built on the same
    host, so its absence is a misconfigured run and not the unavailable
    hardware the conftest rule exempts."""
    assert os.path.isfile(HARNESS), (
        f"built binary absent: {HARNESS}; run "
        f"`cargo build --release -p harness` in {HARNESS_DIR}"
    )


class Artifact:
    """One written alignment.jsonl, parsed, with the run's transcript."""

    def __init__(self, path: str, transcript: str) -> None:
        with open(path, encoding="utf-8") as handle:
            lines = [json.loads(line) for line in handle if line.strip()]
        self.header = lines[0]
        self.rows = lines[1:]
        self.text = self.header["text"]
        self.text_bytes = self.text.encode("utf-8")
        self.transcript = transcript
        # Exactly one: the table is rebuilt whole from the last turn's prompt,
        # and a second turn would make that prompt longer than any one prefill.
        prefills = PREFILL_LINE.findall(transcript)
        assert len(prefills) == 1, (
            f"{len(prefills)} prefills, one turn expected\n{transcript}"
        )
        self.prompt_len = int(prefills[0])

    def span_bytes(self, row: dict) -> bytes:
        return self.text_bytes[row["byte_start"] : row["byte_end"]]

    def groups(self) -> list[tuple[list[int], int, int, list[int]]]:
        """(ids, byte_start, byte_end, positions) per attribution group.

        A trailing run of empty spans, with no non-empty span to end it, is
        its own group over an empty range at the end of the text.
        """
        out = []
        ids: list[int] = []
        positions: list[int] = []
        for row in self.rows:
            ids.append(row["token_id"])
            positions.append(row["position"])
            if row["byte_end"] > row["byte_start"]:
                out.append((ids, row["byte_start"], row["byte_end"], positions))
                ids, positions = [], []
        if ids:
            end = len(self.text_bytes)
            out.append((ids, end, end, positions))
        return out


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Artifact:
    """One run of the harness for the module: the checkpoint loads once.

    Its own empty directory, because the harness resumes `session.kv` from the
    directory it starts in and a resumed session would prefill nothing.
    """
    model = e4b_checkpoint()
    if not os.path.isfile(os.path.join(model, "model.safetensors")):
        pytest.skip(f"E4B checkpoint not on disk at {model}")
    require_binary()

    directory = tmp_path_factory.mktemp("rh19")
    out = os.path.join(directory, "align.jsonl")
    result = subprocess.run(
        [
            HARNESS,
            "--model",
            model,
            "--text-only",
            "--seed",
            SEED,
            "--max-tokens",
            str(MAX_TOKENS),
            "--alignment-out",
            out,
            PROMPT,
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )
    transcript = result.stdout + result.stderr
    assert result.returncode == 0, f"harness exited {result.returncode}\n{transcript}"
    written = ALIGNMENT_LINE.search(transcript)
    assert written is not None, f"no [INFO] Alignment line\n{transcript}"
    assert os.path.isfile(out), f"{out} was not written\n{transcript}"
    return Artifact(out, transcript)


@pytest.fixture(scope="module")
def tokenizer(artifact):
    """transformers' tokenizer for the checkpoint the run loaded. Imported
    here rather than at module scope so collection stays torch-free."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(e4b_checkpoint())


def first_difference(a: str, b: str) -> int:
    """Index of the first character where the two strings differ, or the
    shorter length when one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def decode(tokenizer, ids: list[int]) -> str:
    """What the harness's decode does: specials dropped, nothing tidied.

    `clean_up_tokenization_spaces` would rewrite ` .` to `.` and the like,
    which is a presentation rule and not a decoding one; the harness applies
    none and the comparison is byte-exact.
    """
    return tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def test_the_artifact_covers_every_position(artifact):
    rows = artifact.rows
    assert artifact.header["positions"] == len(rows), (
        f"header counts {artifact.header['positions']}, {len(rows)} rows written"
    )
    positions = [row["position"] for row in rows]
    assert positions == list(range(len(rows))), "positions are not dense from 0"
    assert len(rows) > artifact.prompt_len, (
        f"{len(rows)} rows for a {artifact.prompt_len}-token prompt: nothing generated"
    )
    written = ALIGNMENT_LINE.search(artifact.transcript)
    assert int(written.group(1)) == len(rows), written.group(0)


def test_scaffold_rows_are_zero_width_on_the_real_vocabulary(artifact, tokenizer):
    roles = {row["role"] for row in artifact.rows}
    assert roles == {"scaffold", "content"}, f"roles present: {sorted(roles)}"
    scaffold = [row for row in artifact.rows if row["role"] == "scaffold"]
    # The real markers are pieces the decoder drops (alignment.rs module doc),
    # so on this vocabulary a scaffold position produces no bytes, and every
    # id at one is a piece the checkpoint declares special.
    wide = [row for row in scaffold if row["byte_end"] != row["byte_start"]]
    assert not wide, f"scaffold rows carrying bytes: {wide}"
    special = set(tokenizer.all_special_ids)
    ordinary = [row for row in scaffold if row["token_id"] not in special]
    assert not ordinary, f"scaffold rows whose id is not a special piece: {ordinary}"
    # The prompt begins with the sequence boundary, and no content row is one.
    assert artifact.rows[0]["token_id"] == tokenizer.bos_token_id, artifact.rows[0]
    assert artifact.rows[0]["role"] == "scaffold", artifact.rows[0]
    bos_content = [
        row
        for row in artifact.rows
        if row["role"] == "content" and row["token_id"] == tokenizer.bos_token_id
    ]
    assert not bos_content, f"content rows carrying BOS: {bos_content}"
    # Generated positions are content by construction: the harness wrote none.
    generated_scaffold = [r for r in scaffold if r["position"] >= artifact.prompt_len]
    assert not generated_scaffold, f"scaffold past the prompt: {generated_scaffold}"


def test_the_spans_tile_the_text(artifact):
    rows = artifact.rows
    assert rows[0]["byte_start"] == 0, rows[0]
    assert rows[-1]["byte_end"] == len(artifact.text_bytes), (
        f"last span ends at {rows[-1]['byte_end']}, text is "
        f"{len(artifact.text_bytes)} bytes"
    )
    gaps = [
        (prev, row)
        for prev, row in zip(rows, rows[1:], strict=False)
        if row["byte_start"] != prev["byte_end"]
    ]
    assert not gaps, f"spans that do not abut: {gaps[:5]}"
    # Every boundary sits on a codepoint boundary, which is the byte-fallback
    # rule observed from the outside: a span never ends inside a character.
    for row in rows:
        try:
            artifact.span_bytes(row).decode("utf-8")
        except UnicodeDecodeError as error:
            raise AssertionError(f"span splits a codepoint: {row}: {error}") from None


def test_each_span_holds_its_tokens_bytes(artifact, tokenizer):
    ids = [row["token_id"] for row in artifact.rows]
    whole = decode(tokenizer, ids)
    assert whole == artifact.text, (
        "transformers decodes the run's ids to different text than the harness "
        f"wrote: {len(whole)} and {len(artifact.text)} characters, first "
        f"difference at index {first_difference(whole, artifact.text)}"
    )
    # The check the partition checks cannot make: every group's ids decode to
    # exactly the bytes its span selects, so a shift by one token inside a run
    # of content fails at the first shifted row.
    wrong = []
    for group_ids, start, end, positions in artifact.groups():
        expected = artifact.text_bytes[start:end].decode("utf-8")
        got = decode(tokenizer, group_ids)
        if got != expected:
            wrong.append((positions, group_ids, got, expected))
    assert not wrong, (
        f"{len(wrong)} groups whose ids do not decode to their span; first: "
        f"positions {wrong[0][0]} ids {wrong[0][1]} decode to {wrong[0][2]!r}, "
        f"span holds {wrong[0][3]!r}"
    )


def test_the_generated_ids_round_trip_through_transformers(artifact, tokenizer):
    """The model's own text, re-encoded, is the ids the model produced.

    Sampled text is not guaranteed round-trip stable -- a merge the model never
    chose can be the encoder's preference -- so the assertion is over the
    common prefix, which for text this short is in practice all of it. A
    divergence early in the sequence is what a wrong id at a position looks
    like; one late in it is reported with its index and read by hand.
    """
    generated = [row for row in artifact.rows if row["position"] >= artifact.prompt_len]
    assert generated, "no generated rows"
    ids = [row["token_id"] for row in generated]
    text = decode(tokenizer, ids)
    again = tokenizer.encode(text, add_special_tokens=False)
    common = 0
    for a, b in zip(ids, again, strict=False):
        if a != b:
            break
        common += 1
    assert common * 2 >= len(ids), (
        f"generated ids diverge from their re-encoding at index {common} of "
        f"{len(ids)}: run {ids[common : common + 4]}, re-encoded "
        f"{again[common : common + 4]}; text {text!r}"
    )
