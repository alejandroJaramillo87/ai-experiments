"""training/models/head_loss.py: the chunked head loss (KB-87), driven against
a fake torch the way test_peft_utils.py fakes it. The pure helpers run as
they are; the driver is checked for what it calls and in what order, since
the sandbox has no torch."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from training.models.head_loss import (
    IGNORE_INDEX,
    causal_lm_loss,
    chunk_bounds,
    final_logit_softcapping,
    shift_labels,
    unwrap_causal_lm,
)


class TestChunkBounds:
    def test_exact_cover(self):
        assert chunk_bounds(4096, 1024) == [
            (0, 1024),
            (1024, 2048),
            (2048, 3072),
            (3072, 4096),
        ]

    def test_last_chunk_is_partial(self):
        assert chunk_bounds(10, 4) == [(0, 4), (4, 8), (8, 10)]

    def test_chunk_larger_than_tokens_is_one_slice(self):
        assert chunk_bounds(3, 1024) == [(0, 3)]

    def test_nonpositive_chunk_is_refused(self):
        with pytest.raises(ValueError, match="chunk_tokens"):
            chunk_bounds(8, 0)


class TestShiftLabels:
    def test_next_token_alignment(self):
        assert shift_labels([[1, 2, 3], [4, IGNORE_INDEX, 6]]) == [
            [2, 3, IGNORE_INDEX],
            [IGNORE_INDEX, 6, IGNORE_INDEX],
        ]


class TestSoftcapAndUnwrap:
    def test_composite_config_reads_the_text_config(self):
        text = SimpleNamespace(final_logit_softcapping=30.0)
        config = SimpleNamespace(get_text_config=lambda: text)
        assert final_logit_softcapping(config) == 30.0

    def test_plain_config_without_softcap_is_none(self):
        assert final_logit_softcapping(SimpleNamespace(hidden_size=8)) is None

    def test_peft_wrapper_is_unwrapped(self):
        base = object()
        assert unwrap_causal_lm(SimpleNamespace(get_base_model=lambda: base)) is base
        assert unwrap_causal_lm(base) is base


# -- the driver against a fake torch ------------------------------------------


class _Scalar:
    """A loss value: sums and divides like a 0-d tensor, records the device."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.device = "cuda:0"

    def __add__(self, other: _Scalar) -> _Scalar:
        return _Scalar(self.value + other.value)

    def __truediv__(self, other) -> _Scalar:
        return _Scalar(
            self.value / (other.value if isinstance(other, _Scalar) else other)
        )

    def clamp(self, min: int) -> _Scalar:
        return _Scalar(max(self.value, min))

    def to(self, device) -> _Scalar:
        return self


class _Labels:
    """A label tensor as the driver reads it: pad, drop the first, flatten,
    move, compare with the ignore index, count."""

    def __init__(self, rows) -> None:
        self.rows = [list(r) for r in rows]

    def __getitem__(self, key):
        # labels[..., 1:] on a [B, S] tensor drops the first column.
        if isinstance(key, tuple):
            return _Labels([r[key[1]] for r in self.rows])
        return _Flat(self.flat()[key])

    def flat(self):
        return [x for r in self.rows for x in r]

    def reshape(self, *shape):
        assert shape == (-1,)
        return _Flat(self.flat())


class _Flat:
    def __init__(self, values) -> None:
        self.values = list(values)

    def __getitem__(self, key):
        return _Flat(self.values[key])

    def to(self, device):
        return self

    def __ne__(self, other):  # (shifted != ignore_index)
        return _Flat([v != other for v in self.values])

    def sum(self) -> _Scalar:
        return _Scalar(sum(self.values))

    def __len__(self) -> int:
        return len(self.values)


class _Hidden:
    def __init__(self, shape) -> None:
        self.shape = shape
        self.device = "cuda:0"

    def reshape(self, rows, width):
        assert (rows, width) == (self.shape[0] * self.shape[1], self.shape[2])
        return _Hidden((rows, width))

    def __getitem__(self, key: slice):
        return _Hidden((key.stop - key.start, self.shape[1]))


class _Logits:
    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.ops: list = []

    def __truediv__(self, cap):
        self.ops.append(("div", cap))
        return self

    def __mul__(self, cap):
        self.ops.append(("mul", cap))
        return self

    def float(self):
        self.ops.append(("float",))
        return self


class _Head:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def __call__(self, h: _Hidden) -> _Logits:
        self.calls.append(h.shape[0])
        return _Logits(h.shape[0])


class _Decoder:
    def __init__(self, shape) -> None:
        self.shape = shape
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(last_hidden_state=_Hidden(self.shape))


@pytest.fixture
def fake_torch(monkeypatch):
    log: dict = {"checkpoints": [], "tanh": [], "ce": []}

    def pad(labels: _Labels, pads, value):
        assert pads == (0, 1)
        return _Labels([r + [value] for r in labels.rows])

    def cross_entropy(logits: _Logits, y: _Flat, ignore_index, reduction):
        assert reduction == "sum" and ignore_index == IGNORE_INDEX
        log["ce"].append((logits.ops, len(y)))
        # One unit of loss per supervised position in the chunk.
        return _Scalar(sum(1 for v in y.values if v != ignore_index))

    def tanh(x: _Logits) -> _Logits:
        log["tanh"].append(x)
        return x

    def checkpoint(fn, h, y, use_reentrant):
        log["checkpoints"].append((h.shape[0], len(y), use_reentrant))
        return fn(h, y)

    torch = types.ModuleType("torch")
    torch.nn = SimpleNamespace(
        functional=SimpleNamespace(pad=pad, cross_entropy=cross_entropy)
    )
    torch.tanh = tanh
    torch.is_tensor = lambda x: isinstance(x, _Scalar)
    utils = types.ModuleType("torch.utils")
    ckpt = types.ModuleType("torch.utils.checkpoint")
    ckpt.checkpoint = checkpoint
    utils.checkpoint = ckpt
    torch.utils = utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.checkpoint", ckpt)
    return log


def fake_model(batch=2, seq_len=5, softcap=30.0, wrapped=True):
    decoder = _Decoder((batch, seq_len, 4))
    head = _Head()
    text = SimpleNamespace(final_logit_softcapping=softcap)
    hf = SimpleNamespace(
        model=decoder,
        lm_head=head,
        config=SimpleNamespace(get_text_config=lambda: text),
    )
    model = SimpleNamespace(get_base_model=lambda: hf) if wrapped else hf
    return model, decoder, head


class TestCausalLmLoss:
    def test_decoder_once_then_one_checkpointed_chunk_per_slice(self, fake_torch):
        model, decoder, head = fake_model(batch=2, seq_len=5)
        labels = _Labels([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        loss = causal_lm_loss(
            model,
            "ids",
            labels,
            chunk_tokens=4,
            attention_mask="mask",
            position_ids="pos",
        )
        assert decoder.calls == [
            {
                "input_ids": "ids",
                "use_cache": False,
                "attention_mask": "mask",
                "position_ids": "pos",
            }
        ]
        # 10 positions in chunks of 4, each under a non-reentrant checkpoint.
        assert fake_torch["checkpoints"] == [
            (4, 4, False),
            (4, 4, False),
            (2, 2, False),
        ]
        assert head.calls == [4, 4, 2]
        # The shift: the last position of each row predicts the ignore index,
        # so 8 of 10 positions are supervised; one unit each, mean over them.
        assert loss.value == pytest.approx(1.0)

    def test_softcap_wraps_the_head_before_the_fp32_upcast(self, fake_torch):
        model, _, _ = fake_model(batch=1, seq_len=3, softcap=30.0)
        causal_lm_loss(model, "ids", _Labels([[1, 2, 3]]), chunk_tokens=8)
        ((ops, _),) = fake_torch["ce"]
        assert ops == [("div", 30.0), ("mul", 30.0), ("float",)]
        assert len(fake_torch["tanh"]) == 1

    def test_no_softcap_for_a_family_without_one(self, fake_torch):
        model, _, _ = fake_model(batch=1, seq_len=3, softcap=None, wrapped=False)
        causal_lm_loss(model, "ids", _Labels([[1, 2, 3]]), chunk_tokens=8)
        ((ops, _),) = fake_torch["ce"]
        assert ops == [("float",)]
        assert fake_torch["tanh"] == []

    def test_trainer_denominator_wins_over_the_supervised_count(self, fake_torch):
        model, _, _ = fake_model(batch=1, seq_len=5)
        labels = _Labels([[1, 2, 3, 4, 5]])
        by_count = causal_lm_loss(model, "ids", labels, chunk_tokens=8)
        by_batch = causal_lm_loss(
            model, "ids", labels, chunk_tokens=8, num_items_in_batch=_Scalar(16)
        )
        assert by_count.value == pytest.approx(1.0)  # 4 supervised, 4 units
        assert by_batch.value == pytest.approx(4 / 16)

    def test_masked_positions_are_not_counted(self, fake_torch):
        model, _, _ = fake_model(batch=1, seq_len=4)
        labels = _Labels([[IGNORE_INDEX, IGNORE_INDEX, 7, 8]])
        loss = causal_lm_loss(model, "ids", labels, chunk_tokens=2)
        # Shifted: [IGN, 7, 8, IGN] -> 2 supervised positions, one unit each.
        assert loss.value == pytest.approx(1.0)
        assert fake_torch["checkpoints"] == [(2, 2, False), (2, 2, False)]
