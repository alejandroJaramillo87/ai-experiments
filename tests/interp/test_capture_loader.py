"""What the probe capture reads off a hooked decoder block.

The capture registers a forward hook on the block the family registry names
and keeps what the hook is handed. transformers is a workbench dependency
absent from the sandbox, so these tests exercise the unwrap, not a real load.
"""

from __future__ import annotations

from interp.probes.capture import hidden_states


class TestHiddenStates:
    """Which shape a decoder block's hooked output has.

    The tuple form is the older transformers convention and the bare tensor
    the current one. Indexing [0] unconditionally is not a crash on the bare
    form -- it selects sequence 0, so the batch dimension silently becomes a
    token count and the error surfaces one broadcast later.
    """

    def test_unwraps_the_tuple_form(self):
        h, attn = object(), object()
        assert hidden_states((h, attn)) is h

    def test_passes_the_bare_tensor_through(self):
        # Indexable, as a real tensor is -- and that is the whole trap: the
        # old unwrap did not crash here, it returned row 0. A stand-in that
        # refused indexing would let the broken version pass.
        class Tensor:
            def __getitem__(self, i):
                return f"sequence {i}"

        tensor = Tensor()
        assert hidden_states(tensor) is tensor

    def test_a_bare_sequence_type_is_not_mistaken_for_a_tensor(self):
        h, attn = object(), object()
        assert hidden_states([h, attn]) is h
