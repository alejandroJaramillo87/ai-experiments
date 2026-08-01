"""Green test suite for window.py — it passes with the planted defect in place.

Named suite.py rather than test_*.py so `make test` does not collect it. Run it
explicitly: `uv run pytest tests/agents/planted_defect/suite.py`.
"""

from window import window_length, window_start


def test_window_start_clamps_at_zero() -> None:
    assert window_start(0, 4) == 0
    assert window_start(3, 4) == 0


def test_window_length_grows_to_the_window() -> None:
    assert window_length(0, 4) == 1
    assert window_length(10, 4) == 4
