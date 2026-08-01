"""Sliding-window attention bounds, for the agent acceptance fixtures.

One function here carries a planted off-by-one and one does not. The shape is
lifted from the sliding-window bound the Rust CPU backend and its numpy
reference both compute, where a layer at position p attends to exactly
`window` positions ending at p inclusive.

Not imported by any production code. See tests/agents/README.md.
"""


def window_start(position: int, window: int) -> int:
    """First cache position a sliding-window layer at `position` may attend to."""
    return max(0, position - window)


def window_length(position: int, window: int) -> int:
    """Count of positions a sliding-window layer at `position` attends to."""
    return min(window, position + 1)
