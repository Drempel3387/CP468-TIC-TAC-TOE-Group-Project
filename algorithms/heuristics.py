"""Heuristic evaluation for depth-limited search on larger boards."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def _windows(size: int, k: int):
    """Every run of k consecutive cells (row, column, both diagonals) as index tuples."""
    out = []
    for r in range(size):
        for c in range(size):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = r + dr * (k - 1), c + dc * (k - 1)
                if 0 <= er < size and 0 <= ec < size:
                    out.append(tuple((r + dr * i) * size + (c + dc * i) for i in range(k)))
    return tuple(out)


def line_heuristic(state, player: str) -> float:
    """Score a non-terminal state from `player`'s view, strictly between -1 and 1.

    Each winning line still open for only one side is worth 10^(marks - 1), so
    two-in-a-row beats two separate singles. The net score is squashed into
    (-1, 1) so it can never outrank a real win (+1) or loss (-1).
    """
    opponent = "O" if player == "X" else "X"
    cells = state.cells
    net = 0
    for window in _windows(state.size, state.win_length):
        mine = sum(cells[i] == player for i in window)
        theirs = sum(cells[i] == opponent for i in window)
        if mine and not theirs:
            net += 10 ** (mine - 1)
        elif theirs and not mine:
            net -= 10 ** (theirs - 1)
    return net / (abs(net) + 100)
