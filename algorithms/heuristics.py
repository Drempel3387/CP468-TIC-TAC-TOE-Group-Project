"""Heuristic evaluation for depth-limited search on larger boards."""
from functools import lru_cache


@lru_cache(maxsize=None)
def _lines(size, k):
    """All index tuples of k consecutive cells in a row, column, or diagonal."""
    lines = []
    for r in range(size):
        for c in range(size):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                end_r, end_c = r + (k - 1) * dr, c + (k - 1) * dc
                if 0 <= end_r < size and 0 <= end_c < size:
                    lines.append(tuple((r + s * dr) * size + c + s * dc
                                       for s in range(k)))
    return tuple(lines)


def line_heuristic(state, player):
    """Estimated value of `state` from `player`'s point of view, in (-1, 1)."""
    k = state.win_length
    lines = _lines(state.size, k)
    score = 0.0
    for line in lines:
        mine = theirs = 0
        for i in line:
            cell = state.cells[i]
            if cell == player:
                mine += 1
            elif cell is not None:
                theirs += 1
        if mine and not theirs:
            score += (mine / k) ** 2
        elif theirs and not mine:
            score -= (theirs / k) ** 2
    return score / len(lines)