
"""Minimax AI player."""

from algorithms.minimax import minimax_search


class MinimaxAgent:
    name = "Minimax"

    def __init__(self):
        self.last_stats = None

    def choose_move(self, state):
        """Choose the best move using Minimax."""

        move, score, stats = minimax_search(
            state,
            state.current_player
        )

        # Keep statistics for later comparisons.
        self.last_stats = stats

        return move
