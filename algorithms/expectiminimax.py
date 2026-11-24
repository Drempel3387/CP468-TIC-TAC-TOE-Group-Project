"""Expectiminimax search for stochastic Tic-Tac-Toe."""
from functools import lru_cache
from .stats import SearchStats
from game.stochastic import move_outcomes, DEFAULT_INTENDED_PROBABILITY


def expectiminimax_search(state, player, intended_probability=DEFAULT_INTENDED_PROBABILITY):
    """Return (best_intended_move, expected_value, stats).

    Each player selects an intended empty square. The chance node then places
    the mark there with probability p, or uniformly in another empty square
    with probability 1-p. Cached states keep the complete 3x3 search practical.
    """
    if not 0.0 <= intended_probability <= 1.0:
        raise ValueError("intended_probability must be between 0 and 1.")
    stats = SearchStats()
    stats.start()
    if state.is_terminal():
        stats.terminal_evals += 1
        value = state.utility(player)
        stats.stop()
        return None, value, stats

    @lru_cache(maxsize=None)
    def value_for(cells, current_player, is_max):
        from game.board import Board
        position = Board(cells, current_player, state.size, state.win_length)
        stats.nodes_evaluated += 1
        if position.is_terminal():
            stats.terminal_evals += 1
            return position.utility(player)
        moves = position.get_legal_moves()
        best = float("-inf") if is_max else float("inf")
        for intended in moves:
            expected = 0.0
            for actual, chance in move_outcomes(position, intended, intended_probability):
                child = position.apply_move(actual)
                expected += chance * value_for(child.cells, child.current_player, not is_max)
            best = max(best, expected) if is_max else min(best, expected)
        return best

    best_move, best_value = None, float("-inf")
    for intended in state.get_legal_moves():
        expected = 0.0
        for actual, chance in move_outcomes(state, intended, intended_probability):
            child = state.apply_move(actual)
            expected += chance * value_for(child.cells, child.current_player, False)
        if expected > best_value:
            best_move, best_value = intended, expected
    stats.stop()
    return best_move, best_value, stats
