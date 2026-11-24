
"""Minimax algorithm for Tic-Tac-Toe."""

from .stats import SearchStats


def minimax(state, player, is_max, stats):
    """Find the value of a board using recursion."""

    stats.nodes_evaluated += 1

    # Stop if someone won or the game is a draw.
    if state.is_terminal():
        stats.terminal_evals += 1
        return state.utility(player)

    # MAX wants the highest score, MIN wants the lowest.
    if is_max:
        best = float("-inf")
    else:
        best = float("inf")

    # Try every possible move.
    for move in state.get_legal_moves():
        new_state = state.apply_move(move)

        # Switch between MAX and MIN.
        score = minimax(new_state, player, not is_max, stats)

        if is_max:
            best = max(best, score)
        else:
            best = min(best, score)

    return best


def minimax_search(state, player):
    """Find the best move for the AI."""

    stats = SearchStats()
    stats.start()

    stats.nodes_evaluated += 1

    # No move is needed if the game is over.
    if state.is_terminal():
        stats.terminal_evals += 1
        value = state.utility(player)
        stats.stop()
        return None, value, stats

    best_move = None
    best_score = float("-inf")

    # Check every available move.
    for move in state.get_legal_moves():
        new_state = state.apply_move(move)

        # After our move, the opponent plays (MIN).
        score = minimax(new_state, player, False, stats)

        # Save the move with the best score.
        if score > best_score:
            best_score = score
            best_move = move

    stats.stop()

    return best_move, best_score, stats
