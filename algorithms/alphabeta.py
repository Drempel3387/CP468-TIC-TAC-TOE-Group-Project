"""Alpha-Beta Pruning.

State interface (duck-typed, so any board class works):
    state.is_terminal()          -> bool
    state.get_legal_moves()      -> list of moves
    state.apply_move(move)       -> NEW state (must not mutate the original)
    state.utility(player)        -> number for a terminal state from `player`'s
                                    view (e.g. +1 win, 0 draw, -1 loss)

`player` is the maximizing player for the whole search. The state is also
expected to expose `current_player`, used to decide whose turn it is.

Optional:
    max_depth + heuristic(state, player) for depth-limited search on larger
    boards. If max_depth is hit, heuristic scores the state.
    record_tree=True keeps a tree of visited/pruned nodes for ui/tree_viz.py.
"""

from .stats import SearchStats

INF = float("inf")


class TreeNode:
    """One node of the recorded search tree."""

    def __init__(self, move=None, is_max=True):
        self.move = move          # move that led to this node
        self.is_max = is_max      # True if a max node
        self.value = None
        self.alpha = None         # alpha/beta when the node was entered
        self.beta = None
        self.pruned = False       # True if never explored (cut off)
        self.children = []


def alphabeta_search(state, player, max_depth=None, heuristic=None,
                     record_tree=False, order_moves=None):
    """Return (best_move, value, stats, root) for `player` to move at `state`.

    root is the recorded TreeNode tree, or None if record_tree is False.
    order_moves(state, moves) -> moves lets callers sort moves so the best are
    tried first, which makes pruning much more effective.
    """
    stats = SearchStats()
    stats.start()

    root = TreeNode(None, True) if record_tree else None
    best_move, best_value = None, -INF
    alpha, beta = -INF, INF

    moves = state.get_legal_moves()
    if order_moves:
        moves = order_moves(state, moves)
    if root:
        root.alpha, root.beta = alpha, beta
    stats.nodes_evaluated += 1

    for i, move in enumerate(moves):
        child_node = TreeNode(move, False) if record_tree else None
        value = _search(state.apply_move(move), player, 1, alpha, beta, False,
                        max_depth, heuristic, stats, child_node, order_moves)
        if root:
            root.children.append(child_node)
        if value > best_value:
            best_value, best_move = value, move
        alpha = max(alpha, best_value)
        if alpha >= beta:  # cannot happen at root with beta=inf, kept for safety
            _mark_pruned(moves[i + 1:], root, stats, False)
            break

    stats.stop()
    if root:
        root.value = best_value
    return best_move, best_value, stats, root


def _search(state, player, depth, alpha, beta, is_max, max_depth, heuristic,
            stats, node, order_moves):
    stats.nodes_evaluated += 1
    if node:
        node.alpha, node.beta = alpha, beta

    if state.is_terminal():
        stats.terminal_evals += 1
        v = state.utility(player)
        if node:
            node.value = v
        return v
    if max_depth is not None and depth >= max_depth:
        stats.terminal_evals += 1
        v = heuristic(state, player) if heuristic else 0
        if node:
            node.value = v
        return v

    moves = state.get_legal_moves()
    if order_moves:
        moves = order_moves(state, moves)

    value = -INF if is_max else INF
    for i, move in enumerate(moves):
        child_node = TreeNode(move, not is_max) if node else None
        child_value = _search(state.apply_move(move), player, depth + 1, alpha,
                              beta, not is_max, max_depth, heuristic, stats,
                              child_node, order_moves)
        if node:
            node.children.append(child_node)

        if is_max:
            value = max(value, child_value)
            alpha = max(alpha, value)
        else:
            value = min(value, child_value)
            beta = min(beta, value)

        if alpha >= beta:  # cutoff: remaining siblings cannot change the result
            _mark_pruned(moves[i + 1:], node, stats, not is_max)
            break

    if node:
        node.value = value
    return value


def _mark_pruned(skipped_moves, node, stats, child_is_max):
    if not skipped_moves:
        return
    stats.prunes += 1
    stats.pruned_branches += len(skipped_moves)
    if node:
        for m in skipped_moves:
            pruned = TreeNode(m, child_is_max)
            pruned.pruned = True
            node.children.append(pruned)
