"""Alpha-Beta agent.

Standalone for now: exposes choose_move(state). Once agents/base.py defines the
shared interface, make this subclass it.
"""

from algorithms.alphabeta import alphabeta_search


class AlphaBetaAgent:
    name = "AlphaBeta"

    def __init__(self, max_depth=None, heuristic=None, order_moves=None,
                 record_tree=False):
        self.max_depth = max_depth
        self.heuristic = heuristic
        self.order_moves = order_moves
        self.record_tree = record_tree
        self.last_stats = None   # SearchStats from the most recent move
        self.last_tree = None    # TreeNode root, if record_tree is on

    def choose_move(self, state):
        move, _value, stats, root = alphabeta_search(
            state, state.current_player, self.max_depth, self.heuristic,
            self.record_tree, self.order_moves)
        self.last_stats, self.last_tree = stats, root
        return move
