"""Alpha-beta correctness tests, using a self-contained Tic-Tac-Toe state."""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from algorithms.alphabeta import alphabeta_search
from agents.alphabeta_agent import AlphaBetaAgent
from ui.tree_viz import render_tree

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6)]


class TTT:
    """Minimal immutable 3x3 Tic-Tac-Toe implementing the state interface."""

    def __init__(self, cells=None, current_player="X"):
        self.cells = cells or (None,) * 9
        self.current_player = current_player

    def winner(self):
        for a, b, c in LINES:
            if self.cells[a] and self.cells[a] == self.cells[b] == self.cells[c]:
                return self.cells[a]
        return None

    def is_terminal(self):
        return self.winner() is not None or None not in self.cells

    def get_legal_moves(self):
        return [i for i, c in enumerate(self.cells) if c is None]

    def apply_move(self, move):
        cells = list(self.cells)
        cells[move] = self.current_player
        return TTT(tuple(cells), "O" if self.current_player == "X" else "X")

    def utility(self, player):
        w = self.winner()
        return 0 if w is None else (1 if w == player else -1)


def minimax(state, player, is_max, counter):
    counter[0] += 1
    if state.is_terminal():
        return state.utility(player)
    vals = [minimax(state.apply_move(m), player, not is_max, counter)
            for m in state.get_legal_moves()]
    return max(vals) if is_max else min(vals)


class TestAlphaBeta(unittest.TestCase):
    def test_empty_board_is_draw(self):
        _, value, _, _ = alphabeta_search(TTT(), "X")
        self.assertEqual(value, 0)

    def test_matches_minimax_value_and_prunes(self):
        state = TTT()
        counter = [0]
        mm_value = minimax(state, "X", True, counter)
        _, ab_value, stats, _ = alphabeta_search(state, "X")
        self.assertEqual(ab_value, mm_value)
        self.assertLess(stats.nodes_evaluated, counter[0])
        self.assertGreater(stats.prunes, 0)

    def test_takes_immediate_win(self):
        # X X _ / O O _ / _ _ _ , X to move: 2 wins
        cells = ("X", "X", None, "O", "O", None, None, None, None)
        move, value, _, _ = alphabeta_search(TTT(cells, "X"), "X")
        self.assertEqual((move, value), (2, 1))

    def test_blocks_opponent_win(self):
        # O threatens 3-4-5 line; X must block at 5
        cells = ("X", None, None, "O", "O", None, None, None, "X")
        move, _, _, _ = alphabeta_search(TTT(cells, "X"), "X")
        self.assertEqual(move, 5)

    def test_depth_limit_uses_heuristic(self):
        calls = []
        def h(state, player):
            calls.append(1)
            return 0
        alphabeta_search(TTT(), "X", max_depth=2, heuristic=h)
        self.assertTrue(calls)

    def test_agent_and_tree(self):
        agent = AlphaBetaAgent(record_tree=True)
        move = agent.choose_move(TTT())
        self.assertIn(move, range(9))
        self.assertGreater(agent.last_stats.nodes_evaluated, 0)
        text = render_tree(agent.last_tree, max_depth=3)
        self.assertIn("PRUNED", text)


if __name__ == "__main__":
    unittest.main()
