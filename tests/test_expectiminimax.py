"""Expectiminimax and stochastic-move tests."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.expectiminimax_agent import ExpectiminimaxAgent
from algorithms.alphabeta import alphabeta_search
from algorithms.expectiminimax import expectiminimax_search
from game.board import Board
from game.stochastic import move_outcomes, resolve_stochastic_move


def board(rows, player="X"):
    """Build a 3x3 board from a string such as 'XX_/OO_/___'."""
    cells = tuple(None if ch == "_" else ch for ch in rows.replace("/", ""))
    return Board(cells, player, 3)


class TestChanceOutcomes(unittest.TestCase):

    def test_probabilities_sum_to_one(self):
        for state in (Board.empty(), board("XO_/_X_/__O", "O"), board("XOX/OXO/OX_")):
            for move in state.get_legal_moves():
                total = sum(p for _, p in move_outcomes(state, move, 0.9))
                self.assertAlmostEqual(total, 1.0)

    def test_intended_square_gets_p_and_others_share_the_rest(self):
        outcomes = dict(move_outcomes(Board.empty(), 4, 0.9))
        self.assertAlmostEqual(outcomes[4], 0.9)
        self.assertEqual(len(outcomes), 9)
        self.assertAlmostEqual(outcomes[0], 0.1 / 8)

    def test_single_empty_square_is_certain(self):
        state = board("XOX/OXO/OX_")
        self.assertEqual(move_outcomes(state, 8, 0.9), [(8, 1.0)])

    def test_illegal_intended_move_and_bad_probability(self):
        with self.assertRaises(ValueError):
            move_outcomes(board("X__/___/___", "O"), 0)
        with self.assertRaises(ValueError):
            move_outcomes(Board.empty(), 0, 1.5)

    def test_sampling_matches_probabilities(self):
        rng = random.Random(1)
        hits = sum(resolve_stochastic_move(Board.empty(), 4, rng, 0.9) == 4 for _ in range(5000))
        self.assertAlmostEqual(hits / 5000, 0.9, delta=0.03)


class TestExpectiminimax(unittest.TestCase):

    def test_p1_matches_minimax_value_on_empty_board(self):
        _, value, _ = expectiminimax_search(Board.empty(), "X", 1.0)
        self.assertEqual(value, 0)
        _, ab_value, _, _ = alphabeta_search(Board.empty(), "X")
        self.assertEqual(value, ab_value)

    def test_p1_matches_alphabeta_on_midgame_position(self):
        state = board("XO_/_X_/__O", "X")
        _, value, _ = expectiminimax_search(state, "X", 1.0)
        _, ab_value, _, _ = alphabeta_search(state, "X")
        self.assertEqual(value, ab_value)

    def test_takes_immediate_win(self):
        state = board("XX_/OO_/___", "X")
        move, value, _ = expectiminimax_search(state, "X", 1.0)
        self.assertEqual(move, 2)
        self.assertEqual(value, 1)

    def test_blocks_opponent_win(self):
        state = board("OO_/X__/X__", "X")
        move, _, _ = expectiminimax_search(state, "X", 1.0)
        self.assertEqual(move, 2)

    def test_slip_chance_lowers_value_of_a_sure_win(self):
        state = board("XX_/OO_/___", "X")
        _, certain, _ = expectiminimax_search(state, "X", 1.0)
        _, slippery, _ = expectiminimax_search(state, "X", 0.9)
        self.assertGreater(certain, slippery)
        self.assertGreater(slippery, 0)

    def test_value_stays_within_utility_range(self):
        _, value, _ = expectiminimax_search(Board.empty(), "X", 0.9)
        self.assertGreaterEqual(value, -1)
        self.assertLessEqual(value, 1)

    def test_terminal_state_has_no_move(self):
        state = board("XXX/OO_/___", "O")
        move, value, _ = expectiminimax_search(state, "X", 0.9)
        self.assertIsNone(move)
        self.assertEqual(value, 1)

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            expectiminimax_search(Board.empty(), "X", -0.1)

    def test_agent_returns_legal_move_and_stats(self):
        agent = ExpectiminimaxAgent()
        state = board("X__/_O_/___", "X")
        self.assertIn(agent.choose_move(state), state.get_legal_moves())
        self.assertGreater(agent.last_stats.nodes_evaluated, 0)


if __name__ == "__main__":
    unittest.main()
