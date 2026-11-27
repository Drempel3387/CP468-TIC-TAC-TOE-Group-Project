"""Tests for the game environment; board rules, chance mechanics and the game loop"""
import random
import unittest

from game.board import Board, other_player
from game.stochastic import move_outcomes, resolve_stochastic_move
from game.tictactoe import play_game


def board_from(rows, player="X"):
    """Build a board from strings such as 'XX.' where '.' is an empty square."""
    cells = tuple(None if ch == "." else ch for row in rows for ch in row)
    return Board(cells, player, len(rows))


class FirstLegalAgent:
    """Always plays the lowest-numbered empty square."""
    name = "FirstLegal"

    def choose_move(self, state):
        return state.get_legal_moves()[0]


class ScriptedAgent:
    """Plays a fixed list of moves, one per turn."""
    name = "Scripted"

    def __init__(self, moves):
        self.moves = list(moves)

    def choose_move(self, state):
        return self.moves.pop(0)


class BoardRulesTests(unittest.TestCase):
    def test_empty_board(self):
        board = Board.empty()
        self.assertEqual(board.cells, (None,) * 9)
        self.assertEqual(board.current_player, "X")
        self.assertEqual(board.get_legal_moves(), list(range(9)))
        self.assertFalse(board.is_terminal())

    def test_apply_move_places_mark_and_switches_player(self):
        board = Board.empty().apply_move(4)
        self.assertEqual(board.cells[4], "X")
        self.assertEqual(board.current_player, "O")
        self.assertNotIn(4, board.get_legal_moves())

    def test_apply_move_does_not_change_original_board(self):
        original = Board.empty()
        original.apply_move(0)
        self.assertEqual(original.cells, (None,) * 9)
        self.assertEqual(original.current_player, "X")

    def test_illegal_moves_are_rejected(self):
        board = Board.empty().apply_move(0)
        with self.assertRaises(ValueError):
            board.apply_move(0)       
        with self.assertRaises(ValueError):
            board.apply_move(9)       
        with self.assertRaises(ValueError):
            board.apply_move(-1)      
        with self.assertRaises(ValueError):
            board.apply_move("4")     

    def test_every_winning_line_is_detected(self):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8),      
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),      
                 (0, 4, 8), (2, 4, 6)]                 
        for mark in ("X", "O"):
            for line in lines:
                cells = [None] * 9
                for i in line:
                    cells[i] = mark
                board = Board(tuple(cells), "X")
                self.assertEqual(board.winner(), mark, f"{mark} on {line}")
                self.assertTrue(board.is_terminal())

    def test_no_winner_on_unfinished_board(self):
        board = board_from(["XO.", ".X.", "..O"])
        self.assertIsNone(board.winner())
        self.assertFalse(board.is_terminal())

    def test_draw_is_detected(self):
        board = board_from(["XOX", "XOO", "OXX"])
        self.assertIsNone(board.winner())
        self.assertTrue(board.is_draw())
        self.assertTrue(board.is_terminal())
        self.assertEqual(board.utility("X"), 0)
        self.assertEqual(board.utility("O"), 0)

    def test_utility_is_from_the_players_point_of_view(self):
        board = board_from(["XXX", "OO.", "..."], "O")
        self.assertEqual(board.utility("X"), 1)
        self.assertEqual(board.utility("O"), -1)

    def test_other_player(self):
        self.assertEqual(other_player("X"), "O")
        self.assertEqual(other_player("O"), "X")
        with self.assertRaises(ValueError):
            other_player("Z")

    def test_larger_board_needs_a_full_line_by_default(self):
        board = board_from(["XXX.", "OO..", "....", "...."])
        self.assertIsNone(board.winner())
        self.assertEqual(len(board.get_legal_moves()), 16 - 5)

    def test_win_length_can_be_shorter_than_the_board(self):
        cells = ["X", "X", "X", None] + [None] * 12
        board = Board(tuple(cells), "O", size=4, win_length=3)
        self.assertEqual(board.winner(), "X")

    def test_invalid_boards_are_rejected(self):
        with self.assertRaises(ValueError):
            Board((None,) * 8, "X")                
        with self.assertRaises(ValueError):
            Board((None,) * 9, "Z")                
        with self.assertRaises(ValueError):
            Board(("Q",) + (None,) * 8, "X")       


class StochasticTests(unittest.TestCase):
    def test_outcomes_form_a_probability_distribution(self):
        outcomes = move_outcomes(Board.empty(), 4)
        self.assertEqual(len(outcomes), 9)
        self.assertAlmostEqual(sum(p for _, p in outcomes), 1.0)

    def test_intended_square_gets_ninety_percent_and_others_share_the_rest(self):
        outcomes = dict(move_outcomes(Board.empty(), 4))
        self.assertAlmostEqual(outcomes[4], 0.90)
        for square, probability in outcomes.items():
            if square != 4:
                self.assertAlmostEqual(probability, 0.10 / 8)

    def test_custom_probability(self):
        outcomes = dict(move_outcomes(Board.empty(), 0, 0.7))
        self.assertAlmostEqual(outcomes[0], 0.7)
        self.assertAlmostEqual(sum(outcomes.values()), 1.0)

    def test_only_empty_squares_can_receive_the_mark(self):
        board = Board.empty().apply_move(0).apply_move(1)
        squares = [square for square, _ in move_outcomes(board, 4)]
        self.assertNotIn(0, squares)
        self.assertNotIn(1, squares)
        self.assertEqual(sorted(squares), board.get_legal_moves())

    def test_last_empty_square_is_certain(self):
        board = board_from(["XOX", "XOO", "OX."])
        self.assertEqual(move_outcomes(board, 8), [(8, 1.0)])

    def test_illegal_intended_move_is_rejected(self):
        board = Board.empty().apply_move(0)
        with self.assertRaises(ValueError):
            move_outcomes(board, 0)

    def test_probability_must_be_between_zero_and_one(self):
        with self.assertRaises(ValueError):
            move_outcomes(Board.empty(), 0, 1.5)
        with self.assertRaises(ValueError):
            move_outcomes(Board.empty(), 0, -0.1)

    def test_certain_probability_never_slips(self):
        rng = random.Random(0)
        for _ in range(200):
            self.assertEqual(resolve_stochastic_move(Board.empty(), 4, rng, 1.0), 4)

    def test_slip_rate_is_close_to_ten_percent(self):
        rng = random.Random(1)
        trials = 5000
        kept = sum(resolve_stochastic_move(Board.empty(), 4, rng) == 4 for _ in range(trials))
        self.assertAlmostEqual(kept / trials, 0.90, delta=0.03)

    def test_resolved_move_is_always_legal(self):
        rng = random.Random(2)
        board = Board.empty().apply_move(0).apply_move(1)
        for _ in range(300):
            self.assertIn(resolve_stochastic_move(board, 4, rng), board.get_legal_moves())


class GameLoopTests(unittest.TestCase):
    def test_x_wins_a_scripted_game(self):
        result = play_game(ScriptedAgent([0, 1, 2]), ScriptedAgent([3, 4]))
        self.assertEqual(result.winner, "X")
        self.assertFalse(result.is_draw)
        self.assertEqual(result.moves, [0, 3, 1, 4, 2])

    def test_scripted_draw(self):
        result = play_game(ScriptedAgent([0, 2, 3, 7, 8]), ScriptedAgent([1, 4, 5, 6]))
        self.assertIsNone(result.winner)
        self.assertTrue(result.is_draw)
        self.assertEqual(len(result.moves), 9)

    def test_result_matches_final_board(self):
        result = play_game(FirstLegalAgent(), FirstLegalAgent())
        self.assertEqual(result.winner, result.final_state.winner())
        self.assertTrue(result.final_state.is_terminal())

    def test_illegal_move_by_an_agent_is_rejected(self):
        with self.assertRaises(ValueError):
            play_game(ScriptedAgent([0]), ScriptedAgent([0]))

    def test_display_callback_sees_every_position(self):
        seen = []
        result = play_game(FirstLegalAgent(), FirstLegalAgent(), display=seen.append)
        self.assertEqual(len(seen), len(result.moves) + 1)   

    def test_stochastic_game_finishes_with_distinct_moves(self):
        result = play_game(FirstLegalAgent(), FirstLegalAgent(),
                           stochastic=True, rng=random.Random(5))
        self.assertTrue(result.final_state.is_terminal())
        self.assertEqual(len(result.moves), len(set(result.moves)))

    def test_stochastic_games_are_reproducible_with_a_seed(self):
        first = play_game(FirstLegalAgent(), FirstLegalAgent(),
                          stochastic=True, rng=random.Random(9))
        second = play_game(FirstLegalAgent(), FirstLegalAgent(),
                           stochastic=True, rng=random.Random(9))
        self.assertEqual(first.moves, second.moves)

    def test_larger_board_game_finishes(self):
        result = play_game(FirstLegalAgent(), FirstLegalAgent(), size=4, win_length=3)
        self.assertTrue(result.final_state.is_terminal())


if __name__ == "__main__":
    unittest.main()