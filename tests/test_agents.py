"""Tests for agent behaviour; legal moves, tactics, statistics and head-to-head play"""
import random
import unittest

from agents.alphabeta_agent import AlphaBetaAgent
from agents.expectiminimax_agent import ExpectiminimaxAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from algorithms.heuristics import line_heuristic
from game.board import Board
from game.tictactoe import play_game


def board_from(rows, player="X"):
    """Build a board from strings such as 'XX.' where '.' is an empty square."""
    cells = tuple(None if ch == "." else ch for row in rows for ch in row)
    return Board(cells, player, len(rows))


MID_GAME = Board.empty().apply_move(4).apply_move(0)

X_CAN_WIN = board_from(["XX.", "OO.", "..."], "X")

O_MUST_BLOCK = board_from(["XX.", ".O.", "..."], "O")


class InterfaceTests(unittest.TestCase):
    def all_agents(self):
        return [RandomAgent(), MinimaxAgent(), AlphaBetaAgent(), ExpectiminimaxAgent()]

    def test_every_agent_has_a_name_and_returns_a_legal_move(self):
        for agent in self.all_agents():
            move = agent.choose_move(MID_GAME)
            self.assertTrue(getattr(agent, "name", ""), f"{agent!r} needs a name")
            self.assertIn(move, MID_GAME.get_legal_moves(), agent.name)

    def test_agents_do_not_change_the_board_they_are_given(self):
        before = MID_GAME.cells
        for agent in self.all_agents():
            agent.choose_move(MID_GAME)
        self.assertEqual(MID_GAME.cells, before)
        self.assertEqual(MID_GAME.current_player, "X")


class RandomAgentTests(unittest.TestCase):
    def test_always_picks_a_legal_move(self):
        agent = RandomAgent()
        board = board_from(["XO.", ".X.", "..O"])
        for _ in range(100):
            self.assertIn(agent.choose_move(board), board.get_legal_moves())

    def test_plays_the_only_remaining_square(self):
        board = board_from(["XOX", "XOO", "OX."])
        self.assertEqual(RandomAgent().choose_move(board), 8)

    def test_same_seed_gives_the_same_moves(self):
        board = Board.empty()
        first = RandomAgent(random.Random(3))
        second = RandomAgent(random.Random(3))
        self.assertEqual([first.choose_move(board) for _ in range(20)],
                         [second.choose_move(board) for _ in range(20)])

    def test_full_board_has_no_move_to_choose(self):
        board = board_from(["XOX", "XOO", "OXX"])
        with self.assertRaises(ValueError):
            RandomAgent().choose_move(board)


class SearchAgentTacticsTests(unittest.TestCase):
    def test_minimax_takes_an_immediate_win(self):
        self.assertEqual(MinimaxAgent().choose_move(X_CAN_WIN), 2)

    def test_alphabeta_takes_an_immediate_win(self):
        self.assertEqual(AlphaBetaAgent().choose_move(X_CAN_WIN), 2)

    def test_expectiminimax_takes_an_immediate_win(self):
        self.assertEqual(ExpectiminimaxAgent().choose_move(X_CAN_WIN), 2)

    def test_minimax_blocks_an_immediate_loss(self):
        self.assertEqual(MinimaxAgent().choose_move(O_MUST_BLOCK), 2)

    def test_alphabeta_blocks_an_immediate_loss(self):
        self.assertEqual(AlphaBetaAgent().choose_move(O_MUST_BLOCK), 2)


class StatisticsTests(unittest.TestCase):
    def test_search_agents_record_statistics(self):
        for agent in (MinimaxAgent(), AlphaBetaAgent(), ExpectiminimaxAgent()):
            self.assertIsNone(agent.last_stats)
            agent.choose_move(MID_GAME)
            self.assertGreater(agent.last_stats.nodes_evaluated, 0, agent.name)
            self.assertGreaterEqual(agent.last_stats.elapsed, 0.0, agent.name)

    def test_alphabeta_evaluates_fewer_nodes_than_minimax(self):
        minimax, alphabeta = MinimaxAgent(), AlphaBetaAgent()
        move_a = minimax.choose_move(MID_GAME)
        move_b = alphabeta.choose_move(MID_GAME)
        self.assertLess(alphabeta.last_stats.nodes_evaluated,
                        minimax.last_stats.nodes_evaluated)
        self.assertGreater(alphabeta.last_stats.prunes, 0)
        self.assertEqual(move_a, move_b)   


class AlphaBetaOptionTests(unittest.TestCase):
    def test_tree_is_only_recorded_when_requested(self):
        quiet = AlphaBetaAgent()
        quiet.choose_move(MID_GAME)
        self.assertIsNone(quiet.last_tree)

        recording = AlphaBetaAgent(record_tree=True)
        recording.choose_move(MID_GAME)
        self.assertIsNotNone(recording.last_tree)
        self.assertTrue(recording.last_tree.children)

    def test_depth_limited_search_uses_the_heuristic(self):
        calls = []

        def spy(state, player):
            calls.append(1)
            return line_heuristic(state, player)

        agent = AlphaBetaAgent(max_depth=2, heuristic=spy)
        move = agent.choose_move(Board.empty(4))
        self.assertIn(move, Board.empty(4).get_legal_moves())
        self.assertTrue(calls)

    def test_depth_limit_reduces_the_search(self):
        shallow, deep = AlphaBetaAgent(max_depth=2, heuristic=line_heuristic), AlphaBetaAgent()
        shallow.choose_move(MID_GAME)
        deep.choose_move(MID_GAME)
        self.assertLess(shallow.last_stats.nodes_evaluated, deep.last_stats.nodes_evaluated)


class ExpectiminimaxAgentTests(unittest.TestCase):
    def test_default_probability_is_ninety_percent(self):
        self.assertAlmostEqual(ExpectiminimaxAgent().intended_probability, 0.90)

    def test_probability_is_configurable(self):
        self.assertAlmostEqual(ExpectiminimaxAgent(0.75).intended_probability, 0.75)


class HeadToHeadTests(unittest.TestCase):
    def test_alphabeta_against_itself_is_a_draw(self):
        result = play_game(AlphaBetaAgent(), AlphaBetaAgent())
        self.assertTrue(result.is_draw)

    def test_alphabeta_never_loses_to_random(self):
        for game in range(10):
            ab, rnd = AlphaBetaAgent(), RandomAgent(random.Random(game))
            x, o = (ab, rnd) if game % 2 == 0 else (rnd, ab)
            result = play_game(x, o)
            ab_mark = "X" if game % 2 == 0 else "O"
            self.assertNotEqual(result.winner, "O" if ab_mark == "X" else "X",
                                f"Alpha-Beta lost game {game}")


if __name__ == "__main__":
    unittest.main()