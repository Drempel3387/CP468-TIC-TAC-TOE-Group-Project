"""Scalability study: how search cost grows with board size and search depth.

Part 1 (cost):  from the empty board, depth-limited Minimax vs Alpha-Beta on 3x3, 4x4, 5x5.
Part 2 (play):  depth-limited Alpha-Beta with a heuristic vs Random / a shallow searcher.

Writes results/data/scalability_cost.csv and results/data/scalability_games.csv.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from agents.alphabeta_agent import AlphaBetaAgent
from agents.random_agent import RandomAgent
from algorithms.alphabeta import alphabeta_search
from algorithms.heuristics import line_heuristic
from algorithms.stats import SearchStats
from evaluation.metrics import MeteredAgent
from game.board import Board
from game.tictactoe import play_game

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "data"

MINIMAX_NODE_CAP = 8_000_000   # skip plain minimax runs expected to exceed this


def minimax_limited(state, player, depth, max_depth, is_max, stats):
    """Plain depth-limited Minimax (no pruning), same scoring as the Alpha-Beta search."""
    stats.nodes_evaluated += 1
    if state.is_terminal():
        return state.utility(player)
    if depth >= max_depth:
        return line_heuristic(state, player)
    best = float("-inf") if is_max else float("inf")
    for move in state.get_legal_moves():
        v = minimax_limited(state.apply_move(move), player, depth + 1, max_depth, not is_max, stats)
        best = max(best, v) if is_max else min(best, v)
    return best


def plain_minimax_root(state, depth):
    stats = SearchStats()
    stats.start()
    stats.nodes_evaluated += 1
    for move in state.get_legal_moves():
        minimax_limited(state.apply_move(move), state.current_player, 1, depth, False, stats)
    stats.stop()
    return stats


def cost_study(max_depth_by_size):
    rows = []
    for size, depths in max_depth_by_size.items():
        board = Board.empty(size)
        branching = size * size
        for d in depths:
            # a depth-d search from the empty board has at most b*(b-1)*...*(b-d+1) leaves
            est = 1
            for i in range(d):
                est *= branching - i
            if est <= MINIMAX_NODE_CAP:
                mm = plain_minimax_root(board, d)
                mm_nodes, mm_time = mm.nodes_evaluated, mm.elapsed
            else:
                mm_nodes = mm_time = ""
            _, _, ab, _ = alphabeta_search(board, "X", max_depth=d, heuristic=line_heuristic)
            rows.append({"board_size": size, "depth": d, "minimax_nodes": mm_nodes,
                         "minimax_time_s": round(mm_time, 4) if mm_time != "" else "",
                         "alphabeta_nodes": ab.nodes_evaluated,
                         "alphabeta_time_s": round(ab.elapsed, 4)})
            print(f"  {size}x{size} depth {d}: minimax={mm_nodes or 'skipped'} "
                  f"alpha-beta={ab.nodes_evaluated} ({ab.elapsed:.2f}s)", flush=True)
    return rows


def play_study(games, rng):
    """Depth-limited Alpha-Beta (heuristic) against Random and against a depth-1 searcher."""
    rows = []
    for size, depth in ((4, 3), (5, 3)):
        for opp_label, make_opp in (("Random", RandomAgent),
                                    ("Alpha-Beta d=1", lambda: AlphaBetaAgent(1, line_heuristic))):
            label = f"Alpha-Beta d={depth}"
            print(f"  {size}x{size}: {label} vs {opp_label}", flush=True)
            for i in range(games):
                a = MeteredAgent(AlphaBetaAgent(depth, line_heuristic), label)
                b = MeteredAgent(make_opp(), opp_label)
                a_is_x = i % 2 == 0
                x, o = (a, b) if a_is_x else (b, a)
                res = play_game(x, o, size=size, rng=rng)
                seat = "X" if a_is_x else "O"
                outcome = "draw" if res.winner is None else ("win" if res.winner == seat else "loss")
                rows.append({"board_size": size, "agent": label, "opponent": opp_label,
                             "seat": seat, "result": outcome, "moves": a.moves,
                             "nodes": a.nodes, "time_s": round(a.time_s, 4)})
    return rows


def write(rows, name):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=468)
    args = parser.parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)

    print("== Cost from the empty board ==")
    write(cost_study({3: [1, 2, 3, 4, 5, 6, 7, 8, 9], 4: [1, 2, 3, 4, 5], 5: [1, 2, 3, 4, 5]}),
          "scalability_cost.csv")
    print("== Play strength ==")
    write(play_study(args.games, rng), "scalability_games.csv")


if __name__ == "__main__":
    main()
