"""Runs matches between agents and saves the results"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from agents.alphabeta_agent import AlphaBetaAgent
from agents.expectiminimax_agent import ExpectiminimaxAgent
from agents.minimax_agent import MinimaxAgent
from agents.random_agent import RandomAgent
from evaluation.metrics import play_metered, print_summary, summarize, write_rows
from game.board import Board

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "data" / "games.csv"

AGENTS = {
    "Random": RandomAgent,
    "Minimax": MinimaxAgent,
    "Alpha-Beta": AlphaBetaAgent,
    "Expectiminimax": ExpectiminimaxAgent,
}

STANDARD_MATCHUPS = [
    ("Minimax", "Random"),
    ("Alpha-Beta", "Random"),
    ("Minimax", "Alpha-Beta"),
]
STOCHASTIC_MATCHUPS = [
    ("Expectiminimax", "Random"),
    ("Expectiminimax", "Minimax"),
    ("Expectiminimax", "Alpha-Beta"),
]


def run_matchup(label_a, label_b, games, stochastic, rng):
    agent_a, agent_b = AGENTS[label_a](), AGENTS[label_b]()
    rows = []
    start = time.perf_counter()
    for i in range(games):
        rows += play_metered(label_a, agent_a, label_b, agent_b,
                             a_is_x=(i % 2 == 0), stochastic=stochastic, rng=rng)
        if (i + 1) % 10 == 0 or i + 1 == games:
            mode = "stochastic" if stochastic else "standard"
            print(f"  [{mode}] {label_a} vs {label_b}: {i + 1}/{games} games "
                  f"({time.perf_counter() - start:.0f}s)", flush=True)
    return rows


def probe():
    """Time one move on an empty board for each search agent."""
    print("Timing one move on an empty 3x3 board (Ctrl+C if one takes > 1 minute)\n")
    for label in ("Minimax", "Alpha-Beta", "Expectiminimax"):
        agent = AGENTS[label]()
        print(f"{label}...", end=" ", flush=True)
        start = time.perf_counter()
        agent.choose_move(Board.empty())
        secs = time.perf_counter() - start
        stats = getattr(agent, "last_stats", None)
        nodes = stats.nodes_evaluated if stats else "n/a"
        print(f"{secs:.2f}s, nodes evaluated: {nodes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100,
                        help="games per standard matchup (default 100)")
    parser.add_argument("--stoch-games", type=int, default=50,
                        help="games per stochastic matchup (default 50)")
    parser.add_argument("--seed", type=int, default=468)
    parser.add_argument("--probe", action="store_true",
                        help="only time one move per agent, then exit")
    parser.add_argument("--skip-stochastic", action="store_true")
    parser.add_argument("--skip-standard", action="store_true")
    args = parser.parse_args()

    if args.probe:
        probe()
        return

    random.seed(args.seed)          # RandomAgent uses the global generator
    rng = random.Random(args.seed)  # used for the 90/10 chance in stochastic games

    rows = []
    if not args.skip_standard:
        print("== Standard 3x3 Tic-Tac-Toe ==")
        for a, b in STANDARD_MATCHUPS:
            rows += run_matchup(a, b, args.games, False, rng)
    if not args.skip_stochastic:
        print("== Stochastic 3x3 Tic-Tac-Toe (90% intended / 10% elsewhere) ==")
        for a, b in STOCHASTIC_MATCHUPS:
            rows += run_matchup(a, b, args.stoch_games, True, rng)

    write_rows(rows, OUT_CSV)
    print(f"\nSaved {len(rows)} rows to {OUT_CSV}\n")
    print_summary(summarize(rows))


if __name__ == "__main__":
    main()