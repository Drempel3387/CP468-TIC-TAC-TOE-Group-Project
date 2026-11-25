"""Plays Gemini against the search agents and saves results to results/data/gemini_games.csv.

Needs GEMINI_API_KEY in the environment (optionally GEMINI_MODEL). An illegal move or
an API failure that survives retries is a forfeit, counted as a loss for Gemini.
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

from agents.alphabeta_agent import AlphaBetaAgent
from agents.expectiminimax_agent import ExpectiminimaxAgent
from agents.llm_agent import LLMAgent
from agents.random_agent import RandomAgent
from api.gemini_client import GeminiClient
from game.tictactoe import play_game

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "results" / "data" / "gemini_games.csv"
FIELDS = ["agent", "opponent", "seat", "stochastic", "result", "forfeit",
          "moves", "gemini_time_s"]

# (opponent label, builder, stochastic?)
MATCHUPS = [
    ("Random", RandomAgent, False),
    ("Alpha-Beta", AlphaBetaAgent, False),
    ("Expectiminimax", ExpectiminimaxAgent, True),
]


class Forfeit(Exception):
    pass


class GeminiPlayer:
    """LLMAgent wrapper with retries; raises Forfeit on illegal moves or API failure."""

    def __init__(self, model, thinking, retries=12, pause=1.0):
        self.agent = LLMAgent(GeminiClient(model=model, timeout=120, thinking_level=thinking))
        self.retries, self.pause = retries, pause
        self.moves = 0
        self.time_s = 0.0

    def choose_move(self, state):
        start = time.perf_counter()
        try:
            for attempt in range(self.retries):
                try:
                    move = self.agent.choose_move(state)
                    self.moves += 1
                    return move
                except RuntimeError as exc:
                    msg = str(exc)
                    if "illegal move" in msg or "did not contain" in msg:
                        raise Forfeit(msg)
                    if any(c in msg for c in ("HTTP 429", "HTTP 503", "HTTP 500", "Could not reach")):
                        print(f"    retry {attempt + 1}: {msg[:70].replace(chr(10), ' ')}", flush=True)
                        time.sleep(20 if 'HTTP 429' in msg else min(30, 2 ** (attempt + 1)))
                        continue
                    raise Forfeit(msg)
            raise Forfeit("API unavailable after retries")
        finally:
            self.time_s += time.perf_counter() - start
            time.sleep(self.pause)  # stay under free-tier rate limits


def play_one(model, thinking, pause, opp_label, opp_builder, stochastic, gemini_is_x, rng):
    gem, opp = GeminiPlayer(model, thinking, pause=pause), opp_builder()
    x, o = (gem, opp) if gemini_is_x else (opp, gem)
    seat = "X" if gemini_is_x else "O"
    forfeit = 0
    try:
        winner = play_game(x, o, stochastic=stochastic, rng=rng).winner
        result = "draw" if winner is None else ("win" if winner == seat else "loss")
    except Forfeit as exc:
        forfeit, result = 1, "loss"
        print(f"    forfeit: {str(exc)[:100]}")
    return {"agent": "Gemini", "opponent": opp_label, "seat": seat,
            "stochastic": int(stochastic), "result": result, "forfeit": forfeit,
            "moves": gem.moves, "gemini_time_s": round(gem.time_s, 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20, help="games per matchup")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--rpm", type=float, default=5, help="free-tier requests per minute for the model")
    parser.add_argument("--thinking", default="low", help="thinkingLevel: minimal, low, medium, high")
    parser.add_argument("--out", default=str(OUT_CSV))
    parser.add_argument("--seed", type=int, default=468)
    args = parser.parse_args()

    if not GeminiClient(model=args.model).is_configured:
        raise SystemExit("Set GEMINI_API_KEY first.")
    random.seed(args.seed)
    rng = random.Random(args.seed)

    rows = []
    for label, builder, stochastic in MATCHUPS:
        print(f"== Gemini ({args.model}, thinking={args.thinking}) vs {label} "
              f"({'stochastic' if stochastic else 'standard'}) ==", flush=True)
        for i in range(args.games):
            row = play_one(args.model, args.thinking, 60.0 / args.rpm + 0.5, label, builder, stochastic, i % 2 == 0, rng)
            rows.append(row)
            print(f"  game {i + 1}/{args.games}: {row['result']}", flush=True)
        # save progress after each matchup
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)

    print("\nOpponent        Mode        Games  Win  Draw  Loss  Forfeits")
    for label, _, stochastic in MATCHUPS:
        rs = [r for r in rows if r["opponent"] == label]
        c = lambda k: sum(r["result"] == k for r in rs)
        print(f"{label:<15} {'stochastic' if stochastic else 'standard':<11} {len(rs):>5}"
              f"{c('win'):>5}{c('draw'):>6}{c('loss'):>6}{sum(r['forfeit'] for r in rs):>9}")


if __name__ == "__main__":
    main()
