"""Records and summarizes game results, search effort, and move times"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

from game.tictactoe import play_game

FIELDS = ["agent", "opponent", "seat", "stochastic", "board_size",
          "result", "moves", "nodes", "time_s"]


class MeteredAgent:
    """Wraps any agent and records time and nodes for every choose_move call."""

    def __init__(self, agent, label):
        self.agent = agent
        self.label = label
        self.moves = 0
        self.nodes = 0
        self.time_s = 0.0

    def choose_move(self, state):
        start = time.perf_counter()
        move = self.agent.choose_move(state)
        self.time_s += time.perf_counter() - start
        self.moves += 1
        stats = getattr(self.agent, "last_stats", None)
        if stats is not None:
            self.nodes += stats.nodes_evaluated
        return move


def play_metered(label_a, agent_a, label_b, agent_b, a_is_x=True,
                 size=3, stochastic=False, rng=None):
    """Play one game and return two result rows (one per agent)."""
    a = MeteredAgent(agent_a, label_a)
    b = MeteredAgent(agent_b, label_b)
    x, o = (a, b) if a_is_x else (b, a)
    result = play_game(x, o, size=size, stochastic=stochastic, rng=rng)

    rows = []
    for me, opp, seat in ((a, b, "X" if a_is_x else "O"),
                          (b, a, "O" if a_is_x else "X")):
        if result.winner is None:
            outcome = "draw"
        elif result.winner == seat:
            outcome = "win"
        else:
            outcome = "loss"
        rows.append({
            "agent": me.label, "opponent": opp.label, "seat": seat,
            "stochastic": int(stochastic), "board_size": size,
            "result": outcome, "moves": me.moves,
            "nodes": me.nodes, "time_s": round(me.time_s, 6),
        })
    return rows


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["stochastic"] = int(r["stochastic"])
        r["board_size"] = int(r["board_size"])
        r["moves"] = int(r["moves"])
        r["nodes"] = int(r["nodes"])
        r["time_s"] = float(r["time_s"])
    return rows


def summarize(rows):
    """Group rows by (agent, opponent, stochastic, board_size) and compute rates."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["agent"], r["opponent"], r["stochastic"], r["board_size"])].append(r)

    summary = []
    for (agent, opp, stoch, size), rs in groups.items():
        n = len(rs)
        total_moves = sum(r["moves"] for r in rs) or 1
        summary.append({
            "agent": agent, "opponent": opp, "stochastic": stoch,
            "board_size": size, "games": n,
            "win_pct": 100 * sum(r["result"] == "win" for r in rs) / n,
            "draw_pct": 100 * sum(r["result"] == "draw" for r in rs) / n,
            "loss_pct": 100 * sum(r["result"] == "loss" for r in rs) / n,
            "avg_nodes_per_move": sum(r["nodes"] for r in rs) / total_moves,
            "avg_ms_per_move": 1000 * sum(r["time_s"] for r in rs) / total_moves,
        })
    return summary


def print_summary(summary):
    header = (f"{'Agent':<15}{'vs':<15}{'Mode':<12}{'Games':>6}"
              f"{'Win%':>8}{'Draw%':>8}{'Loss%':>8}{'Nodes/move':>14}{'ms/move':>10}")
    print(header)
    print("-" * len(header))
    for s in sorted(summary, key=lambda s: (s["stochastic"], s["agent"], s["opponent"])):
        mode = "stochastic" if s["stochastic"] else "standard"
        print(f"{s['agent']:<15}{s['opponent']:<15}{mode:<12}{s['games']:>6}"
              f"{s['win_pct']:>8.1f}{s['draw_pct']:>8.1f}{s['loss_pct']:>8.1f}"
              f"{s['avg_nodes_per_move']:>14.1f}{s['avg_ms_per_move']:>10.2f}")