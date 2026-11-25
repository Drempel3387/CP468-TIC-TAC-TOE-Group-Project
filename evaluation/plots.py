"""Draws charts of the benchmark results for the report and poster"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from evaluation.metrics import read_rows, summarize

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "data" / "games.csv"
FIG_DIR = ROOT / "results" / "figures"

COLORS = {"win": "#4caf50", "draw": "#9e9e9e", "loss": "#e53935"}


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved {path}")


def plot_win_draw_loss(summary, stochastic, filename):
    rows = sorted((s for s in summary
                   if s["stochastic"] == int(stochastic) and s["agent"] != "Random"),
                  key=lambda s: (s["agent"], s["opponent"]))
    if not rows:
        return
    labels = [f"{s['agent']}\nvs {s['opponent']}\n(n={s['games']})" for s in rows]
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(rows)), 5))
    bottom = [0.0] * len(rows)
    for key in ("win", "draw", "loss"):
        vals = [s[f"{key}_pct"] for s in rows]
        ax.bar(labels, vals, bottom=bottom, label=key.capitalize(), color=COLORS[key])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of games (from first agent's point of view)")
    ax.set_ylim(0, 100)
    mode = "Stochastic (90/10)" if stochastic else "Standard"
    ax.set_title(f"Win / draw / loss rates, {mode} 3x3 Tic-Tac-Toe")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    _save(fig, filename)


def _per_agent(rows, field):
    """Average of `field` per move for each (agent, mode); skips agents with no data."""
    total = defaultdict(float)
    moves = defaultdict(int)
    for r in rows:
        key = (r["agent"], r["stochastic"])
        total[key] += r[field]
        moves[key] += r["moves"]
    return {k: total[k] / moves[k] for k in total if moves[k] and total[k] > 0}


def plot_per_move(rows, field, ylabel, title, filename, scale=1.0, log=False):
    data = _per_agent(rows, field)
    if not data:
        return
    keys = sorted(data)
    labels = [f"{a}\n({'stochastic' if s else 'standard'})" for a, s in keys]
    vals = [data[k] * scale for k in keys]
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * len(keys)), 5))
    bars = ax.bar(labels, vals, color="#5c6bc0")
    if log:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,.1f}",
                ha="center", va="bottom", fontsize=9)
    _save(fig, filename)


def plot_gemini():
    """Gemini win/draw/loss per opponent for each thinking level (gemini_games_<level>.csv)."""
    import csv
    levels = [(lvl, ROOT / "results" / "data" / f"gemini_games_{lvl}.csv") for lvl in ("minimal", "low")]
    levels = [(lvl, p) for lvl, p in levels if p.exists()]
    if not levels:
        return
    groups = []
    for lvl, path in levels:
        rows = list(csv.DictReader(open(path)))
        for opp in dict.fromkeys(r["opponent"] for r in rows):
            rs = [r for r in rows if r["opponent"] == opp]
            groups.append((f"thinking={lvl}\nvs {opp}\n(n={len(rs)})",
                           {k: 100.0 * sum(r["result"] == k for r in rs) / len(rs)
                            for k in ("win", "draw", "loss")}))
    fig, ax = plt.subplots(figsize=(max(6, 2.2 * len(groups)), 5))
    bottom = [0.0] * len(groups)
    for key in ("win", "draw", "loss"):
        vals = [g[1][key] for g in groups]
        ax.bar([g[0] for g in groups], vals, bottom=bottom, label=key.capitalize(), color=COLORS[key])
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of games (forfeits = losses)")
    ax.set_ylim(0, 100)
    ax.set_title("Gemini 3 Flash: win / draw / loss by thinking level")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=3)
    _save(fig, "gemini_win_draw_loss.png")


def main():
    rows = read_rows(CSV_PATH)
    summary = summarize(rows)
    plot_win_draw_loss(summary, False, "win_draw_loss_standard.png")
    plot_win_draw_loss(summary, True, "win_draw_loss_stochastic.png")
    plot_per_move(rows, "nodes", "Average nodes evaluated per move (log scale)",
                  "Search effort per move", "nodes_per_move.png", log=True)
    plot_per_move(rows, "time_s", "Average time per move (ms, log scale)",
                  "Thinking time per move", "time_per_move.png",
                  scale=1000.0, log=True)
    plot_gemini()


if __name__ == "__main__":
    main()