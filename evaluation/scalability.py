"""Scalability test: how search cost grows with board size and search depth."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from algorithms.alphabeta import alphabeta_search
from algorithms.heuristics import line_heuristic
from algorithms.stats import SearchStats
from game.board import Board

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "data" / "scalability.csv"
FIG_DIR = ROOT / "results" / "figures"
FIELDS = ["board_size", "algorithm", "depth", "nodes", "time_s", "best_move", "value"]
ALGOS = ("Minimax", "Alpha-Beta")


def minimax_depth_limited(state, player, max_depth, heuristic):
    """Plain Minimax (no pruning) that stops at max_depth and uses a heuristic."""
    stats = SearchStats()
    stats.start()

    def value(s, depth, is_max):
        stats.nodes_evaluated += 1
        if s.is_terminal():
            stats.terminal_evals += 1
            return s.utility(player)
        if depth >= max_depth:
            stats.terminal_evals += 1
            return heuristic(s, player)
        best = float("-inf") if is_max else float("inf")
        for move in s.get_legal_moves():
            v = value(s.apply_move(move), depth + 1, not is_max)
            best = max(best, v) if is_max else min(best, v)
        return best

    stats.nodes_evaluated += 1  # the root, counted the same way as alphabeta_search
    best_move, best_value = None, float("-inf")
    for move in state.get_legal_moves():
        v = value(state.apply_move(move), 1, False)
        if v > best_value:
            best_move, best_value = move, v
    stats.stop()
    return best_move, best_value, stats


def measure(algo, size, depth):
    state = Board.empty(size)
    player = state.current_player
    if algo == "Alpha-Beta":
        move, value, stats, _ = alphabeta_search(
            state, player, max_depth=depth, heuristic=line_heuristic)
    else:
        move, value, stats = minimax_depth_limited(
            state, player, depth, line_heuristic)
    return {"board_size": size, "algorithm": algo, "depth": depth,
            "nodes": stats.nodes_evaluated, "time_s": round(stats.elapsed, 6),
            "best_move": move, "value": round(value, 4)}


def run(sizes, max_depth, time_limit):
    rows = []
    try:
        for size in sizes:
            cap = min(max_depth, size * size)
            for algo in ALGOS:
                prev = None
                for depth in range(1, cap + 1):
                    # Predict the next search's time from the last one's growth.
                    if prev is not None:
                        ratio = prev["nodes"] / max(prev_prev_nodes, 1) if prev_prev_nodes else size * size
                        if prev["time_s"] * max(ratio, 1) > time_limit:
                            print(f"  {size}x{size} {algo}: stopping before depth {depth} "
                                  f"(predicted > {time_limit:.0f}s)", flush=True)
                            break
                    row = measure(algo, size, depth)
                    rows.append(row)
                    print(f"  {size}x{size} {algo:<10} depth {depth}: "
                          f"{row['nodes']:>12,} nodes  {row['time_s']:>9.3f}s", flush=True)
                    prev_prev_nodes = prev["nodes"] if prev else None
                    prev = row
    except KeyboardInterrupt:
        print("\nStopped early; saving what was measured so far.")
    return rows


def write_csv(rows):
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {CSV_PATH}")


def read_csv():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["board_size"], r["depth"], r["nodes"] = int(r["board_size"]), int(r["depth"]), int(r["nodes"])
        r["time_s"] = float(r["time_s"])
    return rows


def plot(rows, compare_depth=3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    colors = {3: "#1e88e5", 4: "#fb8c00", 5: "#8e24aa"}
    styles = {"Minimax": ("-", "o"), "Alpha-Beta": ("--", "s")}
    sizes = sorted({r["board_size"] for r in rows})

    for field, ylabel, fname, scale in (
            ("nodes", "Nodes evaluated (log scale)", "scalability_nodes_vs_depth.png", 1),
            ("time_s", "Search time in ms (log scale)", "scalability_time_vs_depth.png", 1000)):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for size in sizes:
            for algo in ALGOS:
                pts = sorted((r["depth"], max(r[field] * scale, 1e-3))
                             for r in rows if r["board_size"] == size and r["algorithm"] == algo)
                if pts:
                    ls, mk = styles[algo]
                    ax.plot(*zip(*pts), ls, marker=mk, color=colors.get(size),
                            label=f"{size}x{size} {algo}")
        ax.set_yscale("log")
        ax.set_xlabel("Search depth (moves looked ahead)")
        ax.set_ylabel(ylabel)
        ax.set_title("Search cost vs depth (first move, empty board)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / fname, dpi=200)
        plt.close(fig)
        print(f"saved {FIG_DIR / fname}")

    # Fixed-depth comparison across board sizes.
    pick = {(r["board_size"], r["algorithm"]): r["nodes"]
            for r in rows if r["depth"] == compare_depth}
    if pick:
        fig, ax = plt.subplots(figsize=(7, 5))
        width = 0.38
        xs = range(len(sizes))
        for j, algo in enumerate(ALGOS):
            vals = [pick.get((s, algo), 0) for s in sizes]
            bars = ax.bar([x + (j - 0.5) * width for x in xs], vals, width, label=algo)
            for b, v in zip(bars, vals):
                if v:
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{s}x{s}" for s in sizes])
        ax.set_yscale("log")
        ax.set_ylabel("Nodes evaluated (log scale)")
        ax.set_title(f"Growth with board size at depth {compare_depth}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "scalability_by_board_size.png", dpi=200)
        plt.close(fig)
        print(f"saved {FIG_DIR / 'scalability_by_board_size.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--max-depth", type=int, default=9)
    parser.add_argument("--time-limit", type=float, default=60.0,
                        help="skip a search if it is predicted to take longer than this (seconds)")
    parser.add_argument("--compare-depth", type=int, default=3)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if not args.plot_only:
        rows = run(args.sizes, args.max_depth, args.time_limit)
        write_csv(rows)
    else:
        rows = read_csv()
    plot(rows, args.compare_depth)


if __name__ == "__main__":
    main()