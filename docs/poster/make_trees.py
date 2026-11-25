"""Generates the three game-tree SVGs for the poster from the real project code.

Run from the project root:  python3 docs/poster/make_trees.py
Position: X to move, three empty squares (5, 7, 8 as the board prints them).
Values are from X's perspective (+1 win, 0 draw, -1 loss).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.board import Board
from game.stochastic import move_outcomes
from algorithms.alphabeta import alphabeta_search
from algorithms.minimax import minimax_search

OUT = Path(__file__).parent / "img"
CELLS = ("X", "O", "X", "O", None, "X", None, None, "O")
START = Board(CELLS, "X")
ME = "X"

C_MAX, C_MIN, C_CH = "#1d4e89", "#b3541e", "#6a4c93"
C_PRUNE = "#c0392b"


def n(m):  # 0-based cell -> label the board prints
    return str(m + 1)


def val_txt(v):
    return f"{v:+.2f}".replace("+0.00", "0.00") if isinstance(v, float) and v != int(v) else f"{int(v):+d}".replace("+0", "0")


# ---------- tree builders: dict(kind, value, edge, children, pruned, note) ----------
def build_minimax(state, is_max, edge=None):
    if state.is_terminal():
        return dict(kind="leaf", value=state.utility(ME), edge=edge, children=[])
    kids = [build_minimax(state.apply_move(m), not is_max, n(m)) for m in state.get_legal_moves()]
    vals = [k["value"] for k in kids]
    return dict(kind="max" if is_max else "min", value=max(vals) if is_max else min(vals),
                edge=edge, children=kids)


def build_ab(state, node, edge=None):
    """Convert the project's recorded TreeNode tree to our dict form."""
    if node.pruned:
        return dict(kind="pruned", value=None, edge=edge, children=[])
    if not node.children:
        return dict(kind="leaf", value=node.value, edge=edge, children=[])
    kids = []
    for ch in node.children:
        if ch.pruned:
            kids.append(dict(kind="pruned", value=None, edge=n(ch.move), children=[]))
        else:
            kids.append(build_ab(state, ch, n(ch.move)))
    return dict(kind="max" if node.is_max else "min", value=node.value, edge=edge,
                children=kids, note=f"α={fmt(node.alpha)} β={fmt(node.beta)}")


def fmt(x):
    return "-∞" if x == float("-inf") else "∞" if x == float("inf") else str(int(x))


def expecti_value(state, is_max, depth_left):
    """Return a dict tree down to depth_left player moves, with exact values below."""
    def full(s, mx):
        if s.is_terminal():
            return s.utility(ME)
        best = None
        for m in s.get_legal_moves():
            ev = sum(p * full(s.apply_move(a), not mx) for a, p in move_outcomes(s, m))
            best = ev if best is None else (max(best, ev) if mx else min(best, ev))
        return best

    def build(s, mx, d, edge=None):
        if s.is_terminal() or d == 0:
            return dict(kind="leaf", value=full(s, mx), edge=edge, children=[],
                        exact=not s.is_terminal())
        kids = []
        for m in s.get_legal_moves():
            outs = []
            for a, p in move_outcomes(s, m):
                outs.append(build(s.apply_move(a), not mx, d - 1, f"{p:.2f}"[1:]))
            ev = sum(p * o["value"] for (a, p), o in zip(move_outcomes(s, m), outs))
            kids.append(dict(kind="chance", value=ev, edge=n(m), children=outs))
        vals = [k["value"] for k in kids]
        return dict(kind="max" if mx else "min", value=max(vals) if mx else min(vals),
                    edge=edge, children=kids)
    return build(state, is_max, depth_left)


# ---------- layout + SVG ----------
LEAF_W, LEVEL_H, R = 52, 66, 15


def layout(node, depth=0, x0=0):
    node["depth"] = depth
    if not node["children"]:
        node["x"], node["w"] = x0 + LEAF_W / 2, LEAF_W
        return LEAF_W
    w, cx = 0, []
    for ch in node["children"]:
        cw = layout(ch, depth + 1, x0 + w)
        cx.append(ch["x"])
        w += cw
    node["x"], node["w"] = (cx[0] + cx[-1]) / 2, w
    return w


def shape(node, x, y, color):
    k = node["kind"]
    if k == "max":
        pts = f"{x},{y-R-2} {x-R-3},{y+R-2} {x+R+3},{y+R-2}"
        return f'<polygon points="{pts}" fill="#fff" stroke="{color}" stroke-width="2.5"/>'
    if k == "min":
        pts = f"{x},{y+R+2} {x-R-3},{y-R+2} {x+R+3},{y-R+2}"
        return f'<polygon points="{pts}" fill="#fff" stroke="{color}" stroke-width="2.5"/>'
    if k == "chance":
        return f'<circle cx="{x}" cy="{y}" r="{R}" fill="#f3ecfa" stroke="{C_CH}" stroke-width="2.5"/>'
    if k == "pruned":
        return (f'<rect x="{x-R}" y="{y-R}" width="{2*R}" height="{2*R}" rx="3" fill="#fdecea" '
                f'stroke="{C_PRUNE}" stroke-width="2" stroke-dasharray="4 3"/>'
                f'<path d="M{x-8},{y-8} L{x+8},{y+8} M{x+8},{y-8} L{x-8},{y+8}" stroke="{C_PRUNE}" stroke-width="2.5"/>')
    return f'<rect x="{x-R+2}" y="{y-R+2}" width="{2*R-4}" height="{2*R-4}" rx="3" fill="#eef2f6" stroke="#6b7785" stroke-width="2"/>'


def render(root, title_note=""):
    layout(root)
    width = root["w"] + 40
    maxd = 0
    stack = [root]
    while stack:
        nd = stack.pop()
        maxd = max(maxd, nd["depth"])
        stack += nd["children"]
    height = maxd * LEVEL_H + 90
    parts = []

    def pos(nd):
        return nd["x"] + 20, nd["depth"] * LEVEL_H + 30

    def walk(nd):
        x, y = pos(nd)
        for ch in nd["children"]:
            cx, cy = pos(ch)
            col = C_PRUNE if ch["kind"] == "pruned" else "#8a96a3"
            dash = ' stroke-dasharray="5 4"' if ch["kind"] == "pruned" else ""
            parts.append(f'<line x1="{x}" y1="{y+R}" x2="{cx}" y2="{cy-R}" stroke="{col}" stroke-width="2"{dash}/>')
            if ch.get("edge"):
                mx, my = (x + cx) / 2, (y + cy) / 2
                parts.append(f'<text x="{mx+(-6 if cx<x else 6 if cx>x else 8)}" y="{my}" font-size="15" '
                             f'fill="#4a5560" text-anchor="{"end" if cx<x else "start"}">{ch["edge"]}</text>')
        for ch in nd["children"]:
            walk(ch)
        color = {"max": C_MAX, "min": C_MIN, "chance": C_CH}.get(nd["kind"], "#333")
        parts.append(shape(nd, x, y, color))
        if nd["value"] is not None:
            v = nd["value"]
            txt = f"{v:g}" if abs(v - round(v)) < 1e-9 else f"{v:.2f}"
            parts.append(f'<text x="{x}" y="{y+5}" font-size="14" font-weight="700" text-anchor="middle" '
                         f'fill="#1a1a1a">{txt}</text>')
        if nd.get("note"):
            parts.append("")

    walk(root)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
            f'font-family="Helvetica, Arial, sans-serif">{"".join(parts)}</svg>')


def main():
    OUT.mkdir(exist_ok=True)

    # sanity check against the real algorithms
    mm_move, mm_val, mm_stats = minimax_search(START, ME)[:3]
    ab_move, ab_val, ab_stats, ab_root = alphabeta_search(START, ME, record_tree=True)
    assert mm_val == ab_val == 0, (mm_val, ab_val)

    mm = build_minimax(START, True)
    (OUT / "tree_minimax.svg").write_text(render(mm))
    ab = build_ab(START, ab_root)
    (OUT / "tree_alphabeta.svg").write_text(render(ab))
    ex = expecti_value(START, True, 1)
    (OUT / "tree_expectiminimax.svg").write_text(render(ex))

    print("Minimax nodes:", mm_stats.nodes_evaluated,
          "| Alpha-Beta nodes:", ab_stats.nodes_evaluated, "prunes:", ab_stats.prunes,
          "pruned branches:", getattr(ab_stats, "pruned_branches", "?"))
    print("Expectiminimax root value:", ex["value"])


if __name__ == "__main__":
    main()
