"""Text rendering of a recorded alpha-beta search tree, pruned branches marked."""

def render_tree(root, max_depth=3, fmt_move=str):
    """Return the tree as a string. Pruned branches are shown as [PRUNED]."""
    lines = []

    def fmt(x):
        return "-inf" if x == float("-inf") else "inf" if x == float("inf") else str(x)

    def walk(node, depth, prefix):
        kind = "MAX" if node.is_max else "MIN"
        label = "root" if node.move is None else f"move {fmt_move(node.move)}"
        if node.pruned:
            lines.append(f"{prefix}{label} [PRUNED]")
            return
        lines.append(f"{prefix}{label} {kind} value={node.value} "
                     f"(a={fmt(node.alpha)}, b={fmt(node.beta)})")
        if depth >= max_depth:
            if node.children:
                lines.append(f"{prefix}  ...")
            return
        for child in node.children:
            walk(child, depth + 1, prefix + "  ")

    walk(root, 0, "")
    return "\n".join(lines)
