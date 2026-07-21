"""Stage 0 — see the structure we're about to exploit.

Draws the CIFAR-100 taxonomy (20 superclasses -> 100 fine classes) to
runs/hierarchy_tree.png. No training, no download — pure structure.

    uv run python s0_tree.py
"""

from viz import draw_tree

if __name__ == "__main__":
    path = draw_tree()
    print(f"wrote {path}")
