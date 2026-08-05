"""Render the pipeline architecture diagram (clean left-to-right flowchart).

Produces dissertation/presentation/architecture.png. Drawn with matplotlib so it
needs no Graphviz or Mermaid binary. Edit the box text here and re-run.

    python dissertation/presentation/make_architecture.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "architecture.png"

# Plain, professional palette: one teal accent, dark slate ink, light fills.
INK = "#222831"
COMP_EDGE = "#2b6777"
COMP_FILL = "#ffffff"
IO_FILL = "#e7eff1"
OUT_FILL = "#2b6777"
SUB_FILL = "#f3f5f6"
SUB_EDGE = "#9bb3bb"
ARROW = "#4a5a63"

W, H = 17.5, 6.8


def box(ax, cx, cy, w, h, title, *, fill, edge, tcolor, fs=10.5, bold=True, sub=None):
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(p)
    ty = cy + (0.16 if sub else 0)
    ax.text(cx, ty, title, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tcolor, zorder=3)
    if sub:
        ax.text(cx, cy - 0.32, sub, ha="center", va="center", fontsize=8,
                color=tcolor, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=ARROW, lw=1.7, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, zorder=1,
        shrinkA=0, shrinkB=0,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    n = 8
    margin = 0.3
    gap = 0.44
    bw = (W - 2 * margin - (n - 1) * gap) / n
    bh = 1.3
    ys = 4.7  # spine centre
    xs = [margin + i * (bw + gap) + bw / 2 for i in range(n)]

    # Spine: input, six numbered components, output.
    spine = [
        ("Student\nsubmission", "io"),
        ("1  AI text\ndetector", "comp"),
        ("2  Explainability", "comp"),
        ("3  Argument\nmining", "comp"),
        ("4  Question\ngeneration", "comp"),
        ("5  Bloom's\nclassifier", "comp"),
        ("6  Output\nassembler", "comp"),
        ("Verification\nInterview Guide", "out"),
    ]
    for x, (title, kind) in zip(xs, spine):
        if kind == "io":
            box(ax, x, ys, bw, bh, title, fill=IO_FILL, edge="#9bb3bb", tcolor=INK)
        elif kind == "out":
            box(ax, x, ys, bw, bh, title, fill=OUT_FILL, edge=OUT_FILL, tcolor="white")
        else:
            box(ax, x, ys, bw, bh, title, fill=COMP_FILL, edge=COMP_EDGE, tcolor=INK)

    # Spine arrows.
    for i in range(n - 1):
        arrow(ax, xs[i] + bw / 2, ys, xs[i + 1] - bw / 2, ys)

    # Lower tier: exploded internals for the detector and question generation.
    ysub = 2.35
    sbh = 0.95
    sbw = 1.5

    def sub_pair(parent_x, left_text, right_text):
        lx = parent_x - 0.85
        rx = parent_x + 0.85
        box(ax, lx, ysub, sbw, sbh, left_text, fill=SUB_FILL, edge=SUB_EDGE,
            tcolor=INK, fs=8.2, bold=False)
        box(ax, rx, ysub, sbw, sbh, right_text, fill=SUB_FILL, edge=SUB_EDGE,
            tcolor=INK, fs=8.2, bold=False)
        # Arrows up into the parent spine box.
        arrow(ax, lx, ysub + sbh / 2, parent_x - 0.25, ys - bh / 2, lw=1.3)
        arrow(ax, rx, ysub + sbh / 2, parent_x + 0.25, ys - bh / 2, lw=1.3)

    # Detector is hybrid: transformer + stylometric features.
    sub_pair(xs[1], "DeBERTa-v3\ntransformer", "Stylometric\nfeatures")
    ax.text(xs[1], ysub + sbh / 2 + 0.15, "hybrid", ha="center", va="center",
            fontsize=7.5, color="#7a8a92")

    # Question generation: two backends, compared (a core contribution).
    # Backend B moved from Llama 3 8B to Qwen2.5 3B on the fit-probe evidence, signed off by the
    # supervisor on 3 July 2026. The diagram carried the old name until the pre-final deck.
    sub_pair(xs[4], "A: commercial\nLLM (API)", "B: Qwen2.5 3B\n(QLoRA)")
    ax.text(xs[4], ysub + sbh / 2 + 0.15, "two backends", ha="center",
            va="center", fontsize=7.5, color="#7a8a92")

    # Headline and caption.
    ax.text(W / 2, H - 0.35,
            "Explainable pipeline for academic integrity verification",
            ha="center", va="center", fontsize=13, fontweight="bold", color=INK)
    ax.text(W / 2, H - 0.78,
            "Steps 2 to 6 run on submissions the detector flags as AI.",
            ha="center", va="center", fontsize=9.5, color="#52616b")
    ax.text(W / 2, 0.55,
            "Two-class detection (Human vs AI). Each stage feeds the next; the "
            "explanation, argument provenance and Bloom's labels accumulate into "
            "the lecturer's guide.",
            ha="center", va="center", fontsize=9.5, color="#52616b")

    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
